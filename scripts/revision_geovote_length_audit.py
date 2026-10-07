#!/usr/bin/env python3
"""Retrospective same-pool GeoVote length-bias audit for TACL 11241.

This script must not be used to claim a new held-out result on pools that were
already inspected.  It directly answers the reviewer's confound question by
recomputing raw and length-residualized selection on exactly the same candidate
pool and attaching paired uncertainty for every method versus majority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from geoprobe.revision.geovote import (
    LinearLengthResidualizer,
    correlation_with_length,
    same_pool_outcomes,
)
from geoprobe.revision.stats import paired_binary_summary


def _parse_ids(spec: str) -> set[int]:
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, end = part.split(":", 1)
            values.update(range(int(start), int(end)))
        else:
            values.add(int(part))
    if not values:
        raise ValueError("ID specification is empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-level", type=Path, required=True)
    parser.add_argument("--calibration-ids", required=True, help="e.g. 0:50")
    parser.add_argument("--evaluation-ids", required=True, help="e.g. 50:100")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    calibration_ids = _parse_ids(args.calibration_ids)
    evaluation_ids = _parse_ids(args.evaluation_ids)
    if calibration_ids & evaluation_ids:
        raise SystemExit("calibration and evaluation IDs must be disjoint")
    frame = pd.read_parquet(args.sample_level)
    required = {"sample_id", "sample_idx", "geo_value", "geo_conf", "pred", "correct", "n_gen_tokens", "sequence_logprob"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"input lacks required columns: {sorted(missing)}")
    observed = set(frame["sample_id"].astype(int).unique())
    if not calibration_ids <= observed or not evaluation_ids <= observed:
        raise SystemExit("declared calibration/evaluation IDs are not all present in the candidate pool")
    calibration = frame[frame["sample_id"].isin(calibration_ids)].copy()
    evaluation = frame[frame["sample_id"].isin(evaluation_ids)].copy()
    residualizer = LinearLengthResidualizer.fit(calibration)
    calibration["length_residual_conf"] = residualizer.residualize(calibration)
    evaluation["length_residual_conf"] = residualizer.residualize(evaluation)

    args.out.mkdir(parents=True, exist_ok=True)

    # The locked Table 4 row was generated with inverse weights on geo_value
    # (MeanStepNorm at layer 20, sign min; geo_conf = -geo_value). Pass the
    # sign explicitly so the historical vote reproduces that CSV.
    raw_outcomes = same_pool_outcomes(evaluation, score_column="geo_conf", historical_sign="min").rename(
        columns={"score_pick": "raw_geo_pick", "residual_score_vote": "raw_score_softmax_vote"}
    )
    residual_outcomes = same_pool_outcomes(evaluation, score_column="length_residual_conf").rename(
        columns={"score_pick": "length_residual_geo_pick", "residual_score_vote": "length_residual_geo_vote"}
    )
    keep_raw = ["sample_id", "first", "majority", "shortest", "longest", "raw_geo_pick", "historical_geovote", "logprob_weighted", "oracle"]
    keep_residual = ["sample_id", "length_residual_geo_pick", "length_residual_geo_vote"]
    outcomes = raw_outcomes[keep_raw].merge(residual_outcomes[keep_residual], on="sample_id", validate="one_to_one")
    outcomes.to_parquet(args.out / "per_problem_outcomes.parquet", index=False)

    comparison_rows = []
    baseline = outcomes["majority"].astype(int).to_numpy()
    for method in outcomes.columns:
        if method in {"sample_id", "majority", "oracle"}:
            continue
        summary = paired_binary_summary(baseline, outcomes[method].astype(int).to_numpy())
        comparison_rows.append({"method": method, **summary.__dict__})
    comparison = pd.DataFrame(comparison_rows).sort_values("delta", ascending=False)
    comparison.to_csv(args.out / "paired_comparisons_vs_majority.csv", index=False)

    candidate_audit = pd.DataFrame(
        [
            {"partition": "calibration", "score": "geo_conf", "spearman_vs_length": correlation_with_length(calibration, "geo_conf")},
            {"partition": "evaluation", "score": "geo_conf", "spearman_vs_length": correlation_with_length(evaluation, "geo_conf")},
            {"partition": "calibration", "score": "length_residual_conf", "spearman_vs_length": correlation_with_length(calibration, "length_residual_conf")},
            {"partition": "evaluation", "score": "length_residual_conf", "spearman_vs_length": correlation_with_length(evaluation, "length_residual_conf")},
        ]
    )
    candidate_audit.to_csv(args.out / "candidate_length_audit.csv", index=False)
    manifest = {
        "protocol": "tacl-11241-retrospective-geovote-length-audit-v1",
        "retrospective_only": True,
        "sample_level": str(args.sample_level.resolve()),
        "sample_level_sha256": hashlib.sha256(args.sample_level.read_bytes()).hexdigest(),
        "calibration_ids": sorted(calibration_ids),
        "evaluation_ids": sorted(evaluation_ids),
        "residualizer": {
            "formula": "geo_conf - (intercept + beta * log1p(n_gen_tokens))",
            "intercept": residualizer.intercept,
            "log_length_coefficient": residualizer.log_length_coefficient,
        },
        "same_pool_rule": "Every compared method selects from the identical K candidates per problem.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(comparison.to_string(index=False))
    print(candidate_audit.to_string(index=False))


if __name__ == "__main__":
    main()
