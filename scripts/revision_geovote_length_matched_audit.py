#!/usr/bin/env python3
"""Run the explicit length-matched GeoVote null audit.

This command is intentionally separate from the already completed retrospective
GeoVote report.  It does not alter the manuscript or promote its output to
formal evidence automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from geoprobe.revision.geovote import LinearLengthResidualizer
from geoprobe.revision.geovote_length_matched import run_length_matched_null


def parse_ids(spec: str) -> set[int]:
    """Parse a comma-separated half-open range specification."""
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
    parser.add_argument("--calibration-ids", required=True)
    parser.add_argument("--evaluation-ids", required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    calibration_ids = parse_ids(args.calibration_ids)
    evaluation_ids = parse_ids(args.evaluation_ids)
    if calibration_ids & evaluation_ids:
        raise SystemExit("calibration and evaluation IDs must be disjoint")

    frame = pd.read_parquet(args.sample_level).reset_index(drop=True)
    required = {
        "sample_id",
        "sample_idx",
        "pred",
        "correct",
        "n_gen_tokens",
        "geo_conf",
        "sequence_logprob",
    }
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"input lacks required columns: {sorted(missing)}")
    observed = set(frame["sample_id"].astype(int).unique())
    if not calibration_ids <= observed or not evaluation_ids <= observed:
        raise SystemExit("declared IDs are not all present in the candidate pool")

    calibration = frame[frame["sample_id"].isin(calibration_ids)].copy()
    evaluation = frame[frame["sample_id"].isin(evaluation_ids)].copy()
    residualizer = LinearLengthResidualizer.fit(calibration)
    evaluation["length_residual_conf"] = residualizer.residualize(evaluation)

    rows = run_length_matched_null(
        evaluation,
        score_column="length_residual_conf",
        permutations=args.permutations,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / "length_matched_permutation_rows.csv", index=False)
    summary = {
        "protocol": "tacl-11241-retrospective-geovote-length-matched-v1",
        "retrospective_only": True,
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "calibration_ids": sorted(calibration_ids),
        "evaluation_ids": sorted(evaluation_ids),
        "input_sha256": hashlib.sha256(args.sample_level.read_bytes()).hexdigest(),
        "residualizer": {
            "formula": "geo_conf - (intercept + beta * log1p(n_gen_tokens))",
            "intercept": residualizer.intercept,
            "log_length_coefficient": residualizer.log_length_coefficient,
        },
        "pairing_rule": "sort candidates by (n_gen_tokens, sample_idx), then permute scores within adjacent pairs",
        "baseline_accuracy": float(rows["baseline_accuracy"].iloc[0]),
        "mean_method_accuracy": float(rows["method_accuracy"].mean()),
        "mean_delta_pp": float(rows["delta_pp"].mean()),
        "median_delta_pp": float(rows["delta_pp"].median()),
        "delta_pp_q025": float(rows["delta_pp"].quantile(0.025)),
        "delta_pp_q975": float(rows["delta_pp"].quantile(0.975)),
        "delta_pp_min": float(rows["delta_pp"].min()),
        "delta_pp_max": float(rows["delta_pp"].max()),
    }
    (args.out / "length_matched_permutation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
