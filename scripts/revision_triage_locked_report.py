#!/usr/bin/env python3
"""Triage a locked steering report into a claim-verdict table.

Reads the CSV emitted by `scripts/revision_build_locked_report.py` and maps each
method to the contribution-strengthening decision rule from
`revision/CONTRIBUTION_STRENGTHENING_PLAN.md`:

- A method's claim is "supported" when its 95% bootstrap CI lower bound is
  strictly above zero on locked GSM8K IDs.
- CrossSteer's method-superiority claim additionally requires that its delta
  exceed every fair baseline in the same family under matched conditions.

This is a reporting/decision-support tool only.  It never inspects or alters any
experimental artifact and performs no re-selection of a layer, alpha, or method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASELINE_METHODS = {
    "negative_crosssteer_source",
    "matched_norm_random",
    "caa_target_prompt_final",
    "actadd_target_prompt_final",
    "sae_sparse_activation",
    "sparse_caa_coordinate_10pct",
    "target_calibrated",
}


def _verdict(row: pd.Series) -> str:
    ci_low = float(row["bootstrap_ci_low_pp"])
    ci_high = float(row["bootstrap_ci_high_pp"])
    p_holm = float(row["holm_adjusted_p_value"])
    if ci_low > 0.0:
        return "SUPPORTED_POSITIVE" if p_holm < 0.05 else "POSITIVE_NOT_HOLM"
    if ci_high < 0.0:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True, help="path to locked_summary.csv")
    parser.add_argument("--out", type=Path, required=True, help="output JSON verdict file")
    parser.add_argument("--source-report-required", action="store_true",
                        help="require the report dir to carry no SUPERSEDED marker")
    parser.add_argument("--expected-n", type=int, default=100,
                        help="required locked problem count per method")
    parser.add_argument("--max-truncation-rate", type=float, default=0.25,
                        help="maximum permitted baseline or intervention truncation rate")
    args = parser.parse_args()

    if args.source_report_required and (args.report.parent / "SUPERSEDED").exists():
        raise SystemExit("refusing to triage a SUPERSEDED (invalidated) locked report")
    if args.expected_n <= 0:
        raise SystemExit("--expected-n must be positive")
    if not 0.0 <= args.max_truncation_rate <= 1.0:
        raise SystemExit("--max-truncation-rate must be in [0, 1]")

    frame = pd.read_csv(args.report)
    required = {
        "method", "layer", "alpha", "n", "baseline_accuracy", "method_accuracy",
        "delta_pp", "bootstrap_ci_low_pp", "bootstrap_ci_high_pp",
        "repairs", "breaks", "exact_sign_p_value", "holm_adjusted_p_value",
        "baseline_truncation_rate", "method_truncation_rate",
    }
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"locked summary missing required columns: {sorted(missing)}")

    if not frame["n"].astype(int).eq(args.expected_n).all():
        observed = sorted(set(int(value) for value in frame["n"].tolist()))
        raise SystemExit(f"locked summary n mismatch: observed={observed} expected={args.expected_n}")

    rows: list[dict[str, object]] = []
    crosssteer = frame[frame["method"] == "crosssteer_source"]
    if len(crosssteer) != 1:
        raise SystemExit("report must contain exactly one crosssteer_source row")
    cs = crosssteer.iloc[0]
    cs_delta = float(cs["delta_pp"])
    cs_ci_low = float(cs["bootstrap_ci_low_pp"])
    baselines = frame[frame["method"].isin(BASELINE_METHODS)]
    beaten = {m: cs_delta > float(baselines.loc[baselines["method"] == m, "delta_pp"].iloc[0])
              for m in sorted(baselines["method"].unique())}
    all_beaten = bool(beaten) and all(beaten.values())
    cs_supported = cs_ci_low > 0.0

    for _, row in frame.sort_values("delta_pp", ascending=False).iterrows():
        rows.append(
            {
                "method": str(row["method"]),
                "layer": int(row["layer"]),
                "alpha": float(row["alpha"]),
                "n": int(row["n"]),
                "baseline_accuracy": float(row["baseline_accuracy"]),
                "method_accuracy": float(row["method_accuracy"]),
                "delta_pp": round(float(row["delta_pp"]), 1),
                "ci_pp": [round(float(row["bootstrap_ci_low_pp"]), 1),
                          round(float(row["bootstrap_ci_high_pp"]), 1)],
                "repairs": int(row["repairs"]),
                "breaks": int(row["breaks"]),
                "exact_p": float(row["exact_sign_p_value"]),
                "holm_p": float(row["holm_adjusted_p_value"]),
                "baseline_truncation_rate": round(float(row["baseline_truncation_rate"]), 3),
                "method_truncation_rate": round(float(row["method_truncation_rate"]), 3),
                "generated_token_delta": round(float(row.get("generated_token_delta", 0.0)), 2),
                "baseline_repetition": round(float(row.get("baseline_repetition", 0.0)), 4),
                "method_repetition": round(float(row.get("method_repetition", 0.0)), 4),
                "verdict": _verdict(row),
            }
        )

    max_observed_truncation = float(
        max(frame["baseline_truncation_rate"].max(), frame["method_truncation_rate"].max())
    )
    decoding_envelope_valid = max_observed_truncation <= args.max_truncation_rate
    verdict: dict[str, object] = {
        "protocol": "tacl-11241-locked-report-triage-v1",
        "report_path": str(args.report.resolve()),
        "n_methods": int(len(frame)),
        "expected_n": args.expected_n,
        "max_truncation_rate": args.max_truncation_rate,
        "max_observed_truncation_rate": round(max_observed_truncation, 4),
        "decoding_envelope_valid": decoding_envelope_valid,
        "methods": rows,
        "crosssteer": {
            "delta_pp": round(cs_delta, 1),
            "ci_low_pp": round(cs_ci_low, 1),
            "supported": cs_supported,
            "beats_all_fair_baselines": all_beaten,
            "beaten_by": sorted(
                (m for m, ok in beaten.items() if not ok),
                key=lambda m: -float(baselines.loc[baselines["method"] == m, "delta_pp"].iloc[0]),
            ),
        },
        "narrative": (
            "INVALID_DECODING_ENVELOPE"
            if not decoding_envelope_valid
            else "METHOD_CLAIM_SUPPORTED"
            if (cs_supported and all_beaten)
            else "CROSSSTEER_POSITIVE_NOT_SUPERIOR"
            if cs_supported
            else "NO_METHOD_SUPPORTED"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
