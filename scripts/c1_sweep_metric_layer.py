#!/usr/bin/env python3
"""Sweep (metric, layer, sign) and find the configuration that maximally beats
the majority-vote baseline.

For each candidate, reports: geo_weighted_vote, geo_then_majority,
majority_geo_tiebreak, combined_mult, combined_log. Picks the best per row.

Usage:
    python scripts/c1_sweep_metric_layer.py --run <dir> --dataset {gsm8k,math500}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geoprobe.analysis import evaluate_strategies, write_run_metrics
from geoprobe.datasets import is_correct, is_correct_math500


METRICS = ["trajectory_length", "mean_step_norm", "curvature_mean",
           "curvature_var", "curvature_max", "curvature_p90", "curvature_p99"]
LAYERS  = [4, 8, 12, 14, 16, 20, 24, 28]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    labels = pd.read_parquet(args.run / "labels.parquet")
    if (args.run / "metrics.parquet").exists():
        metrics = pd.read_parquet(args.run / "metrics.parquet")
    else:
        write_run_metrics(args.run)
        metrics = pd.read_parquet(args.run / "metrics.parquet")

    grader = is_correct if args.dataset == "gsm8k" else is_correct_math500
    if args.dataset == "gsm8k":
        labels = labels.copy()
        labels["gold"] = labels["gold"].astype(float)
        labels["pred"] = labels["pred"].apply(lambda x: float(x) if x is not None and x != "None" else None)

    res0 = evaluate_strategies(labels_df=labels, metrics_df=metrics,
                               geo_metric=None, geo_layer=None, grader=grader)
    maj = float(res0[res0["strategy"] == "majority_vote"]["accuracy"].iloc[0])
    print(f"baseline majority_vote = {maj:.3f}")

    rows = []
    for m in METRICS:
        for L in LAYERS:
            for sign in ["min", "max"]:
                res = evaluate_strategies(
                    labels_df=labels, metrics_df=metrics,
                    geo_metric=m, geo_layer=L, geo_sign=sign, grader=grader,
                )
                d = {r["strategy"]: r["accuracy"] for _, r in res.iterrows()}
                row = {
                    "metric": m, "layer": L, "sign": sign,
                    "geo_vote":          d.get(f"geo_{sign}_weighted_vote_{m}_L{L}", float("nan")),
                    "geo_then_majority": d.get(f"geo_then_majority_{m}_L{L}", float("nan")),
                    "majority_tiebreak": d.get(f"majority_geo_tiebreak_{m}_L{L}", float("nan")),
                    "combined_mult":     d.get(f"geo_majority_combined_{m}_L{L}", float("nan")),
                    "combined_log":      d.get(f"geo_majority_logsum_{m}_L{L}", float("nan")),
                }
                row["best"] = max(row["geo_vote"], row["geo_then_majority"],
                                  row["majority_tiebreak"], row["combined_mult"], row["combined_log"])
                row["delta_vs_majority"] = row["best"] - maj
                rows.append(row)

    df = pd.DataFrame(rows).sort_values("delta_vs_majority", ascending=False)
    print("\n=== top 12 by best Δ vs pure majority ===")
    print(df.head(12).to_string(index=False))
    print(f"\nbest delta = {df['delta_vs_majority'].max():+.3f}  (majority alone = {maj:.3f})")
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
