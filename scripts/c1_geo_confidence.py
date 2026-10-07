#!/usr/bin/env python3
"""C1: evaluate answer-selection strategies on a sampling extraction.

Compares:
  - greedy_pass1
  - random_pick
  - majority_vote (self-consistency)
  - logprob_max + logprob_weighted_vote
  - geo strategies using a chosen (metric, layer, sign)
  - oracle

Usage:
    python scripts/c1_geo_confidence.py \
        --run ~/AI/runs/<sampling-run-id> \
        --geo-metric mean_step_norm --geo-layer 20 --geo-sign min
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geoprobe.analysis import evaluate_strategies, write_run_metrics
from geoprobe.datasets import is_correct, is_correct_math500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--geo-metric", default="mean_step_norm")
    ap.add_argument("--geo-layer", default=20, type=int)
    ap.add_argument("--geo-sign", default="min", choices=["min", "max"])
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"],
                    help="Selects the grader: gsm8k uses numeric tol, math500 uses string/float fallback.")
    args = ap.parse_args()

    labels = pd.read_parquet(args.run / "labels.parquet")
    print(f"labels.parquet: {labels.shape[0]} rows, {labels['sample_id'].nunique()} questions, "
          f"N={labels.groupby('sample_id').size().mode()[0]} samples/q")

    if not (args.run / "metrics.parquet").exists():
        print(f"computing metrics for {args.run.name} ...")
        write_run_metrics(args.run)
    metrics = pd.read_parquet(args.run / "metrics.parquet")
    print(f"metrics.parquet: {metrics.shape[0]} rows, "
          f"{metrics['metric'].nunique()} metrics x {metrics['layer'].nunique()} layers")

    grader = is_correct if args.dataset == "gsm8k" else is_correct_math500
    # Convert labels.gold/pred back to float for gsm8k (they're stored as strings by extract_trajectories.py)
    if args.dataset == "gsm8k":
        labels = labels.copy()
        labels["gold"] = labels["gold"].astype(float)
        labels["pred"] = labels["pred"].apply(lambda x: float(x) if x is not None and x != "None" else None)
    result = evaluate_strategies(
        labels_df=labels,
        metrics_df=metrics,
        geo_metric=args.geo_metric,
        geo_layer=args.geo_layer,
        geo_sign=args.geo_sign,
        grader=grader,
    )

    print()
    print(result.to_string(index=False))

    out = args.run / "c1_selection_strategies.csv"
    result.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
