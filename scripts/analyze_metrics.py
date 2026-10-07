#!/usr/bin/env python3
"""Read metrics.parquet + labels.parquet from a run, compute AUC table + plot.

Usage:
    python scripts/analyze_metrics.py --run ~/AI/runs/<exp-id>

Writes:
    <run>/auc_table.csv
    <run>/plots/auc_per_layer.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geoprobe.analysis import compute_auc_table, plot_auc_per_layer, trivial_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()

    metrics_df = pd.read_parquet(args.run / "metrics.parquet")
    labels_df = pd.read_parquet(args.run / "labels.parquet")

    auc_df = compute_auc_table(metrics_df, labels_df, positive_label="incorrect")
    auc_path = args.run / "auc_table.csv"
    auc_df.to_csv(auc_path, index=False)
    print(f"wrote {auc_path}")

    baseline = trivial_baseline(labels_df, column="n_gen_tokens")
    print(f"trivial baseline (n_gen_tokens): AUC={baseline['auc']:.3f} AP={baseline['ap']:.3f}")

    print("\n=== best (metric, layer) by AUC ===")
    print(auc_df.sort_values("auc", ascending=False).head(10).to_string(index=False))

    plots_dir = args.run / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    title = f"AUC per layer  ({args.run.name},  n={labels_df.shape[0]})"
    plot_auc_per_layer(
        auc_df,
        baseline_auc=baseline["auc"],
        title=title,
        out_path=plots_dir / "auc_per_layer.png",
    )
    print(f"wrote {plots_dir / 'auc_per_layer.png'}")


if __name__ == "__main__":
    main()
