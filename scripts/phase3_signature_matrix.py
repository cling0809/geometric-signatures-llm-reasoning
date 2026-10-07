#!/usr/bin/env python3
"""Phase 3a/3b: build geometric signatures + paradigm comparison.

For each run dir, treat its (metric x layer) AUC matrix as the model's
geometric signature. Then:
  - Save each signature as CSV.
  - Plot the signatures side-by-side as heatmaps (one panel per model).
  - Compute pairwise L2 distance between (sig - 0.5) flattened vectors.
  - Save the distance matrix as CSV + heatmap.

Usage:
    python scripts/phase3_signature_matrix.py \
        --runs <run_a> <run_b> ... \
        --labels "Base" "SFT" ... \
        --out ~/AI/runs/<aggregate-exp-id>/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geoprobe.analysis import (
    load_signatures,
    pairwise_signature_distance,
    plot_pairwise_distance,
    plot_signature_grid,
    write_run_metrics,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, nargs="+", type=Path)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Short paradigm labels, one per run, in the same order")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plots").mkdir(exist_ok=True)
    (args.out / "signatures").mkdir(exist_ok=True)

    # Ensure metrics.parquet exists for every run
    for r in args.runs:
        if not (r / "metrics.parquet").exists():
            print(f"[{r.name}] computing metrics ...")
            write_run_metrics(r)

    label_map: dict[str, str] | None = None
    if args.labels:
        if len(args.labels) != len(args.runs):
            raise ValueError("--labels must have same length as --runs")
        label_map = {r.name: lbl for r, lbl in zip(args.runs, args.labels)}

    sigs = load_signatures(args.runs)

    # Save individual signatures
    for name, sig in sigs.items():
        sig.to_csv(args.out / "signatures" / f"{name}.csv")

    # Grid of heatmaps
    grid_path = args.out / "plots" / "signature_grid.png"
    plot_signature_grid(sigs, grid_path, labels=label_map)
    print(f"wrote {grid_path}")

    # Pairwise distance
    dist = pairwise_signature_distance(sigs)
    dist.to_csv(args.out / "pairwise_signature_distance.csv")
    plot_pairwise_distance(dist, args.out / "plots" / "pairwise_distance.png", labels=label_map)
    print(f"wrote {args.out / 'pairwise_signature_distance.csv'}")

    print("\n=== pairwise signature distance ===")
    if label_map:
        ld = dist.copy()
        ld.index = [label_map.get(n, n) for n in ld.index]
        ld.columns = [label_map.get(n, n) for n in ld.columns]
        print(ld.round(3).to_string())
    else:
        print(dist.round(3).to_string())


if __name__ == "__main__":
    main()
