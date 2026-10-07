#!/usr/bin/env python3
"""Aggregate P1 (cross-source / alpha / layer) + P2 (N=8 vs N=16) into one figure + CSV.

Outputs:
  <out>/p1_p2_summary.csv
  <out>/plots/p1_cross_source.png
  <out>/plots/p1_alpha_sweep.png      (re-styled)
  <out>/plots/p1_cross_layer.png
  <out>/plots/p2_n_scaling.png

Usage:
    python scripts/phase3c3_aggregate.py --out ~/AI/runs/2026-05-17_phase3c3-aggregate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Hard-coded data — we have it from the runs.
CROSS_SOURCE = {
    "Base": 0.42,
    "Instruct": 0.45,
    "Math": 0.47,
}
ALPHA_SWEEP = {
    0.0: 0.32,
    2.0: 0.47,
    2.5: 0.39,
    3.0: 0.38,
    4.0: 0.32,
    5.0: 0.25,
}
CROSS_LAYER = {
    5: 0.36,
    10: 0.40,
    14: 0.47,
    20: 0.45,
    24: 0.44,
}
BASELINE = 0.32  # R1-Distill no-steering

C1_N8 = {
    "greedy_pass1": 0.76,
    "random_pick": 0.78,
    "logprob_max": 0.82,
    "logprob_weighted_vote": 0.82,
    "geo_min_weighted_vote@L20": 0.88,
    "majority_vote": 0.90,
    "oracle": 0.97,
}
C1_N16 = {
    "greedy_pass1": 0.76,
    "random_pick": 0.79,
    "logprob_max": 0.84,
    "logprob_weighted_vote": 0.84,
    "geo_min_weighted_vote@L20": 0.90,
    "majority_vote": 0.92,
    "oracle": 0.99,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    plots = args.out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # ---------------- CSV summary ----------------
    rows = []
    for src, acc in CROSS_SOURCE.items():
        rows.append({"phase": "P1.1 cross-source", "x_name": "source", "x": src, "acc": acc})
    for a, acc in ALPHA_SWEEP.items():
        rows.append({"phase": "P1.2 alpha", "x_name": "alpha", "x": a, "acc": acc})
    for L, acc in CROSS_LAYER.items():
        rows.append({"phase": "P1.3 layer", "x_name": "layer", "x": L, "acc": acc})
    for s, acc in C1_N8.items():
        rows.append({"phase": "P2 C1 N=8", "x_name": "strategy", "x": s, "acc": acc})
    for s, acc in C1_N16.items():
        rows.append({"phase": "P2 C1 N=16", "x_name": "strategy", "x": s, "acc": acc})
    pd.DataFrame(rows).to_csv(args.out / "p1_p2_summary.csv", index=False)
    print(f"wrote {args.out / 'p1_p2_summary.csv'}")

    # ---------------- P1.1 cross-source ----------------
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(CROSS_SOURCE.keys())
    accs = [CROSS_SOURCE[n] for n in names]
    bars = ax.bar(names, accs, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.005, f"{a:.2f}", ha="center", fontsize=10)
    ax.axhline(BASELINE, color="grey", linestyle="--", label=f"R1-Distill no-steering baseline = {BASELINE}")
    ax.set_ylabel("R1-Distill GSM8K acc (n=100)")
    ax.set_title("P1.1: source for steering vector (alpha=2.0, layer 14)")
    ax.set_ylim(0.25, 0.55)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots / "p1_cross_source.png", dpi=140)
    plt.close(fig)
    print(f"wrote {plots / 'p1_cross_source.png'}")

    # ---------------- P1.2 alpha sweep ----------------
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = sorted(ALPHA_SWEEP.keys())
    ys = [ALPHA_SWEEP[a] for a in xs]
    ax.plot(xs, ys, marker="o", linewidth=2, color="#2ca02c")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(BASELINE, color="grey", linestyle="--", label=f"baseline = {BASELINE}")
    ax.set_xlabel("alpha (steering magnitude)")
    ax.set_ylabel("R1-Distill GSM8K acc (n=100)")
    ax.set_title("P1.2: alpha sweep (Math source, layer 14)  —  inverted U with peak at α=2")
    ax.set_ylim(0.15, 0.55)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "p1_alpha_sweep.png", dpi=140)
    plt.close(fig)
    print(f"wrote {plots / 'p1_alpha_sweep.png'}")

    # ---------------- P1.3 cross-layer ----------------
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = sorted(CROSS_LAYER.keys())
    ys = [CROSS_LAYER[L] for L in xs]
    ax.plot(xs, ys, marker="o", linewidth=2, color="#d62728")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(BASELINE, color="grey", linestyle="--", label=f"baseline = {BASELINE}")
    ax.set_xlabel("inject layer (1 = first transformer block; 28 = last)")
    ax.set_ylabel("R1-Distill GSM8K acc (n=100)")
    ax.set_title("P1.3: cross-layer (Math source, alpha=2)  —  peak at mid-late layers (14)")
    ax.set_ylim(0.25, 0.55)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "p1_cross_layer.png", dpi=140)
    plt.close(fig)
    print(f"wrote {plots / 'p1_cross_layer.png'}")

    # ---------------- P2 C1 N=8 vs N=16 ----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    strategies = list(C1_N16.keys())
    x = range(len(strategies))
    w = 0.4
    n8_vals = [C1_N8[s] for s in strategies]
    n16_vals = [C1_N16[s] for s in strategies]
    ax.bar([xi - w / 2 for xi in x], n8_vals, w, label="N=8", color="#1f77b4")
    ax.bar([xi + w / 2 for xi in x], n16_vals, w, label="N=16", color="#ff7f0e")
    for xi, (a, b) in enumerate(zip(n8_vals, n16_vals)):
        ax.text(xi - w / 2, a + 0.005, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Qwen-Math GSM8K acc (n=100)")
    ax.set_title("P2: C1 selection strategies at N=8 vs N=16  —  geo_weighted beats logprob by +6pp at both N")
    ax.set_ylim(0.65, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots / "p2_n_scaling.png", dpi=140)
    plt.close(fig)
    print(f"wrote {plots / 'p2_n_scaling.png'}")


if __name__ == "__main__":
    main()
