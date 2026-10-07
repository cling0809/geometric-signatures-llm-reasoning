#!/usr/bin/env python3
"""Plot the C4 alpha sweep alongside the random-vector control distribution.

Shows:
  - Line: actual steering vector accuracy across alphas
  - Box / scatter: random-vector control at alpha=2.0
  - Dashed horizontal: baseline (alpha=0)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-csv", required=True, type=Path,
                    help="c4_alpha_accuracy.csv from the actual-vector sweep")
    ap.add_argument("--random-csv", required=True, type=Path,
                    help="c4_random_summary.csv from random control")
    ap.add_argument("--alpha-of-control", type=float, default=2.0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sweep = pd.read_csv(args.sweep_csv).sort_values("alpha")
    rand = pd.read_csv(args.random_csv)

    baseline = float(sweep[sweep["alpha"] == 0.0]["accuracy"].iloc[0])
    ours_at_a = float(sweep[sweep["alpha"] == args.alpha_of_control]["accuracy"].iloc[0])
    rand_mean = float(rand["accuracy"].mean())
    rand_std = float(rand["accuracy"].std())
    rand_min = float(rand["accuracy"].min())
    rand_max = float(rand["accuracy"].max())
    z = (ours_at_a - rand_mean) / rand_std if rand_std > 0 else float("inf")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(sweep["alpha"], sweep["accuracy"], marker="o", label="ours: (correct − incorrect) direction", linewidth=2)
    ax.axhline(baseline, color="grey", linestyle="--", label=f"baseline (alpha=0): {baseline:.2f}")

    # random control points at alpha=control
    a = args.alpha_of_control
    for v in rand["accuracy"]:
        ax.scatter(a, v, color="orange", alpha=0.6, s=40, zorder=3)
    ax.errorbar([a + 0.08], [rand_mean], yerr=[rand_std], fmt="s", color="darkorange",
                capsize=4, capthick=2, label=f"random ctrl (n={len(rand)}): {rand_mean:.2f} ± {rand_std:.2f}")

    ax.annotate(f"ours\n{ours_at_a:.2f}\n(+{ours_at_a - baseline:+.2f})",
                xy=(a, ours_at_a), xytext=(a + 0.15, ours_at_a + 0.01),
                fontsize=10, color="C0", fontweight="bold")
    ax.annotate(f"z = {z:.1f}σ vs random",
                xy=(a + 0.08, rand_mean), xytext=(a + 0.20, rand_mean - 0.03),
                fontsize=9, color="darkorange",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.5))

    ax.set_xlabel("alpha (steering magnitude)")
    ax.set_ylabel("GSM8K accuracy (n=100)")
    ax.set_title("C4: directional steering beats random-vector control at same magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    print(f"  ours = {ours_at_a:.3f}  random = {rand_mean:.3f} ± {rand_std:.3f}  z = {z:.2f}σ")


if __name__ == "__main__":
    main()
