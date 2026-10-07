#!/usr/bin/env python3
"""Phase 4 B: combine alpha sweep on MATH-500 (8 alphas) and compare to GSM8K (6 alphas).

Inputs: two CSVs from c4_steer_r1.py (already exist), plus the Phase 3c.3 GSM8K
alpha sweep numbers (hard-coded).

Output: phase4b plot showing two curves overlaid (GSM8K vs MATH-500), and the
delta-from-baseline view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# From Phase 3c.3 P1.2 (GSM8K, Math source, layer 14)
GSM8K_ALPHA = {
    0.0: 0.32,
    2.0: 0.47,
    2.5: 0.39,
    3.0: 0.38,
    4.0: 0.32,
    5.0: 0.25,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math500-base-csv", required=True, type=Path,
                    help="alphas 0/1/2 (from /root/AI/runs/2026-05-18_c4-math500-source-math500-qwenmath/c4_alpha_accuracy.csv)")
    ap.add_argument("--math500-extra-csv", required=True, type=Path,
                    help="alphas 0.5/1.5/2.5/3.0/4.0 (from the Tier 2 run)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.math500_base_csv)
    extra = pd.read_csv(args.math500_extra_csv)
    math500 = pd.concat([base, extra]).drop_duplicates("alpha").sort_values("alpha").reset_index(drop=True)
    math500.to_csv(args.out / "phase4b_math500_full_sweep.csv", index=False)
    math500_baseline = float(math500[math500["alpha"] == 0.0]["accuracy"].iloc[0])

    print("=== MATH-500 full alpha sweep (8 points) ===")
    print(math500.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # left: raw acc curves
    ax = axes[0]
    gx = sorted(GSM8K_ALPHA.keys())
    gy = [GSM8K_ALPHA[a] for a in gx]
    ax.plot(gx, gy, marker="o", linewidth=2, label="GSM8K (Math source, baseline 0.32)", color="#1f77b4")
    for x, y in zip(gx, gy):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    mx = math500["alpha"].tolist()
    my = math500["accuracy"].tolist()
    ax.plot(mx, my, marker="s", linewidth=2, label=f"MATH-500 (Math source, baseline {math500_baseline:.2f})", color="#d62728")
    for x, y in zip(mx, my):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#d62728")

    ax.set_xlabel("alpha (steering magnitude)")
    ax.set_ylabel("R1-Distill GSM8K / MATH-500 accuracy")
    ax.set_title("Alpha sweep: GSM8K vs MATH-500 (Math source, layer 14, n=100)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # right: delta-from-baseline (normalized view to compare shapes)
    ax = axes[1]
    gd = [GSM8K_ALPHA[a] - 0.32 for a in gx]
    md = [v - math500_baseline for v in my]
    ax.plot(gx, gd, marker="o", linewidth=2, label="GSM8K Δ", color="#1f77b4")
    ax.plot(mx, md, marker="s", linewidth=2, label="MATH-500 Δ", color="#d62728")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("alpha")
    ax.set_ylabel("Δ accuracy (steered − baseline)")
    ax.set_title("Δ from baseline: inverted-U replicates across datasets")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = args.out / "phase4b_alpha_sweep_gsm8k_vs_math500.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
