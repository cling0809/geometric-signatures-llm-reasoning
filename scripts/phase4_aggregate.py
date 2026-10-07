#!/usr/bin/env python3
"""Aggregate Phase 4 A results into one CSV + figures.

Combines:
  - C1 GSM8K vs MATH-500 selection strategies
  - C4 on R1-Distill: GSM8K vs MATH-500, varying source
  - C4 cross-target: Qwen-Instruct (negative result)

Outputs:
  <out>/phase4_summary.csv
  <out>/plots/p4_c1_dataset_transfer.png
  <out>/plots/p4_c4_target_source_dataset.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# C1 selection: GSM8K vs MATH-500
C1_GSM8K_N8 = {
    "greedy_pass1": 0.76,
    "random_pick": 0.78,
    "logprob_max": 0.82,
    "logprob_weighted_vote": 0.82,
    "geo_min_weighted_vote@L20": 0.88,
    "majority_vote": 0.90,
    "oracle": 0.97,
}
C1_MATH500_N8 = {
    "greedy_pass1": 0.63,
    "random_pick": 0.68,
    "logprob_max": 0.70,
    "logprob_weighted_vote": 0.70,
    "geo_min_weighted_vote@L20": 0.75,
    "majority_vote": 0.77,
    "oracle": 0.87,
}

# C4: R1-Distill target, alpha=2, layer=14, varying source/dataset.
C4_ON_R1_DISTILL = [
    # (dataset, source_label, source_acc, target_acc_baseline, target_acc_alpha2)
    ("GSM8K", "Qwen-Base",                  0.37, 0.32, 0.42),
    ("GSM8K", "Qwen-Instruct",              0.60, 0.32, 0.45),
    ("GSM8K", "Qwen-Math (cross-task best)",0.78, 0.32, 0.47),
    ("MATH-500", "Qwen-Math/GSM8K (cross-task)", 0.78, 0.17, 0.22),
    ("MATH-500", "Qwen-Instruct/MATH-500 (same-task, weak source)", 0.31, 0.17, 0.20),
    # Filled in at runtime from the in-flight strongest-source experiment:
    # ("MATH-500", "Qwen-Math/MATH-500 (same-task, strong source)", 0.63, 0.17, ?),
]

# C4 cross-target: Qwen-Instruct target on MATH-500, Qwen-Math source. Negative result.
C4_CROSS_TARGET = {
    "alpha": [0.0, 1.0, 2.0],
    "acc":   [0.29, 0.26, 0.10],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strongest-source-csv", type=Path, default=None,
                    help="optional path to Qwen-Math/MATH-500 -> R1-Distill csv")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    plots = args.out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # Optional strongest-source row
    c4_r1 = list(C4_ON_R1_DISTILL)
    if args.strongest_source_csv and args.strongest_source_csv.exists():
        df_ss = pd.read_csv(args.strongest_source_csv)
        a2 = df_ss[df_ss["alpha"] == 2.0]["accuracy"]
        if len(a2):
            c4_r1.append(("MATH-500", "Qwen-Math/MATH-500 (same-task, strong source)",
                          0.63, 0.17, float(a2.iloc[0])))

    # ----- CSV -----
    rows = []
    for k, v in C1_GSM8K_N8.items():
        rows.append({"phase": "C1", "dataset": "GSM8K", "strategy": k, "acc": v})
    for k, v in C1_MATH500_N8.items():
        rows.append({"phase": "C1", "dataset": "MATH-500", "strategy": k, "acc": v})
    for ds, src, src_a, t_b, t_a in c4_r1:
        rows.append({"phase": "C4", "dataset": ds, "target": "R1-Distill",
                     "source": src, "source_acc": src_a,
                     "target_baseline": t_b, "target_alpha2": t_a,
                     "delta_pp": round(t_a - t_b, 3)})
    for a, acc in zip(C4_CROSS_TARGET["alpha"], C4_CROSS_TARGET["acc"]):
        rows.append({"phase": "C4 cross-target",
                     "dataset": "MATH-500", "target": "Qwen-Instruct (non-math)",
                     "source": "Qwen-Math/GSM8K",
                     "alpha": a, "acc": acc})
    pd.DataFrame(rows).to_csv(args.out / "phase4_summary.csv", index=False)
    print(f"wrote {args.out / 'phase4_summary.csv'}")

    # ----- Plot 1: C1 GSM8K vs MATH-500 -----
    fig, ax = plt.subplots(figsize=(10, 5))
    strategies = list(C1_MATH500_N8.keys())
    x = range(len(strategies))
    w = 0.4
    g8 = [C1_GSM8K_N8[s] for s in strategies]
    m5 = [C1_MATH500_N8[s] for s in strategies]
    ax.bar([xi - w / 2 for xi in x], g8, w, label="GSM8K", color="#1f77b4")
    ax.bar([xi + w / 2 for xi in x], m5, w, label="MATH-500", color="#d62728")
    for xi, (a, b) in enumerate(zip(g8, m5)):
        ax.text(xi - w / 2, a + 0.005, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Qwen-Math-1.5B acc (n=100, N=8 sampling)")
    ax.set_title("C1 GeoVote replicates from GSM8K to MATH-500  —  +6pp / +5pp over logprob")
    ax.set_ylim(0.55, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots / "p4_c1_dataset_transfer.png", dpi=140)
    plt.close(fig)
    print(f"wrote {plots / 'p4_c1_dataset_transfer.png'}")

    # ----- Plot 2: C4 R1-Distill across source × dataset -----
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    deltas = []
    for ds, src, src_a, t_b, t_a in c4_r1:
        labels.append(f"{ds}\n{src.split(' (')[0]}")
        deltas.append(t_a - t_b)
    colors = ["#1f77b4" if "GSM8K" in lab else "#d62728" for lab in labels]
    bars = ax.barh(labels, deltas, color=colors)
    for b, d in zip(bars, deltas):
        ax.text(d + 0.003, b.get_y() + b.get_height() / 2, f"+{d:.2f}", va="center", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Δ accuracy (R1-Distill steered − R1-Distill baseline)")
    ax.set_title("C4 cross-source × cross-dataset on R1-Distill target  —  layer 14, α=2")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(plots / "p4_c4_target_source_dataset.png", dpi=140)
    plt.close(fig)
    print(f"wrote {plots / 'p4_c4_target_source_dataset.png'}")

    # ----- Plot 3: cross-target negative -----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(C4_CROSS_TARGET["alpha"], C4_CROSS_TARGET["acc"], marker="o", linewidth=2,
            color="#e377c2", label="Qwen-Instruct target (sign-flip not broken)")
    ax.axhline(0.32, color="grey", linestyle=":", linewidth=1, label="R1-Distill baseline (broken sign-flip) for context")
    ax.annotate("steering HURTS\nnon-broken target",
                xy=(2.0, 0.10), xytext=(1.2, 0.02),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
                fontsize=10)
    ax.set_xlabel("alpha (steering magnitude)")
    ax.set_ylabel("Qwen-Instruct MATH-500 acc")
    ax.set_title("C4 cross-target negative result  —  steering hurts already-correct targets")
    ax.set_ylim(0.0, 0.4)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "p4_c4_cross_target_negative.png", dpi=140)
    plt.close(fig)
    print(f"wrote {plots / 'p4_c4_cross_target_negative.png'}")


if __name__ == "__main__":
    main()
