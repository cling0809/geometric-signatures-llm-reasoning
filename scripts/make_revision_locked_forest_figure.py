#!/usr/bin/env python3
"""Render the complete locked CrossSteer comparison as a homogeneous forest plot."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geoprobe.revision.figure_style import (
    BLUE,
    GRID,
    INK,
    MUTED,
    ORANGE,
    RED,
    TEAL,
    clean_axis,
    save_publication_figure,
    setup_publication_style,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "revision" / "evidence" / "locked-gsm8k-qwen-instruct-v3"
FIGURES = ROOT / "paper" / "figures"

METHODS = [
    ("CrossSteer (source)", "crosssteer_source", BLUE),
    ("Target-calibrated", "target_calibrated", TEAL),
    ("CAA", "caa_target_prompt_final", ORANGE),
    ("SAE sparse", "sae_sparse_activation", ORANGE),
    ("Sparse CAA (10%)", "sparse_caa_coordinate_10pct", MUTED),
    ("ActAdd", "actadd_target_prompt_final", MUTED),
    ("Sign-reversed", "negative_crosssteer_source", RED),
    ("Matched-norm random", "matched_norm_random", MUTED),
]


def load_rows() -> dict[str, dict[str, float]]:
    path = EVIDENCE / "locked_summary.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["method"]: row for row in rows}
    expected = {key for _, key, _ in METHODS}
    if set(indexed) != expected:
        raise ValueError(f"locked methods mismatch: expected {sorted(expected)}, got {sorted(indexed)}")
    return {
        key: {
            "delta": float(indexed[key]["delta_pp"]),
            "low": float(indexed[key]["bootstrap_ci_low_pp"]),
            "high": float(indexed[key]["bootstrap_ci_high_pp"]),
        }
        for _, key, _ in METHODS
    }


def main() -> None:
    setup_publication_style()
    values = load_rows()
    labels = [label for label, _, _ in METHODS]
    y = np.arange(len(METHODS))[::-1]

    fig, ax = plt.subplots(figsize=(6.95, 3.25), constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.95, bottom=0.18)
    for ypos, (_label, key, color) in zip(y, METHODS, strict=True):
        row = values[key]
        delta = row["delta"]
        low = row["low"]
        high = row["high"]
        ax.errorbar(
            delta,
            ypos,
            xerr=[[delta - low], [high - delta]],
            fmt="o",
            ms=4.5,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            elinewidth=1.15,
            capsize=2.4,
            capthick=1.0,
            zorder=3,
        )

    ax.axvline(0, color=INK, linewidth=0.85, zorder=1)
    ax.set_xlim(-10, 18)
    ax.set_xticks([-10, -5, 0, 5, 10, 15])
    ax.set_xlabel("paired accuracy change vs. no steering (pp)")
    ax.set_yticks(y, labels)
    ax.set_ylim(-1, len(METHODS))
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    clean_axis(ax, grid=None)
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.tick_params(axis="y", length=0, pad=3)
    save_publication_figure(fig, FIGURES, "fig_locked_comparison_forest")


if __name__ == "__main__":
    main()
