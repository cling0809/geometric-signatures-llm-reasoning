#!/usr/bin/env python3
"""Regenerate the two audited signature figures used in the revision.

Historical exploratory trajectory and CrossSteer plots are intentionally not
regenerated here.  The active script is limited to the length-residualized
signature heatmaps and the paired budget-stability dumbbell plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geoprobe.revision.figure_style import (
    BLUE,
    GRID,
    INK,
    MODEL_COLORS,
    ORANGE,
    panel_title,
    save_publication_figure,
    setup_publication_style,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIG = PAPER / "figures"
EXP_PHEN = ROOT / "experiments" / "2026-05-17_phase3-paradigm-4"
DIVERGING = "RdBu_r"


METRIC_ORDER = [
    "trajectory_length",
    "mean_step_norm",
    "curvature_mean",
    "curvature_var",
    "curvature_max",
    "curvature_p90",
    "curvature_p99",
]
METRIC_LABELS = {
    "trajectory_length": "traj. length",
    "mean_step_norm": "step norm",
    "curvature_mean": "curv. mean",
    "curvature_var": "curv. var",
    "curvature_max": "curv. max",
    "curvature_p90": "curv. p90",
    "curvature_p99": "curv. p99",
}
PHEN_MODELS = [
    ("Base", "2026-05-17_extract-qwen-1.5b-base-100.csv"),
    ("Instruct", "2026-05-17_extract-qwen-1.5b-nomath-100.csv"),
    ("Math-Instruct", "2026-05-17_extract-pilot-gsm8k-100.csv"),
    ("R1-Distill", "2026-05-17_extract-deepseek-r1-distill-1.5b-100.csv"),
]


def setup_style() -> None:
    setup_publication_style()


def save(fig: plt.Figure, stem: str) -> None:
    save_publication_figure(fig, FIG, stem)


def make_signature_figure() -> None:
    """Render the all-cell diagnostic as heatmaps only."""
    setup_style()
    fig = plt.figure(figsize=(6.75, 2.82), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.91,
        top=0.91,
        bottom=0.18,
        wspace=0.08,
        hspace=0.30,
    )
    axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    image = None
    norm = mpl.colors.TwoSlopeNorm(vmin=0.25, vcenter=0.5, vmax=0.75)
    for idx, (ax, (name, csv_name)) in enumerate(zip(axes, PHEN_MODELS, strict=True)):
        table = pd.read_csv(EXP_PHEN / csv_name).set_index("metric").reindex(METRIC_ORDER)
        image = ax.imshow(
            table.to_numpy(dtype=float),
            aspect="auto",
            cmap=DIVERGING,
            norm=norm,
            interpolation="nearest",
        )
        panel_title(ax, chr(65 + idx), name, color=MODEL_COLORS.get(name, INK))
        ax.set_xticks([0, 7, 14, 21, 28])
        ax.set_xticklabels(["0", "7", "14", "21", "28"])
        ax.set_yticks(range(len(METRIC_ORDER)))
        ax.set_yticklabels([METRIC_LABELS[m] for m in METRIC_ORDER] if idx % 2 == 0 else [])
        if idx >= 2:
            ax.set_xlabel("layer", labelpad=1.0)
        ax.tick_params(length=0, pad=1.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("white")
        ax.set_xticks(np.arange(-0.5, 29, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(METRIC_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.28)
        ax.tick_params(which="minor", length=0)

    assert image is not None
    cax = fig.add_axes([0.935, 0.235, 0.018, 0.53])
    colorbar = fig.colorbar(image, cax=cax, orientation="vertical")
    colorbar.set_ticks([0.25, 0.5, 0.75])
    colorbar.set_ticklabels(["0.25", "0.50", "0.75"])
    colorbar.set_label("AUC\n(incorrect)", fontsize=6.5, labelpad=4)
    colorbar.ax.tick_params(labelsize=6.1, length=2)
    save(fig, "fig2_signature_heatmaps")


def make_signature_distance_figure() -> None:
    """Render the budget-sensitivity distance audit as one homogeneous dot plot."""
    setup_style()
    pair_labels = [
        "Base–Instruct",
        "Base–Math",
        "Base–R1",
        "Instruct–Math",
        "Instruct–R1",
        "Math–R1",
    ]
    d512 = np.array([1.83, 2.63, 2.13, 2.18, 1.13, 2.09])
    d2048 = np.array([1.81, 2.27, 2.44, 2.06, 1.54, 1.57])
    y = np.arange(len(pair_labels), dtype=float)
    fig, ax = plt.subplots(figsize=(6.75, 2.18), constrained_layout=False)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.88, bottom=0.22)
    # Dumbbell encoding: each row is one model pair, and the connector shows
    # only that pair's budget change.  Do not connect unrelated rows with a
    # polyline; their ordering is categorical, not a trend axis.
    for old, new, yy in zip(d512, d2048, y, strict=True):
        ax.plot([old, new], [yy, yy], color=GRID, lw=1.15, zorder=0)
    ax.scatter(d512, y, color=BLUE, s=28, zorder=3, edgecolor="white", linewidth=0.55)
    ax.scatter(d2048, y, color=ORANGE, s=28, zorder=3, edgecolor="white", linewidth=0.55)
    for old, new, yy in zip(d512, d2048, y, strict=True):
        ax.text(
            2.94,
            yy,
            f"{new - old:+.2f}",
            ha="right",
            va="center",
            fontsize=6.1,
            color=INK,
        )
    ax.set_yticks(y, pair_labels)
    ax.invert_yaxis()
    ax.set_xlabel("centered signature distance")
    ax.set_xlim(0.0, 3.0)
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.0, pad=1.4)
    ax.text(
        0.985,
        1.03,
        r"$\Delta$ (2,048 $-$ 512)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        color=INK,
    )
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=BLUE,
                markeredgecolor="white",
                markersize=5,
                label="512-token",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=ORANGE,
                markeredgecolor="white",
                markersize=5,
                label="2,048-token",
            ),
        ],
        loc="upper left",
        ncol=2,
        handlelength=1.0,
        columnspacing=1.2,
        borderaxespad=0.0,
        bbox_to_anchor=(0.0, 1.08),
    )
    save(fig, "fig_signature_distance_stability")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate audited real-data figures for the Geoprobe paper."
    )
    parser.add_argument(
        "--which",
        choices=("all", "signature", "distance"),
        default="all",
        help="Regenerate only one named figure family (default: all).",
    )
    args = parser.parse_args()
    if args.which in {"all", "signature"}:
        make_signature_figure()
    if args.which in {"all", "distance"}:
        make_signature_distance_figure()


if __name__ == "__main__":
    main()
