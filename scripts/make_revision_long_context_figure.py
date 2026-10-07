#!/usr/bin/env python3
"""Make homogeneous long-context audit figures for the TACL revision."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geoprobe.revision.figure_style import (
    BLUE,
    INK,
    ORANGE,
    TEAL,
    clean_axis,
    panel_title,
    save_publication_figure,
    setup_publication_style,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "revision" / "evidence" / "long-context-qwen-instruct-v3"
FIGURES = ROOT / "paper" / "figures"

POLICIES = ["constant", "exponential-1024", "prefix-256", "relative-hidden-rms"]
POLICY_LABELS = ["constant", "exp-1024", "prefix-256", "relative-RMS"]
BUDGETS = [4096, 32768]
BUDGET_LABELS = ["4,096", "32,768"]
BUDGET_COLORS = [BLUE, ORANGE]
DIRECTIONS = {
    "source": {"title": "Source CrossSteer", "dir": "crosssteer", "color": BLUE},
    "target": {"title": "Target-calibrated", "dir": "target-calibrated", "color": TEAL},
}


def load(direction: str, budget: int) -> list[dict[str, float | str]]:
    path = EVIDENCE / f"{DIRECTIONS[direction]['dir']}-b{budget}" / "long_context_summary.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rows_by_policy(direction: str) -> dict[str, dict[int, dict[str, float]]]:
    out: dict[str, dict[int, dict[str, float]]] = {}
    for budget in BUDGETS:
        for row in load(direction, budget):
            out.setdefault(row["policy"], {})[budget] = {
                "delta_pp": float(row["delta_pp"]),
                "ci_low": float(row["bootstrap_ci_low_pp"]),
                "ci_high": float(row["bootstrap_ci_high_pp"]),
                "token_delta": float(row["generated_token_delta"]),
                "truncation": float(row["method_truncation_rate"]),
            }
    assert set(out) == set(POLICIES), (direction, sorted(out))
    assert all(set(values) == set(BUDGETS) for values in out.values())
    return out


def draw_accuracy(ax: plt.Axes, direction: str, data: dict[str, dict[int, dict[str, float]]]) -> None:
    """Draw one homogeneous point-and-interval panel."""
    x = np.arange(len(POLICIES), dtype=float)
    offsets = [-0.105, 0.105]
    for offset, budget, color, label in zip(
        offsets, BUDGETS, BUDGET_COLORS, BUDGET_LABELS, strict=True
    ):
        delta = np.array([data[p][budget]["delta_pp"] for p in POLICIES])
        low = np.array([data[p][budget]["ci_low"] for p in POLICIES])
        high = np.array([data[p][budget]["ci_high"] for p in POLICIES])
        yerr = np.vstack([delta - low, high - delta])
        ax.errorbar(
            x + offset,
            delta,
            yerr=yerr,
            fmt="o",
            ms=4.0,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            label=f"{label} tokens",
            zorder=3,
        )
    ax.axhline(0, color=INK, linewidth=0.75)
    ax.set_ylim(-10, 16)
    ax.set_yticks([-10, -5, 0, 5, 10, 15])
    ax.set_xticks(x, POLICY_LABELS, rotation=18, ha="right")
    ax.set_ylabel("paired $\\Delta$ accuracy (pp)")
    panel_title(ax, "", DIRECTIONS[direction]["title"], color=DIRECTIONS[direction]["color"])
    clean_axis(ax, grid="y")
    ax.grid(axis="x", visible=False)


def draw_length(ax: plt.Axes, direction: str, data: dict[str, dict[int, dict[str, float]]]) -> None:
    """Draw one homogeneous lollipop panel for the long-context cost."""
    x = np.arange(len(POLICIES), dtype=float)
    vals = np.array([data[p][32768]["token_delta"] for p in POLICIES])
    color = DIRECTIONS[direction]["color"]
    ax.axhline(0, color=INK, linewidth=0.75, zorder=1)
    ax.vlines(x, 0, vals, color=color, linewidth=1.5, alpha=0.82, zorder=2)
    ax.scatter(x, vals, color=color, edgecolor="white", linewidth=0.45, s=30, zorder=3)
    for xpos, value in zip(x, vals, strict=True):
        offset = 8 if value >= 0 else -8
        ax.annotate(
            f"{value:+.0f}",
            (xpos, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=6.0,
            color=INK,
        )
    ax.set_ylim(-80, 380)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xticks(x, POLICY_LABELS, rotation=18, ha="right")
    ax.set_ylabel("method $-$ baseline tokens")
    panel_title(ax, "", DIRECTIONS[direction]["title"], color=color)
    clean_axis(ax, grid="y")
    ax.grid(axis="x", visible=False)


def main() -> None:
    setup_publication_style()
    source = rows_by_policy("source")
    target = rows_by_policy("target")

    # Figure A: both panels use the same point-and-interval grammar.
    fig_accuracy, axes_accuracy = plt.subplots(1, 2, figsize=(6.95, 2.55), constrained_layout=False)
    fig_accuracy.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.25, wspace=0.30)
    draw_accuracy(axes_accuracy[0], "source", source)
    draw_accuracy(axes_accuracy[1], "target", target)
    handles, labels = axes_accuracy[0].get_legend_handles_labels()
    fig_accuracy.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        columnspacing=1.5,
        frameon=False,
    )
    save_publication_figure(fig_accuracy, FIGURES, "fig_long_context_accuracy")
    plt.close(fig_accuracy)

    # Figure B: both panels use the same lollipop grammar.
    fig_length, axes_length = plt.subplots(1, 2, figsize=(6.95, 2.55), constrained_layout=False)
    fig_length.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.25, wspace=0.30)
    draw_length(axes_length[0], "source", source)
    draw_length(axes_length[1], "target", target)
    save_publication_figure(fig_length, FIGURES, "fig_long_context_length")
    plt.close(fig_length)


if __name__ == "__main__":
    main()
