#!/usr/bin/env python3
"""Render a compact, vector-safe curvature-angle schematic for the methods."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

from geoprobe.revision.figure_style import (
    INK,
    RED,
    TEAL,
    save_publication_figure,
    setup_publication_style,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.45,
            color=color,
            shrinkA=4,
            shrinkB=5,
        )
    )


def path_panel(
    ax: plt.Axes,
    *,
    points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    title: str,
    point_color: str,
    angle_label: str,
    letter: str,
) -> None:
    ax.plot(*zip(*points, strict=True), color=INK, linewidth=1.1, zorder=1)
    ax.scatter(*zip(*points, strict=True), color=point_color, s=28, zorder=2, edgecolor="white", linewidth=0.5)
    offsets = ((3, 7), (4, -17), (4, 7))
    for index, ((x, y), offset) in enumerate(zip(points, offsets, strict=True)):
        label = rf"$h_{{t+{index}}}$" if index else r"$h_t$"
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8.0, color=INK)
    arrow(ax, points[0], points[1], TEAL)
    arrow(ax, points[1], points[2], RED)
    ax.text(0.58, 0.35, r"$v_t$", color=TEAL, fontsize=8.0)
    ax.text(1.78, 0.43 if angle_label == "(small)" else 1.24, r"$v_{t+1}$", color=RED, fontsize=8.0)
    if angle_label == "small":
        theta1, theta2, text_xy = 9, 24, (1.35, 0.83)
    else:
        theta1, theta2, text_xy = 11, 76, (1.67, 0.88)
    ax.add_patch(Arc(points[1], 0.72, 0.72, angle=0, theta1=theta1, theta2=theta2, color=INK, linewidth=0.9))
    ax.text(*text_xy, rf"$\theta_t$ {angle_label}", fontsize=7.5, color=INK)
    ax.text(0.02, 1.04, f"{letter}  {title}", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=INK)
    ax.set_aspect("equal")
    ax.set_xlim(-0.25, 3.2)
    ax.set_ylim(-1.0, 2.15)
    ax.axis("off")


def main() -> None:
    setup_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), constrained_layout=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.16, wspace=0.12)
    path_panel(
        axes[0],
        points=((0.0, 0.0), (1.25, 0.25), (2.55, 0.62)),
        title="small turn",
        point_color=TEAL,
        angle_label="(small)",
        letter="A",
    )
    path_panel(
        axes[1],
        points=((0.0, 0.0), (1.25, 0.25), (1.62, 1.62)),
        title="large turn",
        point_color=RED,
        angle_label="(large)",
        letter="B",
    )
    save_publication_figure(fig, FIGURES, "fig_curvature_schematic")


if __name__ == "__main__":
    main()
