"""Shared publication figure style for the TACL-11241 revision.

The module keeps all data-driven figures visually consistent while leaving the
conceptual overview (Fig. 1) untouched.  It is intentionally small: figure
scripts still own their data transformations and scientific annotations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#172033"
MUTED = "#64748B"
GRID = "#D8E1EA"
BLUE = "#2F6DB0"
ORANGE = "#D77A18"
TEAL = "#1B8A89"
RED = "#C94B3F"
PURPLE = "#7656A6"
PALE_BLUE = "#E8F0F8"
PALE_ORANGE = "#FCEEDC"

MODEL_COLORS = {
    "Base": MUTED,
    "Instruct": BLUE,
    "Math-Instruct": TEAL,
    "R1-Distill": RED,
}


def setup_publication_style() -> None:
    """Apply a compact, vector-safe style suitable for a two-column paper."""
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.1,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.65,
            "axes.edgecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax: plt.Axes, *, grid: str | None = "y") -> None:
    """Apply consistent axes and grid treatment."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(length=2.2, width=0.55, pad=1.8)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.45, alpha=0.9)
        ax.set_axisbelow(True)


def panel_title(ax: plt.Axes, letter: str, title: str, *, color: str = INK) -> None:
    """Draw a compact panel label and title without oversized typography."""
    ax.set_title(f"{letter}  {title}", loc="left", color=color, pad=3.5, fontweight="bold")


def save_publication_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    """Save an editable PDF and a high-resolution PNG preview."""
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf")
    fig.savefig(directory / f"{stem}.png", dpi=600)
    plt.close(fig)
    print(directory / f"{stem}.pdf")
