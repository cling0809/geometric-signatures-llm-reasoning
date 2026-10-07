#!/usr/bin/env python3
"""Render the 512/2048 signature-distance robustness figure from CSV data."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"
CSV_512 = ROOT / "experiments" / "2026-05-17_phase3-paradigm-4" / "pairwise_signature_distance.csv"
CSV_2048 = ROOT / "experiments" / "2026-05-20_phase3-robust2048-distance" / "pairwise_signature_distance.csv"

ORDER = ["Base", "Instruct", "Math", "R1"]

NAME_MAP = {
    "2026-05-17_extract-qwen-1.5b-base-100": "Base",
    "2026-05-17_extract-qwen-1.5b-nomath-100": "Instruct",
    "2026-05-17_extract-pilot-gsm8k-100": "Math",
    "2026-05-17_extract-deepseek-r1-distill-1.5b-100": "R1",
    "2026-05-20_robust2048-qwen-base-gsm8k-100": "Base",
    "2026-05-20_robust2048-qwen-instruct-gsm8k-100": "Instruct",
    "2026-05-20_robust2048-qwen-math-gsm8k-100": "Math",
    "2026-05-20_robust2048-r1-distill-gsm8k-100": "R1",
}


def read_matrix(path: Path) -> pd.DataFrame:
    dist = pd.read_csv(path, index_col=0)
    dist.index = [NAME_MAP[x] for x in dist.index]
    dist.columns = [NAME_MAP[x] for x in dist.columns]
    return dist.loc[ORDER, ORDER]


def draw_matrix(ax, mat: pd.DataFrame, title: str, vmax: float):
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "geoprobe_blues", ["#f8fbff", "#dbeafe", "#93c5fd", "#2563eb", "#0b3b8c"]
    )
    im = ax.imshow(mat.values, vmin=0, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=9.5, pad=7)
    ax.set_xticks(range(len(ORDER)))
    ax.set_yticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, fontsize=8)
    ax.set_yticklabels(ORDER, fontsize=8)
    ax.tick_params(which="major", length=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-0.5, len(ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)

    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            val = mat.iloc[i, j]
            if i == j:
                text = "0"
                color = "#64748b"
            else:
                text = f"{val:.2f}"
                color = "white" if val > vmax * 0.55 else "#0f172a"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=color)
    return im


def main() -> None:
    mat512 = read_matrix(CSV_512)
    mat2048 = read_matrix(CSV_2048)
    vmax = max(float(mat512.values.max()), float(mat2048.values.max()))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    fig = plt.figure(figsize=(7.0, 2.65))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.08, 1, 0.18], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])

    im = draw_matrix(ax1, mat512, "512-token controlled extraction", vmax)
    draw_matrix(ax2, mat2048, "2048-token robustness extraction", vmax)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("signature distance", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_linewidth(0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "signature_distance_robustness_real.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "signature_distance_robustness_real.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
