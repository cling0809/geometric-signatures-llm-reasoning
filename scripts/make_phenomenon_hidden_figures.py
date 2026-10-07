#!/usr/bin/env python3
"""Build real-data phenomenon figures from signatures and hidden states.

This script has two entry points:

```
python scripts/make_phenomenon_hidden_figures.py aggregate --runs-root /root/AI/runs
python scripts/make_phenomenon_hidden_figures.py plot
```

The aggregate step is meant to run on the server, where the full `.pt`
trajectories live. It writes compact projection coordinates. The plot step can
run locally and produces vector PDF plus high-resolution PNG figures.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geoprobe.revision.figure_style import (
    BLUE as PUB_BLUE,
)
from geoprobe.revision.figure_style import (
    GRID as PUB_GRID,
)
from geoprobe.revision.figure_style import (
    INK as PUB_INK,
)
from geoprobe.revision.figure_style import (
    MODEL_COLORS,
    clean_axis,
    panel_title,
    save_publication_figure,
    setup_publication_style,
)
from geoprobe.revision.figure_style import (
    ORANGE as PUB_ORANGE,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIG_DIR = PAPER / "figures"
TMP_DIR = PAPER / "tmp"
EXP_PHEN = ROOT / "experiments" / "2026-05-17_phase3-paradigm-4"

BLUE = "#0f5fc9"
ORANGE = "#df7b18"
TEAL = "#178d8a"
RED = "#d84a3a"
GRAY = "#667085"
INK = "#111827"
GRID = "#d8dee8"
LIGHT_BLUE = "#e9f1ff"
LIGHT_ORANGE = "#fff1dd"
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


@dataclass(frozen=True)
class ModelRun:
    label: str
    run_id: str
    signature_csv: str
    color: str


MODEL_RUNS = [
    ModelRun(
        "Base",
        "2026-05-17_extract-qwen-1.5b-base-100",
        "2026-05-17_extract-qwen-1.5b-base-100.csv",
        GRAY,
    ),
    ModelRun(
        "Instruct",
        "2026-05-17_extract-qwen-1.5b-nomath-100",
        "2026-05-17_extract-qwen-1.5b-nomath-100.csv",
        BLUE,
    ),
    ModelRun(
        "Math-Instruct",
        "2026-05-17_extract-pilot-gsm8k-100",
        "2026-05-17_extract-pilot-gsm8k-100.csv",
        TEAL,
    ),
    ModelRun(
        "R1-Distill",
        "2026-05-17_extract-deepseek-r1-distill-1.5b-100",
        "2026-05-17_extract-deepseek-r1-distill-1.5b-100.csv",
        RED,
    ),
]


def setup_style() -> None:
    setup_publication_style()


def save_figure(fig: plt.Figure, stem: str) -> None:
    save_publication_figure(fig, FIG_DIR, stem)


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def _pca_first_component(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[0]


def _trajectory_file(run_dir: Path, sample_id: int) -> Path:
    candidates = [
        run_dir / "trajectories" / f"sample_{sample_id:04d}.pt",
        run_dir / "trajectories" / f"sample_{sample_id:04d}_idx_0.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no trajectory file for sample_id={sample_id} in {run_dir}")


def aggregate_hidden_projection(runs_root: Path, out_csv: Path) -> None:
    """Project full hidden states into an interpretable two-axis plane.

    For each model, each sample is represented by the mean generated-token hidden
    state at every layer. The x-axis is the supervised correct-minus-wrong
    direction computed only within that model. The y-axis is the first principal
    component of the residual space after removing that direction. This keeps the
    plot faithful to the hidden states while making the correctness contrast
    legible.
    """
    import torch

    rows: list[dict[str, float | int | str | bool]] = []
    for model in MODEL_RUNS:
        run_dir = runs_root / model.run_id
        labels_path = run_dir / "labels.parquet"
        labels = pd.read_parquet(labels_path)
        labels = labels[["sample_id", "correct"]].copy()
        labels["sample_id"] = labels["sample_id"].astype(int)
        labels = labels.sort_values("sample_id")

        pooled: list[np.ndarray] = []
        sample_ids: list[int] = []
        correctness: list[bool] = []
        for rec in labels.to_dict("records"):
            sample_id = int(rec["sample_id"])
            traj_path = _trajectory_file(run_dir, sample_id)
            obj = torch.load(traj_path, map_location="cpu")
            hidden = obj["hidden_states"].float()
            pooled.append(hidden.mean(dim=0).numpy())
            sample_ids.append(sample_id)
            correctness.append(bool(rec["correct"]))

        arr = np.stack(pooled, axis=0)  # [N, L, H]
        n_samples, n_layers, hidden_dim = arr.shape
        flat = arr.reshape(n_samples * n_layers, hidden_dim)
        flat_z = _zscore(flat).reshape(n_samples, n_layers, hidden_dim)
        correct = np.asarray(correctness, dtype=bool)

        correct_mean = flat_z[correct].mean(axis=(0, 1))
        wrong_mean = flat_z[~correct].mean(axis=(0, 1))
        direction = correct_mean - wrong_mean
        direction_norm = np.linalg.norm(direction)
        if not np.isfinite(direction_norm) or direction_norm == 0:
            raise ValueError(f"degenerate correctness direction for {model.label}")
        direction = direction / direction_norm

        all_points = flat_z.reshape(n_samples * n_layers, hidden_dim)
        x_coord = all_points @ direction
        residual = all_points - x_coord[:, None] * direction[None, :]
        residual_pc = _pca_first_component(residual)
        residual_pc = residual_pc - residual_pc.dot(direction) * direction
        residual_pc = residual_pc / np.linalg.norm(residual_pc)

        # Orient the residual axis so layer depth tends upward.
        layer_means = flat_z.mean(axis=0)
        y_by_layer = layer_means @ residual_pc
        if y_by_layer[-1] < y_by_layer[0]:
            residual_pc = -residual_pc

        projected = np.stack(
            [
                flat_z.reshape(n_samples * n_layers, hidden_dim) @ direction,
                flat_z.reshape(n_samples * n_layers, hidden_dim) @ residual_pc,
            ],
            axis=1,
        ).reshape(n_samples, n_layers, 2)

        for i, sample_id in enumerate(sample_ids):
            for layer in range(n_layers):
                rows.append(
                    {
                        "model": model.label,
                        "sample_id": sample_id,
                        "correct": bool(correct[i]),
                        "layer": layer,
                        "x_correctness": float(projected[i, layer, 0]),
                        "y_residual_pc": float(projected[i, layer, 1]),
                    }
                )

        sep = float(
            projected[correct, 14, 0].mean() - projected[~correct, 14, 0].mean()
        )
        print(
            f"{model.label}: samples={n_samples}, layers={n_layers}, "
            f"hidden={hidden_dim}, layer14_sep={sep:.3f}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


def plot_signature_heatmaps() -> None:
    setup_style()
    fig = plt.figure(figsize=(7.05, 3.05), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.082,
        right=0.988,
        top=0.925,
        bottom=0.205,
        wspace=0.035,
        hspace=0.28,
    )
    axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    image = None

    for idx, (ax, model) in enumerate(zip(axes, MODEL_RUNS, strict=True)):
        sig = pd.read_csv(EXP_PHEN / model.signature_csv).set_index("metric")
        sig = sig.reindex(METRIC_ORDER)
        mat = sig.to_numpy(dtype=float)
        image = ax.imshow(
            mat,
            aspect="auto",
            cmap=DIVERGING,
            vmin=0.20,
            vmax=0.80,
            interpolation="nearest",
        )
        ax.set_title(
            f"{chr(65 + idx)}  {model.label}",
            loc="left",
            pad=1.5,
            color=model.color,
            fontsize=8.8,
            fontweight="bold",
        )
        ax.set_xticks([0, 7, 14, 21, 28])
        ax.set_xticklabels(["0", "7", "14", "21", "28"])
        ax.set_yticks(np.arange(len(METRIC_ORDER)))
        if idx % 2 == 0:
            ax.set_yticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
        else:
            ax.set_yticklabels([])
        if idx >= 2:
            ax.set_xlabel("layer", labelpad=1.0)
        ax.tick_params(length=0, pad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for y in np.arange(0.5, len(METRIC_ORDER), 1.0):
            ax.axhline(y, color="white", lw=0.35, alpha=0.55)

        if model.label == "Math-Instruct":
            row_idx = METRIC_ORDER.index("mean_step_norm")
            ax.add_patch(
                mpl.patches.Rectangle(
                    (11.5, row_idx - 0.48),
                    16.9,
                    0.96,
                    fill=False,
                    ec=INK,
                    lw=0.85,
                )
            )
        if model.label == "R1-Distill":
            row_idx = METRIC_ORDER.index("curvature_mean")
            ax.add_patch(
                mpl.patches.Rectangle(
                    (-0.48, row_idx - 0.48),
                    28.96,
                    0.96,
                    fill=False,
                    ec=RED,
                    lw=0.85,
                )
            )

    assert image is not None
    cax = fig.add_axes([0.285, 0.078, 0.48, 0.038])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_ticks([0.3, 0.5, 0.7])
    colorbar.set_label("AUC for incorrect trajectories", labelpad=2, fontsize=7.2)
    colorbar.ax.tick_params(labelsize=6.8, length=2)
    save_figure(fig, "fig2_signature_heatmaps")


def _confidence_interval(vals: np.ndarray) -> tuple[float, float]:
    if len(vals) <= 1:
        return float(vals.mean()), 0.0
    se = vals.std(ddof=1) / math.sqrt(len(vals))
    return float(vals.mean()), float(1.96 * se)


def plot_hidden_projection(coords_csv: Path) -> None:
    setup_style()
    df = pd.read_csv(coords_csv)
    fig = plt.figure(figsize=(7.05, 3.05), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.062,
        right=0.992,
        top=0.89,
        bottom=0.105,
        wspace=0.14,
        hspace=0.30,
    )
    axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    layer_focus = 14

    for idx, (ax, model) in enumerate(zip(axes, MODEL_RUNS, strict=True)):
        sub = df[df["model"] == model.label].copy()
        traj_scale = np.nanpercentile(np.abs(sub["y_residual_pc"]), 95)
        corr_scale = np.nanpercentile(np.abs(sub["x_correctness"]), 95)
        if not np.isfinite(traj_scale) or traj_scale == 0:
            traj_scale = 1.0
        if not np.isfinite(corr_scale) or corr_scale == 0:
            corr_scale = 1.0
        sub["trajectory_axis"] = sub["y_residual_pc"] / traj_scale
        sub["correctness_axis"] = sub["x_correctness"] / corr_scale
        focus = sub[sub["layer"] == layer_focus]

        for correct, color, marker, _label, zorder in [
            (True, BLUE, "o", "correct", 3),
            (False, ORANGE, "s", "wrong", 2),
        ]:
            pts = focus[focus["correct"] == correct]
            ax.scatter(
                pts["trajectory_axis"],
                pts["correctness_axis"],
                s=8,
                color=color,
                alpha=0.18,
                linewidths=0,
                marker=marker,
                zorder=zorder,
            )

        means = (
            sub.groupby(["correct", "layer"], as_index=False)[
                ["trajectory_axis", "correctness_axis"]
            ]
            .mean()
            .sort_values("layer")
        )
        for correct, color, label in [
            (True, BLUE, "correct"),
            (False, ORANGE, "wrong"),
        ]:
            line = means[means["correct"] == correct]
            ax.plot(
                line["trajectory_axis"],
                line["correctness_axis"],
                color=color,
                lw=1.55,
                label=label if idx == 0 else None,
                zorder=6,
            )
            marks = line[line["layer"].isin([0, layer_focus, 28])]
            ax.scatter(
                marks["trajectory_axis"],
                marks["correctness_axis"],
                s=[15 if layer_value != layer_focus else 24 for layer_value in marks["layer"]],
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=7,
            )

        cvals = focus[focus["correct"]]["correctness_axis"].to_numpy()
        wvals = focus[~focus["correct"]]["correctness_axis"].to_numpy()
        delta = cvals.mean() - wvals.mean()

        ax.axvline(0, color=GRID, lw=0.7, zorder=0)
        ax.axhline(0, color=GRID, lw=0.7, zorder=0)
        ax.grid(color=GRID, lw=0.45, alpha=0.55)
        ax.set_title(
            f"{chr(65 + idx)}  {model.label}",
            loc="left",
            color=model.color,
            fontsize=8.8,
            fontweight="bold",
            pad=1.0,
        )
        if idx >= 2:
            ax.set_xlabel("trajectory axis", labelpad=1.5)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if idx % 2 == 0:
            ax.set_ylabel("correctness direction", labelpad=1.5)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.tick_params(length=2.5, pad=1.5)
        ax.text(
            0.985,
            1.015,
            rf"$\Delta_{{L14}}={delta:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.3,
            color="#4b5563",
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.995),
        ncol=2,
        handlelength=2.1,
        columnspacing=1.4,
    )
    save_figure(fig, "fig3_hidden_projection")


def plot_correctness_separation(coords_csv: Path) -> None:
    """Render the descriptive correctness-direction density comparison."""
    setup_style()
    df = pd.read_csv(coords_csv)
    layer_focus = 14
    fig, axes = plt.subplots(1, 4, figsize=(7.10, 1.82), sharey=True)
    # Use one shared x-axis label instead of repeating a long phrase under only
    # two middle panels.  This makes the four density panels read as one
    # homogeneous comparison rather than as four differently annotated plots.
    fig.subplots_adjust(left=0.055, right=0.995, top=0.77, bottom=0.30, wspace=0.13)
    bins = np.linspace(-2.6, 2.6, 45)
    centers = (bins[:-1] + bins[1:]) / 2
    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    kernel /= kernel.sum()

    for idx, (ax, model) in enumerate(zip(axes, MODEL_RUNS, strict=True)):
        sub = df[(df["model"] == model.label) & (df["layer"] == layer_focus)].copy()
        vals = sub["x_correctness"].to_numpy(dtype=float)
        scale = vals.std(ddof=0)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        sub["z"] = (vals - vals.mean()) / scale
        correct_vals = sub.loc[sub["correct"], "z"].to_numpy(dtype=float)
        wrong_vals = sub.loc[~sub["correct"], "z"].to_numpy(dtype=float)

        for arr, color, label in [
            (wrong_vals, PUB_ORANGE, "incorrect"),
            (correct_vals, PUB_BLUE, "correct"),
        ]:
            hist, _ = np.histogram(arr, bins=bins, density=True)
            smooth = np.convolve(hist, kernel, mode="same")
            ax.fill_between(centers, 0, smooth, color=color, alpha=0.16, linewidth=0)
            ax.plot(centers, smooth, color=color, linewidth=1.25, label=label if idx == 0 else None)

        cmean = float(correct_vals.mean())
        wmean = float(wrong_vals.mean())
        ax.axvline(wmean, color=PUB_ORANGE, linewidth=0.8, linestyle="--")
        ax.axvline(cmean, color=PUB_BLUE, linewidth=0.8)
        ax.axvline(0, color=PUB_GRID, linewidth=0.65, zorder=0)
        ax.set_xlim(-2.55, 2.55)
        ax.set_ylim(bottom=0)
        ax.set_xticks([-2, 0, 2])
        ax.grid(axis="x", color=PUB_GRID, linewidth=0.35)
        clean_axis(ax, grid=None)
        panel_title(ax, chr(65 + idx), model.label, color=MODEL_COLORS.get(model.label, PUB_INK))
        ax.text(
            0.98,
            1.025,
            rf"$\Delta_{{14}}={cmean - wmean:.2f}\sigma$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.0,
            color=PUB_INK,
        )
        ax.tick_params(axis="y", left=False, labelleft=False)
        if idx == 0:
            ax.set_ylabel("density", labelpad=2)
        ax.set_xlabel("")

    fig.text(
        0.52,
        0.055,
        "projection on correctness direction",
        ha="center",
        va="center",
        fontsize=7.0,
        color=PUB_INK,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.01),
        ncol=2,
        handlelength=2.0,
        columnspacing=1.3,
    )
    save_figure(fig, "fig3_correctness_separation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--runs-root", type=Path, default=Path("/root/AI/runs"))
    aggregate.add_argument(
        "--out",
        type=Path,
        default=TMP_DIR / "hidden_projection_coords.csv",
    )

    plot = sub.add_parser("plot")
    plot.add_argument(
        "--coords",
        type=Path,
        default=TMP_DIR / "hidden_projection_coords.csv",
    )
    plot.add_argument(
        "--which",
        choices=["all", "heatmap", "projection", "separation"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "aggregate":
        aggregate_hidden_projection(args.runs_root, args.out)
        return

    if args.which in {"all", "heatmap"}:
        plot_signature_heatmaps()
    if args.which in {"all", "projection"}:
        plot_hidden_projection(args.coords)
    if args.which in {"all", "projection", "separation"}:
        plot_correctness_separation(args.coords)


if __name__ == "__main__":
    main()
