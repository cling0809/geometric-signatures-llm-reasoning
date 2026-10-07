#!/usr/bin/env python3
"""Render descriptive all-cell heatmaps for a completed signature confirmation.

This script deliberately consumes the saved partition-wise matrices rather than
recomputing signatures from raw trajectories.  It makes the figure traceable to
the frozen retrieval audit and cannot select a metric, layer, partition or
colour scale from the plotted result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = {
    "partition",
    "model",
    "metric",
    "relative_depth_bin",
    "auc_incorrect_vs_correct",
    "n_incorrect",
    "n_correct",
}
_PARTITIONS = ("A", "B", "C")


def load_partition_cells(audit_dir: Path) -> pd.DataFrame:
    """Load and validate a completed, provenance-gated diagnostic matrix."""
    audit_dir = Path(audit_dir)
    required = {
        "DONE": audit_dir / "DONE",
        "manifest": audit_dir / "manifest.json",
        "input_audit": audit_dir / "input_provenance_audit.json",
        "cells": audit_dir / "partition_signature_cells.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"signature audit is incomplete: missing {missing}")
    manifest = json.loads(required["manifest"].read_text())
    if manifest.get("protocol") != "tacl-11241-diagnostic-generalization-v1":
        raise ValueError("unexpected signature-audit protocol")
    cells = pd.read_csv(required["cells"])
    if missing := _REQUIRED_COLUMNS - set(cells.columns):
        raise ValueError(f"signature cells missing {sorted(missing)}")
    if set(cells["partition"]) != set(_PARTITIONS):
        raise ValueError("signature cells must contain exactly A/B/C partitions")
    if cells.duplicated(["partition", "model", "metric", "relative_depth_bin"]).any():
        raise ValueError("signature cells contain duplicate matrix cells")
    if not cells["auc_incorrect_vs_correct"].between(0.0, 1.0).all():
        raise ValueError("signature AUC values must be in [0, 1]")
    expected_cells = set(
        zip(cells["metric"].astype(str), cells["relative_depth_bin"].astype(int), strict=True)
    )
    for (partition, model), group in cells.groupby(["partition", "model"], sort=True):
        observed_cells = set(
            zip(group["metric"].astype(str), group["relative_depth_bin"].astype(int), strict=True)
        )
        if observed_cells != expected_cells:
            raise ValueError(f"{partition}/{model}: incomplete metric-depth grid")
    support = cells.loc[:, ["n_incorrect", "n_correct"]]
    if (support < int(manifest.get("minimum_per_class", 10))).any().any():
        raise ValueError("signature cells do not satisfy declared class support")
    return cells


def summarize_cells(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return mean and SD across the three immutable problem partitions."""
    keys = ["model", "metric", "relative_depth_bin"]
    grouped = cells.groupby(keys, as_index=False)["auc_incorrect_vs_correct"]
    mean = grouped.mean().rename(columns={"auc_incorrect_vs_correct": "mean_auc"})
    sd = grouped.std(ddof=0).rename(columns={"auc_incorrect_vs_correct": "partition_sd_auc"})
    return mean, sd


def _matrix(frame: pd.DataFrame, *, model: str, value: str) -> tuple[np.ndarray, list[str], list[int]]:
    sub = frame.loc[frame["model"].eq(model)]
    metrics = sorted(sub["metric"].unique())
    depths = sorted(int(value) for value in sub["relative_depth_bin"].unique())
    pivot = sub.pivot(index="metric", columns="relative_depth_bin", values=value)
    pivot = pivot.reindex(index=metrics, columns=depths)
    if pivot.isna().any().any():
        raise ValueError(f"{model}: incomplete metric-depth grid")
    return pivot.to_numpy(dtype=float), metrics, depths


def render_all_cell_heatmaps(mean: pd.DataFrame, sd: pd.DataFrame, out: Path, *, title: str) -> None:
    """Render complete matrix means and partition uncertainty with fixed scales."""
    models = sorted(mean["model"].unique())
    figure, axes = plt.subplots(
        2,
        len(models),
        figsize=(3.1 * len(models), 5.0),
        constrained_layout=True,
        squeeze=False,
    )
    mean_image = None
    sd_image = None
    for column, model in enumerate(models):
        matrix, metrics, depths = _matrix(mean, model=model, value="mean_auc")
        uncertainty, _, _ = _matrix(sd, model=model, value="partition_sd_auc")
        mean_image = axes[0, column].imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=0.30,
            vmax=0.70,
        )
        sd_image = axes[1, column].imshow(
            uncertainty,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.00,
            vmax=0.20,
        )
        axes[0, column].set_title(model, fontsize=9)
        for row, ylabel in enumerate(("mean AUC\n(incorrect vs. correct)", "partition SD")):
            ax = axes[row, column]
            ax.set_xticks(np.linspace(0, len(depths) - 1, 5, dtype=int))
            ax.set_xticklabels(np.linspace(0, 1, 5).round(2))
            ax.set_xlabel("relative depth")
            ax.set_yticks(np.arange(len(metrics)))
            ax.set_yticklabels(metrics if column == 0 else [])
            if column == 0:
                ax.set_ylabel(ylabel)
    assert mean_image is not None and sd_image is not None
    figure.colorbar(mean_image, ax=axes[0, :].tolist(), shrink=0.78, label="AUC")
    figure.colorbar(sd_image, ax=axes[1, :].tolist(), shrink=0.78, label="SD across A/B/C")
    figure.suptitle(title, fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    cells = load_partition_cells(args.audit_dir)
    mean, sd = summarize_cells(cells)
    args.out.mkdir(parents=True, exist_ok=True)
    mean.to_csv(args.out / "signature_matrix_partition_mean.csv", index=False)
    sd.to_csv(args.out / "signature_matrix_partition_sd.csv", index=False)
    render_all_cell_heatmaps(
        mean,
        sd,
        args.out / "signature_all_cells.png",
        title=f"{args.label}: complete length-residualized signature matrices",
    )
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-signature-confirmation-figure-v1",
                "audit_dir": str(args.audit_dir.resolve()),
                "label": args.label,
                "input": "partition_signature_cells.csv",
                "mean_colour_scale": [0.30, 0.70],
                "uncertainty_colour_scale": [0.00, 0.20],
                "outputs": [
                    "signature_matrix_partition_mean.csv",
                    "signature_matrix_partition_sd.csv",
                    "signature_all_cells.png",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
