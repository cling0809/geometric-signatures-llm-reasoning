"""Paradigm-level geometric signatures and inter-model comparison.

A "geometric signature" of a model is a (metric x layer) matrix where each
entry is the raw AUC of that (metric, layer) for predicting *that model's*
incorrect answers.

Interpretation:
  entry > 0.5 -> higher metric value => more likely incorrect
  entry < 0.5 -> higher metric value => more likely correct (sign-flipped)
  entry = 0.5 -> no signal

Two models' signatures are compared by flattening the matrices into vectors
and computing distances (L2, or signed-distance from 0.5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from geoprobe.analysis.auc import compute_auc_table


def signature_matrix(metrics_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Return a (metric x layer) DataFrame of raw AUC values.

    Index = metric name (sorted). Columns = layer index (sorted).
    """
    table = compute_auc_table(metrics_df, labels_df, positive_label="incorrect")
    sig = table.pivot(index="metric", columns="layer", values="auc")
    sig = sig.sort_index().sort_index(axis=1)
    return sig


def load_signatures(runs: list[Path]) -> dict[str, pd.DataFrame]:
    """Load each run's metrics + labels, return {run_name: signature_matrix}."""
    out: dict[str, pd.DataFrame] = {}
    for r in runs:
        m = pd.read_parquet(r / "metrics.parquet")
        l = pd.read_parquet(r / "labels.parquet")
        out[r.name] = signature_matrix(m, l)
    return out


def pairwise_signature_distance(sigs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """L2 distance between flattened (auc - 0.5) signature vectors.

    Subtracting 0.5 means "no-signal cells contribute zero distance" — a
    paradigm-induced difference appears as cells deviating from 0.5 in
    opposite directions across two models.
    """
    names = list(sigs.keys())
    K = len(names)
    M = np.zeros((K, K), dtype=float)
    centred = {n: (sig.values - 0.5).flatten() for n, sig in sigs.items()}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            M[i, j] = float(np.linalg.norm(centred[ni] - centred[nj]))
    return pd.DataFrame(M, index=names, columns=names)


def plot_signature_grid(
    sigs: dict[str, pd.DataFrame],
    out_path: str | Path,
    labels: dict[str, str] | None = None,
    figsize_per: tuple[float, float] = (6.0, 3.5),
):
    """Grid of heatmaps, one per model. Diverging colormap centred on 0.5."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    K = len(sigs)
    ncols = 2 if K > 1 else 1
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False)
    norm = TwoSlopeNorm(vmin=0.20, vcenter=0.50, vmax=0.80)
    for k, (name, sig) in enumerate(sigs.items()):
        ax = axes[k // ncols, k % ncols]
        im = ax.imshow(sig.values, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_yticks(range(len(sig.index)))
        ax.set_yticklabels(sig.index, fontsize=8)
        ax.set_xticks(range(0, sig.shape[1], 4))
        ax.set_xticklabels(range(0, sig.shape[1], 4))
        ax.set_xlabel("layer")
        title = labels[name] if labels and name in labels else name
        ax.set_title(title, fontsize=10)
    # hide unused axes
    for k in range(K, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="AUC (predict incorrect)")
    fig.suptitle("Geometric signatures: (metric × layer) AUC per model", fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def plot_pairwise_distance(
    dist: pd.DataFrame,
    out_path: str | Path,
    labels: dict[str, str] | None = None,
):
    """Heatmap of pairwise L2 signature distances."""
    import matplotlib.pyplot as plt

    n = dist.shape[0]
    fig, ax = plt.subplots(figsize=(1.2 + 0.7 * n, 1.0 + 0.7 * n))
    im = ax.imshow(dist.values, cmap="viridis")
    tick_labels = [labels[x] if labels and x in labels else x for x in dist.index]
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if dist.values[i, j] > dist.values.max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="L2 dist (sig - 0.5)")
    ax.set_title("Pairwise signature distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    import matplotlib.pyplot as _plt
    _plt.close(fig)
