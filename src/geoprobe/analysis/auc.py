"""AUC / AP analysis for (metric, layer) vs correctness.

We measure ranking quality: does the metric value (or its negation) rank
correct trajectories above incorrect ones? Both directions are reported so
the user can see which sign is informative.

We also include a trivial baseline (n_gen_tokens vs correctness) for context —
generation length is often correlated with failure (the model rambles into
max_new_tokens), and any geometric metric must beat it to be interesting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """roc_auc_score but returns NaN when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_auc_table(
    metrics_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    positive_label: str = "incorrect",
) -> pd.DataFrame:
    """For each (metric, layer) pair: compute AUC of the metric as predictor of *incorrectness*.

    Why incorrectness as positive class: H1 says failure trajectories have
    higher curvature. So a higher metric value should rank incorrect samples
    higher. AUC > 0.5 means the metric correctly flags failures.
    """
    if positive_label not in ("incorrect", "correct"):
        raise ValueError(positive_label)
    merged = metrics_df.merge(labels_df[["sample_id", "correct"]], on="sample_id")
    y_pos = (~merged["correct"]) if positive_label == "incorrect" else merged["correct"]

    rows: list[dict] = []
    for (metric, layer), g in merged.groupby(["metric", "layer"]):
        y = y_pos.loc[g.index].to_numpy().astype(int)
        s = g["value"].to_numpy().astype(float)
        rows.append(
            {
                "metric": metric,
                "layer": int(layer),
                "n": int(len(s)),
                "auc": _safe_auc(y, s),
                "ap": _safe_ap(y, s),
            }
        )
    out = pd.DataFrame(rows).sort_values(["metric", "layer"]).reset_index(drop=True)
    return out


def trivial_baseline(labels_df: pd.DataFrame, column: str = "n_gen_tokens") -> dict:
    y = (~labels_df["correct"]).to_numpy().astype(int)
    s = labels_df[column].to_numpy().astype(float)
    return {"feature": column, "auc": _safe_auc(y, s), "ap": _safe_ap(y, s)}


def partial_auc(
    metric_values: np.ndarray,
    confounder: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """AUC of metric_values, after linearly residualizing out `confounder`.

    Use this to ask: does the metric carry signal *beyond* what the
    confounder (e.g. n_gen_tokens) already provides? If partial_auc is near
    0.5, the metric is redundant with the confounder.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    x = np.asarray(metric_values, dtype=float)
    c = np.asarray(confounder, dtype=float)
    # OLS: x = a*c + b ; residual = x - (a*c + b)
    A = np.vstack([c, np.ones_like(c)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    residual = x - A @ coef
    return float(roc_auc_score(y_true, residual))


def compute_partial_auc_table(
    metrics_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    confounder_col: str = "n_gen_tokens",
    positive_label: str = "incorrect",
) -> pd.DataFrame:
    """Like compute_auc_table but reports partial AUC residualizing the confounder."""
    merged = metrics_df.merge(labels_df[["sample_id", "correct", confounder_col]], on="sample_id")
    y_pos = (~merged["correct"]) if positive_label == "incorrect" else merged["correct"]
    rows: list[dict] = []
    for (metric, layer), g in merged.groupby(["metric", "layer"]):
        y = y_pos.loc[g.index].to_numpy().astype(int)
        s = g["value"].to_numpy().astype(float)
        c = g[confounder_col].to_numpy().astype(float)
        rows.append(
            {
                "metric": metric,
                "layer": int(layer),
                "n": int(len(s)),
                "auc": _safe_auc(y, s),
                "partial_auc": partial_auc(s, c, y),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "layer"]).reset_index(drop=True)


def plot_auc_per_layer(
    auc_df: pd.DataFrame,
    baseline_auc: float | None = None,
    title: str = "",
    out_path: str | Path | None = None,
):
    """Line plot: x=layer, y=AUC, one line per metric. Saves to out_path if given."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric, g in auc_df.groupby("metric"):
        ax.plot(g["layer"], g["auc"], marker="o", label=metric)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="chance (0.5)")
    if baseline_auc is not None:
        ax.axhline(baseline_auc, color="black", linestyle="--", linewidth=1,
                   label=f"n_gen_tokens baseline ({baseline_auc:.3f})")
    ax.set_xlabel("layer (0 = embedding)")
    ax.set_ylabel("AUC (predicting incorrect)")
    ax.set_ylim(0.3, 1.0)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
    return fig
