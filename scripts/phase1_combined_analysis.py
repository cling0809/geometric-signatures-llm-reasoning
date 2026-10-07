#!/usr/bin/env python3
"""Phase 1 follow-up: 5-fold CV-AUC for single + combined features.

Tests whether geometric metrics ADD predictive value on top of the trivial
n_gen_tokens baseline. We pick the best layer per metric (by raw AUC against
the per-layer table) and combine them via logistic regression.

Usage:
    python scripts/phase1_combined_analysis.py --run ~/AI/runs/<exp-id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def cv_auc(X: np.ndarray, y: np.ndarray, name: str, k: int = 5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    proba = cross_val_predict(
        LogisticRegression(max_iter=1000),
        X,
        y,
        cv=skf,
        method="predict_proba",
    )[:, 1]
    auc = roc_auc_score(y, proba)
    print(f"  {name:55s} cv-AUC = {auc:.3f}")
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()

    mdf = pd.read_parquet(args.run / "metrics.parquet")
    labels = pd.read_parquet(args.run / "labels.parquet").set_index("sample_id")
    y = (~labels["correct"]).astype(int).to_numpy()

    wide = mdf.pivot_table(index="sample_id", columns=["metric", "layer"], values="value")
    wide.columns = [f"{m}_L{l}" for (m, l) in wide.columns]
    wide["n_gen_tokens"] = labels["n_gen_tokens"]

    # Pick the best layer per metric by raw AUC against y (taking abs(AUC-0.5))
    def best_layer(metric: str) -> str:
        cols = [c for c in wide.columns if c.startswith(f"{metric}_L")]
        scores = {c: abs(roc_auc_score(y, wide[c]) - 0.5) for c in cols}
        return max(scores, key=scores.get)

    tl_col = best_layer("trajectory_length")
    mc_col = best_layer("mean_curvature")
    ngt = "n_gen_tokens"
    n_gen_auc = roc_auc_score(y, wide[ngt])

    print("=== raw single-feature AUC ===")
    print(f"  baseline {ngt:50s} AUC = {n_gen_auc:.3f}")
    for c in (tl_col, mc_col):
        a = roc_auc_score(y, wide[c])
        a_signed = max(a, 1 - a)
        print(f"  {c:50s} AUC = {a:.3f}  (|signed| = {a_signed:.3f})")

    print()
    print("=== 5-fold CV logistic regression ===")
    cv_auc(wide[[ngt]].values, y, f"baseline: {ngt} only")
    cv_auc(wide[[tl_col]].values, y, f"{tl_col} only")
    cv_auc(wide[[mc_col]].values, y, f"{mc_col} only")
    cv_auc(wide[[ngt, mc_col]].values, y, f"{ngt} + {mc_col}")
    cv_auc(wide[[ngt, tl_col]].values, y, f"{ngt} + {tl_col}")
    cv_auc(wide[[ngt, tl_col, mc_col]].values, y, f"{ngt} + length + curvature")

    mc_cols = [c for c in wide.columns if c.startswith("mean_curvature_")]
    tl_cols = [c for c in wide.columns if c.startswith("trajectory_length_")]
    cv_auc(wide[[ngt] + mc_cols].values, y, f"{ngt} + all 29 curvature layers")
    cv_auc(wide[[ngt] + tl_cols + mc_cols].values, y, f"{ngt} + all length + curvature layers")


if __name__ == "__main__":
    main()
