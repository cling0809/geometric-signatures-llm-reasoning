#!/usr/bin/env python3
"""Aggregate per-metric AUC across multiple model runs.

Phase 2 P1 + P2 centerpiece. Given N run directories, for each:
  - compute metrics if missing
  - compute raw AUC and partial-AUC (residualizing n_gen_tokens)
  - look up the best layer for each metric

Then produce:
  - cross_model_summary.csv: row per (model, metric) with best layer + auc + partial_auc
  - cross_model_curvature_signflip.csv: where does mean curvature sign-flip hold?
  - plots/cross_model_curvature_per_layer.png

Usage:
    python scripts/cross_model_compare.py \
        --runs run_a run_b run_c ... \
        --out  ~/AI/runs/<aggregate-exp-id>/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from geoprobe.analysis import (
    compute_auc_table,
    compute_partial_auc_table,
    trivial_baseline,
    write_run_metrics,
)


def _ensure_metrics(run: Path):
    if not (run / "metrics.parquet").exists():
        print(f"[{run.name}] computing metrics ...")
        write_run_metrics(run)


def _per_run_tables(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _ensure_metrics(run)
    metrics = pd.read_parquet(run / "metrics.parquet")
    labels = pd.read_parquet(run / "labels.parquet")
    raw = compute_auc_table(metrics, labels)
    par = compute_partial_auc_table(metrics, labels, confounder_col="n_gen_tokens")
    base = trivial_baseline(labels, column="n_gen_tokens")
    return raw, par, base


def _best_layer_per_metric(raw: pd.DataFrame, par: pd.DataFrame) -> pd.DataFrame:
    """For each metric: best layer by |raw_auc - 0.5| (so sign-flipped also counts)."""
    rows = []
    for metric, g in raw.groupby("metric"):
        g = g.copy()
        g["signed"] = (g["auc"] - 0.5).abs() + 0.5
        idx = g["signed"].idxmax()
        layer = int(g.loc[idx, "layer"])
        raw_auc = float(g.loc[idx, "auc"])
        par_auc = float(par[(par["metric"] == metric) & (par["layer"] == layer)]["partial_auc"].iloc[0])
        rows.append(
            {
                "metric": metric,
                "best_layer": layer,
                "auc": raw_auc,
                "auc_signed": max(raw_auc, 1 - raw_auc),
                "partial_auc": par_auc,
                "partial_auc_signed": max(par_auc, 1 - par_auc),
                "sign_flipped": raw_auc < 0.5,
            }
        )
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, nargs="+", type=Path,
                    help="Run directories (e.g. ~/AI/runs/2026-05-17_extract-*)")
    ap.add_argument("--out", required=True, type=Path, help="Aggregate output directory")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plots").mkdir(exist_ok=True)

    # ---------- per-run computation ----------
    all_summaries = []
    all_per_layer_curv: dict[str, pd.DataFrame] = {}
    baselines = []

    for run in args.runs:
        print(f"\n=== {run.name} ===")
        raw, par, base = _per_run_tables(run)
        summary = _best_layer_per_metric(raw, par)
        summary.insert(0, "model_run", run.name)
        all_summaries.append(summary)

        baselines.append({"model_run": run.name, **base})

        # Save raw + partial AUC tables per run for traceability
        raw.to_csv(run / "auc_table.csv", index=False)
        par.to_csv(run / "auc_partial_table.csv", index=False)

        # Stash per-layer curvature_mean for cross-model plot
        cm = raw[raw["metric"] == "curvature_mean"][["layer", "auc"]].copy()
        cm.columns = ["layer", "auc"]
        all_per_layer_curv[run.name] = cm

        print(summary.to_string(index=False))
        print(f"  baseline n_gen_tokens AUC = {base['auc']:.3f}")

    # ---------- aggregate ----------
    big = pd.concat(all_summaries, ignore_index=True)
    base_df = pd.DataFrame(baselines)
    big = big.merge(base_df.rename(columns={"auc": "baseline_auc"})[["model_run", "baseline_auc"]],
                    on="model_run")
    big["beats_baseline_partial"] = big["partial_auc_signed"] > big["baseline_auc"]
    big["delta_vs_baseline_signed"] = big["partial_auc_signed"] - big["baseline_auc"]
    big.to_csv(args.out / "cross_model_summary.csv", index=False)
    print(f"\nwrote {args.out / 'cross_model_summary.csv'}")

    # Sign-flip table: focus on curvature_mean
    cm_summary = big[big["metric"] == "curvature_mean"][
        ["model_run", "best_layer", "auc", "sign_flipped", "partial_auc_signed", "baseline_auc"]
    ]
    cm_summary.to_csv(args.out / "cross_model_curvature_signflip.csv", index=False)

    # Pretty-print headline
    print("\n=== curvature_mean sign-flip across models ===")
    print(cm_summary.to_string(index=False))

    # ---------- plot ----------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, df in all_per_layer_curv.items():
        ax.plot(df["layer"], df["auc"], marker="o", label=name.replace("2026-05-17_extract-", "").replace("-100", ""))
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="chance")
    ax.set_xlabel("layer (0 = embedding)")
    ax.set_ylabel("AUC of curvature_mean (predict incorrect)")
    ax.set_title("Sign-flip phenomenon across models — curvature_mean per layer")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.30, 0.75)
    fig.tight_layout()
    out_png = args.out / "plots" / "cross_model_curvature_per_layer.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
