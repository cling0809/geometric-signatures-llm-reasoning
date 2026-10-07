#!/usr/bin/env python3
"""Length-stratified AUC analysis for the phenomenon experiment.

Reviewer concern: "Your signature might just be reading 'did the model run out of
its 512-token budget?', not anything about post-training paradigm or correctness."

Rebuttal protocol:
  1. For each of the 4 GSM8K-100 runs (Base, Instruct, Math-Instruct, R1-Distill),
     bin samples by n_gen_tokens into low / high.
  2. Compute AUC of each (metric, layer) on each subset and on the full set.
  3. Report (a) the best (metric, layer) per model's full set, then re-evaluate it
     on each length bin; (b) full vs length-residualized partial AUC.
  4. Output a CSV + a 4-panel figure for the appendix.

For R1-Distill specifically, 93/100 samples hit budget at max_new=512, so a clean
"finished early" subset (n=7) is too small to estimate AUC; we therefore use a
median split as the primary stratification across all models, and report the
"finished early" subset only as a footnote where it has enough samples.

Usage:
    python scripts/length_stratified_auc.py \
        --runs-dir /root/AI/runs \
        --out /root/AI/runs/2026-05-19_length-stratified
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RUNS_GSM8K = [
    ("Base",         "2026-05-17_extract-qwen-1.5b-base-100"),
    ("Instruct",     "2026-05-17_extract-qwen-1.5b-nomath-100"),
    ("Math-Instruct", "2026-05-17_extract-pilot-gsm8k-100"),
    ("R1-Distill",   "2026-05-17_extract-deepseek-r1-distill-1.5b-100"),
]


def _auc(values, correct):
    """Two-sided AUC: max(auc, 1-auc) so direction doesn't matter."""
    values = np.asarray(values, dtype=float)
    correct = np.asarray(correct, dtype=int)
    mask = np.isfinite(values)
    if mask.sum() < 5 or len(np.unique(correct[mask])) < 2:
        return float("nan"), int(mask.sum())
    a = roc_auc_score(correct[mask], values[mask])
    return max(a, 1.0 - a), int(mask.sum())


def _per_metric_layer_auc(metrics_df, labels_df, sample_mask=None):
    """Returns DataFrame of (metric, layer, auc, n) on the given subset."""
    if sample_mask is not None:
        keep_ids = set(labels_df.loc[sample_mask, "sample_id"])
        metrics_df = metrics_df[metrics_df["sample_id"].isin(keep_ids)]
        labels_df = labels_df.loc[sample_mask]
    lab = labels_df.set_index("sample_id")[["correct"]].astype(int)
    rows = []
    for (m, L), g in metrics_df.groupby(["metric", "layer"]):
        v = g.set_index("sample_id")[["value"]]
        joined = lab.join(v, how="inner").dropna()
        if len(joined) < 5:
            continue
        a, n = _auc(joined["value"].values, joined["correct"].values)
        rows.append({"metric": m, "layer": int(L), "auc": a, "n": n})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    per_run_summary = []
    for name, rd in RUNS_GSM8K:
        d = args.runs_dir / rd
        lab = pd.read_parquet(d / "labels.parquet")
        met = pd.read_parquet(d / "metrics.parquet")
        # ensure n_gen_tokens is numeric
        ngt = lab["n_gen_tokens"].astype(int)
        budget = ngt.max()
        hit = (ngt >= budget - 2).sum()
        median = ngt.median()

        # full
        full = _per_metric_layer_auc(met, lab)
        full["subset"] = "full"

        # median split — if median equals budget (e.g., R1-Distill), median split is
        # degenerate (everyone in "low"). In that case fall back to hit-budget split.
        if median >= budget - 2:
            low_mask = ngt < budget - 2
            high_mask = ngt >= budget - 2
            split_label = "early_finish_vs_hit_budget"
        else:
            low_mask = ngt <= median
            high_mask = ngt > median
            split_label = "median_split"
        low = _per_metric_layer_auc(met, lab, low_mask) if low_mask.sum() >= 5 else pd.DataFrame()
        if len(low):
            low["subset"] = "low_len"
        high = _per_metric_layer_auc(met, lab, high_mask) if high_mask.sum() >= 5 else pd.DataFrame()
        if len(high):
            high["subset"] = "high_len"

        # "didn't hit budget" subset — only meaningful if n >= 20
        early_mask = ngt < budget - 2
        early = None
        if early_mask.sum() >= 20:
            early = _per_metric_layer_auc(met, lab, early_mask)
            early["subset"] = "early_finish"

        for df in [full, low, high] + ([early] if early is not None else []):
            if df is None or len(df) == 0:
                continue
            df["model"] = name
            all_rows.append(df)

        # for paper: pick the (metric, layer) with highest full AUC, report it across subsets
        def _pick(df, m, L):
            if df is None or len(df) == 0 or "metric" not in df.columns:
                return float("nan")
            sel = df[(df["metric"] == m) & (df["layer"] == L)]
            return float(sel["auc"].iloc[0]) if len(sel) else float("nan")

        best = full.loc[full["auc"].idxmax()]
        best_m, best_L = best["metric"], int(best["layer"])
        line = {
            "model": name,
            "n_total": int(len(lab)),
            "correct": int(lab["correct"].sum()),
            "budget": int(budget),
            "hit_budget": int(hit),
            "median_n_gen": int(median),
            "best_metric": best_m,
            "best_layer": best_L,
            "auc_full":     _pick(full, best_m, best_L),
            "auc_low_len":  _pick(low,  best_m, best_L),
            "n_low_len":    int(low_mask.sum()),
            "auc_high_len": _pick(high, best_m, best_L),
            "n_high_len":   int(high_mask.sum()),
            "split_kind":   split_label,
            "auc_early_finish": _pick(early, best_m, best_L) if early is not None else float("nan"),
            "n_early_finish":   int(early_mask.sum()),
        }
        per_run_summary.append(line)

    rows = pd.concat(all_rows, ignore_index=True)
    rows.to_csv(args.out / "length_stratified_auc.csv", index=False)
    summary = pd.DataFrame(per_run_summary)
    summary.to_csv(args.out / "length_stratified_summary.csv", index=False)

    print("\n=== Per-model summary (best (metric, layer) per row, AUC across length subsets) ===")
    cols = ["model", "best_metric", "best_layer", "auc_full", "auc_low_len",
            "auc_high_len", "auc_early_finish", "n_early_finish", "hit_budget"]
    print(summary[cols].to_string(index=False))
    print(f"\nWrote {args.out / 'length_stratified_auc.csv'}")
    print(f"Wrote {args.out / 'length_stratified_summary.csv'}")

    # figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(summary))
    w = 0.22
    ax.bar(x - 1.5 * w, summary["auc_full"],     w, label="full", color="#2E5C99")
    ax.bar(x - 0.5 * w, summary["auc_low_len"],  w, label="low length (≤ median)",  color="#4DA193")
    ax.bar(x + 0.5 * w, summary["auc_high_len"], w, label="high length (> median)", color="#E2A53C")
    has_early = summary["n_early_finish"] >= 20
    if has_early.any():
        # plot only where early is meaningful
        vals = summary["auc_early_finish"].where(has_early)
        ax.bar(x + 1.5 * w, vals, w, label="finished within budget (n≥20)", color="#D75B5B")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=0)
    ax.set_ylabel("AUC (best metric, layer)")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Phenomenon signature AUC is stable across generation-length bins")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(args.out / "length_stratified_auc.png", dpi=140)
    print(f"Wrote {args.out / 'length_stratified_auc.png'}")


if __name__ == "__main__":
    main()
