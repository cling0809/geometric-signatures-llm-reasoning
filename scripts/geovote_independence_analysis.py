#!/usr/bin/env python3
"""Analyze whether GeoVote carries information beyond logprob and majority.

Outputs:
  - accuracy_summary.csv
  - sample_signal.csv
  - candidate_signal.csv
  - margin_strata.csv
  - complementarity.csv

Example:
  python scripts/geovote_independence_analysis.py \
      --run /root/AI/runs/2026-05-19_qwen-math-math500-2048-n8 \
      --dataset math500 --geo-metric mean_step_norm --geo-layer 20 --geo-sign min \
      --out /root/AI/runs/2026-05-20_geovote_independence/math500_2048_n8
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geoprobe.datasets import is_correct, is_correct_math500


def weighted_vote(preds: list, weights: list[float]):
    totals = defaultdict(float)
    for pred, weight in zip(preds, weights):
        if pred is not None:
            totals[pred] += float(weight)
    if not totals:
        return None
    return max(totals.items(), key=lambda kv: kv[1])[0]


def majority_vote(preds: list):
    nonnull = [p for p in preds if p is not None and not pd.isna(p)]
    if not nonnull:
        return None
    return Counter(nonnull).most_common(1)[0][0]


def safe_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmax(x)
    w = np.exp(x)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.ones_like(x) / len(x)
    return w / w.sum()


def zscore_per_question(df: pd.DataFrame, col: str) -> pd.Series:
    def z(g: pd.Series) -> pd.Series:
        sd = g.std(ddof=0)
        if sd == 0 or not np.isfinite(sd):
            return pd.Series(np.zeros(len(g)), index=g.index)
        return (g - g.mean()) / sd

    return df.groupby("sample_id")[col].transform(z)


def get_geo_values(metrics: pd.DataFrame, metric: str, layer: int) -> pd.DataFrame:
    cols = ["sample_id", "sample_idx", "geo_value"]
    subset = metrics[(metrics["metric"] == metric) & (metrics["layer"] == layer)].copy()
    subset = subset.rename(columns={"value": "geo_value"})
    return subset[cols]


def question_rows(
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    metric: str,
    layer: int,
    sign: Literal["min", "max"],
    grader: Callable[[object, object], bool],
    prefix_n: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    geo = get_geo_values(metrics, metric, layer)
    df = labels.merge(geo, on=["sample_id", "sample_idx"], how="inner")
    if prefix_n is not None:
        df = df[df["sample_idx"] < prefix_n].copy()

    rows = []
    per_sample = []
    per_candidate = []
    for sample_id, g in df.groupby("sample_id"):
        g = g.sort_values("sample_idx")
        preds = g["pred"].tolist()
        gold = g["gold"].iloc[0]
        correct = g["correct"].astype(bool).to_numpy()
        logprobs = g["sequence_logprob"].astype(float).to_numpy()
        lengths = g["n_gen_tokens"].astype(float).to_numpy()
        vals = g["geo_value"].astype(float).to_numpy()
        n = len(g)
        if n == 0 or np.isnan(vals).any():
            continue

        majority_pred = majority_vote(preds)
        logprob_max_pred = preds[int(np.argmax(logprobs))]
        logprob_weighted_pred = weighted_vote(preds, safe_softmax(logprobs))
        geo_idx = int(np.argmin(vals) if sign == "min" else np.argmax(vals))
        geo_pick_pred = preds[geo_idx]
        if sign == "min":
            geo_weights = 1.0 / (np.maximum(vals, 0) + 1e-8)
        else:
            geo_weights = np.maximum(vals, 0)
        if not np.isfinite(geo_weights).all() or geo_weights.sum() <= 0:
            geo_weights = np.ones_like(vals)
        geo_weights = geo_weights / geo_weights.sum()
        geo_vote_pred = weighted_vote(preds, geo_weights)

        counts = Counter([p for p in preds if p is not None and not pd.isna(p)])
        if counts:
            geo_sum = defaultdict(float)
            for p, w in zip(preds, geo_weights):
                if p is not None:
                    geo_sum[p] += float(w)
            combined = {a: counts[a] * geo_sum[a] for a in counts}
            geo_majority_pred = max(combined.items(), key=lambda kv: kv[1])[0]
            top_counts = sorted(counts.values(), reverse=True)
            top_count = top_counts[0]
            second_count = top_counts[1] if len(top_counts) > 1 else 0
            margin = (top_count - second_count) / max(n, 1)
            top_frac = top_count / max(n, 1)
            n_unique = len(counts)
        else:
            geo_majority_pred = None
            margin = 0.0
            top_frac = 0.0
            n_unique = 0

        row = {
            "sample_id": sample_id,
            "n": n,
            "majority_margin": margin,
            "majority_top_frac": top_frac,
            "n_unique_answers": n_unique,
            "oracle": bool(correct.any()),
            "first": bool(correct[0]),
            "majority": bool(grader(majority_pred, gold)),
            "logprob_max": bool(grader(logprob_max_pred, gold)),
            "logprob_weighted": bool(grader(logprob_weighted_pred, gold)),
            "geo_pick": bool(grader(geo_pick_pred, gold)),
            "geovote": bool(grader(geo_vote_pred, gold)),
            "geo_majority": bool(grader(geo_majority_pred, gold)),
        }
        rows.append(row)

        sg = g.copy()
        sg["geo_conf"] = -sg["geo_value"] if sign == "min" else sg["geo_value"]
        per_sample.append(sg)

        if counts:
            lp_weights = safe_softmax(logprobs)
            for answer in counts:
                mask = np.array([p == answer for p in preds], dtype=bool)
                if mask.sum() == 0:
                    continue
                per_candidate.append({
                    "sample_id": sample_id,
                    "answer": answer,
                    "correct_answer": bool(grader(answer, gold)),
                    "count": int(mask.sum()),
                    "count_frac": float(mask.mean()),
                    "logprob_weight_sum": float(lp_weights[mask].sum()),
                    "geo_weight_sum": float(geo_weights[mask].sum()),
                    "mean_length": float(lengths[mask].mean()),
                    "min_geo_value": float(vals[mask].min()),
                    "max_geo_conf": float(((-vals) if sign == "min" else vals)[mask].max()),
                    "n_candidates": int(len(counts)),
                })

    qdf = pd.DataFrame(rows)
    sdf = pd.concat(per_sample, ignore_index=True) if per_sample else pd.DataFrame()
    cdf = pd.DataFrame(per_candidate)
    if not sdf.empty:
        sdf["geo_conf_z"] = zscore_per_question(sdf, "geo_conf")
        sdf["logprob_z"] = zscore_per_question(sdf, "sequence_logprob")
        sdf["length_z"] = zscore_per_question(sdf, "n_gen_tokens")
    if not cdf.empty:
        for col in ["count_frac", "logprob_weight_sum", "geo_weight_sum", "mean_length", "max_geo_conf"]:
            cdf[col + "_z"] = zscore_per_question(cdf, col)
    return qdf, sdf, cdf


def grouped_cv_auc(sdf: pd.DataFrame, features: list[str]) -> float:
    y = sdf["correct"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    x = sdf[features].astype(float).to_numpy()
    groups = sdf["sample_id"].to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    preds = np.zeros(len(sdf), dtype=float)
    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(x, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            preds[test_idx] = y[train_idx].mean()
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    return float(roc_auc_score(y, preds))


def grouped_cv_candidate_choice(cdf: pd.DataFrame, features: list[str]) -> float:
    """Cross-validated answer selection accuracy from candidate-level features."""
    d = cdf.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["correct_answer"])
    if d.empty:
        return float("nan")
    groups = d["sample_id"].to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    scores = np.zeros(len(d), dtype=float)
    y = d["correct_answer"].astype(int).to_numpy()
    x = d[features].astype(float).to_numpy()
    if len(np.unique(y)) < 2 or n_splits < 2:
        return float("nan")
    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(x, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            scores[test_idx] = y[train_idx].mean()
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(x[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(x[test_idx])[:, 1]

    correct = 0
    n_questions = 0
    d = d.copy()
    d["_score"] = scores
    for _, g in d.groupby("sample_id"):
        pick = g.loc[g["_score"].idxmax()]
        correct += int(bool(pick["correct_answer"]))
        n_questions += 1
    return correct / n_questions if n_questions else float("nan")


def partial_corr(sdf: pd.DataFrame) -> tuple[float, float]:
    cols = ["geo_conf_z", "logprob_z", "length_z", "correct"]
    d = sdf[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if d["correct"].nunique() < 2 or len(d) < 10:
        return float("nan"), float("nan")
    controls = d[["logprob_z", "length_z"]].to_numpy()
    geo_resid = d["geo_conf_z"].to_numpy() - LinearRegression().fit(
        controls, d["geo_conf_z"].to_numpy()
    ).predict(controls)
    y = d["correct"].astype(int).to_numpy()
    y_resid = y - LinearRegression().fit(controls, y).predict(controls)
    r, p = pearsonr(geo_resid, y_resid)
    return float(r), float(p)


def candidate_partial_corr(cdf: pd.DataFrame) -> tuple[float, float]:
    cols = ["geo_weight_sum_z", "count_frac_z", "logprob_weight_sum_z", "correct_answer"]
    d = cdf[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if d["correct_answer"].nunique() < 2 or len(d) < 10:
        return float("nan"), float("nan")
    controls = d[["count_frac_z", "logprob_weight_sum_z"]].to_numpy()
    geo_resid = d["geo_weight_sum_z"].to_numpy() - LinearRegression().fit(
        controls, d["geo_weight_sum_z"].to_numpy()
    ).predict(controls)
    y = d["correct_answer"].astype(int).to_numpy()
    y_resid = y - LinearRegression().fit(controls, y).predict(controls)
    r, p = pearsonr(geo_resid, y_resid)
    return float(r), float(p)


def make_accuracy_summary(qdf: pd.DataFrame, run_name: str, dataset: str, prefix_n: int) -> pd.DataFrame:
    strategies = [
        "first",
        "majority",
        "logprob_max",
        "logprob_weighted",
        "geo_pick",
        "geovote",
        "geo_majority",
        "oracle",
    ]
    rows = []
    for s in strategies:
        rows.append({
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "strategy": s,
            "accuracy": float(qdf[s].mean()),
            "n_questions": int(len(qdf)),
        })
    return pd.DataFrame(rows)


def make_sample_signal(sdf: pd.DataFrame, run_name: str, dataset: str, prefix_n: int) -> pd.DataFrame:
    d = sdf.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["geo_conf_z", "logprob_z", "length_z", "correct"]
    )
    y = d["correct"].astype(int)
    geo_spear = spearmanr(d["geo_conf_z"], y).statistic
    lp_spear = spearmanr(d["logprob_z"], y).statistic
    length_spear = spearmanr(d["length_z"], y).statistic
    pc_r, pc_p = partial_corr(d)
    rows = [
        {
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "analysis": "spearman_geo_correct",
            "value": float(geo_spear),
        },
        {
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "analysis": "spearman_logprob_correct",
            "value": float(lp_spear),
        },
        {
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "analysis": "spearman_length_correct",
            "value": float(length_spear),
        },
        {
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "analysis": "partial_corr_geo_correct_ctrl_logprob_length",
            "value": pc_r,
            "p_value": pc_p,
        },
    ]
    for features in [
        ["logprob_z"],
        ["geo_conf_z"],
        ["length_z"],
        ["logprob_z", "length_z"],
        ["logprob_z", "length_z", "geo_conf_z"],
    ]:
        rows.append({
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "analysis": "group_cv_logistic_auc:" + "+".join(features),
            "value": grouped_cv_auc(d, features),
        })
    return pd.DataFrame(rows)


def make_candidate_signal(cdf: pd.DataFrame, run_name: str, dataset: str, prefix_n: int) -> pd.DataFrame:
    d = cdf.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "geo_weight_sum_z",
            "count_frac_z",
            "logprob_weight_sum_z",
            "mean_length_z",
            "correct_answer",
        ]
    )
    y = d["correct_answer"].astype(int)
    pc_r, pc_p = candidate_partial_corr(d)
    rows = [
        {
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "analysis": "candidate_spearman_geo_correct",
            "value": float(spearmanr(d["geo_weight_sum_z"], y).statistic),
        },
        {
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "analysis": "candidate_spearman_count_correct",
            "value": float(spearmanr(d["count_frac_z"], y).statistic),
        },
        {
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "analysis": "candidate_spearman_logprob_correct",
            "value": float(spearmanr(d["logprob_weight_sum_z"], y).statistic),
        },
        {
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "analysis": "candidate_partial_corr_geo_ctrl_count_logprob",
            "value": pc_r,
            "p_value": pc_p,
        },
    ]
    for features in [
        ["count_frac_z"],
        ["logprob_weight_sum_z"],
        ["geo_weight_sum_z"],
        ["count_frac_z", "logprob_weight_sum_z"],
        ["count_frac_z", "geo_weight_sum_z"],
        ["count_frac_z", "logprob_weight_sum_z", "geo_weight_sum_z"],
    ]:
        rows.append({
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "analysis": "candidate_group_cv_choice_acc:" + "+".join(features),
            "value": grouped_cv_candidate_choice(d, features),
        })
    return pd.DataFrame(rows)


def make_margin_strata(qdf: pd.DataFrame, run_name: str, dataset: str, prefix_n: int) -> pd.DataFrame:
    bins = [
        ("unanimous_or_near", qdf["majority_top_frac"] >= 0.75),
        ("medium_margin", (qdf["majority_top_frac"] < 0.75) & (qdf["majority_margin"] >= 0.25)),
        ("low_margin", qdf["majority_margin"] < 0.25),
    ]
    rows = []
    for name, mask in bins:
        part = qdf[mask]
        if part.empty:
            continue
        row = {
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "stratum": name,
            "n_questions": int(len(part)),
            "mean_top_frac": float(part["majority_top_frac"].mean()),
            "mean_unique_answers": float(part["n_unique_answers"].mean()),
        }
        for s in ["majority", "logprob_weighted", "geovote", "geo_majority", "oracle"]:
            row[s] = float(part[s].mean())
        row["geovote_minus_majority"] = row["geovote"] - row["majority"]
        row["geo_majority_minus_majority"] = row["geo_majority"] - row["majority"]
        rows.append(row)
    return pd.DataFrame(rows)


def make_complementarity(qdf: pd.DataFrame, run_name: str, dataset: str, prefix_n: int) -> pd.DataFrame:
    rows = []
    pairs = [
        ("geovote", "majority"),
        ("geo_majority", "majority"),
        ("geovote", "logprob_weighted"),
        ("geo_majority", "logprob_weighted"),
    ]
    for a, b in pairs:
        a_ok = qdf[a].astype(bool)
        b_ok = qdf[b].astype(bool)
        rows.append({
            "run": run_name,
            "dataset": dataset,
            "prefix_n": prefix_n,
            "method_a": a,
            "method_b": b,
            "both_correct": int((a_ok & b_ok).sum()),
            "a_only_recovery": int((a_ok & ~b_ok).sum()),
            "b_only_breakage": int((~a_ok & b_ok).sum()),
            "both_wrong": int((~a_ok & ~b_ok).sum()),
            "net_a_minus_b": int((a_ok & ~b_ok).sum() - (~a_ok & b_ok).sum()),
            "n_questions": int(len(qdf)),
        })
    return pd.DataFrame(rows)


def parse_prefixes(prefix: str | None, observed_n: int) -> list[int]:
    if not prefix:
        return [observed_n]
    out = []
    for item in prefix.split(","):
        n = int(item)
        if n <= observed_n:
            out.append(n)
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math500"])
    ap.add_argument("--geo-metric", default="mean_step_norm")
    ap.add_argument("--geo-layer", type=int, default=20)
    ap.add_argument("--geo-sign", choices=["min", "max"], default="min")
    ap.add_argument("--prefixes", default=None, help="Comma-separated sample prefixes, e.g. 8,16,32")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    labels = pd.read_parquet(args.run / "labels.parquet")
    metrics = pd.read_parquet(args.run / "metrics.parquet")
    if args.dataset == "gsm8k":
        labels = labels.copy()
        labels["gold"] = labels["gold"].astype(float)
        labels["pred"] = labels["pred"].apply(
            lambda x: float(x) if x is not None and x != "None" else None
        )
        grader = is_correct
    else:
        grader = is_correct_math500

    observed_n = int(labels.groupby("sample_id").size().mode().iloc[0])
    prefixes = parse_prefixes(args.prefixes, observed_n)
    args.out.mkdir(parents=True, exist_ok=True)

    accuracy, signal, candidate_signal, strata, comp = [], [], [], [], []
    for prefix_n in prefixes:
        qdf, sdf, cdf = question_rows(
            labels=labels,
            metrics=metrics,
            metric=args.geo_metric,
            layer=args.geo_layer,
            sign=args.geo_sign,
            grader=grader,
            prefix_n=prefix_n,
        )
        qdf.to_csv(args.out / f"question_level_N{prefix_n}.csv", index=False)
        sdf.to_parquet(args.out / f"sample_level_N{prefix_n}.parquet", index=False)
        cdf.to_csv(args.out / f"candidate_level_N{prefix_n}.csv", index=False)
        accuracy.append(make_accuracy_summary(qdf, args.run.name, args.dataset, prefix_n))
        signal.append(make_sample_signal(sdf, args.run.name, args.dataset, prefix_n))
        candidate_signal.append(make_candidate_signal(cdf, args.run.name, args.dataset, prefix_n))
        strata.append(make_margin_strata(qdf, args.run.name, args.dataset, prefix_n))
        comp.append(make_complementarity(qdf, args.run.name, args.dataset, prefix_n))

    pd.concat(accuracy, ignore_index=True).to_csv(args.out / "accuracy_summary.csv", index=False)
    pd.concat(signal, ignore_index=True).to_csv(args.out / "sample_signal.csv", index=False)
    pd.concat(candidate_signal, ignore_index=True).to_csv(
        args.out / "candidate_signal.csv", index=False
    )
    pd.concat(strata, ignore_index=True).to_csv(args.out / "margin_strata.csv", index=False)
    pd.concat(comp, ignore_index=True).to_csv(args.out / "complementarity.csv", index=False)

    print(f"wrote {args.out}")
    print(pd.concat(accuracy, ignore_index=True).to_string(index=False))
    print()
    print(pd.concat(signal, ignore_index=True).to_string(index=False))
    print()
    print(pd.concat(candidate_signal, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
