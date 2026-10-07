#!/usr/bin/env python3
"""Candidate-level probe for length-residualized GeoVote signals.

This is a follow-up to ``geovote_residual_length_rescue.py``. The direct
residualized vote is weak against majority at N=16. This script asks a narrower
question: after count, log-probability, and answer length are available, does the
frozen residual geometry score still add information for choosing among candidate
answers?

Protocol:
  * use the residualized metric/layer/sign selected on calibration;
  * build one row per candidate answer per question;
  * train small logistic answer selectors on calibration questions only;
  * evaluate the selected answer on held-out questions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geoprobe.datasets import is_correct, is_correct_math500
from geovote_independence_analysis import safe_softmax


FEATURE_SETS = {
    "count": ["count_frac_z"],
    "logprob": ["logprob_weight_sum_z"],
    "length": ["mean_length_z"],
    "resid_geo": ["resid_geo_weight_sum_z"],
    "count+logprob": ["count_frac_z", "logprob_weight_sum_z"],
    "count+logprob+length": [
        "count_frac_z",
        "logprob_weight_sum_z",
        "mean_length_z",
    ],
    "count+logprob+length+resid_geo": [
        "count_frac_z",
        "logprob_weight_sum_z",
        "mean_length_z",
        "resid_geo_weight_sum_z",
    ],
}


def parse_range(spec: str, observed_ids: list[int]) -> set[int]:
    if spec == "all":
        return set(observed_ids)
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        start = int(start_s) if start_s else min(observed_ids)
        end = int(end_s) if end_s else max(observed_ids) + 1
        return {i for i in observed_ids if start <= i < end}
    return {int(x) for x in spec.split(",") if x.strip()}


def zscore_per_question(df: pd.DataFrame, col: str) -> pd.Series:
    def zscore(group: pd.Series) -> pd.Series:
        std = group.std(ddof=0)
        if std == 0 or not np.isfinite(std):
            return pd.Series(np.zeros(len(group)), index=group.index)
        return (group - group.mean()) / std

    return df.groupby("sample_id")[col].transform(zscore)


def load_run(run: Path, dataset: str):
    labels = pd.read_parquet(run / "labels.parquet")
    metrics = pd.read_parquet(run / "metrics.parquet")
    if dataset == "gsm8k":
        labels = labels.copy()
        labels["gold"] = labels["gold"].astype(float)
        labels["pred"] = labels["pred"].apply(
            lambda x: float(x) if x is not None and x != "None" else None
        )
        grader = is_correct
    else:
        grader = is_correct_math500
    return labels, metrics, grader


def fit_residualizer(df: pd.DataFrame) -> tuple[float, float]:
    x = df["n_gen_tokens"].astype(float).to_numpy()
    y = df["geo_value"].astype(float).to_numpy()
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(((x - x_mean) ** 2).sum())
    slope = 0.0 if denom <= 0 else float(((x - x_mean) * (y - y_mean)).sum() / denom)
    intercept = y_mean - slope * x_mean
    return intercept, slope


def candidate_rows(
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    metric: str,
    layer: int,
    sign: str,
    prefix_n: int,
    residualizer: tuple[float, float],
    grader: Callable[[object, object], bool],
) -> pd.DataFrame:
    metric_rows = metrics[
        (metrics["metric"] == metric) & (metrics["layer"] == layer)
    ][["sample_id", "sample_idx", "value"]].rename(columns={"value": "geo_value"})
    df = labels.merge(metric_rows, on=["sample_id", "sample_idx"], how="inner")
    df = df[df["sample_idx"] < prefix_n].copy()
    intercept, slope = residualizer
    df["resid_value"] = (
        df["geo_value"].astype(float)
        - intercept
        - slope * df["n_gen_tokens"].astype(float)
    )

    rows: list[dict] = []
    for sample_id, group in df.groupby("sample_id"):
        group = group.sort_values("sample_idx")
        preds = group["pred"].tolist()
        gold = group["gold"].iloc[0]
        logprobs = group["sequence_logprob"].astype(float).to_numpy()
        lengths = group["n_gen_tokens"].astype(float).to_numpy()
        residuals = group["resid_value"].astype(float).to_numpy()
        if len(group) == 0 or not np.isfinite(residuals).all():
            continue

        counts = Counter([p for p in preds if p is not None and not pd.isna(p)])
        if not counts:
            continue
        logprob_weights = safe_softmax(logprobs)
        confidence = -residuals if sign == "min" else residuals
        residual_weights = safe_softmax(confidence)

        for answer, count in counts.items():
            mask = np.array([p == answer for p in preds], dtype=bool)
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "answer": answer,
                    "correct_answer": bool(grader(answer, gold)),
                    "count": int(count),
                    "count_frac": float(count / len(group)),
                    "logprob_weight_sum": float(logprob_weights[mask].sum()),
                    "mean_length": float(lengths[mask].mean()),
                    "resid_geo_weight_sum": float(residual_weights[mask].sum()),
                    "n_candidates": int(len(counts)),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in [
        "count_frac",
        "logprob_weight_sum",
        "mean_length",
        "resid_geo_weight_sum",
    ]:
        out[f"{col}_z"] = zscore_per_question(out, col)
    return out.replace([np.inf, -np.inf], np.nan)


def choose_by_feature(candidates: pd.DataFrame, feature: str) -> float:
    correct = 0
    total = 0
    for _, group in candidates.groupby("sample_id"):
        group = group.dropna(subset=[feature])
        if group.empty:
            continue
        pick = group.loc[group[feature].idxmax()]
        correct += int(bool(pick["correct_answer"]))
        total += 1
    return correct / total if total else float("nan")


def train_eval_selector(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[float, float]:
    cols = features + ["correct_answer"]
    train_clean = train.dropna(subset=cols).copy()
    test_clean = test.dropna(subset=cols).copy()
    if train_clean.empty or test_clean.empty:
        return float("nan"), float("nan")

    y_train = train_clean["correct_answer"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        return float("nan"), float("nan")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(train_clean[features].astype(float).to_numpy(), y_train)

    def grouped_accuracy(frame: pd.DataFrame) -> float:
        scores = model.predict_proba(frame[features].astype(float).to_numpy())[:, 1]
        scored = frame.copy()
        scored["_score"] = scores
        correct = 0
        total = 0
        for _, group in scored.groupby("sample_id"):
            pick = group.loc[group["_score"].idxmax()]
            correct += int(bool(pick["correct_answer"]))
            total += 1
        return correct / total if total else float("nan")

    return grouped_accuracy(train_clean), grouped_accuracy(test_clean)


def partial_corr(frame: pd.DataFrame) -> tuple[float, float, int]:
    cols = [
        "correct_answer",
        "count_frac_z",
        "logprob_weight_sum_z",
        "mean_length_z",
        "resid_geo_weight_sum_z",
    ]
    data = frame.dropna(subset=cols).copy()
    if len(data) < 10 or data["correct_answer"].nunique() < 2:
        return float("nan"), float("nan"), int(len(data))
    controls = data[["count_frac_z", "logprob_weight_sum_z", "mean_length_z"]]
    geo = data["resid_geo_weight_sum_z"].astype(float)
    target = data["correct_answer"].astype(float)
    geo_resid = geo - LinearRegression().fit(controls, geo).predict(controls)
    target_resid = target - LinearRegression().fit(controls, target).predict(controls)
    r, p = pearsonr(geo_resid, target_resid)
    return float(r), float(p), int(len(data))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], required=True)
    parser.add_argument("--selected-config", required=True, type=Path)
    parser.add_argument("--prefixes", default="8,16")
    parser.add_argument("--calib-ids", default="0:100")
    parser.add_argument("--test-ids", default="100:500")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    labels, metrics, grader = load_run(args.run, args.dataset)
    observed_ids = sorted(int(x) for x in labels["sample_id"].unique())
    calib_ids = parse_range(args.calib_ids, observed_ids)
    test_ids = parse_range(args.test_ids, observed_ids)

    labels_cal = labels[labels["sample_id"].isin(calib_ids)].copy()
    labels_test = labels[labels["sample_id"].isin(test_ids)].copy()
    metrics_cal = metrics[metrics["sample_id"].isin(calib_ids)].copy()
    metrics_test = metrics[metrics["sample_id"].isin(test_ids)].copy()
    selected = pd.read_csv(args.selected_config)
    args.out.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    partial_rows: list[dict] = []
    for prefix_n in [int(x) for x in args.prefixes.split(",") if x.strip()]:
        cfg = selected[selected["prefix_n"] == prefix_n].iloc[0]
        metric = str(cfg["metric"])
        layer = int(cfg["layer"])
        sign = str(cfg["sign"])

        cal_metric = metrics_cal[
            (metrics_cal["metric"] == metric) & (metrics_cal["layer"] == layer)
        ][["sample_id", "sample_idx", "value"]].rename(columns={"value": "geo_value"})
        cal_join = labels_cal.merge(cal_metric, on=["sample_id", "sample_idx"])
        cal_join = cal_join[cal_join["sample_idx"] < prefix_n].copy()
        residualizer = fit_residualizer(cal_join)

        train = candidate_rows(
            labels_cal,
            metrics_cal,
            metric,
            layer,
            sign,
            prefix_n,
            residualizer,
            grader,
        )
        test = candidate_rows(
            labels_test,
            metrics_test,
            metric,
            layer,
            sign,
            prefix_n,
            residualizer,
            grader,
        )
        train.to_csv(args.out / f"candidate_rows_calib_N{prefix_n}.csv", index=False)
        test.to_csv(args.out / f"candidate_rows_heldout_N{prefix_n}.csv", index=False)

        direct_features = {
            "argmax_count": "count_frac",
            "argmax_logprob": "logprob_weight_sum",
            "argmax_length": "mean_length",
            "argmax_resid_geo": "resid_geo_weight_sum",
        }
        for name, feature in direct_features.items():
            summary_rows.append(
                {
                    "prefix_n": prefix_n,
                    "selector": name,
                    "features": feature,
                    "calibration_acc": choose_by_feature(train, feature),
                    "heldout_acc": choose_by_feature(test, feature),
                    "metric": metric,
                    "layer": layer,
                    "sign": sign,
                }
            )

        for name, features in FEATURE_SETS.items():
            cal_acc, heldout_acc = train_eval_selector(train, test, features)
            summary_rows.append(
                {
                    "prefix_n": prefix_n,
                    "selector": f"logistic:{name}",
                    "features": "+".join(features),
                    "calibration_acc": cal_acc,
                    "heldout_acc": heldout_acc,
                    "metric": metric,
                    "layer": layer,
                    "sign": sign,
                }
            )

        r, p, n = partial_corr(test)
        partial_rows.append(
            {
                "prefix_n": prefix_n,
                "partial_corr_resid_geo_given_count_logprob_length": r,
                "p_value": p,
                "n_candidate_rows": n,
                "metric": metric,
                "layer": layer,
                "sign": sign,
            }
        )

    summary = pd.DataFrame(summary_rows)
    partial = pd.DataFrame(partial_rows)
    summary.to_csv(args.out / "candidate_selector_summary.csv", index=False)
    partial.to_csv(args.out / "candidate_partial_corr.csv", index=False)
    print(summary.to_string(index=False))
    print()
    print(partial.to_string(index=False))


if __name__ == "__main__":
    main()
