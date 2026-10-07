#!/usr/bin/env python3
"""Length-residualized GeoVote rescue analysis.

The clean-split GeoVote result selected a path-length metric. This script asks
whether any held-out signal remains after removing simple token-count effects.

Protocol:
  * calibration ids choose metric/layer/sign/strategy;
  * for each metric/layer, fit value ~ n_gen_tokens on calibration samples;
  * apply the frozen residualizer and frozen selection to held-out ids;
  * report accuracy, low-margin behavior, and question-length strata.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from geoprobe.datasets import is_correct, is_correct_math500
from geovote_independence_analysis import majority_vote, safe_softmax, weighted_vote


DEFAULT_METRICS = [
    "trajectory_length",
    "mean_step_norm",
    "curvature_mean",
    "curvature_var",
    "curvature_max",
    "curvature_p90",
    "curvature_p99",
]
DEFAULT_LAYERS = [4, 8, 12, 14, 16, 20, 24, 28]
STRATEGIES = ["resid_geovote", "resid_geo_majority"]


def parse_range(spec: str, observed_ids: list[int]) -> set[int]:
    if spec == "all":
        return set(observed_ids)
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        start = int(start_s) if start_s else min(observed_ids)
        end = int(end_s) if end_s else max(observed_ids) + 1
        return {i for i in observed_ids if start <= i < end}
    return {int(x) for x in spec.split(",") if x.strip()}


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


def fit_residualizer(
    labels_cal: pd.DataFrame,
    metrics_cal: pd.DataFrame,
    metric: str,
    layer: int,
    prefix_n: int,
) -> tuple[float, float]:
    subset = metrics_cal[
        (metrics_cal["metric"] == metric) & (metrics_cal["layer"] == layer)
    ][["sample_id", "sample_idx", "value"]].copy()
    subset = subset.rename(columns={"value": "geo_value"})
    df = labels_cal.merge(subset, on=["sample_id", "sample_idx"], how="inner")
    df = df[df["sample_idx"] < prefix_n].copy()
    if df.empty:
        raise ValueError(f"empty calibration data for {metric}@{layer}")
    x = df["n_gen_tokens"].astype(float).to_numpy()
    y = df["geo_value"].astype(float).to_numpy()
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(((x - x_mean) ** 2).sum())
    slope = 0.0 if denom <= 0 else float(((x - x_mean) * (y - y_mean)).sum() / denom)
    intercept = y_mean - slope * x_mean
    return intercept, slope


def make_rows(
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    metric: str,
    layer: int,
    sign: str,
    grader: Callable[[object, object], bool],
    prefix_n: int,
    residualizer: tuple[float, float],
) -> pd.DataFrame:
    intercept, slope = residualizer
    subset = metrics[
        (metrics["metric"] == metric) & (metrics["layer"] == layer)
    ][["sample_id", "sample_idx", "value"]].copy()
    subset = subset.rename(columns={"value": "geo_value"})
    df = labels.merge(subset, on=["sample_id", "sample_idx"], how="inner")
    df = df[df["sample_idx"] < prefix_n].copy()
    df["resid_value"] = (
        df["geo_value"].astype(float)
        - intercept
        - slope * df["n_gen_tokens"].astype(float)
    )

    rows = []
    for sample_id, g in df.groupby("sample_id"):
        g = g.sort_values("sample_idx")
        preds = g["pred"].tolist()
        gold = g["gold"].iloc[0]
        correct = g["correct"].astype(bool).to_numpy()
        logprobs = g["sequence_logprob"].astype(float).to_numpy()
        lengths = g["n_gen_tokens"].astype(float).to_numpy()
        vals = g["resid_value"].astype(float).to_numpy()
        n = len(g)
        if n == 0 or not np.isfinite(vals).all():
            continue

        majority_pred = majority_vote(preds)
        logprob_max_pred = preds[int(np.argmax(logprobs))]
        logprob_weighted_pred = weighted_vote(preds, safe_softmax(logprobs))

        # Residuals can be negative, so use a per-question softmax over signed
        # confidence instead of the positive reciprocal used by raw GeoVote.
        confidence = -vals if sign == "min" else vals
        resid_weights = safe_softmax(confidence)
        resid_vote_pred = weighted_vote(preds, resid_weights)

        counts = Counter([p for p in preds if p is not None and not pd.isna(p)])
        if counts:
            resid_sum = defaultdict(float)
            for pred, weight in zip(preds, resid_weights):
                if pred is not None and not pd.isna(pred):
                    resid_sum[pred] += float(weight)
            combined = {answer: counts[answer] * resid_sum[answer] for answer in counts}
            resid_geo_majority_pred = max(combined.items(), key=lambda kv: kv[1])[0]
            top_counts = sorted(counts.values(), reverse=True)
            top_count = top_counts[0]
            second_count = top_counts[1] if len(top_counts) > 1 else 0
            majority_margin = (top_count - second_count) / max(n, 1)
            majority_top_frac = top_count / max(n, 1)
            n_unique = len(counts)
        else:
            resid_geo_majority_pred = None
            majority_margin = 0.0
            majority_top_frac = 0.0
            n_unique = 0

        rows.append(
            {
                "sample_id": int(sample_id),
                "n": n,
                "mean_n_gen_tokens": float(lengths.mean()),
                "majority_margin": float(majority_margin),
                "majority_top_frac": float(majority_top_frac),
                "n_unique_answers": int(n_unique),
                "oracle": bool(correct.any()),
                "first": bool(correct[0]),
                "majority": bool(grader(majority_pred, gold)),
                "logprob_max": bool(grader(logprob_max_pred, gold)),
                "logprob_weighted": bool(grader(logprob_weighted_pred, gold)),
                "resid_geovote": bool(grader(resid_vote_pred, gold)),
                "resid_geo_majority": bool(grader(resid_geo_majority_pred, gold)),
            }
        )
    return pd.DataFrame(rows)


def accuracy_rows(qdf: pd.DataFrame, prefix_n: int, cfg: dict, split: str) -> list[dict]:
    rows = []
    for strategy in [
        "first",
        "majority",
        "logprob_max",
        "logprob_weighted",
        "resid_geovote",
        "resid_geo_majority",
        "oracle",
    ]:
        rows.append(
            {
                "strategy": strategy,
                "accuracy": float(qdf[strategy].mean()),
                "n_questions": int(len(qdf)),
                "prefix_n": int(prefix_n),
                "frozen_metric": cfg["metric"],
                "frozen_layer": int(cfg["layer"]),
                "frozen_sign": cfg["sign"],
                "selected_strategy": cfg["strategy"],
                "resid_intercept": cfg["resid_intercept"],
                "resid_slope": cfg["resid_slope"],
                "split": split,
            }
        )
    return rows


def margin_strata(qdf: pd.DataFrame, prefix_n: int) -> pd.DataFrame:
    bins = [
        ("unanimous_or_near", qdf["majority_top_frac"] >= 0.75),
        (
            "medium_margin",
            (qdf["majority_top_frac"] < 0.75) & (qdf["majority_margin"] >= 0.25),
        ),
        ("low_margin", qdf["majority_margin"] < 0.25),
    ]
    rows = []
    for name, mask in bins:
        part = qdf[mask]
        if part.empty:
            continue
        row = {
            "prefix_n": int(prefix_n),
            "stratum": name,
            "n_questions": int(len(part)),
            "mean_top_frac": float(part["majority_top_frac"].mean()),
            "mean_unique_answers": float(part["n_unique_answers"].mean()),
            "mean_n_gen_tokens": float(part["mean_n_gen_tokens"].mean()),
        }
        for strategy in [
            "majority",
            "logprob_weighted",
            "resid_geovote",
            "resid_geo_majority",
            "oracle",
        ]:
            row[strategy] = float(part[strategy].mean())
        row["resid_geovote_minus_majority"] = row["resid_geovote"] - row["majority"]
        row["resid_geo_majority_minus_majority"] = (
            row["resid_geo_majority"] - row["majority"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def length_strata(qdf: pd.DataFrame, prefix_n: int) -> pd.DataFrame:
    qs = qdf["mean_n_gen_tokens"].quantile([1 / 3, 2 / 3]).to_list()
    lo, hi = float(qs[0]), float(qs[1])
    bins = [
        ("short_questions", qdf["mean_n_gen_tokens"] <= lo),
        (
            "medium_questions",
            (qdf["mean_n_gen_tokens"] > lo) & (qdf["mean_n_gen_tokens"] <= hi),
        ),
        ("long_questions", qdf["mean_n_gen_tokens"] > hi),
    ]
    rows = []
    for name, mask in bins:
        part = qdf[mask]
        if part.empty:
            continue
        row = {
            "prefix_n": int(prefix_n),
            "stratum": name,
            "n_questions": int(len(part)),
            "mean_n_gen_tokens": float(part["mean_n_gen_tokens"].mean()),
            "mean_top_frac": float(part["majority_top_frac"].mean()),
        }
        for strategy in [
            "majority",
            "logprob_weighted",
            "resid_geovote",
            "resid_geo_majority",
            "oracle",
        ]:
            row[strategy] = float(part[strategy].mean())
        row["resid_geovote_minus_majority"] = row["resid_geovote"] - row["majority"]
        row["resid_geo_majority_minus_majority"] = (
            row["resid_geo_majority"] - row["majority"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def complementarity(qdf: pd.DataFrame, prefix_n: int) -> pd.DataFrame:
    rows = []
    for method_a, method_b in [
        ("resid_geovote", "majority"),
        ("resid_geo_majority", "majority"),
        ("resid_geovote", "logprob_weighted"),
        ("resid_geo_majority", "logprob_weighted"),
    ]:
        a_ok = qdf[method_a].astype(bool)
        b_ok = qdf[method_b].astype(bool)
        rows.append(
            {
                "prefix_n": int(prefix_n),
                "method_a": method_a,
                "method_b": method_b,
                "both_correct": int((a_ok & b_ok).sum()),
                "a_only_recovery": int((a_ok & ~b_ok).sum()),
                "b_only_breakage": int((~a_ok & b_ok).sum()),
                "both_wrong": int((~a_ok & ~b_ok).sum()),
                "net_a_minus_b": int(
                    (a_ok & ~b_ok).sum() - (~a_ok & b_ok).sum()
                ),
                "n_questions": int(len(qdf)),
            }
        )
    return pd.DataFrame(rows)


def candidate_sweep(
    labels_cal: pd.DataFrame,
    metrics_cal: pd.DataFrame,
    grader: Callable[[object, object], bool],
    prefix_n: int,
    metrics_list: list[str],
    layers: list[int],
) -> pd.DataFrame:
    rows = []
    available = set(zip(metrics_cal["metric"], metrics_cal["layer"]))
    for metric in metrics_list:
        for layer in layers:
            if (metric, layer) not in available:
                continue
            residualizer = fit_residualizer(
                labels_cal, metrics_cal, metric, layer, prefix_n
            )
            for sign in ["min", "max"]:
                qdf = make_rows(
                    labels_cal,
                    metrics_cal,
                    metric,
                    layer,
                    sign,
                    grader,
                    prefix_n,
                    residualizer,
                )
                if qdf.empty:
                    continue
                for strategy in STRATEGIES:
                    acc = float(qdf[strategy].mean())
                    majority = float(qdf["majority"].mean())
                    rows.append(
                        {
                            "metric": metric,
                            "layer": int(layer),
                            "sign": sign,
                            "strategy": strategy,
                            "accuracy": acc,
                            "majority": majority,
                            "logprob_weighted": float(
                                qdf["logprob_weighted"].mean()
                            ),
                            "oracle": float(qdf["oracle"].mean()),
                            "delta_vs_majority": acc - majority,
                            "n_questions": int(len(qdf)),
                            "resid_intercept": residualizer[0],
                            "resid_slope": residualizer[1],
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], required=True)
    parser.add_argument("--prefixes", default="8,16")
    parser.add_argument("--calib-ids", default="0:100")
    parser.add_argument("--test-ids", default="100:500")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    args = parser.parse_args()

    labels, metrics, grader = load_run(args.run, args.dataset)
    observed_ids = sorted(int(x) for x in labels["sample_id"].unique())
    calib_ids = parse_range(args.calib_ids, observed_ids)
    test_ids = parse_range(args.test_ids, observed_ids)
    overlap = calib_ids & test_ids
    if overlap:
        raise SystemExit(f"calibration/test overlap: {sorted(overlap)[:5]}")
    labels_cal = labels[labels["sample_id"].isin(calib_ids)].copy()
    metrics_cal = metrics[metrics["sample_id"].isin(calib_ids)].copy()
    labels_test = labels[labels["sample_id"].isin(test_ids)].copy()
    metrics_test = metrics[metrics["sample_id"].isin(test_ids)].copy()

    prefixes = [int(x) for x in args.prefixes.split(",") if x.strip()]
    metrics_list = [x.strip() for x in args.metrics.split(",") if x.strip()]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    selected_rows = []
    accuracy = []
    margin_parts = []
    length_parts = []
    comp_parts = []
    for prefix_n in prefixes:
        sweep = candidate_sweep(
            labels_cal, metrics_cal, grader, prefix_n, metrics_list, layers
        )
        if sweep.empty:
            raise SystemExit(f"no residualized candidates for N={prefix_n}")
        sweep = sweep.sort_values(
            ["delta_vs_majority", "accuracy", "metric", "layer", "sign", "strategy"],
            ascending=[False, False, True, True, True, True],
        ).reset_index(drop=True)
        sweep.to_csv(args.out / f"calibration_sweep_N{prefix_n}.csv", index=False)
        cfg = sweep.iloc[0].to_dict()
        cfg["prefix_n"] = int(prefix_n)
        cfg["calib_ids"] = args.calib_ids
        cfg["test_ids"] = args.test_ids
        selected_rows.append(cfg)

        residualizer = (float(cfg["resid_intercept"]), float(cfg["resid_slope"]))
        q_cal = make_rows(
            labels_cal,
            metrics_cal,
            cfg["metric"],
            int(cfg["layer"]),
            cfg["sign"],
            grader,
            prefix_n,
            residualizer,
        )
        q_test = make_rows(
            labels_test,
            metrics_test,
            cfg["metric"],
            int(cfg["layer"]),
            cfg["sign"],
            grader,
            prefix_n,
            residualizer,
        )
        q_test.to_csv(args.out / f"heldout_question_level_N{prefix_n}.csv", index=False)
        accuracy.extend(accuracy_rows(q_cal, prefix_n, cfg, "calibration"))
        accuracy.extend(accuracy_rows(q_test, prefix_n, cfg, "heldout"))
        margin_parts.append(margin_strata(q_test, prefix_n))
        length_parts.append(length_strata(q_test, prefix_n))
        comp_parts.append(complementarity(q_test, prefix_n))

    pd.DataFrame(selected_rows).to_csv(args.out / "selected_config.csv", index=False)
    pd.DataFrame(accuracy).to_csv(args.out / "accuracy_by_split.csv", index=False)
    pd.concat(margin_parts, ignore_index=True).to_csv(
        args.out / "heldout_margin_strata.csv", index=False
    )
    pd.concat(length_parts, ignore_index=True).to_csv(
        args.out / "heldout_length_strata.csv", index=False
    )
    pd.concat(comp_parts, ignore_index=True).to_csv(
        args.out / "heldout_complementarity.csv", index=False
    )

    print("selected configs")
    print(pd.DataFrame(selected_rows).to_string(index=False))
    print("\naccuracy by split")
    print(pd.DataFrame(accuracy).to_string(index=False))


if __name__ == "__main__":
    main()
