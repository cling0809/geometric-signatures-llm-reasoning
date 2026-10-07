#!/usr/bin/env python3
"""Clean-split GeoVote evaluation.

This script prevents oracle tuning by separating:

1. calibration questions: choose metric/layer/sign/aggregation strategy;
2. held-out questions: evaluate the frozen choice exactly once.

It is intended to replace full-test metric/layer sweeps in paper tables.
Sweeps remain useful as exploratory appendix material, but main claims should
come from this clean-split protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geoprobe.datasets import is_correct, is_correct_math500

from geovote_independence_analysis import (
    make_complementarity,
    make_margin_strata,
    question_rows,
)


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
STRATEGIES = ["geovote", "geo_majority"]


def _parse_range(spec: str, observed_ids: list[int]) -> set[int]:
    if spec == "all":
        return set(observed_ids)
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        start = int(start_s) if start_s else min(observed_ids)
        end = int(end_s) if end_s else max(observed_ids) + 1
        return {i for i in observed_ids if start <= i < end}
    return {int(x) for x in spec.split(",") if x.strip()}


def _filter_by_ids(labels: pd.DataFrame, metrics: pd.DataFrame, ids: set[int]):
    labels_part = labels[labels["sample_id"].isin(ids)].copy()
    metrics_part = metrics[metrics["sample_id"].isin(ids)].copy()
    return labels_part, metrics_part


def _load_run(run: Path, dataset: str):
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


def _candidate_rows(
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    grader,
    prefix_n: int,
    metrics_list: list[str],
    layers: list[int],
) -> pd.DataFrame:
    rows = []
    available = set(zip(metrics["metric"], metrics["layer"]))
    for metric in metrics_list:
        for layer in layers:
            if (metric, layer) not in available:
                continue
            for sign in ["min", "max"]:
                qdf, _, _ = question_rows(
                    labels=labels,
                    metrics=metrics,
                    metric=metric,
                    layer=layer,
                    sign=sign,
                    grader=grader,
                    prefix_n=prefix_n,
                )
                if qdf.empty:
                    continue
                majority = float(qdf["majority"].mean())
                logprob = float(qdf["logprob_weighted"].mean())
                oracle = float(qdf["oracle"].mean())
                for strategy in STRATEGIES:
                    acc = float(qdf[strategy].mean())
                    rows.append(
                        {
                            "metric": metric,
                            "layer": int(layer),
                            "sign": sign,
                            "strategy": strategy,
                            "accuracy": acc,
                            "majority": majority,
                            "logprob_weighted": logprob,
                            "oracle": oracle,
                            "delta_vs_majority": acc - majority,
                            "n_questions": int(len(qdf)),
                        }
                    )
    return pd.DataFrame(rows)


def _evaluate_frozen(
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    grader,
    cfg: dict,
    prefix_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qdf, _, _ = question_rows(
        labels=labels,
        metrics=metrics,
        metric=cfg["metric"],
        layer=int(cfg["layer"]),
        sign=cfg["sign"],
        grader=grader,
        prefix_n=prefix_n,
    )
    rows = []
    for strategy in [
        "first",
        "majority",
        "logprob_max",
        "logprob_weighted",
        "geovote",
        "geo_majority",
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
            }
        )
    return pd.DataFrame(rows), qdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math500"])
    ap.add_argument("--prefixes", default="8", help="Comma-separated N prefixes, e.g. 8,16")
    ap.add_argument("--calib-ids", default="0:50", help="Question ids used for tuning, e.g. 0:100")
    ap.add_argument("--test-ids", default="50:", help="Held-out question ids, e.g. 100:500")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    args = ap.parse_args()

    labels, metrics, grader = _load_run(args.run, args.dataset)
    observed_ids = sorted(int(x) for x in labels["sample_id"].unique())
    calib_ids = _parse_range(args.calib_ids, observed_ids)
    test_ids = _parse_range(args.test_ids, observed_ids)
    overlap = calib_ids & test_ids
    if overlap:
        raise SystemExit(f"calibration/test overlap is forbidden: {sorted(overlap)[:10]}")
    if not calib_ids or not test_ids:
        raise SystemExit("calibration and test splits must both be non-empty")

    labels_cal, metrics_cal = _filter_by_ids(labels, metrics, calib_ids)
    labels_test, metrics_test = _filter_by_ids(labels, metrics, test_ids)
    prefixes = [int(x) for x in args.prefixes.split(",") if x.strip()]
    metrics_list = [x.strip() for x in args.metrics.split(",") if x.strip()]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    selected_rows = []
    eval_rows = []
    for prefix_n in prefixes:
        sweep = _candidate_rows(
            labels_cal,
            metrics_cal,
            grader,
            prefix_n,
            metrics_list=metrics_list,
            layers=layers,
        )
        if sweep.empty:
            raise SystemExit(f"no calibration candidates for prefix N={prefix_n}")
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

        calib_eval, _ = _evaluate_frozen(labels_cal, metrics_cal, grader, cfg, prefix_n)
        calib_eval["split"] = "calibration"
        test_eval, qdf_test = _evaluate_frozen(labels_test, metrics_test, grader, cfg, prefix_n)
        test_eval["split"] = "heldout"
        eval_rows.extend(calib_eval.to_dict(orient="records"))
        eval_rows.extend(test_eval.to_dict(orient="records"))

        qdf_test.to_csv(args.out / f"heldout_question_level_N{prefix_n}.csv", index=False)
        make_margin_strata(qdf_test, args.run.name, args.dataset, prefix_n).to_csv(
            args.out / f"heldout_margin_strata_N{prefix_n}.csv", index=False
        )
        make_complementarity(qdf_test, args.run.name, args.dataset, prefix_n).to_csv(
            args.out / f"heldout_complementarity_N{prefix_n}.csv", index=False
        )

    selected = pd.DataFrame(selected_rows)
    eval_df = pd.DataFrame(eval_rows)
    selected.to_csv(args.out / "selected_config.csv", index=False)
    eval_df.to_csv(args.out / "accuracy_by_split.csv", index=False)

    protocol = [
        "Clean-split GeoVote protocol",
        f"run={args.run}",
        f"dataset={args.dataset}",
        f"calib_ids={args.calib_ids}",
        f"test_ids={args.test_ids}",
        "Selection rule: choose metric/layer/sign/strategy on calibration by max delta_vs_majority.",
        "Held-out rule: evaluate the frozen selection exactly once.",
    ]
    (args.out / "PROTOCOL.txt").write_text("\n".join(protocol) + "\n")

    print("selected configs")
    print(selected.to_string(index=False))
    print("\naccuracy by split")
    print(eval_df.to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
