#!/usr/bin/env python3
"""Slice-level robustness for the n=300 signature diagnostic.

This is a no-inference analysis. It uses the copied n=300 metric/label CSVs
from the server run and recomputes signature distances on multiple 100-question
slices to check whether the main relations are specific to the first-100 slice.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = (
    ROOT
    / "results/2026-05-26_tacl_server_artifacts/signature_n300/raw_runs"
)
DEFAULT_OUT = ROOT / "results/2026-05-26_tacl_slice_robustness"


RUN_DIRS = {
    "Base": "qwen-base",
    "Instruct": "qwen-instruct",
    "Math-Instruct": "qwen-math",
    "R1-Distill": "r1-distill",
}


def auc_1d(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    sum_pos = float(ranks[y].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass
class Run:
    metrics: pd.DataFrame
    labels: pd.DataFrame


def load_runs(raw: Path) -> dict[str, Run]:
    runs: dict[str, Run] = {}
    for label, dirname in RUN_DIRS.items():
        run_dir = raw / dirname
        metrics = pd.read_csv(run_dir / "metrics.csv")
        labels = pd.read_csv(run_dir / "labels.csv")
        labels = labels.sort_values("sample_id").reset_index(drop=True)
        runs[label] = Run(metrics=metrics, labels=labels)
    return runs


def signature(run: Run, sample_ids: np.ndarray) -> pd.DataFrame:
    labels = run.labels[run.labels["sample_id"].isin(sample_ids)][
        ["sample_id", "correct"]
    ]
    joined = run.metrics.merge(labels, on="sample_id", how="inner")
    rows = []
    for (metric, layer), group in joined.groupby(["metric", "layer"], sort=True):
        rows.append(
            {
                "metric": metric,
                "layer": int(layer),
                "auc": auc_1d((~group["correct"]).to_numpy(), group["value"].to_numpy()),
            }
        )
    return (
        pd.DataFrame(rows)
        .pivot(index="metric", columns="layer", values="auc")
        .sort_index()
        .sort_index(axis=1)
    )


def distance(a: pd.DataFrame, b: pd.DataFrame) -> float:
    a2, b2 = a.align(b, join="inner", axis=0)
    a2, b2 = a2.align(b2, join="inner", axis=1)
    diff = (a2.to_numpy() - 0.5) - (b2.to_numpy() - 0.5)
    return float(np.linalg.norm(diff.ravel()))


def summarize_slice(name: str, ids: np.ndarray, runs: dict[str, Run]) -> dict[str, float | str | int]:
    sigs = {model: signature(run, ids) for model, run in runs.items()}
    d_bi = distance(sigs["Base"], sigs["Instruct"])
    d_bm = distance(sigs["Base"], sigs["Math-Instruct"])
    d_br = distance(sigs["Base"], sigs["R1-Distill"])
    d_im = distance(sigs["Instruct"], sigs["Math-Instruct"])
    d_ir = distance(sigs["Instruct"], sigs["R1-Distill"])
    d_mr = distance(sigs["Math-Instruct"], sigs["R1-Distill"])
    math_min = min(d_bm, d_im, d_mr)
    nonmath_max = max(d_bi, d_br, d_ir)
    return {
        "slice": name,
        "n": int(len(ids)),
        "Base--Instruct": d_bi,
        "Base--Math": d_bm,
        "Base--R1": d_br,
        "Instruct--Math": d_im,
        "Instruct--R1": d_ir,
        "Math--R1": d_mr,
        "BaseMath_minus_BaseInstruct": d_bm - d_bi,
        "R1Math_minus_R1Instruct": d_mr - d_ir,
        "R1Math_minus_R1Base": d_mr - d_br,
        "Math_min_minus_nonmath_max": math_min - nonmath_max,
        "math_separated_from_nonmath": bool(math_min > nonmath_max),
        "R1_closer_to_nonmath_than_math": bool(min(d_br, d_ir) < d_mr),
    }


def bootstrap_n300(
    runs: dict[str, Run],
    all_ids: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    for _ in range(n_boot):
        ids = rng.choice(all_ids, size=len(all_ids), replace=True)
        row = summarize_slice("bootstrap", ids, runs)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out: Path, rows: pd.DataFrame, boot: pd.DataFrame | None = None) -> None:
    def pct_bool(col: str) -> str:
        return f"{int(rows[col].sum())}/{len(rows)}"

    margin_cols = [
        "BaseMath_minus_BaseInstruct",
        "R1Math_minus_R1Instruct",
        "R1Math_minus_R1Base",
        "Math_min_minus_nonmath_max",
    ]
    lines = [
        "# Signature slice robustness",
        "",
        "This no-inference analysis recomputes the n=300 signature diagnostic on",
        "three disjoint 100-question blocks and five random 100-question slices.",
        "It also bootstraps the full n=300 diagnostic over question ids.",
        "",
        "## Stability summary",
        "",
        f"- Math separated from the non-math cluster: {pct_bool('math_separated_from_nonmath')} slices.",
        f"- R1 closer to a non-math variant than to Math-Instruct: {pct_bool('R1_closer_to_nonmath_than_math')} slices.",
        "",
        "## Mean margins across slices",
        "",
    ]
    for col in margin_cols:
        lines.append(
            f"- `{col}`: mean {rows[col].mean():.3f}, "
            f"min {rows[col].min():.3f}, max {rows[col].max():.3f}"
        )
    lines.extend(
        [
            "",
            "The 100-question slices are intentionally stress tests and are noisy. We",
            "therefore use them only to bound the claim: the exact nearest neighbor can",
            "vary by slice, while the full n=300 diagnostic is the robustness result.",
            "",
        ]
    )
    if boot is not None:
        lines.extend(
            [
                "## Full n=300 bootstrap",
                "",
            ]
        )
        for col in margin_cols:
            values = boot[col].to_numpy(dtype=float)
            lo, hi = np.percentile(values, [2.5, 97.5])
            lines.append(
                f"- `{col}`: mean {values.mean():.3f}, "
                f"95% CI [{lo:.3f}, {hi:.3f}], support {100*np.mean(values > 0):.1f}%"
            )
        lines.append("")
        lines.append(
            "Paper use: the Base--Math > Base--Instruct relation is the only n=300"
        )
        lines.append(
            "bootstrap relation with high support here. Treat the other slice-level"
        )
        lines.append(
            "relations as internal stress tests rather than additional paper claims."
        )
        lines.append("")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `slice_distance_relations.csv`",
            "- `n300_relation_bootstrap.csv`",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--random-slices", type=int, default=5)
    parser.add_argument("--slice-size", type=int, default=100)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.raw)
    all_ids = np.array(sorted(runs["Base"].labels["sample_id"].unique()))
    rows = []
    for start in range(0, len(all_ids), args.slice_size):
        ids = all_ids[start : start + args.slice_size]
        if len(ids) == args.slice_size:
            rows.append(summarize_slice(f"block_{start}_{start + args.slice_size - 1}", ids, runs))

    rng = np.random.default_rng(args.seed)
    for i in range(args.random_slices):
        ids = np.sort(rng.choice(all_ids, size=args.slice_size, replace=False))
        rows.append(summarize_slice(f"random_{i}", ids, runs))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out / "slice_distance_relations.csv", index=False)
    boot = bootstrap_n300(runs, all_ids, rng, args.n_boot)
    boot.to_csv(args.out / "n300_relation_bootstrap.csv", index=False)
    write_report(args.out, out_df, boot)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
