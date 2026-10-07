#!/usr/bin/env python3
"""Bootstrap and paired-test summaries for the paper.

This script does not launch model inference. It reads existing per-question /
per-sample artifacts and writes uncertainty summaries used to harden the paper.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


RUNS = {
    "Base": ROOT / "local_runs/2026-05-17_extract-qwen-1.5b-base-100",
    "Instruct": ROOT / "local_runs/2026-05-17_extract-qwen-1.5b-nomath-100",
    "Math-Instruct": ROOT / "local_runs/2026-05-17_extract-pilot-gsm8k-100",
    "R1-Distill": ROOT / "local_runs/2026-05-17_extract-deepseek-r1-distill-1.5b-100",
}


def ci(values: np.ndarray, q: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return tuple(float(x) for x in np.percentile(values, q))


def bootstrap_mean(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return ci(x[idx].mean(axis=1))


def bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    return ci((a[idx] - b[idx]).mean(axis=1))


def exact_paired_sign_p(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact sign test on discordant paired binary outcomes."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return a_only, b_only, min(1.0, 2.0 * tail)


def auc_1d(y: np.ndarray, s: np.ndarray) -> float:
    """Rank AUC with average ranks for ties; returns nan for single-class samples."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    sum_pos = float(ranks[y].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def signature_from_sample(metrics: pd.DataFrame, labels: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Build raw AUC signature for a bootstrap sample of question ids."""
    sampled_labels = labels.iloc[ids].reset_index(drop=True)
    sampled_labels["boot_id"] = np.arange(len(sampled_labels))
    sampled_metrics = metrics.merge(
        sampled_labels[["sample_id", "boot_id", "correct"]],
        on="sample_id",
        how="inner",
    )
    rows = []
    for (metric, layer), g in sampled_metrics.groupby(["metric", "layer"], sort=True):
        rows.append(
            {
                "metric": metric,
                "layer": int(layer),
                "auc": auc_1d((~g["correct"]).to_numpy(), g["value"].to_numpy()),
            }
        )
    out = pd.DataFrame(rows)
    return out.pivot(index="metric", columns="layer", values="auc").sort_index().sort_index(axis=1)


def signature_distance(a: pd.DataFrame, b: pd.DataFrame) -> float:
    a2, b2 = a.align(b, join="inner", axis=0)
    a2, b2 = a2.align(b2, join="inner", axis=1)
    return float(np.linalg.norm((a2.to_numpy() - 0.5).ravel() - (b2.to_numpy() - 0.5).ravel()))


@dataclass
class LoadedRun:
    metrics: pd.DataFrame
    labels: pd.DataFrame
    point_sig: pd.DataFrame


def load_run(run_dir: Path) -> LoadedRun:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    labels = pd.read_csv(run_dir / "labels.csv").sort_values("sample_id").reset_index(drop=True)
    sig = signature_from_sample(metrics, labels, np.arange(len(labels)))
    return LoadedRun(metrics=metrics, labels=labels, point_sig=sig)


def geovote_stats(out: Path, rng: np.random.Generator, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = ROOT / "results/2026-05-22_geovote_math500_2048_n16_clean_split_full"
    token_dir = ROOT / "results/2026-05-23_geovote_math500_2048_n16_length_controls/token_count_only"
    no_traj_dir = ROOT / "results/2026-05-23_geovote_math500_2048_n16_length_controls/no_trajlen"
    resid_dir = ROOT / "results/2026-05-24_geovote_residual_length_rescue/all_metrics"

    rows_acc = []
    rows_delta = []
    for n in (8, 16):
        q = pd.read_csv(base_dir / f"heldout_question_level_N{n}.csv").sort_values("sample_id")
        token = pd.read_csv(token_dir / f"heldout_question_level_N{n}.csv").sort_values("sample_id")
        no_traj = pd.read_csv(no_traj_dir / f"heldout_question_level_N{n}.csv").sort_values("sample_id")
        resid = pd.read_csv(resid_dir / f"heldout_question_level_N{n}.csv").sort_values("sample_id")
        methods = {
            "first": q["first"],
            "majority": q["majority"],
            "logprob_weighted": q["logprob_weighted"],
            "geovote": q["geovote"],
            "geo_majority": q["geo_majority"],
            "token_majority": token["token_majority"],
            "no_trajlen_geo_majority": no_traj["geo_majority"],
            "resid_geo_majority": resid["resid_geo_majority"],
            "oracle": q["oracle"],
        }
        for name, vals in methods.items():
            arr = vals.to_numpy(dtype=bool)
            lo, hi = bootstrap_mean(arr, rng, n_boot)
            rows_acc.append(
                {
                    "prefix_n": n,
                    "method": name,
                    "accuracy": arr.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_questions": len(arr),
                }
            )
        refs = {
            "majority": q["majority"].to_numpy(dtype=bool),
            "logprob_weighted": q["logprob_weighted"].to_numpy(dtype=bool),
        }
        for method_name in (
            "geovote",
            "geo_majority",
            "token_majority",
            "no_trajlen_geo_majority",
            "resid_geo_majority",
        ):
            arr = methods[method_name].to_numpy(dtype=bool)
            for ref_name, ref in refs.items():
                lo, hi = bootstrap_delta(arr, ref, rng, n_boot)
                a_only, b_only, p = exact_paired_sign_p(arr, ref)
                rows_delta.append(
                    {
                        "prefix_n": n,
                        "method": method_name,
                        "reference": ref_name,
                        "delta": arr.mean() - ref.mean(),
                        "ci_low": lo,
                        "ci_high": hi,
                        "method_only": a_only,
                        "reference_only": b_only,
                        "exact_sign_p": p,
                    }
                )

    acc = pd.DataFrame(rows_acc)
    delta = pd.DataFrame(rows_delta)
    acc.to_csv(out / "geovote_accuracy_ci.csv", index=False)
    delta.to_csv(out / "geovote_paired_delta_ci.csv", index=False)
    return acc, delta


def crosssteer_holdout_stats(out: Path, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    roots = {
        "Qwen-Math -> R1-Distill": ROOT
        / "results/server_2026-05-24_c4-holdout-baselines/cross_source_qwen_math_to_r1/c4_per_sample.jsonl",
        "R1-Distill -> R1-Distill": ROOT
        / "results/server_2026-05-24_c4-holdout-baselines/same_target_r1_to_r1/c4_per_sample.jsonl",
    }
    rows = []
    for name, path in roots.items():
        df = pd.read_json(path, lines=True)
        base = df[df["alpha"] == 0].sort_values("sample_id")
        for alpha in sorted(a for a in df["alpha"].unique() if a != 0):
            steered = df[df["alpha"] == alpha].sort_values("sample_id")
            merged = base[["sample_id", "correct"]].merge(
                steered[["sample_id", "correct"]],
                on="sample_id",
                suffixes=("_base", "_steered"),
            )
            b = merged["correct_base"].to_numpy(dtype=bool)
            s = merged["correct_steered"].to_numpy(dtype=bool)
            lo, hi = bootstrap_delta(s, b, rng, n_boot)
            s_only, b_only, p = exact_paired_sign_p(s, b)
            rows.append(
                {
                    "setting": name,
                    "alpha": alpha,
                    "baseline_accuracy": b.mean(),
                    "steered_accuracy": s.mean(),
                    "delta": s.mean() - b.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "steered_only": s_only,
                    "baseline_only": b_only,
                    "exact_sign_p": p,
                    "n_questions": len(merged),
                }
            )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out / "crosssteer_holdout_ci.csv", index=False)
    return out_df


def random_control_stats(out: Path) -> pd.DataFrame:
    path = ROOT / "results/2026-05-26_tacl_server_artifacts/random_control/c4_random_summary.csv"
    df = pd.read_csv(path)
    values = df["accuracy"].to_numpy(dtype=float)
    source_accuracy = 0.47
    baseline_accuracy = 0.32
    row = {
        "setting": "Qwen-Math -> R1-Distill random controls",
        "n_random": int(len(values)),
        "baseline_accuracy": baseline_accuracy,
        "source_accuracy": source_accuracy,
        "random_mean": float(values.mean()),
        "random_median": float(np.median(values)),
        "random_sd": float(values.std(ddof=1)),
        "random_q25": float(np.percentile(values, 25)),
        "random_q75": float(np.percentile(values, 75)),
        "random_min": float(values.min()),
        "random_max": float(values.max()),
        "random_below_source": int(np.sum(values < source_accuracy)),
        "source_minus_random_mean": float(source_accuracy - values.mean()),
        "empirical_percentile": float(np.mean(values < source_accuracy)),
    }
    out_df = pd.DataFrame([row])
    out_df.to_csv(out / "crosssteer_random_control_k20.csv", index=False)
    return out_df


def signature_bootstrap(out: Path, rng: np.random.Generator, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    loaded = {name: load_run(path) for name, path in RUNS.items()}
    names = list(loaded.keys())
    point_pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            point_pairs.append(
                {
                    "pair": f"{a}--{b}",
                    "point_distance": signature_distance(loaded[a].point_sig, loaded[b].point_sig),
                }
            )

    pair_values: dict[str, list[float]] = {row["pair"]: [] for row in point_pairs}
    relation_rows = []
    relation_values = {
        "Base-Math > Base-Instruct": [],
        "R1 closer to Instruct than Math": [],
        "R1 closer to Instruct than Base": [],
    }
    n = len(next(iter(loaded.values())).labels)
    for _ in range(n_boot):
        ids = rng.integers(0, n, size=n)
        sigs = {
            name: signature_from_sample(run.metrics, run.labels, ids)
            for name, run in loaded.items()
        }
        distances = {}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                key = f"{a}--{b}"
                distances[key] = signature_distance(sigs[a], sigs[b])
                pair_values[key].append(distances[key])
        relation_values["Base-Math > Base-Instruct"].append(
            distances["Base--Math-Instruct"] - distances["Base--Instruct"]
        )
        relation_values["R1 closer to Instruct than Math"].append(
            distances["Math-Instruct--R1-Distill"] - distances["Instruct--R1-Distill"]
        )
        relation_values["R1 closer to Instruct than Base"].append(
            distances["Base--R1-Distill"] - distances["Instruct--R1-Distill"]
        )

    pair_rows = []
    for row in point_pairs:
        values = np.asarray(pair_values[row["pair"]])
        lo, hi = ci(values)
        pair_rows.append(
            {
                "pair": row["pair"],
                "point_distance": row["point_distance"],
                "bootstrap_mean": float(values.mean()),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    for relation, values in relation_values.items():
        values = np.asarray(values)
        lo, hi = ci(values)
        relation_rows.append(
            {
                "relation": relation,
                "point_margin": {
                    "Base-Math > Base-Instruct": point_distance_lookup(point_pairs, "Base--Math-Instruct")
                    - point_distance_lookup(point_pairs, "Base--Instruct"),
                    "R1 closer to Instruct than Math": point_distance_lookup(
                        point_pairs, "Math-Instruct--R1-Distill"
                    )
                    - point_distance_lookup(point_pairs, "Instruct--R1-Distill"),
                    "R1 closer to Instruct than Base": point_distance_lookup(
                        point_pairs, "Base--R1-Distill"
                    )
                    - point_distance_lookup(point_pairs, "Instruct--R1-Distill"),
                }[relation],
                "bootstrap_mean_margin": float(values.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "support_rate_margin_gt_0": float(np.mean(values > 0)),
            }
        )

    pairs = pd.DataFrame(pair_rows)
    rels = pd.DataFrame(relation_rows)
    pairs.to_csv(out / "signature_pair_distance_ci.csv", index=False)
    rels.to_csv(out / "signature_relation_bootstrap.csv", index=False)
    return pairs, rels


def point_distance_lookup(rows: list[dict], key: str) -> float:
    for row in rows:
        if row["pair"] == key:
            return float(row["point_distance"])
    raise KeyError(key)


def write_report(
    out: Path,
    geovote_acc: pd.DataFrame,
    geovote_delta: pd.DataFrame,
    cross: pd.DataFrame,
    random_control: pd.DataFrame,
    sig_pairs: pd.DataFrame,
    sig_rels: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> None:
    def pct(x: float) -> str:
        return f"{100*x:.1f}"

    n16_geo = geovote_delta[
        (geovote_delta["prefix_n"] == 16)
        & (geovote_delta["method"] == "geo_majority")
        & (geovote_delta["reference"] == "majority")
    ].iloc[0]
    n16_geo_lp = geovote_delta[
        (geovote_delta["prefix_n"] == 16)
        & (geovote_delta["method"] == "geo_majority")
        & (geovote_delta["reference"] == "logprob_weighted")
    ].iloc[0]
    n16_token = geovote_delta[
        (geovote_delta["prefix_n"] == 16)
        & (geovote_delta["method"] == "token_majority")
        & (geovote_delta["reference"] == "majority")
    ].iloc[0]
    cross_math = cross[
        (cross["setting"] == "Qwen-Math -> R1-Distill") & (cross["alpha"] == 2)
    ].iloc[0]
    same_r1 = cross[
        (cross["setting"] == "R1-Distill -> R1-Distill") & (cross["alpha"] == 2)
    ].iloc[0]
    random_row = random_control.iloc[0]

    lines = [
        "# Statistical hardening report",
        "",
        f"- Seed: `{seed}`",
        f"- Bootstrap replicates: `{n_boot}`",
        "",
        "## GeoVote held-out MATH-500",
        "",
        (
            "At N=16, selected Geo+Majority vs majority is "
            f"{pct(n16_geo.delta)} pp, 95% CI "
            f"[{pct(n16_geo.ci_low)}, {pct(n16_geo.ci_high)}] pp, "
            f"exact paired sign p={n16_geo.exact_sign_p:.4g}."
        ),
        (
            "At N=16, selected Geo+Majority vs logprob-weighted is "
            f"{pct(n16_geo_lp.delta)} pp, 95% CI "
            f"[{pct(n16_geo_lp.ci_low)}, {pct(n16_geo_lp.ci_high)}] pp, "
            f"exact paired sign p={n16_geo_lp.exact_sign_p:.4g}."
        ),
        (
            "Token-count+Majority vs majority is "
            f"{pct(n16_token.delta)} pp, 95% CI "
            f"[{pct(n16_token.ci_low)}, {pct(n16_token.ci_high)}] pp, "
            "supporting the paper's length-mediated interpretation."
        ),
        "",
        "## CrossSteer clean held-out GSM8K",
        "",
        (
            "Qwen-Math -> R1-Distill at alpha=2 gives "
            f"{pct(cross_math.delta)} pp, 95% CI "
            f"[{pct(cross_math.ci_low)}, {pct(cross_math.ci_high)}] pp, "
            f"exact paired sign p={cross_math.exact_sign_p:.4g}."
        ),
        (
            "R1-Distill -> R1-Distill at alpha=2 gives "
            f"{pct(same_r1.delta)} pp, 95% CI "
            f"[{pct(same_r1.ci_low)}, {pct(same_r1.ci_high)}] pp, "
            f"exact paired sign p={same_r1.exact_sign_p:.4g}."
        ),
        (
            "Matched-norm random controls: the Qwen-Math source direction reaches "
            f"{random_row.source_accuracy:.3f}, exceeding "
            f"{int(random_row.random_below_source)}/{int(random_row.n_random)} random directions; "
            f"the random center is mean {random_row.random_mean:.3f}, "
            f"median {random_row.random_median:.3f}, IQR "
            f"[{random_row.random_q25:.3f}, {random_row.random_q75:.3f}]."
        ),
        "",
        "## Signature-distance bootstrap",
        "",
    ]
    for _, row in sig_rels.iterrows():
        lines.append(
            f"- {row.relation}: point margin {row.point_margin:.3f}, "
            f"95% CI [{row.ci_low:.3f}, {row.ci_high:.3f}], "
            f"support {100*row.support_rate_margin_gt_0:.1f}%."
        )
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `geovote_accuracy_ci.csv`",
            "- `geovote_paired_delta_ci.csv`",
            "- `crosssteer_holdout_ci.csv`",
            "- `crosssteer_random_control_k20.csv`",
            "- `signature_pair_distance_ci.csv`",
            "- `signature_relation_bootstrap.csv`",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/2026-05-25_statistical_hardening",
    )
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--signature-boot", type=int, default=1000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    geovote_acc, geovote_delta = geovote_stats(args.out, rng, args.n_boot)
    cross = crosssteer_holdout_stats(args.out, rng, args.n_boot)
    random_control = random_control_stats(args.out)
    sig_pairs, sig_rels = signature_bootstrap(args.out, rng, args.signature_boot)
    write_report(
        args.out,
        geovote_acc,
        geovote_delta,
        cross,
        random_control,
        sig_pairs,
        sig_rels,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(f"Wrote statistical summaries to {args.out}")


if __name__ == "__main__":
    main()
