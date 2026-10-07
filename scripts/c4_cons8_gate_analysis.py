#!/usr/bin/env python3
"""Post-hoc C4 cons@N gate and mixed-ensemble analysis.

This script reads c4_steer_r1.py JSONL outputs. It does not run generation.

It compares alpha=0 self-consistency against one or more steered alpha runs:
  - paired transitions: fixed vs broken samples
  - threshold gates: switch only when baseline vote is uncertain
  - mixed ensembles: combine k baseline samples with N-k steered samples

The goal is to test whether steering should be used selectively under cons@N.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from geoprobe.datasets import is_correct, is_correct_math500


@dataclass(frozen=True)
class Row:
    alpha: float
    sample_id: int
    gold: str
    pred: str | None
    correct: bool
    samples_preds: tuple[str | None, ...]


def _parse_alpha_label(label: str) -> float:
    return float(label.replace("p", "."))


def _norm_alpha(x: object) -> float:
    return float(x)


def _pred_to_gsm_float(pred: str | None) -> float | None:
    if pred is None or pred == "None":
        return None
    try:
        return float(str(pred).replace(",", ""))
    except ValueError:
        return None


def _gold_to_gsm_float(gold: str) -> float:
    return float(str(gold).replace(",", ""))


def _grader(dataset: str):
    if dataset == "gsm8k":
        return lambda pred, gold: is_correct(_pred_to_gsm_float(pred), _gold_to_gsm_float(gold))
    if dataset == "math500":
        return lambda pred, gold: is_correct_math500(pred, gold)
    raise ValueError(f"unknown dataset {dataset!r}")


def _majority(preds: Iterable[str | None]) -> str | None:
    nonnull = [p for p in preds if p is not None and p != "None"]
    if not nonnull:
        return None
    return Counter(nonnull).most_common(1)[0][0]


def _vote_stats(preds: Iterable[str | None]) -> dict:
    nonnull = [p for p in preds if p is not None and p != "None"]
    counts = Counter(nonnull)
    if not counts:
        return {
            "top_count": 0,
            "second_count": 0,
            "margin": 0,
            "entropy": 0.0,
            "n_unique": 0,
        }
    ordered = counts.most_common()
    top = ordered[0][1]
    second = ordered[1][1] if len(ordered) > 1 else 0
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p + 1e-12)
    return {
        "top_count": top,
        "second_count": second,
        "margin": top - second,
        "entropy": entropy,
        "n_unique": len(counts),
    }


def _load_jsonl(path: Path, alpha: float) -> dict[int, Row]:
    rows: dict[int, Row] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if abs(_norm_alpha(d["alpha"]) - alpha) > 1e-9:
                continue
            sid = int(d["sample_id"])
            rows[sid] = Row(
                alpha=float(d["alpha"]),
                sample_id=sid,
                gold=str(d["gold"]),
                pred=d.get("pred"),
                correct=bool(d["correct"]),
                samples_preds=tuple(d.get("samples_preds") or []),
            )
    return rows


def _evaluate_pick(rows: Iterable[tuple[str | None, str]], dataset: str) -> tuple[int, int, float]:
    grade = _grader(dataset)
    n = 0
    correct = 0
    for pred, gold in rows:
        n += 1
        correct += int(grade(pred, gold))
    return correct, n, correct / n if n else float("nan")


def analyze_pair(
    dataset: str,
    baseline: dict[int, Row],
    steered: dict[int, Row],
    alpha: float,
    max_cons_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grade = _grader(dataset)
    common = sorted(set(baseline) & set(steered))
    if max_cons_n is None:
        max_cons_n = min(
            min((len(baseline[i].samples_preds) for i in common), default=0),
            min((len(steered[i].samples_preds) for i in common), default=0),
        )

    transition = Counter()
    stats_rows = []
    for sid in common:
        b = baseline[sid]
        s = steered[sid]
        if b.correct and s.correct:
            transition["both_correct"] += 1
        elif (not b.correct) and s.correct:
            transition["fixed"] += 1
        elif b.correct and (not s.correct):
            transition["broken"] += 1
        else:
            transition["both_wrong"] += 1
        vst = _vote_stats(b.samples_preds)
        stats_rows.append({
            "sample_id": sid,
            "alpha": alpha,
            "baseline_correct": b.correct,
            "steered_correct": s.correct,
            **vst,
        })

    transition_df = pd.DataFrame([{
        "alpha": alpha,
        "n_common": len(common),
        "baseline_correct": sum(int(baseline[i].correct) for i in common),
        "steered_correct": sum(int(steered[i].correct) for i in common),
        "baseline_acc": sum(int(baseline[i].correct) for i in common) / len(common) if common else float("nan"),
        "steered_acc": sum(int(steered[i].correct) for i in common) / len(common) if common else float("nan"),
        "fixed": transition["fixed"],
        "broken": transition["broken"],
        "net_fixed_minus_broken": transition["fixed"] - transition["broken"],
        "both_correct": transition["both_correct"],
        "both_wrong": transition["both_wrong"],
        "oracle_switch_acc": (
            sum(int(baseline[i].correct or steered[i].correct) for i in common) / len(common)
            if common else float("nan")
        ),
    }])

    gate_rows = []
    # Baselines for this common subset.
    base_correct, n, base_acc = _evaluate_pick(((baseline[i].pred, baseline[i].gold) for i in common), dataset)
    steer_correct, _, steer_acc = _evaluate_pick(((steered[i].pred, steered[i].gold) for i in common), dataset)
    gate_rows.append({
        "alpha": alpha,
        "strategy": "baseline_alpha0",
        "threshold_type": "none",
        "threshold": None,
        "n_switched": 0,
        "correct": base_correct,
        "n": n,
        "accuracy": base_acc,
    })
    gate_rows.append({
        "alpha": alpha,
        "strategy": "all_steered",
        "threshold_type": "none",
        "threshold": None,
        "n_switched": n,
        "correct": steer_correct,
        "n": n,
        "accuracy": steer_acc,
    })

    # Switch from baseline majority to steered majority only when baseline is uncertain.
    for top_threshold in range(1, max_cons_n + 1):
        picks = []
        switched = 0
        for sid in common:
            b = baseline[sid]
            s = steered[sid]
            st = _vote_stats(b.samples_preds)
            use_steered = st["top_count"] <= top_threshold
            switched += int(use_steered)
            picks.append((s.pred if use_steered else b.pred, b.gold))
        c, n, acc = _evaluate_pick(picks, dataset)
        gate_rows.append({
            "alpha": alpha,
            "strategy": f"gate_top_count_le_{top_threshold}",
            "threshold_type": "top_count",
            "threshold": top_threshold,
            "n_switched": switched,
            "correct": c,
            "n": n,
            "accuracy": acc,
        })

    for margin_threshold in range(0, max_cons_n + 1):
        picks = []
        switched = 0
        for sid in common:
            b = baseline[sid]
            s = steered[sid]
            st = _vote_stats(b.samples_preds)
            use_steered = st["margin"] <= margin_threshold
            switched += int(use_steered)
            picks.append((s.pred if use_steered else b.pred, b.gold))
        c, n, acc = _evaluate_pick(picks, dataset)
        gate_rows.append({
            "alpha": alpha,
            "strategy": f"gate_margin_le_{margin_threshold}",
            "threshold_type": "margin",
            "threshold": margin_threshold,
            "n_switched": switched,
            "correct": c,
            "n": n,
            "accuracy": acc,
        })

    # Mixed ensemble: k baseline samples + N-k steered samples.
    mixed_rows = []
    for base_k in range(0, max_cons_n + 1):
        picks = []
        for sid in common:
            b = baseline[sid]
            s = steered[sid]
            preds = list(b.samples_preds[:base_k]) + list(s.samples_preds[: max_cons_n - base_k])
            picks.append((_majority(preds), b.gold))
        c, n, acc = _evaluate_pick(picks, dataset)
        mixed_rows.append({
            "alpha": alpha,
            "strategy": f"mixed_base{base_k}_steer{max_cons_n - base_k}",
            "base_k": base_k,
            "steer_k": max_cons_n - base_k,
            "correct": c,
            "n": n,
            "accuracy": acc,
        })

    # Gate to mixed instead of all-steered.
    gated_mixed_rows = []
    for base_k in range(0, max_cons_n + 1):
        for top_threshold in range(1, max_cons_n + 1):
            picks = []
            switched = 0
            for sid in common:
                b = baseline[sid]
                s = steered[sid]
                st = _vote_stats(b.samples_preds)
                use_mixed = st["top_count"] <= top_threshold
                switched += int(use_mixed)
                if use_mixed:
                    preds = list(b.samples_preds[:base_k]) + list(s.samples_preds[: max_cons_n - base_k])
                    pick = _majority(preds)
                else:
                    pick = b.pred
                picks.append((pick, b.gold))
            c, n, acc = _evaluate_pick(picks, dataset)
            gated_mixed_rows.append({
                "alpha": alpha,
                "strategy": f"gate_top_count_le_{top_threshold}_mixed_base{base_k}_steer{max_cons_n - base_k}",
                "threshold_type": "top_count",
                "threshold": top_threshold,
                "base_k": base_k,
                "steer_k": max_cons_n - base_k,
                "n_switched": switched,
                "correct": c,
                "n": n,
                "accuracy": acc,
            })

    return (
        transition_df,
        pd.DataFrame(gate_rows),
        pd.DataFrame(mixed_rows),
        pd.DataFrame(gated_mixed_rows),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math500"])
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--baseline-alpha", type=float, default=0.0)
    ap.add_argument(
        "--steered",
        nargs="+",
        required=True,
        help="One or more ALPHA=PATH specs. Example: 0.5=/runs/a0p5/c4_per_sample.jsonl",
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    baseline = _load_jsonl(args.baseline, args.baseline_alpha)
    print(f"baseline alpha={args.baseline_alpha}: {len(baseline)} rows from {args.baseline}")

    transitions = []
    gates = []
    mixed = []
    gated_mixed = []

    for spec in args.steered:
        if "=" not in spec:
            raise ValueError(f"--steered item must be ALPHA=PATH, got {spec!r}")
        label, path_s = spec.split("=", 1)
        alpha = _parse_alpha_label(label)
        steered = _load_jsonl(Path(path_s), alpha)
        print(f"steered alpha={alpha}: {len(steered)} rows from {path_s}")
        if not steered:
            continue
        t, g, m, gm = analyze_pair(args.dataset, baseline, steered, alpha)
        transitions.append(t)
        gates.append(g)
        mixed.append(m)
        gated_mixed.append(gm)

    transition_df = pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()
    gate_df = pd.concat(gates, ignore_index=True) if gates else pd.DataFrame()
    mixed_df = pd.concat(mixed, ignore_index=True) if mixed else pd.DataFrame()
    gated_mixed_df = pd.concat(gated_mixed, ignore_index=True) if gated_mixed else pd.DataFrame()

    transition_df.to_csv(args.out / "transition_matrix.csv", index=False)
    gate_df.to_csv(args.out / "gated_switch.csv", index=False)
    mixed_df.to_csv(args.out / "mixed_ensemble.csv", index=False)
    gated_mixed_df.to_csv(args.out / "gated_mixed_ensemble.csv", index=False)

    def _show_best(name: str, df: pd.DataFrame, k: int = 12) -> None:
        print(f"\n=== {name} ===")
        if df.empty:
            print("(empty)")
        else:
            cols = [c for c in [
                "alpha", "strategy", "accuracy", "correct", "n", "n_switched",
                "base_k", "steer_k", "threshold_type", "threshold",
                "baseline_acc", "steered_acc", "fixed", "broken", "net_fixed_minus_broken",
                "oracle_switch_acc",
            ] if c in df.columns]
            print(df.sort_values("accuracy" if "accuracy" in df.columns else "alpha", ascending=False)[cols].head(k).to_string(index=False))

    _show_best("transition", transition_df)
    _show_best("gated switch", gate_df)
    _show_best("mixed ensemble", mixed_df)
    _show_best("gated mixed ensemble", gated_mixed_df)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
