#!/usr/bin/env python3
"""Summarize whether signature orientation predicts CrossSteer applicability.

This is a zero-GPU analysis over existing experiment artifacts.  The goal is to
turn the cross-target negative result into a compact, auditable boundary claim:
the same source direction helps targets whose curvature orientation is opposite
to the source family, and hurts an already-aligned target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE3 = ROOT / "experiments" / "2026-05-17_phase3-paradigm-4"
PHASE4 = ROOT / "experiments" / "2026-05-18_phase4-math500-and-cross-target"
OUT = ROOT / "results" / "2026-05-24_crosssteer_compatibility"


SIGNATURES = {
    "Base": PHASE3 / "2026-05-17_extract-qwen-1.5b-base-100.csv",
    "Instruct": PHASE3 / "2026-05-17_extract-qwen-1.5b-nomath-100.csv",
    "Math": PHASE3 / "2026-05-17_extract-pilot-gsm8k-100.csv",
    "R1-Distill": PHASE3 / "2026-05-17_extract-deepseek-r1-distill-1.5b-100.csv",
}


def _orientation(sig_csv: Path) -> dict:
    sig = pd.read_csv(sig_csv, index_col=0)
    row = sig.loc["curvature_mean"].astype(float)
    centred = row - 0.5
    best_layer = int(centred.abs().idxmax())
    auc = float(row.loc[str(best_layer)] if str(best_layer) in row.index else row.loc[best_layer])
    signed = auc - 0.5
    if signed < 0:
        label = "correct-more-curved"
    elif signed > 0:
        label = "wrong-more-curved"
    else:
        label = "flat"
    return {
        "curvmean_layer": best_layer,
        "curvmean_auc": auc,
        "curvmean_signed": signed,
        "orientation": label,
    }


def _norm_source(source: str) -> str:
    s = source.lower()
    if "qwen-base" in s or "base" in s:
        return "Base"
    if "qwen-instruct" in s:
        return "Instruct"
    if "qwen-math" in s or "math (cross-task" in s:
        return "Math"
    return source


def _norm_target(target: str) -> str:
    t = target.lower()
    if "r1" in t:
        return "R1-Distill"
    if "instruct" in t:
        return "Instruct"
    return target


def _effect_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    c4 = summary[summary["phase"].eq("C4")].copy()
    for rec in c4.to_dict("records"):
        rows.append(
            {
                "dataset": rec["dataset"],
                "target": rec["target"],
                "source": rec["source"],
                "source_acc": float(rec["source_acc"]),
                "baseline": float(rec["target_baseline"]),
                "steered": float(rec["target_alpha2"]),
                "delta": float(rec["delta_pp"]),
            }
        )

    neg = summary[summary["phase"].eq("C4 cross-target")].copy()
    if not neg.empty:
        base = neg.loc[neg["alpha"].astype(float).eq(0.0), "acc"].iloc[0]
        alpha2 = neg.loc[neg["alpha"].astype(float).eq(2.0), "acc"].iloc[0]
        src = neg["source"].dropna().iloc[0]
        tgt = neg["target"].dropna().iloc[0]
        rows.append(
            {
                "dataset": "MATH-500",
                "target": tgt,
                "source": src,
                "source_acc": np.nan,
                "baseline": float(base),
                "steered": float(alpha2),
                "delta": float(alpha2 - base),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    orientations = pd.DataFrame(
        [{"model": name, **_orientation(path)} for name, path in SIGNATURES.items()]
    )
    orientations.to_csv(OUT / "curvature_orientation.csv", index=False)
    orient_map = orientations.set_index("model")["curvmean_signed"].to_dict()
    label_map = orientations.set_index("model")["orientation"].to_dict()

    effects = _effect_rows(pd.read_csv(PHASE4 / "phase4_summary.csv"))
    rows: list[dict] = []
    for rec in effects.to_dict("records"):
        source_family = _norm_source(str(rec["source"]))
        target_family = _norm_target(str(rec["target"]))
        source_signed = orient_map.get(source_family, np.nan)
        target_signed = orient_map.get(target_family, np.nan)
        source_orientation = label_map.get(source_family, "unknown")
        target_orientation = label_map.get(target_family, "unknown")

        relation = "unknown"
        prediction = "unknown"
        if np.isfinite(source_signed) and np.isfinite(target_signed):
            if np.sign(source_signed) == np.sign(target_signed):
                relation = "same orientation"
                prediction = "overshoot risk"
            else:
                relation = "opposite orientation"
                prediction = "repair candidate"

        observed = "positive" if float(rec["delta"]) > 0 else "negative" if float(rec["delta"]) < 0 else "zero"
        predicted_sign = "positive" if prediction == "repair candidate" else "negative" if prediction == "overshoot risk" else "unknown"
        rows.append(
            {
                **rec,
                "source_family": source_family,
                "target_family": target_family,
                "source_orientation": source_orientation,
                "target_orientation": target_orientation,
                "orientation_relation": relation,
                "compatibility_prediction": prediction,
                "predicted_sign": predicted_sign,
                "observed_sign": observed,
                "sign_match": predicted_sign == observed,
            }
        )

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "compatibility_summary.csv", index=False)

    gsm = comp[(comp["dataset"].eq("GSM8K")) & (comp["target_family"].eq("R1-Distill"))]
    pearson = float("nan")
    if len(gsm) >= 2:
        pearson = float(np.corrcoef(gsm["source_acc"].astype(float), gsm["delta"].astype(float))[0, 1])

    metrics = {
        "n_cells": int(len(comp)),
        "n_sign_matches": int(comp["sign_match"].sum()),
        "sign_match_rate": float(comp["sign_match"].mean()),
        "gsm8k_source_quality_delta_pearson": pearson,
    }
    (OUT / "compatibility_metrics.json").write_text(json.dumps(metrics, indent=2))

    # A compact table suitable for manual inclusion in the paper.
    table = comp[
        [
            "dataset",
            "target_family",
            "source_family",
            "orientation_relation",
            "compatibility_prediction",
            "baseline",
            "steered",
            "delta",
        ]
    ].copy()
    table["delta"] = table["delta"].map(lambda x: f"{x:+.2f}")
    table["baseline"] = table["baseline"].map(lambda x: f"{x:.2f}")
    table["steered"] = table["steered"].map(lambda x: f"{x:.2f}")
    table.to_csv(OUT / "compatibility_table.csv", index=False)

    print(f"wrote {OUT / 'curvature_orientation.csv'}")
    print(f"wrote {OUT / 'compatibility_summary.csv'}")
    print(f"wrote {OUT / 'compatibility_metrics.json'}")
    print(table.to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
