#!/usr/bin/env python3
"""Behavior-column comparison: steered vs baseline, from a steering-grid run.

Answers the reviewer B5 / editor E3 question directly: does a steering method
improve answers by injecting correctness, or merely by changing output length,
repetition, formatting, or stop behavior?  It reads only the per-problem
behavior fields (never re-scores correctness; correctness column is used only
to split repairs/breaks).

Usage is intended for a completed locked run directory (per_sample.jsonl with
a baseline and one method per problem), but it also works on a partial
validation grid for an engineering preview.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd

BEHAVIOR_COLUMNS = (
    "text_tokens",
    "text_tokens_whitespace",
    "n_generated_tokens",
    "truncated",
    "repeated_4gram_fraction",
    "distinct_4gram_ratio",
    "answer_marker_relative_position",
)


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    required = {"method", "sample_id", "correct", "text_tokens"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"per_sample missing required columns: {sorted(missing)}")
    return frame



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-truncation-rate", type=float, default=0.25)
    args = parser.parse_args()

    frame = _load(args.per_sample)
    baseline = frame[frame["method"] == "baseline"].set_index("sample_id").sort_index()
    if baseline.empty:
        raise SystemExit("per_sample has no baseline rows")
    if baseline.index.duplicated().any():
        raise SystemExit("per_sample baseline rows are not unique per sample_id")

    baseline_truncation = float(
        baseline["truncated"].astype(float).mean() if "truncated" in baseline.columns else float("nan")
    )

    summaries: list[dict[str, object]] = []
    for method, cell in frame[frame["method"] != "baseline"].groupby("method", sort=True):
        cell = cell.set_index("sample_id").sort_index()
        if not cell.index.equals(baseline.index):
            raise SystemExit(f"{method} does not cover the same problems as baseline")
        if cell.index.duplicated().any():
            raise SystemExit(f"{method} has duplicate per-sample rows")

        deltas: dict[str, list[float]] = {}
        for column in BEHAVIOR_COLUMNS:
            if column in baseline and column in cell:
                b = baseline[column].astype(float)
                c = cell[column].astype(float)
                deltas[column] = (c - b).tolist()

        repairs_mask = (~baseline["correct"].astype(bool)) & cell["correct"].astype(bool)
        breaks_mask = baseline["correct"].astype(bool) & (~cell["correct"].astype(bool))
        both_correct = baseline["correct"].astype(bool) & cell["correct"].astype(bool)

        method_truncation = (
            float(cell["truncated"].astype(float).mean()) if "truncated" in cell.columns else float("nan")
        )

        def mean(values: list[float]) -> float | None:
            return statistics.mean(values) if values else None

        row: dict[str, object] = {"method": method}
        for column in BEHAVIOR_COLUMNS:
            d = deltas.get(column, [])
            row[f"delta_{column}"] = mean(d)
        row["n"] = int(len(cell))
        row["repairs"] = int(repairs_mask.sum())
        row["breaks"] = int(breaks_mask.sum())
        row["both_correct"] = int(both_correct.sum())
        row["baseline_truncation_rate"] = round(baseline_truncation, 3)
        row["method_truncation_rate"] = round(method_truncation, 3)
        # Length signature of repairs vs breaks (dlen on discordant pairs).
        if "text_tokens" in deltas:
            dlen = pd.Series(deltas["text_tokens"], index=cell.index)
            row["repair_delta_text_tokens"] = round(float(dlen[repairs_mask].mean()), 1) if repairs_mask.any() else None
            row["break_delta_text_tokens"] = round(float(dlen[breaks_mask].mean()), 1) if breaks_mask.any() else None
        summaries.append(row)

    out = pd.DataFrame(summaries).sort_values("delta_text_tokens", ascending=False)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print()
    print(f"baseline truncation rate: {baseline_truncation:.3f} (decoding envelope valid: {baseline_truncation <= args.max_truncation_rate})")


if __name__ == "__main__":
    main()
