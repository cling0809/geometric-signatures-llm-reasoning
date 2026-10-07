#!/usr/bin/env python3
"""Merge metric-only extraction shards into a standard run directory.

Each shard is produced by ``extract_metrics_streaming.py`` and contains
labels/metrics JSONL files keyed by (sample_id, sample_idx). This script
concatenates shards, drops duplicate keys by keeping the last occurrence, writes
both JSONL and parquet outputs, and emits a DONE summary compatible with the
downstream GeoVote analysis scripts.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _merge_frames(paths: list[Path], filename: str, dedupe: list[str]) -> pd.DataFrame:
    frames = [_read_jsonl(p / filename) for p in paths]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(dedupe, keep="last")
    return out.sort_values(dedupe).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    if (args.runs[0] / "config.yaml").exists():
        shutil.copyfile(args.runs[0] / "config.yaml", args.out / "config.yaml")

    labels = _merge_frames(args.runs, "labels.jsonl", ["sample_id", "sample_idx"])
    metrics = _merge_frames(args.runs, "metrics.jsonl", ["sample_id", "sample_idx", "metric", "layer"])
    generations = _merge_frames(args.runs, "generations.jsonl", ["sample_id", "sample_idx"])

    if labels.empty or metrics.empty:
        raise SystemExit("no labels/metrics found to merge")

    _write_jsonl(labels, args.out / "labels.jsonl")
    _write_jsonl(metrics, args.out / "metrics.jsonl")
    labels.to_parquet(args.out / "labels.parquet", index=False)
    metrics.to_parquet(args.out / "metrics.parquet", index=False)

    if not generations.empty:
        _write_jsonl(generations, args.out / "generations.jsonl")

    completed = labels[["sample_id", "sample_idx"]].drop_duplicates()
    _write_jsonl(completed, args.out / "completed.jsonl")

    pass1 = labels[labels["sample_idx"] == 0]["correct"].astype(bool).mean()
    n_per_question = labels.groupby("sample_id")["sample_idx"].nunique()
    done_lines = [
        f"pass1={pass1:.4f}",
        f"n_labels={len(labels)}",
        f"n_metrics={len(metrics)}",
        f"n_questions={labels['sample_id'].nunique()}",
        f"min_samples_per_question={int(n_per_question.min())}",
        f"max_samples_per_question={int(n_per_question.max())}",
    ]
    if int(n_per_question.min()) > 1:
        oracle = labels.groupby("sample_id")["correct"].max().astype(bool).mean()
        done_lines.append(f"oracle={oracle:.4f}")
    (args.out / "DONE").write_text("\n".join(done_lines) + "\n")

    print(f"merged labels={len(labels)} metrics={len(metrics)} questions={labels['sample_id'].nunique()}")
    print(args.out / "DONE")


if __name__ == "__main__":
    main()
