#!/usr/bin/env python3
"""Audit a compact R1 official-context readiness run without inspecting scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from geoprobe.extractors import load_trajectory
from geoprobe.revision.pooled_readiness import audit_pooled_readiness_frame
from geoprobe.revision.vector_registry import require_completed_run, sha256_file


def _parse_range(value: str) -> range:
    try:
        start_text, stop_text = value.split(":", 1)
        start, stop = int(start_text), int(stop_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--expected-id-range must be START:STOP") from exc
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("expected range must satisfy 0 <= START < STOP")
    return range(start, stop)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-id-range", type=_parse_range, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-budget-hit-rate", type=float, default=0.25)
    parser.add_argument("--max-severe-repetition-rate", type=float, default=0.05)
    parser.add_argument("--severe-repetition-fraction", type=float, default=0.95)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run = require_completed_run(args.run)
    labels_path = run / "labels.parquet"
    labels = pd.read_parquet(labels_path)
    texts: dict[int, str] = {}
    for sample_id in labels["sample_id"].astype(int).tolist():
        trajectory = load_trajectory(run / "trajectories" / f"sample_{sample_id:04d}_idx_0.pt")
        texts[sample_id] = trajectory.generated_text
    audit = audit_pooled_readiness_frame(
        labels,
        generated_text_by_id=texts,
        expected_ids=args.expected_id_range,
        max_new_tokens=args.max_new_tokens,
        max_budget_hit_rate=args.max_budget_hit_rate,
        max_severe_repetition_rate=args.max_severe_repetition_rate,
        severe_repetition_fraction=args.severe_repetition_fraction,
    )
    payload = {
        "protocol": "tacl-11241-official-context-pooled-readiness-v1",
        "run": str(run.resolve()),
        "config_sha256": sha256_file(run / "config.yaml"),
        "labels_sha256": sha256_file(labels_path),
        "expected_ids": list(args.expected_id_range),
        "max_new_tokens": args.max_new_tokens,
        "max_budget_hit_rate": args.max_budget_hit_rate,
        "max_severe_repetition_rate": args.max_severe_repetition_rate,
        "severe_repetition_fraction": args.severe_repetition_fraction,
        "audit": audit.to_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not audit.eligible:
        raise SystemExit("pooled official-context readiness audit failed")


if __name__ == "__main__":
    main()
