#!/usr/bin/env python3
"""Audit a completed source-only sampled CAA/ActAdd contrast pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from geoprobe.revision.contrast_pool_audit import audit_contrast_pool_frame
from geoprobe.revision.vector_registry import require_completed_run, sha256_file


def _parse_id_range(value: str) -> range:
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
    parser.add_argument("--expected-id-range", type=_parse_id_range, required=True)
    parser.add_argument("--n-samples-per-problem", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument("--min-same-question-pairs", type=int, default=2)
    parser.add_argument("--max-budget-hit-rate", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run = require_completed_run(args.run)
    labels = run / "labels.parquet"
    audit = audit_contrast_pool_frame(
        pd.read_parquet(labels),
        expected_ids=args.expected_id_range,
        n_samples_per_problem=args.n_samples_per_problem,
        max_new_tokens=args.max_new_tokens,
        min_class_count=args.min_class_count,
        min_same_question_pairs=args.min_same_question_pairs,
        max_budget_hit_rate=args.max_budget_hit_rate,
    )
    payload = {
        "protocol": "tacl-11241-contrast-pool-audit-v1",
        **audit.to_dict(),
        "run": str(run.resolve()),
        "labels_sha256": sha256_file(labels),
        "config_sha256": sha256_file(run / "config.yaml"),
        "expected_ids": list(args.expected_id_range),
        "max_new_tokens": args.max_new_tokens,
        "max_budget_hit_rate": args.max_budget_hit_rate,
        "min_class_count": args.min_class_count,
        "min_same_question_pairs": args.min_same_question_pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not audit.eligible_for_prompt_contrast:
        raise SystemExit("contrast-pool audit failed; do not use these prompt-control vectors")


if __name__ == "__main__":
    main()
