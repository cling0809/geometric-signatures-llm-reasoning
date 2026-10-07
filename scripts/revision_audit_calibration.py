#!/usr/bin/env python3
"""Audit train-only calibration labels before building revision steering vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from geoprobe.revision.calibration_provenance import audit_calibration_generation_provenance
from geoprobe.revision.capability_audit import audit_calibration_frame
from geoprobe.revision.vector_registry import require_completed_run, sha256_file


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must have NAME=/path/to/run")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("--run must have non-empty name and path")
    return name, Path(raw_path)


def _parse_id_range(value: str) -> range:
    try:
        start_text, stop_text = value.split(":", 1)
        start, stop = int(start_text), int(stop_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--expected-id-range must be START:STOP") from exc
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("expected ID range must satisfy 0 <= START < STOP")
    return range(start, stop)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--expected-id-range", type=_parse_id_range, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument("--max-budget-hit-rate", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--require-generation-provenance",
        action="store_true",
        help="Require a pre-generation multi-EOS manifest and stop telemetry for every run.",
    )
    args = parser.parse_args()
    if len(args.run) != len({name for name, _ in args.run}):
        raise SystemExit("run names must be unique")

    expected_ids = list(args.expected_id_range)
    audits: list[dict[str, object]] = []
    for name, raw_path in args.run:
        run = require_completed_run(raw_path)
        labels = run / "labels.parquet"
        audit = audit_calibration_frame(
            pd.read_parquet(labels),
            name=name,
            expected_ids=expected_ids,
            max_new_tokens=args.max_new_tokens,
            min_class_count=args.min_class_count,
            max_budget_hit_rate=args.max_budget_hit_rate,
        )
        entry = {
            **audit.to_dict(),
            "run": str(run.resolve()),
            "labels_sha256": sha256_file(labels),
            "config_sha256": sha256_file(run / "config.yaml"),
        }
        if args.require_generation_provenance:
            entry["generation_provenance"] = audit_calibration_generation_provenance(
                run,
                max_new_tokens=args.max_new_tokens,
            )
        audits.append(entry)
    payload = {
        "protocol": "tacl-11241-calibration-capability-audit-v1",
        "expected_ids": expected_ids,
        "max_new_tokens": args.max_new_tokens,
        "min_class_count": args.min_class_count,
        "max_budget_hit_rate": args.max_budget_hit_rate,
        "runs": audits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    failures = [audit for audit in audits if not bool(audit["eligible_for_direction"])]
    if failures:
        raise SystemExit("calibration capability audit failed; do not build steering vectors")


if __name__ == "__main__":
    main()
