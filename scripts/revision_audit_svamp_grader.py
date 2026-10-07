#!/usr/bin/env python3
"""Audit the frozen official SVAMP numeric evaluator before OOD generation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from geoprobe.datasets.svamp import SVAMP_EXPECTED_COUNT, SVAMP_SHA256, SVAMP_URL, load_svamp

_PROTOCOL = "tacl-11241-svamp-numeric-grader-audit-v1"


def audit_svamp() -> dict[str, object]:
    samples = load_svamp()
    expected_ids = list(range(SVAMP_EXPECTED_COUNT))
    if [sample.id for sample in samples] != expected_ids:
        raise ValueError("SVAMP loader does not preserve the frozen official row order")
    if any(not math.isfinite(sample.gold_answer) for sample in samples):
        raise ValueError("SVAMP contains a non-finite gold answer")
    fixture_passed = {
        "integer_exact": abs(51.0 - samples[0].gold_answer) < 1e-4,
        "numeric_negative": abs(50.0 - samples[0].gold_answer) >= 1e-4,
        "fraction_tolerance": abs(0.5 - 0.50001) <= 1e-4,
    }
    return {
        "protocol": _PROTOCOL,
        "dataset": "SVAMP",
        "dataset_url": SVAMP_URL,
        "dataset_sha256": SVAMP_SHA256,
        "n_items": len(samples),
        "evaluator": {"name": "svamp_numeric_boxed_last_number", "version": "revision-v1"},
        "fixtures": fixture_passed,
        "passed": all(fixture_passed.values()) and len(samples) == SVAMP_EXPECTED_COUNT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_svamp()
    if not payload["passed"]:
        raise SystemExit("SVAMP numeric grader audit failed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out.exists() and args.out.read_text() != encoded:
        raise SystemExit("refuse to overwrite a different SVAMP grader audit")
    args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
