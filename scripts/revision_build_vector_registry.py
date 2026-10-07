#!/usr/bin/env python3
"""Create the train-only CrossSteer and target-calibrated vector registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from geoprobe.revision.protocol import RevisionSplit
from geoprobe.revision.vector_registry import build_vector_registry, save_vector_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--source-count", type=int, default=100)
    parser.add_argument("--source-calibration-audit", type=Path)
    parser.add_argument("--target-calibration-audit", type=Path)
    args = parser.parse_args()
    if args.source_start != 0 or args.source_count != 100:
        raise SystemExit(
            "This formal registry is frozen to GSM8K source IDs 0--99; use a separate protocol "
            "and revision document before changing this split."
        )
    split = RevisionSplit.gsm8k_formal()
    vectors, metadata = build_vector_registry(
        args.source_run,
        args.target_run,
        split.source_train,
        split=split,
        source_calibration_audit=args.source_calibration_audit,
        target_calibration_audit=args.target_calibration_audit,
    )
    save_vector_registry(vectors, metadata, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
