#!/usr/bin/env python3
"""Render all frozen Qwen/R1 validation cells after selection is immutable."""

from __future__ import annotations

import argparse
from pathlib import Path

from geoprobe.revision.validation_report import write_validation_family_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report, manifest = write_validation_family_report(args.selection, args.out)
    print(report.to_string(index=False))
    print(f"wrote {manifest['n_cells']} cells for {manifest['n_methods']} methods")


if __name__ == "__main__":
    main()
