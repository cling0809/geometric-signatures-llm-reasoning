#!/usr/bin/env python3
"""Export a hash-verified, content-free per-problem revision audit table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geoprobe.revision.per_problem_release import export_anonymous_per_problem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument("--report-manifest", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    args = parser.parse_args()

    manifest = export_anonymous_per_problem(
        suite=args.suite,
        suite_root=args.suite_root,
        report_manifest=args.report_manifest,
        out_csv=args.out_csv,
        out_manifest=args.out_manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
