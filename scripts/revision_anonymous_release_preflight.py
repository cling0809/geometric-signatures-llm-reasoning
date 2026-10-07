#!/usr/bin/env python3
"""Check whether the TACL-11241 revision tree is ready for anonymous release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geoprobe.revision.release import anonymous_release_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root to inspect",
    )
    parser.add_argument(
        "--evidence",
        action="append",
        type=Path,
        default=[],
        help="formal report/manifests to inventory; repeat for every required artifact",
    )
    parser.add_argument(
        "--require-evidence",
        action="store_true",
        help="fail unless at least one supplied evidence artifact exists",
    )
    parser.add_argument("--out", type=Path, required=True, help="JSON preflight manifest")
    args = parser.parse_args()

    report = anonymous_release_preflight(
        args.root, evidence=args.evidence, require_evidence=args.require_evidence
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["source_tree_ready"]:
        raise SystemExit("anonymous release source tree is not ready")
    if args.require_evidence and not report["ready_for_final_anonymous_release"]:
        raise SystemExit("final anonymous release requires supplied formal evidence")


if __name__ == "__main__":
    main()
