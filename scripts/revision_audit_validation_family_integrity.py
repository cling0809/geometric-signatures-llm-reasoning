#!/usr/bin/env python3
"""Write a read-only integrity audit for completed frozen validation grids."""

from __future__ import annotations

import argparse

from geoprobe.revision.validation_family_integrity import write_validation_family_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="METHOD=DIR")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_dirs = {}
    for value in args.run:
        if "=" not in value:
            raise SystemExit("--run must be METHOD=DIR")
        method, directory = value.split("=", 1)
        if not method or not directory or method in run_dirs:
            raise SystemExit("--run must contain unique non-empty METHOD=DIR values")
        run_dirs[method] = directory
    print(write_validation_family_audit(run_dirs, args.out))


if __name__ == "__main__":
    main()
