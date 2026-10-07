#!/usr/bin/env python3
"""Compute per-layer trajectory metrics for an extracted run.

Usage:
    python scripts/compute_metrics.py --run ~/AI/runs/<exp-id>

Writes:
    <run>/metrics.parquet    long format: sample_id, metric, layer, value
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geoprobe.analysis import write_run_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()

    out = write_run_metrics(args.run)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
