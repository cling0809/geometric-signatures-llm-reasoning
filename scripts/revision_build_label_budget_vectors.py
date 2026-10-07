#!/usr/bin/env python3
"""Build deterministic target-calibrated vectors for the frozen label-efficiency curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from geoprobe.revision.crosssteer import compute_crosssteer_direction
from geoprobe.revision.label_efficiency import stratified_label_budget_ids
from geoprobe.revision.protocol import RevisionSplit
from geoprobe.revision.vector_registry import require_completed_run, sha256_file


def _parse_budgets(value: str) -> list[int]:
    budgets = [int(item) for item in value.split(",") if item.strip()]
    if not budgets or len(set(budgets)) != len(budgets):
        raise argparse.ArgumentTypeError("--budgets must be a non-empty unique comma list")
    if any(budget <= 1 for budget in budgets):
        raise argparse.ArgumentTypeError("each budget must exceed one")
    return budgets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--budgets", type=_parse_budgets, default=[5, 10, 20, 50, 100])
    parser.add_argument("--salt", default="tacl-11241-target-label-efficiency-v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = require_completed_run(args.target_run)
    split = RevisionSplit.gsm8k_formal()
    labels_path = run / "labels.parquet"
    ids_by_budget = stratified_label_budget_ids(
        pd.read_parquet(labels_path),
        budgets=args.budgets,
        allowed_ids=split.source_train,
        salt=args.salt,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for budget, ids in ids_by_budget.items():
        direction, metadata = compute_crosssteer_direction(run, ids, split=split)
        directory = args.out / f"k{budget:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        vector_path = directory / f"target_calibrated_k{budget:03d}.pt"
        torch.save(direction.cpu(), vector_path)
        manifest = {
            "protocol": "tacl-11241-target-label-efficiency-v1",
            "target_run": str(run.resolve()),
            "target_run_config_sha256": sha256_file(run / "config.yaml"),
            "target_labels_sha256": sha256_file(labels_path),
            "budget": budget,
            "source_ids": ids,
            "selection": "stratified_hash_by_correctness_label_only",
            "salt": args.salt,
            "direction": metadata,
            "vector": vector_path.name,
            "vector_sha256": sha256_file(vector_path),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        entries.append({"budget": budget, "directory": str(directory.resolve()), "manifest": manifest})
    aggregate = {
        "protocol": "tacl-11241-target-label-efficiency-v1",
        "target_run": str(run.resolve()),
        "budgets": sorted(ids_by_budget),
        "salt": args.salt,
        "entries": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
