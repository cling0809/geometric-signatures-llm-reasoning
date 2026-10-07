#!/usr/bin/env python3
"""Launch the frozen target-label-efficiency curve on the GSM8K locked set.

The curve is deliberately not a second validation search.  Every nonzero point
uses the layer, alpha, decoder, schedule and locked IDs selected for the full
``target_calibrated`` direction.  The zero-label point uses the external source
vector under exactly those target-selected intervention settings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from geoprobe.revision.locked_run import (
    frozen_generation_cli_args,
    load_locked_run_spec,
    sha256_file,
)

_PROTOCOL = "tacl-11241-target-label-efficiency-locked-v1"


def _parse_budgets(value: str) -> list[int]:
    budgets = [int(item) for item in value.split(",") if item.strip()]
    if not budgets or len(set(budgets)) != len(budgets):
        raise argparse.ArgumentTypeError("--budgets must be a non-empty unique comma list")
    if any(budget <= 1 for budget in budgets):
        raise argparse.ArgumentTypeError("all nonzero label budgets must exceed one")
    return sorted(budgets)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _vector_entry(vector_path: Path) -> dict[str, object]:
    """Load audited sibling provenance for a vector artifact."""
    if not vector_path.exists():
        raise FileNotFoundError(vector_path)
    manifest_path = vector_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{vector_path} lacks sibling manifest.json")
    manifest = _read_json(manifest_path)
    ids = manifest.get("source_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{manifest_path} lacks non-empty source_ids")
    return {
        "vector_path": str(vector_path.resolve()),
        "vector_sha256": sha256_file(vector_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_ids": [int(value) for value in ids],
        "manifest": manifest,
    }


def _label_vector_entry(
    root: Path, budget: int, *, target_registry: dict[str, object]
) -> dict[str, object]:
    directory = root / f"k{budget:03d}"
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("protocol") != "tacl-11241-target-label-efficiency-v1":
        raise ValueError(f"{manifest_path} has unexpected protocol")
    if int(manifest.get("budget", -1)) != budget:
        raise ValueError(f"{manifest_path} does not describe budget {budget}")
    if manifest.get("target_run_config_sha256") != target_registry.get("target_run_config_sha256"):
        raise ValueError(
            "label-budget vector target configuration differs from the selected target vector"
        )
    if manifest.get("target_labels_sha256") != target_registry.get("target_labels_sha256"):
        raise ValueError("label-budget vector target labels differ from the selected target vector")
    vector_name = manifest.get("vector")
    if not isinstance(vector_name, str) or not vector_name:
        raise ValueError(f"{manifest_path} has no vector filename")
    vector = directory / vector_name
    entry = _vector_entry(vector)
    if entry["vector_sha256"] != manifest.get("vector_sha256"):
        raise ValueError(f"{manifest_path} vector SHA-256 mismatch")
    if entry["source_ids"] != [int(value) for value in manifest.get("source_ids", [])]:
        raise ValueError(f"{manifest_path} source IDs disagree with its vector provenance")
    return {**entry, "budget": budget, "label_manifest": manifest}


def _command(
    args: argparse.Namespace, spec, *, method: str, vector_path: str, out: Path
) -> list[str]:
    command = [
        sys.executable,
        "scripts/revision_steering_validation_grid.py",
        "--vector",
        f"{method}={vector_path}",
        "--target-model",
        args.target_model,
        "--target-source",
        args.target_source,
        "--target-device",
        args.target_device,
        "--eval-start",
        str(spec.locked_eval_start),
        "--eval-count",
        str(spec.locked_eval_count),
        "--layers",
        str(spec.layer),
        "--alphas",
        str(spec.alpha),
        "--schedule",
        spec.schedule,
        "--injection-mode",
        spec.injection_mode,
        "--max-new-tokens",
        str(spec.max_new_tokens),
    ]
    command.extend(frozen_generation_cli_args(spec))
    command.extend(["--out", str(out)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--target-method", default="target_calibrated")
    parser.add_argument("--source-vector", type=Path, required=True)
    parser.add_argument("--label-vector-root", type=Path, required=True)
    parser.add_argument("--budgets", type=_parse_budgets, default=[5, 10, 20, 50, 100])
    parser.add_argument("--target-model", required=True)
    parser.add_argument(
        "--target-source", choices=["modelscope", "huggingface"], default="huggingface"
    )
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    spec = load_locked_run_spec(args.selection, args.target_method)
    selected_entry = _vector_entry(Path(spec.vector_path))
    target_registry = selected_entry["manifest"]
    if target_registry.get("protocol") != "tacl-11241-vector-registry-v1":
        raise ValueError("selected target vector must come from the audited vector registry")

    source_entry = _vector_entry(args.source_vector)
    label_entries = [
        _label_vector_entry(args.label_vector_root, budget, target_registry=target_registry)
        for budget in args.budgets
    ]

    methods: list[dict[str, object]] = [
        {
            "label_budget": 0,
            "method": "target_labels_k000_source_transfer",
            "vector": source_entry,
            "out": str((args.out / "k000_source_transfer").resolve()),
        }
    ]
    methods.extend(
        {
            "label_budget": entry["budget"],
            "method": f"target_labels_k{int(entry['budget']):03d}",
            "vector": entry,
            "out": str((args.out / f"k{int(entry['budget']):03d}").resolve()),
        }
        for entry in label_entries
    )

    plan = {
        "protocol": _PROTOCOL,
        "selection_sha256": sha256_file(args.selection),
        "target_method": args.target_method,
        "frozen_target_spec": spec.to_dict(),
        "target_model": args.target_model,
        "target_source": args.target_source,
        "target_device": args.target_device,
        "rule": "all curve points use target-calibrated frozen layer/alpha/decode/locked IDs; k000 is source transfer",
        "methods": methods,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "label_efficiency_locked_launch_manifest.json"
    if plan_path.exists() and _read_json(plan_path) != plan:
        raise SystemExit("refuse to overwrite a different label-efficiency locked launch plan")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    for entry in methods:
        command = _command(
            args,
            spec,
            method=str(entry["method"]),
            vector_path=str(entry["vector"]["vector_path"]),
            out=Path(str(entry["out"])),
        )
        print(" ".join(command))
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
