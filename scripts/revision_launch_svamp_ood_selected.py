#!/usr/bin/env python3
"""Launch exactly one frozen all-SVAMP OOD evaluation from a locked selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from geoprobe.datasets.svamp import SVAMP_EXPECTED_COUNT, SVAMP_SHA256
from geoprobe.revision.locked_run import (
    frozen_generation_cli_args,
    load_locked_run_spec,
    sha256_file,
)

_AUDIT_PROTOCOL = "tacl-11241-svamp-numeric-grader-audit-v1"
_PROTOCOL = "tacl-11241-frozen-svamp-ood-launch-v1"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _require_grader_audit(path: Path) -> dict[str, object]:
    audit = _read_json(path)
    evaluator = audit.get("evaluator")
    if (
        audit.get("protocol") != _AUDIT_PROTOCOL
        or not bool(audit.get("passed"))
        or audit.get("dataset_sha256") != SVAMP_SHA256
        or int(audit.get("n_items", 0)) != SVAMP_EXPECTED_COUNT
        or not isinstance(evaluator, dict)
        or evaluator.get("name") != "svamp_numeric_boxed_last_number"
    ):
        raise ValueError("SVAMP numeric grader audit has not passed the pinned 1,000-item check")
    return audit


def _command(args: argparse.Namespace, spec, out: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/revision_steering_validation_grid.py",
        "--vector",
        f"{spec.method}={spec.vector_path}",
        "--target-model",
        args.target_model,
        "--target-source",
        args.target_source,
        "--target-device",
        args.target_device,
        "--dataset",
        "svamp",
        "--allow-cross-dataset-eval",
        "--eval-start",
        "0",
        "--eval-count",
        str(SVAMP_EXPECTED_COUNT),
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
    parser.add_argument("--method", required=True)
    parser.add_argument("--grader-audit", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument(
        "--target-source", choices=["modelscope", "huggingface"], default="huggingface"
    )
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    spec = load_locked_run_spec(args.selection, args.method)
    audit = _require_grader_audit(args.grader_audit)
    plan = {
        "protocol": _PROTOCOL,
        "selection_sha256": sha256_file(args.selection),
        "grader_audit_sha256": sha256_file(args.grader_audit),
        "grader_audit": {
            "protocol": audit["protocol"],
            "dataset_sha256": audit["dataset_sha256"],
            "n_items": audit["n_items"],
            "evaluator": audit["evaluator"],
        },
        "dataset": "SVAMP",
        "frozen_target_spec": spec.to_dict(),
        "target_model": args.target_model,
        "target_source": args.target_source,
        "target_device": args.target_device,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "ood_launch_manifest.json"
    if manifest.exists() and _read_json(manifest) != plan:
        raise SystemExit("refuse to overwrite a different SVAMP OOD launch plan")
    manifest.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    command = _command(args, spec, args.out)
    print(" ".join(command))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
