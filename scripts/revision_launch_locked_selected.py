#!/usr/bin/env python3
"""Launch exactly one frozen GSM8K locked-test evaluation per eligible method."""

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
    command.extend([
        "--out",
        str(out),
    ])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-source", choices=["modelscope", "huggingface"], default="huggingface")
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    spec = load_locked_run_spec(args.selection, args.method)
    plan = {
        "protocol": "tacl-11241-locked-steering-v1",
        "selection_sha256": sha256_file(args.selection),
        "spec": spec.to_dict(),
        "target_model": args.target_model,
        "target_source": args.target_source,
        "target_device": args.target_device,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "locked_launch_manifest.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise SystemExit("refuse to overwrite a different locked-test launch plan")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    command = _command(args, spec, args.out)
    print(" ".join(command))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
