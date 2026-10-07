#!/usr/bin/env python3
"""Launch predeclared long-context stability runs from a frozen 512-token selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from geoprobe.revision.locked_run import (
    frozen_generation_cli_args,
    load_locked_run_spec,
    sha256_file,
)


@dataclass(frozen=True)
class LongContextPolicy:
    name: str
    schedule: str
    schedule_parameter: float
    injection_mode: str


_POLICIES = (
    LongContextPolicy("constant", "constant", 64.0, "absolute"),
    LongContextPolicy("prefix-256", "prefix", 256.0, "absolute"),
    LongContextPolicy("exponential-1024", "exponential", 1024.0, "absolute"),
    LongContextPolicy("relative-hidden-rms", "constant", 64.0, "relative_hidden_rms"),
)


def _command(args: argparse.Namespace, spec, policy: LongContextPolicy, out: Path) -> list[str]:
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
        policy.schedule,
        "--schedule-parameter",
        str(policy.schedule_parameter),
        "--injection-mode",
        policy.injection_mode,
        "--max-new-tokens",
        str(args.budget),
    ]
    command.extend(frozen_generation_cli_args(spec))
    command.extend(["--out", str(out)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-source", choices=["modelscope", "huggingface"], default="huggingface")
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--budget", type=int, choices=[4096, 32768], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    spec = load_locked_run_spec(args.selection, args.method)
    plan = {
        "protocol": "tacl-11241-long-context-v1",
        "long_context_protocol": "revision/LONG_CONTEXT_PROTOCOL.md",
        "selection_sha256": sha256_file(args.selection),
        "spec": spec.to_dict(),
        "target_model": args.target_model,
        "target_source": args.target_source,
        "target_device": args.target_device,
        "budget": args.budget,
        "policies": [policy.__dict__ for policy in _POLICIES],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "long_context_launch_manifest.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise SystemExit("refuse to overwrite a different long-context launch plan")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    commands = [
        _command(args, spec, policy, args.out / policy.name)
        for policy in _POLICIES
    ]
    for command in commands:
        print(" ".join(command))
    if args.execute:
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
