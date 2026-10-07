#!/usr/bin/env python3
"""Compare compact replay pooling with one stored legacy full trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from geoprobe.extractors import load_trajectory
from geoprobe.models import load_model_and_tokenizer
from geoprobe.revision.pooled_trajectory import replay_generated_state_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-source", choices=["modelscope", "huggingface"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.max_abs_tolerance <= 0:
        raise SystemExit("chunk size and tolerance must be positive")
    legacy = load_trajectory(args.trajectory)
    if legacy.hidden_states.ndim != 3:
        raise SystemExit("parity reference must be a legacy full trajectory")
    model, tokenizer = load_model_and_tokenizer(
        args.model, source=args.model_source, dtype="bfloat16", device=args.device
    )
    inputs = tokenizer(legacy.prompt, return_tensors="pt").to(next(model.parameters()).device)
    pooled, metadata = replay_generated_state_mean(
        model,
        prompt_input_ids=inputs["input_ids"],
        prompt_attention_mask=inputs.get("attention_mask"),
        generated_token_ids=legacy.generated_token_ids,
        chunk_size=args.chunk_size,
    )
    reference = legacy.hidden_states.float().mean(dim=0)
    delta = pooled.cpu() - reference.cpu()
    max_abs = float(delta.abs().max())
    rmse = float(delta.square().mean().sqrt())
    report = {
        "protocol": "tacl-11241-pooled-replay-parity-v1",
        "trajectory": str(args.trajectory.resolve()),
        "trajectory_sha256": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
        "state_count": metadata.state_count,
        "chunk_size": metadata.chunk_size,
        "max_abs_error": max_abs,
        "rmse": rmse,
        "max_abs_tolerance": args.max_abs_tolerance,
        "passed": max_abs <= args.max_abs_tolerance,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("compact replay does not match the legacy trajectory mean")


if __name__ == "__main__":
    main()
