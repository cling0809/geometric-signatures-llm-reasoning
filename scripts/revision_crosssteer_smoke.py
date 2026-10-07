#!/usr/bin/env python3
"""Leak-audited CrossSteer smoke for the TACL major revision.

This is an engineering smoke, not a locked-test result.  It intentionally
uses one frozen source ID set and one disjoint evaluation interval.  The
formal protocol requires all vector construction, layer, schedule and alpha
choices to be frozen on validation before locked testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import GenerationConfig

from geoprobe.datasets import is_correct, load_gsm8k, parse_predicted
from geoprobe.models import load_model_and_tokenizer
from geoprobe.revision import RevisionSplit, paired_binary_summary
from geoprobe.revision.behavior import summarize_text
from geoprobe.revision.crosssteer import compute_crosssteer_direction
from geoprobe.steering import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    LinearDecaySchedule,
    PrefixSchedule,
    SteeringHook,
)

_PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)


def _make_schedule(name: str, decay_parameter: float):
    if name == "constant":
        return ConstantSchedule()
    if name == "prefix":
        return PrefixSchedule(int(decay_parameter))
    if name == "linear":
        return LinearDecaySchedule(int(decay_parameter))
    if name == "exponential":
        return ExponentialDecaySchedule(float(decay_parameter))
    raise ValueError(f"unknown schedule: {name}")


@torch.inference_mode()
def _generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    hook: SteeringHook | None,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Use the model's full EOS set (Qwen2.5-Instruct declares both <|im_end|>
    # and <|endoftext|>; bare generation terminates on the latter).  The
    # tokenizer reports only the first id, which would force 512-token outputs.
    eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        use_cache=True,
        output_scores=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_ids,
    )
    if hook is None:
        output = model.generate(**inputs, generation_config=generation_config)
    else:
        with hook:
            output = model.generate(**inputs, generation_config=generation_config)
    input_len = int(inputs["input_ids"].shape[1])
    return tokenizer.decode(output[0, input_len:], skip_special_tokens=True)


def _parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-source", choices=["modelscope", "huggingface"], default="modelscope")
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--source-count", type=int, default=100)
    parser.add_argument("--eval-start", type=int, default=200)
    parser.add_argument("--eval-count", type=int, default=20)
    parser.add_argument("--layer", type=int, required=True, help="1-based transformer layer index")
    parser.add_argument("--alphas", default="0,1")
    parser.add_argument("--schedule", choices=["constant", "prefix", "linear", "exponential"], default="constant")
    parser.add_argument("--schedule-parameter", type=float, default=64)
    parser.add_argument("--position-mode", choices=["decode_last", "last", "all"], default="decode_last")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.source_start < 0 or args.source_count <= 0 or args.eval_start < 0 or args.eval_count <= 0:
        raise SystemExit("source/eval intervals must be non-negative with positive counts")
    if args.layer < 1:
        raise SystemExit("--layer is 1-based and must be >= 1")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    split = RevisionSplit(
        source_train=tuple(range(args.source_start, args.source_start + args.source_count)),
        validation=tuple(range(10_000, 10_100)),
        locked_test=tuple(range(args.eval_start, args.eval_start + args.eval_count)),
        replication_reserve=tuple(range(20_000, 20_100)),
    )
    source_ids = tuple(range(args.source_start, args.source_start + args.source_count))
    eval_ids = tuple(range(args.eval_start, args.eval_start + args.eval_count))
    if set(source_ids) & set(eval_ids):
        raise SystemExit("source/evaluation IDs overlap")
    direction, direction_meta = compute_crosssteer_direction(
        args.source_run, source_ids, split=split
    )
    if args.layer >= direction.shape[0]:
        raise SystemExit(f"layer {args.layer} outside direction shape {tuple(direction.shape)}")

    args.out.mkdir(parents=True, exist_ok=True)
    schedule = _make_schedule(args.schedule, args.schedule_parameter)
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.target_model,
        source=args.target_source,
        dtype="bfloat16",
        device=args.target_device,
    )
    samples = load_gsm8k(split="test", n=args.eval_start + args.eval_count, source="oss")[args.eval_start:]
    if len(samples) != args.eval_count:
        raise SystemExit(f"expected {args.eval_count} evaluation samples, got {len(samples)}")
    alphas = _parse_float_list(args.alphas)
    rows: list[dict[str, object]] = []
    vector = direction[args.layer]
    metadata = {
        "protocol": "tacl-11241-major-revision-crosssteer-smoke-v1",
        "source_run": str(args.source_run.resolve()),
        "source_ids": list(source_ids),
        "eval_ids": list(eval_ids),
        "target_model": args.target_model,
        "target_source": args.target_source,
        "target_device": args.target_device,
        "layer": args.layer,
        "alphas": alphas,
        "schedule": args.schedule,
        "schedule_parameter": args.schedule_parameter,
        "position_mode": args.position_mode,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": _PROMPT_TEMPLATE,
        "direction": direction_meta,
        "direction_sha256": hashlib.sha256(direction.cpu().numpy().tobytes()).hexdigest(),
    }
    (args.out / "resolved_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for alpha in alphas:
        for sample in samples:
            prompt = _PROMPT_TEMPLATE.format(question=sample.question)
            hook = None
            if alpha != 0.0:
                hook = SteeringHook(
                    model,
                    layer_idx=args.layer,
                    vector=vector,
                    alpha=alpha,
                    schedule=schedule,
                    position_mode=args.position_mode,
                )
            text = _generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, hook=hook)
            pred = parse_predicted(text)
            correct = is_correct(pred, sample.gold_answer)
            rows.append(
                {
                    "alpha": alpha,
                    "sample_id": int(sample.id),
                    "gold": float(sample.gold_answer),
                    "pred": None if pred is None else float(pred),
                    "correct": bool(correct),
                    "text_chars": len(text),
                    "text_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                    **summarize_text(text),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False))

    frame = pd.DataFrame(rows)
    frame.to_parquet(args.out / "per_sample.parquet", index=False)
    summaries = []
    base = frame[frame["alpha"] == 0.0].set_index("sample_id")["correct"].astype(int).to_numpy()
    for alpha in alphas:
        current = frame[frame["alpha"] == alpha].set_index("sample_id")["correct"].astype(int).to_numpy()
        summary = paired_binary_summary(base, current, bootstrap_reps=10_000, seed=0)
        summaries.append({"alpha": alpha, **summary.__dict__})
    pd.DataFrame(summaries).to_csv(args.out / "summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
