#!/usr/bin/env python3
"""C4 control: compare our steering vector at alpha=2 vs K random vectors at same L2 norm.

If our (correct - incorrect) direction is special, our accuracy should
be higher than the distribution of accuracies from random vectors at the
same magnitude.

Usage:
    python scripts/c4_random_control.py \
        --steering-vector ~/AI/runs/.../steering_vector.pt \
        --target-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --target-source modelscope \
        --layer 14 --alpha 2.0 --n-questions 100 --k-random 5 \
        --out ~/AI/runs/2026-05-17_c4-random-control/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GenerationConfig

from geoprobe.datasets import is_correct, load_gsm8k, parse_predicted
from geoprobe.models import load_model_and_tokenizer
from geoprobe.steering import SteeringHook
from geoprobe.steering.vectors import load_steering_vector

_PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)


@torch.inference_mode()
def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    device = next(model.parameters()).device
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inp,
        generation_config=GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        ),
    )
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steering-vector", required=True, type=Path,
                    help="Path to the actual steering_vector.pt to compare against.")
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--target-source", default="modelscope")
    ap.add_argument("--target-device", default="cuda:1")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--n-questions", type=int, default=100)
    ap.add_argument("--k-random", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    vec, _ = load_steering_vector(args.steering_vector)
    target_vec = vec[args.layer]
    target_norm = float(target_vec.norm())
    H = target_vec.shape[0]
    print(f"target_vec layer {args.layer}, L2 norm = {target_norm:.3f}, dim = {H}")

    print(f"loading target model {args.target_model}")
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.target_model, source=args.target_source,
        dtype="bfloat16", device=args.target_device,
    )

    samples = load_gsm8k(split="test", n=args.n_questions, source="oss")
    print(f"{len(samples)} questions")

    rows = []
    summary = []

    # Generate K random unit vectors, scale to same L2 norm
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    for k in range(args.k_random):
        r = torch.randn(H, generator=g, dtype=torch.float32)
        r = r / r.norm() * target_norm
        n_correct = 0
        for s in tqdm(samples, desc=f"random#{k}"):
            prompt = _PROMPT_TEMPLATE.format(question=s.question)
            with SteeringHook(model, layer_idx=args.layer, vector=r, alpha=args.alpha):
                text = _generate(model, tokenizer, prompt)
            pred = parse_predicted(text)
            ok = is_correct(pred, s.gold_answer)
            n_correct += int(ok)
            rows.append({"random_seed_k": k, "sample_id": s.id, "gold": s.gold_answer,
                         "pred": pred, "correct": ok})
        acc = n_correct / len(samples)
        summary.append({"vector": f"random_{k}", "accuracy": acc, "norm": target_norm})
        print(f"  random#{k}: acc={acc:.3f}")

    sdf = pd.DataFrame(summary)
    pd.DataFrame(rows).to_parquet(args.out / "c4_random_per_sample.parquet", index=False)

    # add reference rows: 0 and 2.0 we already have from earlier sweep, but recomputing
    # is cheap insurance against any device-state confound.
    # actually keep them external for clean separation; consumer compares manually.
    sdf.to_csv(args.out / "c4_random_summary.csv", index=False)
    meta = {
        "target_vec_layer": args.layer,
        "target_vec_norm": target_norm,
        "alpha": args.alpha,
        "k_random": args.k_random,
        "n_questions": args.n_questions,
    }
    (args.out / "control_metadata.json").write_text(json.dumps(meta, indent=2))

    print("\n=== random-vector accuracies at alpha=%s ===" % args.alpha)
    print(sdf.to_string(index=False))
    print(f"mean={sdf['accuracy'].mean():.3f}  std={sdf['accuracy'].std():.3f}  "
          f"min={sdf['accuracy'].min():.3f}  max={sdf['accuracy'].max():.3f}")


if __name__ == "__main__":
    main()
