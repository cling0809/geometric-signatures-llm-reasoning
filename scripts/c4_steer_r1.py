#!/usr/bin/env python3
"""C4: steering intervention on R1-Distill using Qwen-Base/Instruct geometric direction.

Pipeline:
  1. Load source-run trajectories (e.g. Qwen2.5-1.5B-Instruct) + labels.
  2. Compute steering vector v_layer = mean(correct) - mean(incorrect) per layer.
  3. Load target model (e.g. DeepSeek-R1-Distill-Qwen-1.5B).
  4. For each alpha in --alphas, generate GSM8K[:n] with a forward hook adding
     alpha * v[layer] to a chosen layer's output. Greedy decoding.
  5. Save accuracy per alpha; also save the steering vector for re-use.

Usage:
    python scripts/c4_steer_r1.py \
        --source-run ~/AI/runs/2026-05-17_extract-qwen-1.5b-nomath-100 \
        --target-model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --target-source modelscope \
        --layer 14 \
        --alphas 0 0.5 1.0 1.5 2.0 -0.5 -1.0 \
        --n-questions 100 \
        --out ~/AI/runs/2026-05-17_c4-steer-r1-distill/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GenerationConfig

from geoprobe.datasets import (
    is_correct,
    is_correct_math500,
    load_gsm8k,
    load_math500,
    parse_predicted,
    parse_predicted_math500,
)
from geoprobe.models import load_model_and_tokenizer
from geoprobe.steering import SteeringHook, compute_steering_vector
from geoprobe.steering.vectors import load_steering_vector, save_steering_vector

_PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)


def _load_dataset_and_grader(name: str, n: int, start: int = 0):
    """Returns (samples, text_grader, parser, pred_grader).
    text_grader: takes raw model output text + gold, returns bool correct.
    parser:      takes raw text, returns parsed prediction (float for gsm8k, str for math500).
    pred_grader: takes already-parsed pred + gold, returns bool correct.
    """
    name = name.lower()
    if name == "gsm8k":
        samples = load_gsm8k(split="test", n=start + n, source="oss")[start:]
        return (
            samples,
            (lambda txt, g: is_correct(parse_predicted(txt), g)),
            (lambda txt: parse_predicted(txt)),
            (lambda pred, g: is_correct(pred, g)),
        )
    elif name in ("math500", "math-500"):
        samples = load_math500(split="test", n=start + n)[start:]
        return (
            samples,
            (lambda txt, g: is_correct_math500(parse_predicted_math500(txt), g)),
            (lambda txt: parse_predicted_math500(txt)),
            (lambda pred, g: is_correct_math500(pred, g)),
        )
    else:
        raise ValueError(f"unknown dataset {name!r}")


@torch.inference_mode()
def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 512,
              do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0) -> str:
    device = next(model.parameters()).device
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **inp,
        generation_config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        ),
    )
    gen_ids = gen[0][inp["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _majority(preds):
    """Majority vote ignoring None. Returns most common parsed pred (first wins on tie)."""
    from collections import Counter
    nonnull = [p for p in preds if p is not None]
    if not nonnull:
        return None
    return Counter(nonnull).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-run", type=Path,
                    help="Run dir from which to compute steering vector (mean correct - mean incorrect). "
                         "Requires trajectories/ dir to be intact.")
    src.add_argument("--steering-vector", type=Path,
                    help="Path to a precomputed steering_vector.pt (saved by an earlier c4_steer_r1.py run).")
    ap.add_argument("--target-model", required=True,
                    help="HF / ModelScope id of the model to steer.")
    ap.add_argument("--target-source", default="modelscope")
    ap.add_argument("--target-device", default="cuda:1")
    ap.add_argument("--layer", type=int, default=14,
                    help="Layer index (1-based; 0 = embedding) where to inject the steering.")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0])
    ap.add_argument("--n-questions", type=int, default=100)
    ap.add_argument("--eval-start", type=int, default=0,
                    help="Start offset in the benchmark split. Use this for clean held-out "
                         "steering baselines when the source vector is calibrated on earlier ids.")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="Generation budget per sample. R1-style reasoning models often need >=4096.")
    ap.add_argument("--cons-n", type=int, default=1,
                    help="cons@N: if >1, sample N times with temperature and majority-vote. Default 1 = greedy single-shot.")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="Sampling temperature (only used when cons-n > 1).")
    ap.add_argument("--top-p", type=float, default=0.95,
                    help="top-p (only used when cons-n > 1).")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. compute or load steering vector
    if args.source_run is not None:
        print(f"=> computing steering vector from {args.source_run.name}")
        v = compute_steering_vector(args.source_run)
        source_label = str(args.source_run)
    else:
        print(f"=> loading precomputed steering vector from {args.steering_vector}")
        v, _meta = load_steering_vector(args.steering_vector)
        source_label = str(args.steering_vector)

    print(f"   vector shape: {tuple(v.shape)}  (L+1={v.shape[0]}, H={v.shape[1]})")
    v_l2 = v.norm(dim=-1).tolist()
    print(f"   per-layer L2 norms: min={min(v_l2):.3f} max={max(v_l2):.3f} L{args.layer}={v_l2[args.layer]:.3f}")

    metadata = {
        "source": source_label,
        "computed_at": time.time(),
        "layer_norms": v_l2,
        "eval_start": args.eval_start,
    }
    save_steering_vector(v, args.out / "steering_vector.pt", metadata=metadata)
    (args.out / "steering_metadata.json").write_text(json.dumps(metadata, indent=2))

    # 2. load target model
    print(f"=> loading target model {args.target_model}")
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.target_model,
        source=args.target_source,
        dtype="bfloat16",
        device=args.target_device,
    )

    # 3. load dataset
    samples, grader, parser, pred_grader = _load_dataset_and_grader(
        args.dataset, args.n_questions, start=args.eval_start
    )
    print(f"=> {len(samples)} questions (dataset={args.dataset}, eval_start={args.eval_start})")

    # 4. sweep alphas — with resume (incremental JSONL append per sample)
    layer_vec = v[args.layer]
    incremental_path = args.out / "c4_per_sample.jsonl"

    # Resume: load any previously-completed (alpha, sample_id) pairs
    done: set[tuple[float, int]] = set()
    existing_rows: list[dict] = []
    if incremental_path.exists():
        with incremental_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                done.add((float(d["alpha"]), int(d["sample_id"])))
                existing_rows.append(d)
        print(f"=> resume: {len(done)} (alpha, sample) pairs already done")

    rows = list(existing_rows)
    summary = []
    use_sampling = args.cons_n > 1
    if use_sampling:
        print(f"=> cons@{args.cons_n}: temperature={args.temperature} top_p={args.top_p}")
    for alpha in args.alphas:
        print(f"\n=> alpha={alpha}")
        n_correct = 0
        for s in tqdm(samples, desc=f"alpha={alpha}"):
            key = (float(alpha), int(s.id))
            if key in done:
                cached = next(r for r in existing_rows if float(r["alpha"]) == float(alpha) and int(r["sample_id"]) == int(s.id))
                n_correct += int(bool(cached["correct"]))
                continue
            prompt = _PROMPT_TEMPLATE.format(question=s.question)

            if not use_sampling:
                # Original greedy single-shot path
                if alpha == 0.0:
                    text = _generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
                else:
                    with SteeringHook(model, layer_idx=args.layer, vector=layer_vec, alpha=alpha):
                        text = _generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
                ok = grader(text, s.gold_answer)
                pred = parser(text)
                row = {
                    "alpha": float(alpha), "sample_id": int(s.id), "gold": str(s.gold_answer),
                    "pred": str(pred) if pred is not None else None,
                    "correct": bool(ok), "len_text": int(len(text)),
                    "cons_n": 1,
                }
            else:
                # cons@N: sample N times (with steering if alpha>0), majority vote
                texts = []
                preds_n = []
                for k in range(args.cons_n):
                    torch.manual_seed(42 + s.id * 1009 + k)
                    if alpha == 0.0:
                        text = _generate(model, tokenizer, prompt,
                                         max_new_tokens=args.max_new_tokens,
                                         do_sample=True,
                                         temperature=args.temperature,
                                         top_p=args.top_p)
                    else:
                        with SteeringHook(model, layer_idx=args.layer, vector=layer_vec, alpha=alpha):
                            text = _generate(model, tokenizer, prompt,
                                             max_new_tokens=args.max_new_tokens,
                                             do_sample=True,
                                             temperature=args.temperature,
                                             top_p=args.top_p)
                    texts.append(text)
                    preds_n.append(parser(text))
                voted = _majority(preds_n)
                ok = pred_grader(voted, s.gold_answer)
                pred = voted
                row = {
                    "alpha": float(alpha), "sample_id": int(s.id), "gold": str(s.gold_answer),
                    "pred": str(pred) if pred is not None else None,
                    "correct": bool(ok),
                    "len_text": int(sum(len(t) for t in texts)),
                    "cons_n": args.cons_n,
                    "samples_preds": [str(p) if p is not None else None for p in preds_n],
                }
            n_correct += int(row["correct"])
            rows.append(row)
            with incremental_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        acc = n_correct / len(samples)
        print(f"   alpha={alpha} acc={acc:.3f}")
        summary.append({"alpha": alpha, "accuracy": acc, "n": len(samples), "cons_n": args.cons_n})

    df = pd.DataFrame(rows)
    df.to_parquet(args.out / "c4_per_sample.parquet", index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(args.out / "c4_alpha_accuracy.csv", index=False)
    print("\n=== alpha vs accuracy ===")
    print(sdf.to_string(index=False))

    # plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    sdf_sorted = sdf.sort_values("alpha")
    ax.plot(sdf_sorted["alpha"], sdf_sorted["accuracy"], marker="o")
    baseline_rows = sdf[sdf["alpha"] == 0.0]
    if not baseline_rows.empty:
        ax.axhline(baseline_rows["accuracy"].iloc[0], color="grey", linestyle="--",
                   label="baseline (alpha=0)")
    ax.set_xlabel("alpha (steering magnitude)")
    ax.set_ylabel(f"{args.dataset} acc (n={args.n_questions})")
    ax.set_title(f"C4: steering target by source-run mean-diff at layer {args.layer}")
    ax.grid(True, alpha=0.3)
    if not baseline_rows.empty:
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "c4_alpha_accuracy.png", dpi=140)
    print(f"\nwrote {args.out / 'c4_alpha_accuracy.png'}")


if __name__ == "__main__":
    main()
