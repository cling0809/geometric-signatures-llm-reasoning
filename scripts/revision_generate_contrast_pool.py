#!/usr/bin/env python3
"""Generate a lightweight source-training completion pool for CAA/ActAdd.

Unlike trajectory extraction, this runner stores no hidden-state tensor.  It is
only for creating same-question correct/incorrect contrast completions from the
frozen source-training IDs.  The later registry replays the exact saved text
through the target model to capture genuine prompt-final activations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from geoprobe.datasets import is_correct, load_gsm8k, parse_predicted
from geoprobe.models import load_model_and_tokenizer
from geoprobe.revision.pooled_trajectory import _generate_with_local_seed
from geoprobe.utils import dump_config, exp_dir, load_config

_PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)


def _setup_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("geoprobe.revision.contrast_pool")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, mode="a"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _load_existing(path: Path) -> dict[tuple[int, int], dict[str, object]]:
    if not path.exists():
        return {}
    values: dict[tuple[int, int], dict[str, object]] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = (int(row["sample_id"]), int(row["sample_idx"]))
        if key in values:
            raise ValueError(f"duplicate completion row for {key}")
        values[key] = row
    return values


def _contrast_seed(base_seed: int, sample_id: int, sample_idx: int) -> int:
    """Frozen completion-level seed; independent of loop/resume order."""
    return base_seed * 100003 + sample_id * 1009 + sample_idx


@torch.inference_mode()
def _generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    generation_seed: int | None,
):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Use the model's full EOS set.  Qwen2.5-Instruct's generation config
    # declares both <|im_end|> (151645) and <|endoftext|> (151643); bare
    # sampling terminates on the latter.  Passing only tokenizer.eos_token_id
    # would drop 151643 and force every completion to max_new_tokens.
    eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    generation_kwargs: dict[str, object] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": eos_ids,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})
    output = _generate_with_local_seed(
        model,
        generation_kwargs,
        generation_seed=generation_seed if do_sample else None,
        device=device,
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    generated_ids = output[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True), int(generated_ids.numel()), prompt_len


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = config.get("dataset", {})
    sampling = config.get("sampling", {}) or {}
    extraction = config.get("extraction", {}) or {}
    if dataset.get("name", "").lower() != "gsm8k" or int(dataset.get("n_samples", 0)) != 100:
        raise SystemExit("the formal CAA pool is frozen to the first 100 GSM8K source-training IDs")
    num_samples = int(sampling.get("num_samples", 1))
    if num_samples < 2:
        raise SystemExit("CAA/ActAdd contrast pool requires at least two samples per source question")
    max_new_tokens = int(extraction.get("max_new_tokens", 512))
    if max_new_tokens <= 0:
        raise SystemExit("max_new_tokens must be positive")

    out = exp_dir(config["exp_id"], create=True)
    out.mkdir(parents=True, exist_ok=True)
    dump_config(config, out / "config.yaml")
    logger = _setup_logging(out / "logs" / "run.log")
    existing_path = out / "completions.jsonl"
    rows = _load_existing(existing_path)
    logger.info("resuming %d existing completion rows", len(rows))

    samples = load_gsm8k(split=dataset.get("split", "test"), n=100, source=dataset.get("source", "oss"))
    model_config = config["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_id=model_config["name"],
        source=model_config.get("source", "modelscope"),
        dtype=model_config.get("dtype", "bfloat16"),
        device=model_config.get("device", "cuda:0"),
    )
    seed = int(config.get("seed", 42))
    do_sample = bool(sampling.get("do_sample", num_samples > 1))
    temperature = float(sampling.get("temperature", 1.0))
    top_p = float(sampling.get("top_p", 1.0))

    with existing_path.open("a") as handle:
        for sample in tqdm(samples, desc="CAA contrast pool"):
            prompt = _PROMPT_TEMPLATE.format(question=sample.question)
            for sample_idx in range(num_samples):
                key = (int(sample.id), sample_idx)
                if key in rows:
                    continue
                generation_seed = _contrast_seed(seed, int(sample.id), sample_idx) if do_sample else None
                text, n_tokens, prompt_len = _generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    generation_seed=generation_seed,
                )
                prediction = parse_predicted(text)
                row = {
                    "sample_id": int(sample.id),
                    "sample_idx": sample_idx,
                    "gold": str(sample.gold_answer),
                    "pred": None if prediction is None else str(prediction),
                    "correct": bool(is_correct(prediction, sample.gold_answer)),
                    "n_gen_tokens": n_tokens,
                    "prompt": prompt,
                    "generated_text": text,
                    "prompt_len": prompt_len,
                    "generation_seed": generation_seed,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows[key] = row

    expected = {(int(sample.id), sample_idx) for sample in samples for sample_idx in range(num_samples)}
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        raise RuntimeError(f"incomplete contrast pool; missing {missing[:10]}")
    frame = pd.DataFrame(rows.values()).sort_values(["sample_id", "sample_idx"])
    # Keep two separate provenance views, consistent with trajectory runs.
    frame.drop(columns=["prompt", "generated_text", "prompt_len"]).to_parquet(out / "labels.parquet", index=False)
    frame.to_parquet(out / "completions.parquet", index=False)
    pass1 = float(frame[frame["sample_idx"] == 0]["correct"].mean())
    oracle = float(frame.groupby("sample_id")["correct"].max().mean())
    (out / "DONE").write_text(
        f"pass1={pass1:.4f}\noracle{num_samples}={oracle:.4f}\nn_questions={len(samples)}\n"
    )
    logger.info("complete: pass1=%.4f oracle@%d=%.4f", pass1, num_samples, oracle)


if __name__ == "__main__":
    main()
