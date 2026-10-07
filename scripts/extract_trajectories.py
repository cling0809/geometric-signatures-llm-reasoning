#!/usr/bin/env python3
"""Orchestrator: load model + dataset per config, extract hidden-state trajectories.

Supports greedy (default) or sampling with multiple samples per question.
Sampling block in config:
  sampling:
    num_samples: 4
    temperature: 0.7
    top_p: 0.95

Writes:
  $GEOPROBE_RUNS/<exp_id>/
    config.yaml
    trajectories/sample_NNNN_idx_M.pt    (greedy uses idx_0)
    labels.parquet  [sample_id, sample_idx, gold, pred, correct, n_gen_tokens, sequence_logprob]
    logs/run.log
    DONE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from geoprobe.datasets import (
    is_correct,
    is_correct_math500,
    load_gsm8k,
    load_math500,
    parse_predicted,
    parse_predicted_math500,
)
from geoprobe.extractors import extract_trajectory, load_trajectory, save_trajectory
from geoprobe.models import load_model_and_tokenizer
from geoprobe.utils import dump_config, exp_dir, load_config

_PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)


def _load_dataset(ds_cfg: dict):
    """Dispatch to the right loader based on config. Returns list of samples
    where each sample has .id, .question, .gold_answer."""
    name = ds_cfg.get("name", "gsm8k").lower()
    if name == "gsm8k":
        return load_gsm8k(
            split=ds_cfg.get("split", "test"),
            n=ds_cfg.get("n_samples"),
            source=ds_cfg.get("source", "oss"),
        ), "gsm8k"
    elif name in ("math500", "math-500"):
        return load_math500(
            split=ds_cfg.get("split", "test"),
            n=ds_cfg.get("n_samples"),
            source=ds_cfg.get("source", "hf-mirror"),
        ), "math500"
    else:
        raise ValueError(f"unknown dataset name: {name!r}")


def _grade(dataset_kind: str, pred_text: str, gold) -> tuple[object, bool]:
    """Return (pred_parsed, correct_bool)."""
    if dataset_kind == "gsm8k":
        pred = parse_predicted(pred_text)
        return pred, is_correct(pred, gold)
    elif dataset_kind == "math500":
        pred = parse_predicted_math500(pred_text)
        return pred, is_correct_math500(pred, gold)
    else:
        raise ValueError(f"unknown dataset_kind: {dataset_kind!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_token_ids(value: object) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [int(value)]
    if isinstance(value, (list, tuple)) and value:
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("model generation EOS values must be integer token ids")
        ids = [int(item) for item in value]
        if len(set(ids)) != len(ids):
            raise ValueError("model generation EOS values must be unique")
        return ids
    raise ValueError("model generation config has no EOS token id")


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("geoprobe.extract")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = exp_dir(cfg["exp_id"], create=True)
    logger = _setup_logging(out_dir / "logs" / "run.log")
    logger.info("config: %s", cfg)
    dump_config(cfg, out_dir / "config.yaml")

    # 1. dataset
    ds_cfg = cfg["dataset"]
    samples, dataset_kind = _load_dataset(ds_cfg)
    logger.info("loaded %d samples (dataset=%s)", len(samples), dataset_kind)

    # 2. model
    m_cfg = cfg["model"]
    logger.info("loading model: %s", m_cfg["name"])
    model, tokenizer = load_model_and_tokenizer(
        model_id=m_cfg["name"],
        dtype=m_cfg.get("dtype", "bfloat16"),
        device=m_cfg.get("device", "cuda:0"),
        source=m_cfg.get("source", "modelscope"),
    )
    logger.info("model loaded; n_layers=%d hidden_dim=%d", model.config.num_hidden_layers, model.config.hidden_size)

    # 3. sampling settings (default: greedy single sample)
    samp_cfg = cfg.get("sampling", {}) or {}
    num_samples = int(samp_cfg.get("num_samples", 1))
    do_sample = num_samples > 1 or samp_cfg.get("do_sample", False)
    temperature = float(samp_cfg.get("temperature", 1.0))
    top_p = float(samp_cfg.get("top_p", 1.0))
    seed = int(cfg.get("seed", 42))
    if do_sample:
        logger.info("sampling: N=%d temp=%.2f top_p=%.2f", num_samples, temperature, top_p)
    else:
        logger.info("greedy decoding (single sample)")

    # 4. freeze the concrete generation envelope before writing any trajectory.
    max_new = int(cfg.get("extraction", {}).get("max_new_tokens", 512))
    if max_new <= 0:
        raise ValueError("extraction.max_new_tokens must be positive")
    eos_ids = _normalize_token_ids(getattr(model.generation_config, "eos_token_id", None))
    generation_manifest = {
        "protocol": "tacl-11241-calibration-generation-provenance-v1",
        "exp_id": cfg["exp_id"],
        "config_sha256": _sha256(out_dir / "config.yaml"),
        "model": m_cfg["name"],
        "model_generation_eos_token_ids": eos_ids,
        "generation": {
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "eos_token_id": "model.generation_config.eos_token_id",
            "pad_token_id": tokenizer.eos_token_id,
        },
        "prompt_template_sha256": hashlib.sha256(_PROMPT_TEMPLATE.encode()).hexdigest(),
        "extractor_sha256": _sha256(Path(__file__)),
    }
    manifest_path = out_dir / "resolved_generation_manifest.json"
    encoded_manifest = json.dumps(generation_manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != encoded_manifest:
        raise ValueError(f"refuse to overwrite a different generation provenance manifest: {manifest_path}")
    if not manifest_path.exists() and any((out_dir / "trajectories").glob("*.pt")):
        raise ValueError("refuse to add generation provenance after trajectory extraction has begun")
    manifest_path.write_text(encoded_manifest)
    logger.info("generation envelope: max_new=%d eos_ids=%s", max_new, eos_ids)

    # 5. loop
    labels = []
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    n_resumed = 0
    n_generated = 0
    n_corrupt = 0
    for s in tqdm(samples, desc="extract"):
        prompt = _PROMPT_TEMPLATE.format(question=s.question)
        for idx in range(num_samples):
            traj_path = traj_dir / f"sample_{s.id:04d}_idx_{idx}.pt"
            loaded = None
            if traj_path.exists():
                try:
                    loaded = load_trajectory(traj_path)
                except Exception as e:
                    # Corrupt file (e.g. partial write from a prior crash) — drop and regenerate.
                    logger.warning("corrupt traj %s: %s — regenerating", traj_path.name, type(e).__name__)
                    traj_path.unlink()
                    n_corrupt += 1
            if loaded is not None:
                traj = loaded
                n_resumed += 1
            else:
                # Seed per (sample_id, sample_idx) so samples are independent + reproducible
                torch.manual_seed(seed * 100003 + s.id * 1009 + idx)
                traj = extract_trajectory(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    sample_id=s.id,
                    sample_idx=idx,
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    model_id=m_cfg["name"],
                    dtype=m_cfg.get("dtype", "bfloat16"),
                )
                save_trajectory(traj, traj_path)
                n_generated += 1

            pred, correct = _grade(dataset_kind, traj.generated_text, s.gold_answer)
            n_gen_tokens = int(traj.generated_token_ids.shape[0])
            terminal_id = int(traj.generated_token_ids[-1].item())
            if terminal_id in eos_ids:
                stop_reason = "eos"
            elif n_gen_tokens >= max_new:
                stop_reason = "max_new_tokens"
            else:
                stop_reason = "other"
            labels.append(
                {
                    "sample_id": s.id,
                    "sample_idx": idx,
                    "gold": str(s.gold_answer),
                    "pred": str(pred) if pred is not None else None,
                    "correct": correct,
                    "n_gen_tokens": n_gen_tokens,
                    "sequence_logprob": traj.sequence_logprob,
                    "stop_reason": stop_reason,
                    "truncated": stop_reason == "max_new_tokens",
                }
            )
            logger.info(
                "sample %d/%d: gold=%s pred=%s correct=%s (T=%d, logprob=%.2f)",
                s.id, idx, s.gold_answer, pred, correct,
                traj.generated_token_ids.shape[0], traj.sequence_logprob,
            )
    logger.info("resumed=%d generated=%d corrupt=%d", n_resumed, n_generated, n_corrupt)

    df = pd.DataFrame(labels)
    df.to_parquet(out_dir / "labels.parquet", index=False)
    # Headline accuracies: greedy (idx=0) vs at-least-one (oracle)
    pass1 = df[df["sample_idx"] == 0]["correct"].mean()
    if num_samples > 1:
        oracle = df.groupby("sample_id")["correct"].max().mean()
        logger.info("done. pass@1=%.3f oracle@%d=%.3f over %d questions",
                    pass1, num_samples, oracle, df["sample_id"].nunique())
        (out_dir / "DONE").write_text(f"pass1={pass1:.4f}\noracle{num_samples}={oracle:.4f}\nn_questions={df['sample_id'].nunique()}\n")
    else:
        logger.info("done. accuracy=%.3f over %d samples", pass1, len(df))
        (out_dir / "DONE").write_text(f"accuracy={pass1:.4f}\nn={len(df)}\n")


if __name__ == "__main__":
    main()
