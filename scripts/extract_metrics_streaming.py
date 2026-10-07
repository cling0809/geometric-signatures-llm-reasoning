#!/usr/bin/env python3
"""Metric-only trajectory extraction for large best-of-N experiments.

This is the storage-safe sibling of ``extract_trajectories.py``. It still
generates with hidden states, but it immediately reduces each trajectory to the
long-form geometry metrics needed by C1/GeoVote and discards the full
``[T, L, H]`` tensor instead of saving ``trajectories/*.pt``.

Writes:
  $GEOPROBE_RUNS/<exp_id>/
    config.yaml
    labels.jsonl / labels.parquet
    metrics.jsonl / metrics.parquet
    completed.jsonl
    logs/run.log
    DONE

The JSONL files are append-only so an interrupted run can resume. Final parquet
files are regenerated from JSONL after each flush and at completion.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

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
from geoprobe.extractors import extract_trajectory
from geoprobe.metrics import trajectory_metrics
from geoprobe.models import load_model_and_tokenizer
from geoprobe.utils import dump_config, exp_dir, load_config


_PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)


def _build_prompt(tokenizer, question: str, prompt_cfg: dict[str, Any]) -> str:
    mode = (prompt_cfg.get("mode") or "plain").lower()
    user_text = prompt_cfg.get("user_template") or _PROMPT_TEMPLATE
    user_prompt = user_text.format(question=question)

    if mode == "plain":
        return user_prompt
    if mode == "chat_template":
        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError("prompt.mode=chat_template requested, but tokenizer has no chat_template")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"unknown prompt.mode: {mode!r}")


def _load_dataset(ds_cfg: dict[str, Any]):
    name = ds_cfg.get("name", "gsm8k").lower()
    start = int(ds_cfg.get("start_sample", 0) or 0)
    n_samples = ds_cfg.get("n_samples")
    load_n = None if n_samples is None else start + int(n_samples)
    if name == "gsm8k":
        samples = load_gsm8k(
            split=ds_cfg.get("split", "test"),
            n=load_n,
            source=ds_cfg.get("source", "oss"),
        )
        return samples[start:], "gsm8k"
    if name in ("math500", "math-500"):
        samples = load_math500(
            split=ds_cfg.get("split", "test"),
            n=load_n,
            source=ds_cfg.get("source", "hf-mirror"),
        )
        return samples[start:], "math500"
    raise ValueError(f"unknown dataset name: {name!r}")


def _grade(dataset_kind: str, pred_text: str, gold) -> tuple[object, bool]:
    if dataset_kind == "gsm8k":
        pred = parse_predicted(pred_text)
        return pred, is_correct(pred, gold)
    if dataset_kind == "math500":
        pred = parse_predicted_math500(pred_text)
        return pred, is_correct_math500(pred, gold)
    raise ValueError(f"unknown dataset_kind: {dataset_kind!r}")


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("geoprobe.extract_metrics_streaming")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
        f.flush()


def _append_jsonl_many(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        f.flush()


def _read_completed(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    done: set[tuple[int, int]] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((int(row["sample_id"]), int(row["sample_idx"])))
    return done


def _jsonl_to_parquet(jsonl_path: Path, parquet_path: Path, dedupe: list[str]) -> int:
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return 0
    df = pd.read_json(jsonl_path, lines=True)
    if df.empty:
        return 0
    df = df.drop_duplicates(dedupe, keep="last")
    df.to_parquet(parquet_path, index=False)
    return len(df)


def _finalize_outputs(out_dir: Path) -> tuple[int, int]:
    n_labels = _jsonl_to_parquet(
        out_dir / "labels.jsonl",
        out_dir / "labels.parquet",
        ["sample_id", "sample_idx"],
    )
    n_metrics = _jsonl_to_parquet(
        out_dir / "metrics.jsonl",
        out_dir / "metrics.parquet",
        ["sample_id", "sample_idx", "metric", "layer"],
    )
    return n_labels, n_metrics


def _metric_rows(sample_id: int, sample_idx: int, hidden_states) -> list[dict[str, Any]]:
    metrics = trajectory_metrics(hidden_states)
    rows: list[dict[str, Any]] = []
    for name, vec in metrics.items():
        for layer_idx, value in enumerate(vec.tolist()):
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "sample_idx": int(sample_idx),
                    "metric": name,
                    "layer": int(layer_idx),
                    "value": float(value),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--flush-every", default=25, type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = exp_dir(cfg["exp_id"], create=True)
    logger = _setup_logging(out_dir / "logs" / "run.log")
    logger.info("config: %s", cfg)
    dump_config(cfg, out_dir / "config.yaml")

    labels_jsonl = out_dir / "labels.jsonl"
    metrics_jsonl = out_dir / "metrics.jsonl"
    generations_jsonl = out_dir / "generations.jsonl"
    completed_jsonl = out_dir / "completed.jsonl"
    completed = _read_completed(completed_jsonl)
    logger.info("resume: %d completed sample trajectories", len(completed))

    ds_cfg = cfg["dataset"]
    samples, dataset_kind = _load_dataset(ds_cfg)
    logger.info("loaded %d samples (dataset=%s)", len(samples), dataset_kind)

    m_cfg = cfg["model"]
    logger.info("loading model: %s", m_cfg["name"])
    model, tokenizer = load_model_and_tokenizer(
        model_id=m_cfg["name"],
        dtype=m_cfg.get("dtype", "bfloat16"),
        device=m_cfg.get("device", "cuda:0"),
        source=m_cfg.get("source", "modelscope"),
    )
    logger.info(
        "model loaded; n_layers=%d hidden_dim=%d",
        model.config.num_hidden_layers,
        model.config.hidden_size,
    )

    samp_cfg = cfg.get("sampling", {}) or {}
    num_samples = int(samp_cfg.get("num_samples", 1))
    start_sample_idx = int(samp_cfg.get("start_sample_idx", 0) or 0)
    end_sample_idx = int(samp_cfg.get("end_sample_idx", num_samples) or num_samples)
    if start_sample_idx < 0 or end_sample_idx > num_samples or start_sample_idx >= end_sample_idx:
        raise ValueError(
            "sampling.start_sample_idx/end_sample_idx must satisfy "
            f"0 <= start < end <= num_samples; got {start_sample_idx}, "
            f"{end_sample_idx}, num_samples={num_samples}"
        )
    do_sample = num_samples > 1 or bool(samp_cfg.get("do_sample", False))
    temperature = float(samp_cfg.get("temperature", 1.0))
    top_p = float(samp_cfg.get("top_p", 1.0))
    seed = int(cfg.get("seed", 42))
    max_new = int(cfg.get("extraction", {}).get("max_new_tokens", 512))
    prompt_cfg = cfg.get("prompt", {}) or {}
    prompt_mode = (prompt_cfg.get("mode") or "plain").lower()
    logger.info(
        "decode: N=%d sample_idx=[%d,%d) do_sample=%s temp=%.2f top_p=%.2f max_new=%d prompt_mode=%s",
        num_samples,
        start_sample_idx,
        end_sample_idx,
        do_sample,
        temperature,
        top_p,
        max_new,
        prompt_mode,
    )

    n_generated = 0
    n_skipped = 0
    for s in tqdm(samples, desc="extract-metrics"):
        prompt = _build_prompt(tokenizer, s.question, prompt_cfg)
        for idx in range(start_sample_idx, end_sample_idx):
            key = (int(s.id), int(idx))
            if key in completed:
                n_skipped += 1
                continue

            torch.manual_seed(seed * 100003 + int(s.id) * 1009 + idx)
            traj = extract_trajectory(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                sample_id=int(s.id),
                sample_idx=idx,
                max_new_tokens=max_new,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                model_id=m_cfg["name"],
                dtype=m_cfg.get("dtype", "bfloat16"),
            )

            pred, correct = _grade(dataset_kind, traj.generated_text, s.gold_answer)
            label_row = {
                "sample_id": int(s.id),
                "sample_idx": int(idx),
                "gold": str(s.gold_answer),
                "pred": str(pred) if pred is not None else None,
                "correct": bool(correct),
                "n_gen_tokens": int(traj.generated_token_ids.shape[0]),
                "sequence_logprob": float(traj.sequence_logprob),
            }
            generation_row = {
                **label_row,
                "generated_text": traj.generated_text,
            }

            _append_jsonl_many(
                metrics_jsonl,
                _metric_rows(int(s.id), int(idx), traj.hidden_states),
            )
            _append_jsonl(labels_jsonl, label_row)
            _append_jsonl(generations_jsonl, generation_row)
            _append_jsonl(
                completed_jsonl,
                {"sample_id": int(s.id), "sample_idx": int(idx)},
            )
            completed.add(key)
            n_generated += 1

            logger.info(
                "sample %d/%d: gold=%s pred=%s correct=%s T=%d logprob=%.2f",
                s.id,
                idx,
                s.gold_answer,
                pred,
                correct,
                traj.generated_token_ids.shape[0],
                traj.sequence_logprob,
            )

            del traj
            if torch.cuda.is_available() and n_generated % 4 == 0:
                torch.cuda.empty_cache()
            if args.flush_every > 0 and n_generated % args.flush_every == 0:
                n_labels, n_metrics = _finalize_outputs(out_dir)
                logger.info(
                    "checkpoint: labels=%d metrics=%d skipped=%d generated=%d",
                    n_labels,
                    n_metrics,
                    n_skipped,
                    n_generated,
                )

    n_labels, n_metrics = _finalize_outputs(out_dir)
    labels = pd.read_parquet(out_dir / "labels.parquet")
    pass1_series = labels[labels["sample_idx"] == 0]["correct"]
    pass1 = pass1_series.mean() if len(pass1_series) else float("nan")
    done_lines = [
        f"pass1={pass1:.4f}",
        f"n_labels={n_labels}",
        f"n_metrics={n_metrics}",
        f"n_questions={labels['sample_id'].nunique()}",
        f"num_samples={num_samples}",
        f"sample_idx_range=[{start_sample_idx},{end_sample_idx})",
    ]
    if num_samples > 1:
        oracle = labels.groupby("sample_id")["correct"].max().mean()
        done_lines.append(f"oracle{num_samples}={oracle:.4f}")
        logger.info(
            "done. pass@1=%.3f oracle@%d=%.3f labels=%d metrics=%d",
            pass1,
            num_samples,
            oracle,
            n_labels,
            n_metrics,
        )
    else:
        logger.info("done. accuracy=%.3f labels=%d metrics=%d", pass1, n_labels, n_metrics)
    (out_dir / "DONE").write_text("\n".join(done_lines) + "\n")


if __name__ == "__main__":
    main()
