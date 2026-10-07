#!/usr/bin/env python3
"""Extract compact generated-state means for a long-context formal calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from geoprobe.datasets import is_correct, load_gsm8k, parse_predicted
from geoprobe.extractors import load_trajectory, save_trajectory
from geoprobe.models import load_model_and_tokenizer
from geoprobe.revision.pooled_trajectory import generate_pooled_trajectory
from geoprobe.utils import dump_config, exp_dir, load_config

_PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"geoprobe.pooled.{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _seed(base_seed: int, sample_id: int) -> int:
    return base_seed * 100003 + sample_id * 1009


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("dataset", {}).get("name", "").lower() != "gsm8k":
        raise SystemExit("pooled formal calibration currently supports GSM8K only")
    generation = cfg.get("generation", {})
    max_new_tokens = int(generation.get("max_new_tokens", 32768))
    do_sample = bool(generation.get("do_sample", True))
    temperature = float(generation.get("temperature", 0.6))
    top_p = float(generation.get("top_p", 0.95))
    replay_chunk_size = int(cfg.get("pooled_replay", {}).get("chunk_size", 128))
    if max_new_tokens <= 512:
        raise SystemExit("official-context pooled calibration requires max_new_tokens > 512")
    if not do_sample:
        raise SystemExit("official-context pooled calibration requires the declared sampling decoder")

    out = exp_dir(cfg["exp_id"], create=True)
    logger = _logger(out / "logs" / "run.log")
    dump_config(cfg, out / "config.yaml")
    samples = load_gsm8k(
        split=cfg["dataset"].get("split", "test"),
        n=cfg["dataset"].get("n_samples"),
        source=cfg["dataset"].get("source", "oss"),
    )
    model_cfg = cfg["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_id=model_cfg["name"],
        source=model_cfg.get("source", "modelscope"),
        dtype=model_cfg.get("dtype", "bfloat16"),
        device=model_cfg.get("device", "cuda:0"),
    )
    labels: list[dict[str, object]] = []
    trajectories = out / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=True)
    base_seed = int(cfg.get("seed", 11241))
    replay_metadata: dict[str, object] | None = None
    for sample in tqdm(samples, desc="pooled-calibration"):
        path = trajectories / f"sample_{sample.id:04d}_idx_0.pt"
        if path.exists():
            trajectory = load_trajectory(path)
            if trajectory.hidden_states.ndim != 2:
                raise SystemExit(f"existing {path} is not a compact pooled trajectory")
        else:
            trajectory, metadata = generate_pooled_trajectory(
                model,
                tokenizer,
                prompt=_PROMPT_TEMPLATE.format(question=sample.question),
                sample_id=sample.id,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                generation_seed=_seed(base_seed, sample.id),
                replay_chunk_size=replay_chunk_size,
                model_id=model_cfg["name"],
                dtype=model_cfg.get("dtype", "bfloat16"),
            )
            replay_metadata = {
                "state_count": metadata.state_count,
                "chunk_size": metadata.chunk_size,
                "state_position_rule": metadata.state_position_rule,
                "aggregation": metadata.aggregation,
            }
            save_trajectory(trajectory, path)
        prediction = parse_predicted(trajectory.generated_text)
        labels.append(
            {
                "sample_id": sample.id,
                "sample_idx": 0,
                "gold": str(sample.gold_answer),
                "pred": str(prediction) if prediction is not None else None,
                "correct": is_correct(prediction, sample.gold_answer),
                "n_gen_tokens": int(trajectory.generated_token_ids.numel()),
                "sequence_logprob": float("nan"),
                "trajectory_representation": "pooled_generated_state_mean_v1",
            }
        )
    frame = pd.DataFrame(labels).sort_values("sample_id").reset_index(drop=True)
    frame.to_parquet(out / "labels.parquet", index=False)
    manifest = {
        "protocol": "tacl-11241-pooled-long-context-calibration-v1",
        "config_sha256": hashlib.sha256((out / "config.yaml").read_bytes()).hexdigest(),
        "n_problems": len(frame),
        "decoder": {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "seed_rule": "seed * 100003 + sample_id * 1009",
        },
        "pooling": replay_metadata
        or {
            "chunk_size": replay_chunk_size,
            "state_position_rule": "prompt_final_plus_generated_prefix",
            "aggregation": "mean_over_generated_state_positions",
        },
        "representation": "[layers, hidden] generated-state mean only",
    }
    (out / "pooled_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "DONE").write_text(f"n={len(frame)}\nrepresentation=pooled_generated_state_mean_v1\n")
    logger.info("completed compact pooled calibration for %d problems", len(frame))


if __name__ == "__main__":
    main()
