"""Auditable prompt-final CAA and ActAdd registry construction.

These baselines are intentionally separate from trajectory mean-difference
vectors.  They replay completed *source-training* completions through the
*target* checkpoint, capture genuine prompt-final residual states, and build:

* CAA from multiple same-question correct/incorrect contrast pairs;
* ActAdd from one deterministic pair among that fixed contrast set;
* a clearly labelled coordinate-sparse CAA ablation.

The module never consumes validation, locked-test, or OOD IDs.  It also avoids
claiming that the coordinate-sparse ablation is an SAE method.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from geoprobe.extractors import load_trajectory
from geoprobe.revision.activation_capture import capture_prompt_final_states
from geoprobe.revision.protocol import RevisionSplit
from geoprobe.revision.vector_registry import require_completed_run, sha256_file
from geoprobe.steering import paired_activation_addition_direction, sparse_topk_direction


@dataclass(frozen=True)
class ContrastCompletion:
    """One labeled source-training completion used as a contrast endpoint."""

    sample_id: int
    sample_idx: int
    correct: bool
    n_gen_tokens: int
    prompt_len: int
    full_prompt: str

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.full_prompt.encode("utf-8")).hexdigest()


def _trajectory_path(run: Path, sample_id: int, sample_idx: int) -> Path:
    candidates = (
        run / "trajectories" / f"sample_{sample_id:04d}_idx_{sample_idx}.pt",
        run / "trajectories" / f"sample_{sample_id:04d}_idx_{sample_idx:02d}.pt",
        run / "trajectories" / f"sample_{sample_id:04d}.pt",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"trajectory for source ID {sample_id}, sample index {sample_idx} not found in {run}"
    )


def load_labeled_source_completions(
    run_dir: str | Path,
    source_ids: tuple[int, ...],
    *,
    split: RevisionSplit,
) -> list[ContrastCompletion]:
    """Load only declared source-training completions from a completed run."""
    run = Path(run_dir)
    completion_path = run / "completions.parquet"
    if completion_path.exists():
        required_run = [run / "DONE", run / "config.yaml", run / "labels.parquet"]
        missing_run = [str(path) for path in required_run if not path.exists()]
        if missing_run:
            raise FileNotFoundError(f"incomplete contrast pool; missing {missing_run}")
    else:
        run = require_completed_run(run)
    split.assert_direction_ids(source_ids)
    labels = pd.read_parquet(run / "labels.parquet")
    required = {"sample_id", "sample_idx", "correct", "n_gen_tokens"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels are missing required columns: {sorted(missing)}")
    rows = labels[labels["sample_id"].isin(source_ids)].copy()
    present = set(int(value) for value in rows["sample_id"].unique())
    absent = sorted(set(source_ids) - present)
    if absent:
        raise ValueError(f"declared source IDs are absent from labels: {absent[:10]}")
    if rows.duplicated(["sample_id", "sample_idx"]).any():
        raise ValueError("labels contain duplicate source completion rows")

    completion_rows: dict[tuple[int, int], object] = {}
    if completion_path.exists():
        completion_frame = pd.read_parquet(completion_path)
        completion_required = {"sample_id", "sample_idx", "prompt", "generated_text", "prompt_len"}
        missing_completion = completion_required - set(completion_frame.columns)
        if missing_completion:
            raise ValueError(f"completions are missing required columns: {sorted(missing_completion)}")
        completion_rows = {
            (int(row.sample_id), int(row.sample_idx)): row
            for row in completion_frame.itertuples(index=False)
        }

    records: list[ContrastCompletion] = []
    for row in rows.sort_values(["sample_id", "sample_idx"]).itertuples(index=False):
        sample_id = int(row.sample_id)
        sample_idx = int(row.sample_idx)
        if completion_path.exists():
            completion = completion_rows.get((sample_id, sample_idx))
            if completion is None:
                raise ValueError(f"completion row missing for source ID {sample_id}/{sample_idx}")
            full_prompt = str(completion.prompt) + str(completion.generated_text)
            prompt_len = int(completion.prompt_len)
        else:
            trajectory = load_trajectory(_trajectory_path(run, sample_id, sample_idx))
            if trajectory.sample_id != sample_id or trajectory.sample_idx != sample_idx:
                raise ValueError(f"trajectory identity mismatch for source ID {sample_id}/{sample_idx}")
            full_prompt = trajectory.prompt + trajectory.generated_text
            prompt_len = int(trajectory.prompt_len)
        if not full_prompt:
            raise ValueError(f"empty prompt/completion for source ID {sample_id}/{sample_idx}")
        records.append(
            ContrastCompletion(
                sample_id=sample_id,
                sample_idx=sample_idx,
                correct=bool(row.correct),
                n_gen_tokens=int(row.n_gen_tokens),
                prompt_len=prompt_len,
                full_prompt=full_prompt,
            )
        )
    if not any(item.correct for item in records) or not any(not item.correct for item in records):
        raise ValueError("source-training completions must include correct and incorrect classes")
    return records


def same_question_pairs(records: list[ContrastCompletion]) -> list[tuple[ContrastCompletion, ContrastCompletion]]:
    """Return one deterministic correct/incorrect contrast pair per question.

    CAA and ActAdd require aligned contrasts.  Therefore both endpoints must
    share the exact original GSM8K question (same ``sample_id``).  The input
    collection is produced with multiple *source-training* completions per
    question.  When a question has both outcomes, choose the pair with the
    smallest generated-length difference, breaking ties by sample indices.
    No hidden activations, validation outcomes, or steering scores participate
    in this selection.
    """
    pairs: list[tuple[ContrastCompletion, ContrastCompletion]] = []
    frame: dict[int, list[ContrastCompletion]] = {}
    for item in records:
        frame.setdefault(item.sample_id, []).append(item)
    for sample_id in sorted(frame):
        values = frame[sample_id]
        correct = [item for item in values if item.correct]
        incorrect = [item for item in values if not item.correct]
        if not correct or not incorrect:
            continue
        pairs.append(
            min(
                ((positive, negative) for positive in correct for negative in incorrect),
                key=lambda pair: (
                    abs(pair[0].n_gen_tokens - pair[1].n_gen_tokens),
                    pair[0].sample_idx,
                    pair[1].sample_idx,
                ),
            )
        )
    return pairs


def _select_actadd_pair(
    pairs: list[tuple[ContrastCompletion, ContrastCompletion]], *, salt: str
) -> tuple[ContrastCompletion, ContrastCompletion]:
    if not pairs:
        raise ValueError("at least one CAA contrast pair is required for ActAdd")
    return min(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"{salt}:{pair[0].sample_id}:{pair[0].sample_idx}:{pair[1].sample_id}:{pair[1].sample_idx}".encode()
        ).hexdigest(),
    )


def build_prompt_contrast_vectors(
    model,
    tokenizer,
    *,
    target_run: str | Path,
    source_ids: tuple[int, ...],
    split: RevisionSplit,
    sparse_keep_fraction: float,
    actadd_selection_salt: str = "tacl-11241-actadd-v1",
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Capture frozen prompt-final contrasts and build CAA/ActAdd directions."""
    if not 0.0 < sparse_keep_fraction <= 1.0:
        raise ValueError("sparse_keep_fraction must be in (0, 1]")
    run = Path(target_run)
    required = [run / "DONE", run / "config.yaml", run / "labels.parquet"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete prompt contrast input; missing {missing}")
    records = load_labeled_source_completions(run, source_ids, split=split)
    pairs = same_question_pairs(records)
    if len(pairs) < 2:
        raise ValueError("at least two same-question contrast pairs are required for CAA")

    positive_states: list[torch.Tensor] = []
    negative_states: list[torch.Tensor] = []
    for positive, negative in pairs:
        positive_states.append(capture_prompt_final_states(model, tokenizer, positive.full_prompt))
        negative_states.append(capture_prompt_final_states(model, tokenizer, negative.full_prompt))
    positive_stack = torch.stack(positive_states)
    negative_stack = torch.stack(negative_states)
    caa = paired_activation_addition_direction(positive_stack, negative_stack)

    actadd_positive, actadd_negative = _select_actadd_pair(pairs, salt=actadd_selection_salt)
    actadd = capture_prompt_final_states(model, tokenizer, actadd_positive.full_prompt) - capture_prompt_final_states(
        model, tokenizer, actadd_negative.full_prompt
    )
    if caa.shape != actadd.shape or caa.ndim != 2:
        raise ValueError("prompt-final directions must share [layers, hidden] shape")
    sparse = torch.stack(
        [sparse_topk_direction(caa[layer], sparse_keep_fraction) for layer in range(caa.shape[0])]
    )

    def serialise(item: ContrastCompletion) -> dict[str, object]:
        return {
            "sample_id": item.sample_id,
            "sample_idx": item.sample_idx,
            "n_gen_tokens": item.n_gen_tokens,
            "prompt_len": item.prompt_len,
            "prompt_sha256": item.prompt_sha256,
        }

    target_labels = pd.read_parquet(run / "labels.parquet")
    metadata: dict[str, object] = {
        "protocol": "tacl-11241-prompt-contrast-registry-v1",
        "source_ids": list(source_ids),
        "target_run": str(run.resolve()),
        "target_run_config_sha256": sha256_file(run / "config.yaml"),
        "target_labels_sha256": sha256_file(run / "labels.parquet"),
        "target_completion_source": "completions.parquet" if (run / "completions.parquet").exists() else "trajectory_artifacts",
        "target_completions_sha256": (
            sha256_file(run / "completions.parquet") if (run / "completions.parquet").exists() else None
        ),
        "capture_position": "true_prompt_final_after_source_completion",
        "source_label_budget_questions": len(source_ids),
        "source_label_budget_completion_rows": int(
            len(target_labels[target_labels["sample_id"].isin(source_ids)])
        ),
        "direction_shape": list(caa.shape),
        "caa": {
            "method": "multi_pair_prompt_final_contrastive_activation_addition",
            "n_pairs": len(pairs),
            "pairing_rule": "same_question_min_abs_generated_length_then_sample_indices",
            "pairs": [{"positive": serialise(pos), "negative": serialise(neg)} for pos, neg in pairs],
        },
        "actadd": {
            "method": "single_prompt_final_activation_addition_pair",
            "selection_rule": "minimum_sha256_over_frozen_length_matched_pairs",
            "selection_salt": actadd_selection_salt,
            "positive": serialise(actadd_positive),
            "negative": serialise(actadd_negative),
        },
        "sparse_caa_coordinate_ablation": {
            "keep_fraction": sparse_keep_fraction,
            "warning": "coordinate-sparse ablation; not SAE-based sparse activation steering",
        },
    }
    return {
        "caa_target_prompt_final": caa.cpu(),
        "actadd_target_prompt_final": actadd.cpu(),
        "sparse_caa_coordinate_10pct": sparse.cpu(),
    }, metadata


def save_prompt_contrast_registry(
    vectors: dict[str, torch.Tensor], metadata: dict[str, object], out_dir: str | Path
) -> None:
    """Write fingerprinted contrast vectors and their complete provenance manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, vector in vectors.items():
        if vector.ndim != 2 or not torch.isfinite(vector).all():
            raise ValueError(f"{name} must be a finite [layers, hidden] tensor")
        path = out / f"{name}.pt"
        torch.save(vector.cpu(), path)
        hashes[name] = sha256_file(path)
    payload = {**metadata, "vector_sha256": hashes}
    (out / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
