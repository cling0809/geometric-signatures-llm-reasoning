"""Leak-audited CrossSteer direction construction for the TACL revision.

The historical helper in ``geoprobe.steering.vectors`` consumes every labeled
row in a run.  That is unsafe for the revision because a run may contain
validation or locked-test IDs.  This module requires an explicit source ID
set and records exactly which rows contributed to the direction.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch

from geoprobe.extractors import load_trajectory
from geoprobe.revision.protocol import RevisionSplit


def compute_crosssteer_direction(
    run_dir: str | Path,
    source_ids: Iterable[int],
    *,
    split: RevisionSplit | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Compute per-layer mean(correct)-mean(incorrect) using only ``source_ids``.

    Each trajectory contributes one time-averaged activation, so long
    completions cannot dominate the source direction.  If ``split`` is given,
    it rejects any non-source ID before reading trajectories.
    """
    run = Path(run_dir)
    ids = tuple(int(x) for x in source_ids)
    if split is not None:
        split.assert_direction_ids(ids)
    if not ids:
        raise ValueError("source_ids must not be empty")
    if len(ids) != len(set(ids)):
        raise ValueError("source_ids contains duplicate problem IDs")

    labels_path = run / "labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    labels = pd.read_parquet(labels_path)
    required = {"sample_id", "correct"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels.parquet is missing columns: {sorted(missing)}")
    if "sample_idx" in labels.columns:
        labels = labels[labels["sample_idx"] == 0]
    labels = labels[labels["sample_id"].isin(ids)]
    present = {int(x) for x in labels["sample_id"].tolist()}
    missing_ids = sorted(set(ids) - present)
    if missing_ids:
        raise ValueError(f"requested source IDs are absent from labels: {missing_ids[:10]}")

    correct: list[torch.Tensor] = []
    incorrect: list[torch.Tensor] = []
    used_ids: list[int] = []
    for _, row in labels.sort_values("sample_id").iterrows():
        sample_id = int(row["sample_id"])
        path = run / "trajectories" / f"sample_{sample_id:04d}_idx_0.pt"
        if not path.exists():
            path = run / "trajectories" / f"sample_{sample_id:04d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"trajectory for source ID {sample_id} not found in {run}")
        trajectory = load_trajectory(path)
        hidden = trajectory.hidden_states.float()
        if hidden.ndim == 3 and hidden.shape[0] > 0:
            per_problem = hidden.mean(dim=0)
        elif hidden.ndim == 2:
            # Compact long-context extraction stores the same time mean directly.
            per_problem = hidden
        else:
            raise ValueError(f"invalid hidden state shape for source ID {sample_id}: {tuple(hidden.shape)}")
        (correct if bool(row["correct"]) else incorrect).append(per_problem)
        used_ids.append(sample_id)

    if not correct or not incorrect:
        raise ValueError(
            f"source IDs must include both classes; got correct={len(correct)} incorrect={len(incorrect)}"
        )
    correct_stack = torch.stack(correct)
    incorrect_stack = torch.stack(incorrect)
    if correct_stack.shape[1:] != incorrect_stack.shape[1:]:
        raise ValueError("correct and incorrect source trajectories have incompatible shapes")
    direction = correct_stack.mean(dim=0) - incorrect_stack.mean(dim=0)
    metadata: dict[str, object] = {
        "run_dir": str(run.resolve()),
        "source_ids": used_ids,
        "source_count": len(used_ids),
        "correct_count": len(correct),
        "incorrect_count": len(incorrect),
        "aggregation": "mean_over_tokens_per_problem_then_mean_over_class",
        "direction_shape": list(direction.shape),
    }
    return direction, metadata
