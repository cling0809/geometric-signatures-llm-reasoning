"""Compute steering vectors from a source run's trajectories.

Given a run with extracted trajectories + correctness labels, compute the
mean-difference direction:
    v_layer = mean(hidden_states_correct[:, layer, :]) - mean(hidden_states_incorrect[:, layer, :])

Returns a [n_layers, hidden_dim] tensor. Each row is the "correct minus
incorrect" direction at that layer, computed from the source model.

Apply this vector at inference time on a *target* model (typically a different
model with the same hidden dimension) to push its hidden state in the direction
that, in the source model, was associated with correctness.

Aggregation: we mean over tokens *within* a trajectory first (each trajectory
contributes one vector), then mean across trajectories within the correct /
incorrect groups. This avoids long trajectories dominating.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from geoprobe.extractors import load_trajectory


def compute_steering_vector(run_dir: str | Path) -> torch.Tensor:
    """Return [n_layers, hidden_dim] tensor: mean(correct) - mean(incorrect) per layer."""
    run = Path(run_dir)
    labels = pd.read_parquet(run / "labels.parquet")
    # Default to greedy (sample_idx=0) if column present
    if "sample_idx" in labels.columns:
        labels = labels[labels["sample_idx"] == 0]

    correct_vecs: list[torch.Tensor] = []
    incorrect_vecs: list[torch.Tensor] = []

    for _, row in labels.iterrows():
        sid = int(row["sample_id"])
        path = run / "trajectories" / f"sample_{sid:04d}_idx_0.pt"
        if not path.exists():
            path = run / "trajectories" / f"sample_{sid:04d}.pt"  # back-compat naming
        if not path.exists():
            continue
        t = load_trajectory(path)
        hs = t.hidden_states.float()  # [T, L+1, H]
        traj_mean = hs.mean(dim=0)     # [L+1, H]
        if bool(row["correct"]):
            correct_vecs.append(traj_mean)
        else:
            incorrect_vecs.append(traj_mean)

    if not correct_vecs or not incorrect_vecs:
        raise ValueError(f"need both correct and incorrect samples; got "
                         f"{len(correct_vecs)} / {len(incorrect_vecs)}")

    c = torch.stack(correct_vecs).mean(dim=0)   # [L+1, H]
    w = torch.stack(incorrect_vecs).mean(dim=0)  # [L+1, H]
    return c - w  # [L+1, H]


def save_steering_vector(vec: torch.Tensor, path: str | Path, metadata: dict | None = None):
    payload = {"vector": vec, "metadata": metadata or {}}
    torch.save(payload, path)


def load_steering_vector(path: str | Path) -> tuple[torch.Tensor, dict]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["vector"], d.get("metadata", {})
