"""Tests for steering vectors + hooks.

We don't load a real HF model in tests (slow + needs GPU); we test the math
of compute_steering_vector against synthetic data, and we test SteeringHook
against a tiny nn.Module that mimics the transformer-block interface.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from geoprobe.extractors import Trajectory, save_trajectory
from geoprobe.steering import SteeringHook, compute_steering_vector


class _FakeBlock(nn.Module):
    """Mimics a transformer block: forward returns (hidden, ...) tuple."""

    def __init__(self, dim=4):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return (self.lin(x), None)


class _FakeBackbone(nn.Module):
    def __init__(self, n_layers=3, dim=4):
        super().__init__()
        self.layers = nn.ModuleList([_FakeBlock(dim) for _ in range(n_layers)])


class _FakeModel(nn.Module):
    def __init__(self, n_layers=3, dim=4):
        super().__init__()
        self.model = _FakeBackbone(n_layers=n_layers, dim=dim)


def _write_fake_run(run: Path, n_correct: int, n_incorrect: int, dim=4, n_layers=3, T=5):
    (run / "trajectories").mkdir(parents=True, exist_ok=True)
    rows = []
    sid = 0
    for ok, n in ((True, n_correct), (False, n_incorrect)):
        for _ in range(n):
            # correct trajectories pushed toward +1, incorrect toward -1
            base = 1.0 if ok else -1.0
            hs = torch.full((T, n_layers, dim), base, dtype=torch.bfloat16)
            traj = Trajectory(
                sample_id=sid, sample_idx=0, hidden_states=hs,
                generated_token_ids=torch.zeros(T, dtype=torch.int64),
                generated_text="", prompt="", prompt_len=0,
                sequence_logprob=0.0, model_id="fake", dtype="bfloat16",
            )
            save_trajectory(traj, run / "trajectories" / f"sample_{sid:04d}_idx_0.pt")
            rows.append({"sample_id": sid, "sample_idx": 0, "gold": 0.0,
                         "pred": 0.0, "correct": ok, "n_gen_tokens": T,
                         "sequence_logprob": 0.0})
            sid += 1
    pd.DataFrame(rows).to_parquet(run / "labels.parquet")


def test_compute_steering_vector_direction(tmp_path):
    _write_fake_run(tmp_path, n_correct=4, n_incorrect=4, dim=4, n_layers=3, T=5)
    v = compute_steering_vector(tmp_path)
    # correct mean = 1, incorrect mean = -1 -> v should be ~2 everywhere
    assert v.shape == (3, 4)
    assert torch.allclose(v, torch.full_like(v, 2.0), atol=1e-3)


def test_compute_steering_requires_both_groups(tmp_path):
    _write_fake_run(tmp_path, n_correct=4, n_incorrect=0)
    try:
        compute_steering_vector(tmp_path)
    except ValueError:
        return
    assert False, "expected ValueError when no incorrect samples"


def test_steering_hook_adds_alpha_times_vector():
    torch.manual_seed(0)
    model = _FakeModel(n_layers=3, dim=4)
    x = torch.zeros(1, 2, 4)

    # baseline forward (no steering)
    with torch.no_grad():
        h0 = model.model.layers[0](x)[0]
        h1 = model.model.layers[1](h0)[0]
        h2 = model.model.layers[2](h1)[0]
    baseline = h2.clone()

    vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
    alpha = 0.5
    # inject at layer_idx=1 (i.e. block index 0)
    with SteeringHook(model, layer_idx=1, vector=vec, alpha=alpha):
        with torch.no_grad():
            h0p = model.model.layers[0](x)[0]
            # h0p should be h0 + alpha*vec (broadcasted)
            assert torch.allclose(h0p, h0 + alpha * vec, atol=1e-5)
            # downstream is then propagated normally
            h1p = model.model.layers[1](h0p)[0]
            h2p = model.model.layers[2](h1p)[0]

    # after exit, hook should be removed
    with torch.no_grad():
        h0b = model.model.layers[0](x)[0]
        assert torch.allclose(h0b, h0, atol=1e-6)
