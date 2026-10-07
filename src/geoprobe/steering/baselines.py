"""Auditable vector primitives for formal activation-steering baselines.

These functions deliberately separate *vector construction* from dataset and
hook details.  Formal CAA/ActAdd/Sparse experiments must save the contrastive
examples, position rule, source-label budget, vector norm and random seed in
an accompanying manifest.
"""

from __future__ import annotations

import math

import torch


def _as_batched(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(
            f"{name} must have shape [n_examples, ...feature_dims], got {tuple(x.shape)}"
        )
    if x.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one example")
    return x.float()


def mean_difference_direction(
    positive_states: torch.Tensor,
    negative_states: torch.Tensor,
) -> torch.Tensor:
    """Return mean(positive) - mean(negative) for one layer/position.

    This is the basic contrastive direction used by CAA-style constructions.
    The caller records whether states came from a prompt-final token, a
    generated token, or a trajectory aggregate.
    """
    positive = _as_batched(positive_states, "positive_states")
    negative = _as_batched(negative_states, "negative_states")
    if positive.shape[1:] != negative.shape[1:]:
        raise ValueError("positive and negative states must share feature dimensions")
    return positive.mean(dim=0) - negative.mean(dim=0)


def paired_activation_addition_direction(
    positive_states: torch.Tensor,
    negative_states: torch.Tensor,
) -> torch.Tensor:
    """Return the mean paired positive-minus-negative activation direction.

    ActAdd-style contrast pairs must be aligned by row: row ``i`` of the
    positive and negative tensors represents the same prompt frame with a
    positive/negative continuation or contrast.
    """
    positive = _as_batched(positive_states, "positive_states")
    negative = _as_batched(negative_states, "negative_states")
    if positive.shape != negative.shape:
        raise ValueError("paired activation addition requires equal tensor shapes")
    return (positive - negative).mean(dim=0)


def match_l2_norm(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Scale ``vector`` to the L2 norm of ``reference`` without mutating inputs."""
    vector = vector.float()
    reference = reference.float()
    vector_norm = torch.linalg.vector_norm(vector)
    reference_norm = torch.linalg.vector_norm(reference)
    if not torch.isfinite(vector_norm) or not torch.isfinite(reference_norm):
        raise ValueError("vector norms must be finite")
    if vector_norm <= 0:
        raise ValueError("cannot norm-match a zero vector")
    return vector * (reference_norm / vector_norm)


def normalize_direction_rms(vector: torch.Tensor, *, rms: float = 1.0) -> torch.Tensor:
    """Scale a direction to a declared root-mean-square coordinate magnitude.

    For a hidden size ``H``, target L2 norm is ``rms * sqrt(H)``.  Formal
    comparisons use this normalization before selecting alpha so raw vector
    magnitude cannot give one steering method a larger perturbation budget.
    """
    if rms <= 0 or not math.isfinite(rms):
        raise ValueError("rms must be finite and positive")
    vector = vector.float()
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("cannot normalize a non-finite or zero vector")
    target_norm = rms * math.sqrt(vector.numel())
    return vector * (target_norm / norm)


def sparse_topk_direction(vector: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    """Keep the largest-magnitude coordinates of a vector and zero the rest.

    This utility is for a *declared sparse-direction ablation*.  It is not a
    substitute for an official sparse-activation method; formal experiments
    must document the source method and select ``keep_fraction`` on validation.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    flat = vector.float().reshape(-1)
    k = max(1, math.ceil(flat.numel() * keep_fraction))
    indices = torch.topk(flat.abs(), k=k, sorted=False).indices
    sparse = torch.zeros_like(flat)
    sparse[indices] = flat[indices]
    return sparse.reshape_as(vector)


def matched_norm_random_direction(reference: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Create a deterministic Gaussian direction with the reference L2 norm."""
    reference = reference.float()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    return match_l2_norm(random, reference)


def opposite_direction(reference: torch.Tensor) -> torch.Tensor:
    """Return the exact sign-reversed control for a nonzero reference vector."""
    reference = reference.float()
    norm = torch.linalg.vector_norm(reference)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("cannot reverse a non-finite or zero direction")
    return -reference
