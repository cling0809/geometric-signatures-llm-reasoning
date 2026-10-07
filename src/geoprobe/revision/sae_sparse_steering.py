"""Audited SAE-space sparse-activation steering baseline.

This is a target-model sparse-activation baseline for the TACL-11241 revision,
not a coordinate top-k ablation presented as an SAE method.  A small sparse
autoencoder is trained *without correctness labels* on fixed relative-progress
states from source-training calibration trajectories.  Correct/incorrect
same-question completion pairs then select sparse SAE features, whose decoder
vectors form a steering direction at each predeclared layer.

The construction keeps every learned statistic inside the source-training
split.  Validation, locked, OOD and steering outcomes are never read here.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from geoprobe.extractors import load_trajectory
from geoprobe.revision.activation_capture import capture_prompt_final_states
from geoprobe.revision.prompt_contrast_registry import (
    ContrastCompletion,
    load_labeled_source_completions,
    same_question_pairs,
)
from geoprobe.revision.protocol import RevisionSplit
from geoprobe.revision.vector_registry import require_completed_run, sha256_file


@dataclass(frozen=True)
class SAETrainingConfig:
    """Frozen construction settings for the sparse-activation baseline."""

    layers: tuple[int, ...] = (8, 14, 20, 24)
    relative_progress_positions: int = 128
    latent_multiplier: int = 4
    feature_keep_fraction: float = 0.02
    train_steps: int = 600
    batch_size: int = 256
    learning_rate: float = 3e-4
    l1_coefficient: float = 5e-4
    seed: int = 11241

    def __post_init__(self) -> None:
        if not self.layers or len(set(self.layers)) != len(self.layers) or min(self.layers) < 0:
            raise ValueError("layers must be non-empty, unique and non-negative")
        if self.relative_progress_positions <= 1:
            raise ValueError("relative_progress_positions must exceed one")
        if self.latent_multiplier <= 1:
            raise ValueError("latent_multiplier must exceed one")
        if not 0.0 < self.feature_keep_fraction <= 1.0:
            raise ValueError("feature_keep_fraction must be in (0, 1]")
        if self.train_steps <= 0 or self.batch_size <= 0:
            raise ValueError("train_steps and batch_size must be positive")
        if self.learning_rate <= 0 or self.l1_coefficient < 0:
            raise ValueError("learning_rate must be positive and l1_coefficient non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "layers": list(self.layers),
            "relative_progress_positions": self.relative_progress_positions,
            "latent_multiplier": self.latent_multiplier,
            "feature_keep_fraction": self.feature_keep_fraction,
            "train_steps": self.train_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "l1_coefficient": self.l1_coefficient,
            "seed": self.seed,
        }


def _trajectory_path(run: Path, sample_id: int) -> Path:
    paths = sorted((run / "trajectories").glob(f"sample_{sample_id:04d}_idx_*.pt"))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"expected exactly one calibration trajectory for source ID {sample_id}, found {paths}"
        )
    return paths[0]


def relative_progress_indices(length: int, n_positions: int) -> torch.Tensor:
    """Return deterministic, evenly-spaced non-padding positions for one trajectory."""
    if length <= 0:
        raise ValueError("trajectory length must be positive")
    if n_positions <= 1:
        raise ValueError("n_positions must exceed one")
    if length <= n_positions:
        return torch.arange(length, dtype=torch.long)
    return torch.linspace(0, length - 1, steps=n_positions).round().to(torch.long).unique()


def load_unlabeled_sae_states(
    calibration_run: str | Path,
    source_ids: Iterable[int],
    *,
    config: SAETrainingConfig,
) -> dict[int, torch.Tensor]:
    """Read only train-split generated states with fixed per-question sampling.

    Exactly the same maximum number of relative-progress states is retained per
    question.  This prevents long source completions from dominating unsupervised
    SAE training.  The calibration labels file is checked for provenance but is
    never read for correctness values.
    """
    run = require_completed_run(calibration_run)
    ids = tuple(int(value) for value in source_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("source_ids must be non-empty and unique")
    hidden_by_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in config.layers}
    expected_shape: tuple[int, int] | None = None
    for sample_id in sorted(ids):
        trajectory = load_trajectory(_trajectory_path(run, sample_id))
        states = trajectory.hidden_states.float()
        if states.ndim != 3:
            raise ValueError(f"trajectory {sample_id} must have [steps,layers,hidden] states")
        n_steps, n_layers, hidden_size = states.shape
        if max(config.layers) >= n_layers:
            raise ValueError(
                f"requested layer {max(config.layers)} but trajectory has only {n_layers} layers"
            )
        if expected_shape is None:
            expected_shape = (n_layers, hidden_size)
        elif expected_shape != (n_layers, hidden_size):
            raise ValueError("calibration trajectories do not share [layers,hidden] shape")
        positions = relative_progress_indices(n_steps, config.relative_progress_positions)
        for layer in config.layers:
            hidden_by_layer[layer].append(states[positions, layer, :].cpu())
    return {layer: torch.cat(values, dim=0) for layer, values in hidden_by_layer.items()}


def _train_one_sae(
    states: torch.Tensor,
    *,
    config: SAETrainingConfig,
    layer: int,
    device: str | torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Fit a deterministic ReLU SAE with unit-norm decoder features."""
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("SAE states must be [n_examples, hidden] with at least two rows")
    if not torch.isfinite(states).all():
        raise ValueError("SAE states contain non-finite values")
    hidden_size = int(states.shape[1])
    latent_size = hidden_size * config.latent_multiplier
    center = states.float().mean(dim=0)
    centered = states.float() - center
    scale = centered.square().mean().sqrt()
    if not torch.isfinite(scale) or float(scale) <= 0:
        raise ValueError("SAE training states have zero or invalid global scale")
    x = (centered / scale).to(device=device, dtype=torch.float32)

    # All initialization and row sampling are local to this layer, so adding a
    # different layer cannot alter the direction at an existing layer.
    torch.manual_seed(config.seed + layer * 1_000_003)
    encoder = torch.empty((latent_size, hidden_size), device=device, requires_grad=True)
    decoder = torch.empty((latent_size, hidden_size), device=device, requires_grad=True)
    encoder_bias = torch.zeros(latent_size, device=device, requires_grad=True)
    torch.nn.init.kaiming_uniform_(encoder, a=math.sqrt(5))
    torch.nn.init.kaiming_uniform_(decoder, a=math.sqrt(5))
    with torch.no_grad():
        decoder.div_(decoder.norm(dim=1, keepdim=True).clamp_min(1e-8))

    optimizer = torch.optim.Adam((encoder, decoder, encoder_bias), lr=config.learning_rate)
    index_generator = torch.Generator(device="cpu").manual_seed(config.seed + layer * 10_000_019)
    final_loss = final_reconstruction = final_l1 = 0.0
    for _ in range(config.train_steps):
        indices = torch.randint(
            x.shape[0], (min(config.batch_size, x.shape[0]),), generator=index_generator
        ).to(device)
        batch = x.index_select(0, indices)
        activations = F.relu(F.linear(batch, encoder, encoder_bias))
        reconstruction = F.linear(activations, decoder.T)
        reconstruction_loss = F.mse_loss(reconstruction, batch)
        sparse_loss = activations.abs().mean()
        loss = reconstruction_loss + config.l1_coefficient * sparse_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            decoder.div_(decoder.norm(dim=1, keepdim=True).clamp_min(1e-8))
        final_loss = float(loss.detach().cpu())
        final_reconstruction = float(reconstruction_loss.detach().cpu())
        final_l1 = float(sparse_loss.detach().cpu())

    state = {
        "center": center.cpu(),
        "scale": scale.detach().cpu(),
        "encoder": encoder.detach().cpu(),
        "encoder_bias": encoder_bias.detach().cpu(),
        "decoder": decoder.detach().cpu(),
    }
    audit = {
        "n_training_states": float(x.shape[0]),
        "hidden_size": float(hidden_size),
        "latent_size": float(latent_size),
        "final_loss": final_loss,
        "final_reconstruction_loss": final_reconstruction,
        "final_mean_feature_activation": final_l1,
    }
    return state, audit


def _encode(states: torch.Tensor, sae: dict[str, torch.Tensor]) -> torch.Tensor:
    scale = sae["scale"].float()
    normalized = (states.float() - sae["center"].float()) / scale
    return F.relu(F.linear(normalized, sae["encoder"].float(), sae["encoder_bias"].float()))


def _pair_metadata(pair: tuple[ContrastCompletion, ContrastCompletion]) -> dict[str, object]:
    positive, negative = pair
    return {
        "sample_id": positive.sample_id,
        "positive_sample_idx": positive.sample_idx,
        "negative_sample_idx": negative.sample_idx,
        "positive_prompt_sha256": positive.prompt_sha256,
        "negative_prompt_sha256": negative.prompt_sha256,
    }


def build_sae_sparse_activation_vectors(
    model,
    tokenizer,
    *,
    calibration_run: str | Path,
    contrast_run: str | Path,
    source_ids: Iterable[int],
    split: RevisionSplit,
    config: SAETrainingConfig,
    device: str | torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Construct an SAE-decoded sparse activation steering direction per layer."""
    ids = tuple(int(value) for value in source_ids)
    split.assert_direction_ids(ids)
    calibration = require_completed_run(calibration_run)
    contrast = Path(contrast_run)
    required = (contrast / "DONE", contrast / "config.yaml", contrast / "labels.parquet")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete contrast run; missing {missing}")

    unlabeled_states = load_unlabeled_sae_states(calibration, ids, config=config)
    records = load_labeled_source_completions(contrast, ids, split=split)
    pairs = same_question_pairs(records)
    if len(pairs) < 2:
        raise ValueError("SAE sparse steering requires at least two same-question contrast pairs")
    positive = torch.stack(
        [capture_prompt_final_states(model, tokenizer, item[0].full_prompt) for item in pairs]
    )
    negative = torch.stack(
        [capture_prompt_final_states(model, tokenizer, item[1].full_prompt) for item in pairs]
    )
    if positive.shape != negative.shape or positive.ndim != 3:
        raise ValueError("contrast states must share [pairs,layers,hidden] shape")

    directions: list[torch.Tensor] = []
    layer_audits: dict[str, object] = {}
    for layer in config.layers:
        sae, audit = _train_one_sae(unlabeled_states[layer], config=config, layer=layer, device=device)
        positive_features = _encode(positive[:, layer, :], sae)
        negative_features = _encode(negative[:, layer, :], sae)
        feature_delta = (positive_features - negative_features).mean(dim=0)
        n_features = feature_delta.numel()
        n_keep = max(1, math.ceil(n_features * config.feature_keep_fraction))
        selected = torch.topk(feature_delta.abs(), k=n_keep, sorted=False).indices
        sparse_delta = torch.zeros_like(feature_delta)
        sparse_delta[selected] = feature_delta[selected]
        # Decoder rows map sparse feature activations back to residual space.
        direction = (sparse_delta @ sae["decoder"].float()) * sae["scale"].float()
        if not torch.isfinite(direction).all() or torch.linalg.vector_norm(direction) <= 0:
            raise ValueError(f"SAE sparse direction at layer {layer} is invalid")
        directions.append(direction.cpu())
        layer_audits[str(layer)] = {
            **audit,
            "selected_feature_count": n_keep,
            "selected_feature_indices_sha256": hashlib.sha256(
                selected.cpu().numpy().tobytes()
            ).hexdigest(),
            "feature_delta_l2": float(torch.linalg.vector_norm(feature_delta).item()),
            "direction_raw_l2": float(torch.linalg.vector_norm(direction).item()),
        }
    matrix = torch.zeros((positive.shape[1], positive.shape[2]), dtype=torch.float32)
    for layer, direction in zip(config.layers, directions, strict=True):
        matrix[layer] = direction

    metadata: dict[str, object] = {
        "protocol": "tacl-11241-sae-sparse-activation-v1",
        "method": "sae_sparse_activation",
        "claim_boundary": (
            "target-model SAE-space sparse-activation baseline; not an exact reproduction of "
            "a third-party pretrained-SAE release"
        ),
        "source_ids": list(ids),
        "calibration_run": str(calibration.resolve()),
        "calibration_config_sha256": sha256_file(calibration / "config.yaml"),
        "calibration_labels_sha256": sha256_file(calibration / "labels.parquet"),
        "contrast_run": str(contrast.resolve()),
        "contrast_config_sha256": sha256_file(contrast / "config.yaml"),
        "contrast_labels_sha256": sha256_file(contrast / "labels.parquet"),
        "contrast_completions_sha256": (
            sha256_file(contrast / "completions.parquet")
            if (contrast / "completions.parquet").is_file()
            else None
        ),
        "sae_training_uses_correctness_labels": False,
        "contrast_feature_selection_uses_same_question_correct_incorrect_pairs": True,
        "same_question_pairs": [_pair_metadata(pair) for pair in pairs],
        "training": config.to_dict(),
        "layer_audits": layer_audits,
        "shape": list(matrix.shape),
        "evaluation_normalization": "direction RMS=1 before alpha",
    }
    return {"sae_sparse_activation": matrix}, metadata
