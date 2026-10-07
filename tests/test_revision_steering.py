"""Regression tests for the TACL-11241 major-revision steering protocol."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from geoprobe.extractors import Trajectory, save_trajectory
from geoprobe.revision import RevisionSplit, paired_binary_summary
from geoprobe.revision.behavior import (
    distinct_ngram_ratio,
    marker_position,
    repeated_ngram_fraction,
    summarize_text,
)
from geoprobe.revision.crosssteer import compute_crosssteer_direction
from geoprobe.steering import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    LinearDecaySchedule,
    PrefixSchedule,
    SteeringHook,
    match_l2_norm,
    matched_norm_random_direction,
    mean_difference_direction,
    normalize_direction_rms,
    opposite_direction,
    paired_activation_addition_direction,
    sparse_topk_direction,
)


class _IdentityBlock(nn.Module):
    def forward(self, hidden):
        return (hidden, None)


class _Backbone(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_IdentityBlock() for _ in range(n_layers)])


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Hooks infer device/dtype from parameters, like a real HuggingFace model.
        self.anchor = nn.Parameter(torch.zeros(1))
        self.model = _Backbone()


def test_schedules_are_declared_and_validate_steps():
    assert ConstantSchedule()(0) == 1.0
    assert PrefixSchedule(2)(0) == 1.0
    assert PrefixSchedule(2)(1) == 1.0
    assert PrefixSchedule(2)(2) == 0.0
    assert LinearDecaySchedule(4)(0) == 1.0
    assert LinearDecaySchedule(4)(1) == pytest.approx(0.75)
    assert LinearDecaySchedule(4)(4) == 0.0
    assert ExponentialDecaySchedule(2.0)(2) == pytest.approx(np.exp(-1.0))

    with pytest.raises(ValueError):
        PrefixSchedule(-1)
    with pytest.raises(ValueError):
        LinearDecaySchedule(0)
    with pytest.raises(ValueError):
        ExponentialDecaySchedule(0)
    with pytest.raises(ValueError):
        ConstantSchedule()(-1)


def test_last_position_hook_does_not_modify_earlier_prompt_positions():
    model = _Model()
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    vector = torch.tensor([1.0, -2.0, 0.5, 3.0])

    with SteeringHook(
        model,
        layer_idx=1,
        vector=vector,
        alpha=2.0,
        position_mode="last",
    ) as hook:
        steered = model.model.layers[0](hidden)[0]
        assert hook.generation_step == 1

    assert torch.equal(steered[:, :-1, :], hidden[:, :-1, :])
    assert torch.allclose(steered[:, -1, :], hidden[:, -1, :] + 2.0 * vector)


def test_schedule_is_advanced_once_per_hooked_forward():
    model = _Model()
    hidden = torch.zeros(1, 2, 4)
    vector = torch.ones(4)

    with SteeringHook(
        model,
        layer_idx=1,
        vector=vector,
        schedule=PrefixSchedule(1),
        position_mode="last",
    ) as hook:
        first = model.model.layers[0](hidden)[0]
        second = model.model.layers[0](hidden)[0]
        assert hook.generation_step == 2

    assert torch.allclose(first[:, -1, :], torch.ones(1, 4))
    assert torch.equal(second, hidden)


def test_vector_primitives_have_auditable_geometry():
    positive = torch.tensor([[3.0, 2.0, 1.0], [5.0, 4.0, 3.0]])
    negative = torch.tensor([[1.0, 0.0, -1.0], [3.0, 2.0, 1.0]])
    expected = torch.tensor([2.0, 2.0, 2.0])

    assert torch.equal(mean_difference_direction(positive, negative), expected)
    assert torch.equal(paired_activation_addition_direction(positive, negative), expected)

    reference = torch.tensor([3.0, 4.0, 0.0])
    matched = match_l2_norm(torch.tensor([1.0, 1.0, 1.0]), reference)
    assert torch.linalg.vector_norm(matched).item() == pytest.approx(5.0)

    normalized = normalize_direction_rms(torch.tensor([3.0, 4.0, 0.0]), rms=2.0)
    assert torch.linalg.vector_norm(normalized).item() == pytest.approx(2.0 * np.sqrt(3))

    sparse = sparse_topk_direction(torch.tensor([1.0, -4.0, 3.0, 2.0]), keep_fraction=0.5)
    assert torch.count_nonzero(sparse).item() == 2
    assert torch.equal(sparse, torch.tensor([0.0, -4.0, 3.0, 0.0]))

    random_one = matched_norm_random_direction(reference, seed=7)
    random_two = matched_norm_random_direction(reference, seed=7)
    assert torch.equal(random_one, random_two)
    assert torch.linalg.vector_norm(random_one).item() == pytest.approx(5.0)
    assert torch.equal(opposite_direction(reference), -reference)


def test_vector_primitives_reject_incompatible_inputs():
    with pytest.raises(ValueError):
        mean_difference_direction(torch.ones(2, 3), torch.ones(2, 4))
    with pytest.raises(ValueError):
        paired_activation_addition_direction(torch.ones(2, 3), torch.ones(3, 3))
    with pytest.raises(ValueError):
        normalize_direction_rms(torch.ones(3), rms=0.0)
    with pytest.raises(ValueError):
        sparse_topk_direction(torch.ones(3), keep_fraction=0.0)
    with pytest.raises(ValueError):
        match_l2_norm(torch.zeros(3), torch.ones(3))


def test_revision_split_blocks_leakage_and_exposes_formal_roles():
    split = RevisionSplit.gsm8k_formal()
    assert split.role_for(0) == "source_train"
    assert split.role_for(100) == "validation"
    assert split.role_for(200) == "locked_test"
    assert split.role_for(300) == "replication_reserve"
    split.assert_direction_ids([0, 1, 99])

    with pytest.raises(ValueError, match="overlap"):
        RevisionSplit((0,), (0,), (1,), (2,))
    with pytest.raises(ValueError, match="leaked"):
        split.assert_direction_ids([0, 100])
    with pytest.raises(ValueError, match="no IDs"):
        split.assert_direction_ids([])


def test_paired_binary_summary_reports_effect_and_exact_test():
    summary = paired_binary_summary(
        np.array([0, 1, 0, 1]),
        np.array([1, 1, 1, 0]),
        bootstrap_reps=1_000,
        seed=17,
    )
    assert summary.n == 4
    assert summary.baseline_accuracy == pytest.approx(0.5)
    assert summary.method_accuracy == pytest.approx(0.75)
    assert summary.delta == pytest.approx(0.25)
    assert summary.repairs == 2
    assert summary.breaks == 1
    assert summary.exact_sign_p_value == pytest.approx(1.0)
    assert summary.bootstrap_ci_low <= summary.delta <= summary.bootstrap_ci_high


def test_paired_binary_summary_handles_no_discordant_pairs_and_rejects_bad_data():
    summary = paired_binary_summary(np.array([0, 1]), np.array([0, 1]), bootstrap_reps=50)
    assert summary.exact_sign_p_value == 1.0
    assert summary.repairs == 0
    assert summary.breaks == 0

    with pytest.raises(ValueError):
        paired_binary_summary(np.array([0, 2]), np.array([0, 1]))
    with pytest.raises(ValueError):
        paired_binary_summary(np.array([0]), np.array([0, 1]))
    with pytest.raises(ValueError):
        paired_binary_summary(np.array([]), np.array([]))


def test_crosssteer_direction_uses_only_declared_source_ids(tmp_path):
    run = tmp_path / "run"
    trajectories = run / "trajectories"
    trajectories.mkdir(parents=True)
    labels = []
    for sample_id, correct in ((0, True), (1, False), (2, True)):
        value = 2.0 if correct else -1.0
        trajectory = Trajectory(
            sample_id=sample_id,
            sample_idx=0,
            hidden_states=torch.full((3, 2, 4), value, dtype=torch.float32),
            generated_token_ids=torch.zeros(3, dtype=torch.int64),
            generated_text="",
            prompt="",
            prompt_len=0,
            sequence_logprob=0.0,
            model_id="fake",
            dtype="float32",
        )
        save_trajectory(trajectory, trajectories / f"sample_{sample_id:04d}_idx_0.pt")
        labels.append({"sample_id": sample_id, "sample_idx": 0, "correct": correct})
    pd.DataFrame(labels).to_parquet(run / "labels.parquet")

    split = RevisionSplit((0, 1), (10,), (2,), (11,))
    direction, metadata = compute_crosssteer_direction(run, [0, 1], split=split)
    assert direction.shape == (2, 4)
    assert torch.equal(direction, torch.full((2, 4), 3.0))
    assert metadata["source_ids"] == [0, 1]

    with pytest.raises(ValueError, match="leaked"):
        compute_crosssteer_direction(run, [0, 2], split=split)


def test_behavior_controls_measure_surface_effects_without_labels():
    tokens = "a b a b c".split()
    assert distinct_ngram_ratio(tokens, 2) == pytest.approx(0.75)
    assert repeated_ngram_fraction(tokens, 2) == pytest.approx(0.25)
    assert marker_position(tokens, ("b",)) == 1
    summary = summarize_text(r"work work work \boxed{3}")
    assert summary["text_tokens_whitespace"] == 4
    assert summary["repeated_4gram_fraction"] == 0.0
    assert summary["answer_marker_position"] == 3
    assert summary["answer_marker_relative_position"] == pytest.approx(0.75)

    with pytest.raises(ValueError):
        distinct_ngram_ratio(tokens, 0)
    with pytest.raises(ValueError):
        repeated_ngram_fraction(tokens, -1)


def test_geovote_length_residualizer_and_same_pool_outcomes():
    from geoprobe.revision.geovote import (
        LinearLengthResidualizer,
        correlation_with_length,
        same_pool_outcomes,
    )

    frame = pd.DataFrame(
        [
            # question 0: raw geometry favors an incorrect long candidate;
            # residualization removes the declared length trend.
            {
                "sample_id": 0,
                "sample_idx": 0,
                "pred": "1",
                "correct": True,
                "n_gen_tokens": 10,
                "sequence_logprob": -1.0,
                "geo_value": 5.0,
                "geo_conf": -5.0,
            },
            {
                "sample_id": 0,
                "sample_idx": 1,
                "pred": "2",
                "correct": False,
                "n_gen_tokens": 100,
                "sequence_logprob": -2.0,
                "geo_value": 1.0,
                "geo_conf": -1.0,
            },
            {
                "sample_id": 1,
                "sample_idx": 0,
                "pred": "3",
                "correct": True,
                "n_gen_tokens": 20,
                "sequence_logprob": -1.0,
                "geo_value": 4.0,
                "geo_conf": -4.0,
            },
            {
                "sample_id": 1,
                "sample_idx": 1,
                "pred": "4",
                "correct": False,
                "n_gen_tokens": 200,
                "sequence_logprob": -2.0,
                "geo_value": 1.0,
                "geo_conf": -1.0,
            },
        ]
    )
    residualizer = LinearLengthResidualizer.fit(frame)
    frame["residual"] = residualizer.residualize(frame)
    # The fixture orients geo_conf = -geo_value, i.e., the frozen sign is "min";
    # the historical vote therefore weights candidates inversely by geo_value.
    outcomes = same_pool_outcomes(frame, score_column="geo_conf", historical_sign="min")
    assert outcomes.shape[0] == 2
    assert outcomes["score_pick"].tolist() == [False, False]
    assert outcomes["majority"].tolist() == [True, True]
    assert outcomes["historical_geovote"].tolist() == [False, False]
    assert correlation_with_length(frame, "geo_conf") > 0.9
    assert abs(correlation_with_length(frame, "residual")) < abs(
        correlation_with_length(frame, "geo_conf")
    )

    with pytest.raises(ValueError):
        LinearLengthResidualizer.fit(frame.iloc[0:0])


class _Encoding(dict):
    def to(self, device):
        return _Encoding({key: value.to(device) for key, value in self.items()})


class _ToyTokenizer:
    def __call__(self, prompt, return_tensors):
        del return_tensors
        width = 2 if prompt == "short" else 4
        return _Encoding(
            {
                "input_ids": torch.arange(width, dtype=torch.long).reshape(1, width),
                "attention_mask": torch.ones(1, width, dtype=torch.long),
            }
        )


class _ActivationOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _ActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache, return_dict):
        assert output_hidden_states is True
        assert use_cache is False
        assert return_dict is True
        del attention_mask
        base = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 3)
        return _ActivationOutput((base, base + 10.0))


def test_prompt_final_capture_uses_true_prompt_final_state_and_aligned_pairs():
    from geoprobe.revision.activation_capture import (
        capture_prompt_final_states,
        capture_prompt_pairs,
    )

    model = _ActivationModel()
    tokenizer = _ToyTokenizer()
    states = capture_prompt_final_states(model, tokenizer, "long")
    assert states.shape == (2, 3)
    assert torch.equal(states[0], torch.full((3,), 3.0))
    assert torch.equal(states[1], torch.full((3,), 13.0))

    positive, negative = capture_prompt_pairs(model, tokenizer, ["short"], ["long"])
    assert positive.shape == negative.shape == (1, 2, 3)
    assert torch.equal(positive[0, 0], torch.full((3,), 1.0))
    assert torch.equal(negative[0, 0], torch.full((3,), 3.0))
    with pytest.raises(ValueError, match="equal length"):
        capture_prompt_pairs(model, tokenizer, ["short"], [])


def test_vector_registry_requires_completed_train_only_runs_and_fingerprints_artifacts(tmp_path):
    from geoprobe.revision.vector_registry import build_vector_registry, save_vector_registry

    def make_run(name, positive, negative):
        run = tmp_path / name
        (run / "trajectories").mkdir(parents=True)
        (run / "DONE").write_text("ok\n")
        (run / "config.yaml").write_text("model: fake\n")
        rows = []
        for sample_id, value, correct in ((0, positive, True), (1, negative, False)):
            trajectory = Trajectory(
                sample_id=sample_id,
                sample_idx=0,
                hidden_states=torch.full((2, 2, 3), value),
                generated_token_ids=torch.zeros(2, dtype=torch.int64),
                generated_text="",
                prompt="",
                prompt_len=0,
                sequence_logprob=0.0,
                model_id="fake",
                dtype="float32",
            )
            save_trajectory(trajectory, run / "trajectories" / f"sample_{sample_id:04d}_idx_0.pt")
            rows.append({"sample_id": sample_id, "sample_idx": 0, "correct": correct})
        pd.DataFrame(rows).to_parquet(run / "labels.parquet")
        return run

    source = make_run("source", 2.0, -1.0)
    target = make_run("target", 4.0, -2.0)
    split = RevisionSplit((0, 1), (2,), (3,), (4,))
    vectors, metadata = build_vector_registry(source, target, [0, 1], split=split)
    assert torch.equal(vectors["crosssteer_source"], torch.full((2, 3), 3.0))
    assert torch.equal(vectors["target_calibrated"], torch.full((2, 3), 6.0))
    registry = tmp_path / "registry"
    save_vector_registry(vectors, metadata, registry)
    assert (registry / "crosssteer_source.pt").exists()
    assert (registry / "target_calibrated.pt").exists()
    manifest = (registry / "manifest.json").read_text()
    assert "vector_sha256" in manifest

    (target / "DONE").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        build_vector_registry(source, target, [0, 1], split=split)


def test_decode_last_hook_skips_multitoken_prefill_then_steers_decode_positions():
    model = _Model()
    vector = torch.tensor([2.0, 0.0, 0.0, 0.0])
    prefill = torch.zeros(1, 3, 4)
    decode = torch.zeros(1, 1, 4)

    with SteeringHook(
        model,
        layer_idx=1,
        vector=vector,
        position_mode="decode_last",
    ) as hook:
        untouched = model.model.layers[0](prefill)[0]
        steered = model.model.layers[0](decode)[0]
        assert hook.generation_step == 1

    assert torch.equal(untouched, prefill)
    assert torch.allclose(steered, decode + vector)


def test_signature_bootstrap_reports_distance_and_nearest_neighbor_probabilities():
    from geoprobe.revision.signature_stability import SignatureData, bootstrap_topology

    cells = (("metric_a", 1), ("metric_b", 2))
    ids = np.arange(6)
    incorrect = np.array([False, True, False, True, False, True])
    left = SignatureData(
        "left",
        ids,
        incorrect,
        np.array([[0, 1], [3, 2], [1, 1], [4, 3], [2, 0], [5, 4]], dtype=float),
        cells,
    )
    near = SignatureData(
        "near",
        ids,
        incorrect,
        np.array([[0, 1.2], [3, 2.2], [1, 1.1], [4, 3.1], [2, 0.1], [5, 4.1]], dtype=float),
        cells,
    )
    far = SignatureData(
        "far",
        ids,
        incorrect,
        np.array([[5, 4], [2, 3], [4, 4], [1, 2], [3, 5], [0, 1]], dtype=float),
        cells,
    )
    distances, neighbors, signature_cells = bootstrap_topology(
        [left, near, far], n_bootstrap=20, seed=1
    )
    assert distances.shape[0] == 3
    assert signature_cells.shape[0] == 6
    assert neighbors.groupby("model")["bootstrap_probability"].sum().eq(1.0).all()
    assert set(neighbors["model"]) == {"left", "near", "far"}


def test_validation_grid_requires_tracked_vectors_and_computes_paired_summary(tmp_path):
    from scripts.revision_steering_validation_grid import _registry_source_ids, _summarize

    vector_one = tmp_path / "crosssteer_source.pt"
    vector_two = tmp_path / "target_calibrated.pt"
    torch.save(torch.ones(3, 4), vector_one)
    torch.save(torch.ones(3, 4), vector_two)
    (tmp_path / "manifest.json").write_text(json.dumps({"source_ids": [0, 1]}))
    assert _registry_source_ids([vector_one, vector_two]) == {0, 1}

    rows = pd.DataFrame(
        [
            {
                "method": "baseline",
                "layer": 0,
                "alpha": 0.0,
                "sample_id": 2,
                "correct": False,
                "text_tokens": 4,
                "repeated_4gram_fraction": 0.0,
            },
            {
                "method": "baseline",
                "layer": 0,
                "alpha": 0.0,
                "sample_id": 3,
                "correct": True,
                "text_tokens": 6,
                "repeated_4gram_fraction": 0.0,
            },
            {
                "method": "cross",
                "layer": 1,
                "alpha": 0.1,
                "sample_id": 2,
                "correct": True,
                "text_tokens": 5,
                "repeated_4gram_fraction": 0.1,
            },
            {
                "method": "cross",
                "layer": 1,
                "alpha": 0.1,
                "sample_id": 3,
                "correct": True,
                "text_tokens": 7,
                "repeated_4gram_fraction": 0.2,
            },
        ]
    )
    summary = _summarize(rows)
    assert summary.shape[0] == 1
    assert summary.loc[0, "delta"] == pytest.approx(0.5)
    assert summary.loc[0, "repairs"] == 1
    assert summary.loc[0, "breaks"] == 0

    (tmp_path / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="untracked"):
        _registry_source_ids([vector_one])


def test_preregistered_validation_selection_uses_guard_then_fixed_ties():
    from geoprobe.revision.validation_selection import select_validation_cell

    summary = pd.DataFrame(
        [
            # Highest accuracy, but it is ineligible for excessive repetition.
            {
                "method": "cross",
                "layer": 8,
                "alpha": 0.05,
                "method_accuracy": 0.90,
                "delta": 0.2,
                "mean_text_tokens": 9.0,
                "mean_repeated_4gram_fraction": 0.07,
            },
            # Two tied eligible cells: smaller alpha wins before lower layer.
            {
                "method": "cross",
                "layer": 20,
                "alpha": 0.10,
                "method_accuracy": 0.80,
                "delta": 0.1,
                "mean_text_tokens": 9.0,
                "mean_repeated_4gram_fraction": 0.01,
            },
            {
                "method": "cross",
                "layer": 14,
                "alpha": 0.05,
                "method_accuracy": 0.80,
                "delta": 0.1,
                "mean_text_tokens": 11.0,
                "mean_repeated_4gram_fraction": 0.02,
            },
        ]
    )
    result = select_validation_cell(
        summary,
        method="cross",
        baseline_mean_text_tokens=10.0,
        baseline_mean_repeated_4gram_fraction=0.0,
    )
    assert result.eligible
    assert result.layer == 14
    assert result.alpha == pytest.approx(0.05)
    assert result.cells_passing_behavior_guard == 2


def test_preregistered_validation_selection_can_declare_all_cells_ineligible():
    from geoprobe.revision.validation_selection import select_validation_cell

    summary = pd.DataFrame(
        [
            {
                "method": "cross",
                "layer": 8,
                "alpha": 0.05,
                "method_accuracy": 0.9,
                "delta": 0.1,
                "mean_text_tokens": 16.0,
                "mean_repeated_4gram_fraction": 0.0,
            }
        ]
    )
    result = select_validation_cell(
        summary,
        method="cross",
        baseline_mean_text_tokens=10.0,
        baseline_mean_repeated_4gram_fraction=0.0,
    )
    assert not result.eligible
    assert result.reason == "no_cell_passed_preregistered_behavior_guard"


def test_prompt_contrast_pairing_and_registry_are_train_only_and_deterministic(
    tmp_path, monkeypatch
):
    import geoprobe.revision.prompt_contrast_registry as registry
    from geoprobe.revision.prompt_contrast_registry import (
        ContrastCompletion,
        build_prompt_contrast_vectors,
        same_question_pairs,
        save_prompt_contrast_registry,
    )

    records = [
        ContrastCompletion(0, 0, True, 11, 20, "p0"),
        ContrastCompletion(0, 1, False, 10, 20, "p1"),
        ContrastCompletion(1, 0, True, 50, 20, "p2"),
        ContrastCompletion(1, 1, False, 60, 20, "p3"),
        # A same-class-only question must not become an unaligned pair.
        ContrastCompletion(2, 0, True, 7, 20, "p4"),
    ]
    pairs = same_question_pairs(records)
    assert [(left.sample_id, right.sample_id) for left, right in pairs] == [(0, 0), (1, 1)]

    run = tmp_path / "target"
    (run / "trajectories").mkdir(parents=True)
    (run / "DONE").write_text("complete\n")
    (run / "config.yaml").write_text("model: fake\n")
    labels = []
    for item in records:
        trajectory = Trajectory(
            sample_id=item.sample_id,
            sample_idx=item.sample_idx,
            hidden_states=torch.ones(2, 2, 3),
            generated_token_ids=torch.ones(item.n_gen_tokens, dtype=torch.int64),
            generated_text=f" completion-{item.sample_id * 10 + item.sample_idx}",
            prompt=f"prompt-{item.sample_id * 10 + item.sample_idx}",
            prompt_len=item.prompt_len,
            sequence_logprob=0.0,
            model_id="fake",
            dtype="float32",
        )
        save_trajectory(
            trajectory,
            run / "trajectories" / f"sample_{item.sample_id:04d}_idx_{item.sample_idx}.pt",
        )
        labels.append(
            {
                "sample_id": item.sample_id,
                "sample_idx": item.sample_idx,
                "correct": item.correct,
                "n_gen_tokens": item.n_gen_tokens,
            }
        )
    pd.DataFrame(labels).to_parquet(run / "labels.parquet")

    def fake_capture(_model, _tokenizer, prompt):
        value = float(int(prompt.rsplit("-", 1)[1]))
        return torch.full((2, 3), value)

    monkeypatch.setattr(registry, "capture_prompt_final_states", fake_capture)
    split = RevisionSplit((0, 1, 2), (3,), (4,), (5,))
    vectors, metadata = build_prompt_contrast_vectors(
        object(),
        object(),
        target_run=run,
        source_ids=split.source_train,
        split=split,
        sparse_keep_fraction=0.10,
    )
    # Same-question pairs are (0,1) and (2,3): mean(pos-neg) = (-1 + -1) / 2.
    assert torch.equal(vectors["caa_target_prompt_final"], torch.full((2, 3), -1.0))
    assert vectors["actadd_target_prompt_final"].shape == (2, 3)
    assert (vectors["sparse_caa_coordinate_10pct"] != 0).sum() == 2
    assert metadata["caa"]["n_pairs"] == 2
    assert metadata["target_completion_source"] == "trajectory_artifacts"
    assert metadata["target_completions_sha256"] is None
    assert metadata["source_ids"] == [0, 1, 2]

    out = tmp_path / "registry"
    save_prompt_contrast_registry(vectors, metadata, out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["protocol"] == "tacl-11241-prompt-contrast-registry-v1"
    assert set(manifest["vector_sha256"]) == set(vectors)


def test_relative_hidden_rms_mode_preserves_a_fixed_relative_perturbation():
    model = _Model()
    hidden = torch.full((1, 1, 4), 3.0)
    vector = torch.ones(4)
    with SteeringHook(
        model,
        layer_idx=1,
        vector=vector,
        alpha=0.2,
        position_mode="decode_last",
        injection_mode="relative_hidden_rms",
    ):
        output = model.model.layers[0](hidden)[0]
    # Coordinate RMS of the residual is 3, so the unit-RMS vector gets a
    # per-coordinate addition of alpha * 3 = 0.6.
    assert torch.allclose(output, hidden + 0.6)

    with pytest.raises(ValueError, match="injection_mode"):
        SteeringHook(model, layer_idx=1, vector=vector, injection_mode="invalid")


def test_prompt_contrast_registry_accepts_a_lightweight_completion_pool(tmp_path, monkeypatch):
    import geoprobe.revision.prompt_contrast_registry as registry
    from geoprobe.revision.prompt_contrast_registry import build_prompt_contrast_vectors

    run = tmp_path / "pool"
    run.mkdir()
    (run / "DONE").write_text("complete\n")
    (run / "config.yaml").write_text("model: fake\n")
    rows = []
    for sample_id in (0, 1):
        for sample_idx, correct in ((0, True), (1, False)):
            rows.append(
                {
                    "sample_id": sample_id,
                    "sample_idx": sample_idx,
                    "correct": correct,
                    "n_gen_tokens": 10 + sample_idx,
                    "prompt": f"prompt-{sample_id}",
                    "generated_text": f" completion-{sample_id * 10 + sample_idx}",
                    "prompt_len": 8,
                }
            )
    frame = pd.DataFrame(rows)
    frame.drop(columns=["prompt", "generated_text", "prompt_len"]).to_parquet(
        run / "labels.parquet"
    )
    frame.to_parquet(run / "completions.parquet")
    monkeypatch.setattr(
        registry,
        "capture_prompt_final_states",
        lambda _model, _tokenizer, prompt: torch.full((2, 3), float(int(prompt.rsplit("-", 1)[1]))),
    )
    split = RevisionSplit((0, 1), (2,), (3,), (4,))
    vectors, metadata = build_prompt_contrast_vectors(
        object(),
        object(),
        target_run=run,
        source_ids=split.source_train,
        split=split,
        sparse_keep_fraction=0.10,
    )
    assert torch.equal(vectors["caa_target_prompt_final"], torch.full((2, 3), -1.0))
    assert metadata["caa"]["n_pairs"] == 2
    assert metadata["target_completion_source"] == "completions.parquet"
    assert metadata["target_completions_sha256"]


def test_validation_grid_signature_records_injection_mode():
    from scripts.revision_steering_validation_grid import _run_signature

    args = Namespace(
        target_model="fake",
        target_source="huggingface",
        target_device="cuda:0",
        eval_start=100,
        eval_count=100,
        layers=[8],
        alphas=[0.1],
        schedule="constant",
        schedule_parameter=64.0,
        injection_mode="absolute",
        max_new_tokens=512,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        seed=11241,
    )
    signature = _run_signature(args, {"cross": Path("/tmp/cross.pt")})
    assert signature["position_mode"] == "decode_last"
    assert signature["injection_mode"] == "absolute"
    assert signature["generation"]["do_sample"] is False
    assert signature["generation"]["seed_rule"] is None


def test_locked_run_spec_requires_frozen_eligible_full_validation_selection(tmp_path):
    from geoprobe.revision.locked_run import load_locked_run_spec

    vector = tmp_path / "vector.pt"
    torch.save(torch.ones(2, 3), vector)
    run = tmp_path / "validation"
    run.mkdir()
    (run / "resolved_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "vectors": {"cross": str(vector)},
                "source_ids": list(range(100)),
                "position_mode": "decode_last",
                "schedule": "constant",
                "max_new_tokens": 512,
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "selection_status": "locked_before_locked_test_generation",
                "methods": [
                    {
                        "method": "cross",
                        "run_dir": str(run),
                        "evaluation_ids": list(range(100, 200)),
                        "decision": {
                            "eligible": True,
                            "reason": "selected_by_preregistered_validation_rule",
                            "layer": 14,
                            "alpha": 0.05,
                            "validation_accuracy": 0.7,
                            "validation_delta": 0.1,
                        },
                    }
                ],
            }
        )
    )
    spec = load_locked_run_spec(selection, "cross")
    assert spec.layer == 14
    assert spec.alpha == pytest.approx(0.05)
    assert spec.locked_eval_start == 200
    assert spec.source_ids == tuple(range(100))

    payload = json.loads(selection.read_text())
    payload["methods"][0]["evaluation_ids"] = list(range(100, 199))
    selection.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="entire frozen validation"):
        load_locked_run_spec(selection, "cross")


def test_validation_family_report_retains_all_frozen_cells_and_rejects_tampering(tmp_path):
    from geoprobe.revision.validation_report import (
        build_validation_family_report,
        sha256_file,
        write_validation_family_report,
    )
    from geoprobe.revision.validation_selection import select_validation_cell

    entries = []
    for method, accuracies in (("cross", (0.61, 0.66)), ("caa", (0.62, 0.64))):
        run = tmp_path / method
        run.mkdir()
        summary = pd.DataFrame(
            [
                {
                    "method": method,
                    "layer": 8,
                    "alpha": 0.05,
                    "mean_text_tokens": 12.0,
                    "mean_repeated_4gram_fraction": 0.01,
                    "n": 100,
                    "baseline_accuracy": 0.58,
                    "method_accuracy": accuracies[0],
                    "delta": accuracies[0] - 0.58,
                    "repairs": 8,
                    "breaks": 5,
                    "bootstrap_ci_low": -0.01,
                    "bootstrap_ci_high": 0.08,
                    "exact_sign_p_value": 0.3,
                },
                {
                    "method": method,
                    "layer": 14,
                    "alpha": 0.1,
                    "mean_text_tokens": 12.5,
                    "mean_repeated_4gram_fraction": 0.02,
                    "n": 100,
                    "baseline_accuracy": 0.58,
                    "method_accuracy": accuracies[1],
                    "delta": accuracies[1] - 0.58,
                    "repairs": 10,
                    "breaks": 3,
                    "bootstrap_ci_low": 0.0,
                    "bootstrap_ci_high": 0.11,
                    "exact_sign_p_value": 0.1,
                },
            ]
        )
        summary_path = run / "summary.csv"
        summary.to_csv(summary_path, index=False)
        decision = select_validation_cell(
            summary,
            method=method,
            baseline_mean_text_tokens=10.0,
            baseline_mean_repeated_4gram_fraction=0.01,
        ).to_dict()
        entries.append(
            {
                "method": method,
                "run_dir": str(run),
                "decision": decision,
                "evidence_hashes": {"summary.csv": sha256_file(summary_path)},
            }
        )

    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "selection_status": "locked_before_locked_test_generation",
                "methods": entries,
            }
        )
    )
    report, manifest = build_validation_family_report(selection)
    assert manifest["n_methods"] == 2
    assert manifest["n_cells"] == 4
    assert manifest["selected_cells"] == 2
    assert report["behavior_guard_passed"].all()
    assert report[report["selected"]]["layer"].tolist() == [14, 14]

    out = tmp_path / "report"
    write_validation_family_report(selection, out)
    assert (out / "validation_complete_summary.csv").is_file()
    assert "cross & 14" in (out / "validation_selected_rows.tex").read_text()
    assert (out / "validation_family_report_manifest.json").is_file()

    summary = pd.read_csv(tmp_path / "cross" / "summary.csv")
    summary.loc[0, "method_accuracy"] = 0.99
    summary.to_csv(tmp_path / "cross" / "summary.csv", index=False)
    with pytest.raises(ValueError, match="hash differs"):
        build_validation_family_report(selection)


def test_locked_run_preserves_frozen_sampled_decoder_for_official_context(tmp_path):
    from geoprobe.revision.locked_run import load_locked_run_spec
    from scripts.revision_launch_locked_selected import _command

    vector = tmp_path / "vector.pt"
    torch.save(torch.ones(3, 4), vector)
    run = tmp_path / "validation"
    run.mkdir()
    (run / "resolved_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "vectors": {"cross": str(vector)},
                "source_ids": list(range(100)),
                "position_mode": "decode_last",
                "injection_mode": "absolute",
                "schedule": "constant",
                "max_new_tokens": 32768,
                "generation": {
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "base_seed": 11241,
                    "seed_rule": "seed * 100003 + sample_id * 1009",
                    "num_beams": 1,
                    "use_cache": True,
                    "eos_token_id": "model.generation_config.eos_token_id",
                },
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "selection_status": "locked_before_locked_test_generation",
                "methods": [
                    {
                        "method": "cross",
                        "run_dir": str(run),
                        "evaluation_ids": list(range(100, 200)),
                        "decision": {
                            "eligible": True,
                            "reason": "selected_by_preregistered_validation_rule",
                            "layer": 14,
                            "alpha": 0.05,
                            "validation_accuracy": 0.7,
                            "validation_delta": 0.1,
                        },
                    }
                ],
            }
        )
    )
    spec = load_locked_run_spec(selection, "cross")
    assert spec.max_new_tokens == 32768
    assert spec.do_sample is True
    assert spec.temperature == pytest.approx(0.6)
    assert spec.top_p == pytest.approx(0.95)
    assert spec.base_seed == 11241
    command = _command(
        Namespace(target_model="r1", target_source="huggingface", target_device="cuda:0"),
        spec,
        tmp_path / "locked",
    )
    assert command[command.index("--max-new-tokens") + 1] == "32768"
    assert command[command.index("--temperature") + 1] == "0.6"
    assert command[command.index("--top-p") + 1] == "0.95"
    assert command[command.index("--seed") + 1] == "11241"


def test_all_frozen_launchers_preserve_sampled_decoder_arguments(tmp_path):
    from geoprobe.revision.locked_run import LockedRunSpec
    from scripts.revision_launch_label_efficiency_locked import _command as label_command
    from scripts.revision_launch_locked_selected import _command as locked_command
    from scripts.revision_launch_long_context import _POLICIES
    from scripts.revision_launch_long_context import _command as long_command
    from scripts.revision_launch_ood_selected import _command as math_ood_command
    from scripts.revision_launch_svamp_ood_selected import _command as svamp_ood_command

    spec = LockedRunSpec(
        method="cross",
        vector_path=str(tmp_path / "vector.pt"),
        source_ids=tuple(range(100)),
        layer=14,
        alpha=0.05,
        validation_accuracy=0.7,
        validation_delta=0.1,
        validation_run_dir=str(tmp_path / "validation"),
        max_new_tokens=32768,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        base_seed=11241,
    )
    args = Namespace(target_model="r1", target_source="huggingface", target_device="cuda:0")
    commands = [
        locked_command(args, spec, tmp_path / "locked"),
        long_command(Namespace(**vars(args), budget=32768), spec, _POLICIES[0], tmp_path / "long"),
        math_ood_command(args, spec, tmp_path / "math-ood"),
        svamp_ood_command(args, spec, tmp_path / "svamp-ood"),
        label_command(
            args,
            spec,
            method="target_labels_k005",
            vector_path=str(tmp_path / "label-vector.pt"),
            out=tmp_path / "labels",
        ),
    ]
    for command in commands:
        assert "--do-sample" in command
        assert command[command.index("--temperature") + 1] == "0.6"
        assert command[command.index("--top-p") + 1] == "0.95"
        assert command[command.index("--seed") + 1] == "11241"
        assert command.index("--do-sample") < command.index("--out")


def test_locked_run_rejects_sampled_validation_with_incomplete_decoder_metadata(tmp_path):
    from geoprobe.revision.locked_run import load_locked_run_spec

    vector = tmp_path / "vector.pt"
    torch.save(torch.ones(2, 3), vector)
    run = tmp_path / "validation"
    run.mkdir()
    (run / "resolved_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "vectors": {"cross": str(vector)},
                "source_ids": list(range(100)),
                "position_mode": "decode_last",
                "schedule": "constant",
                "max_new_tokens": 32768,
                "generation": {"do_sample": True, "temperature": 0.6},
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "selection_status": "locked_before_locked_test_generation",
                "methods": [
                    {
                        "method": "cross",
                        "run_dir": str(run),
                        "evaluation_ids": list(range(100, 200)),
                        "decision": {
                            "eligible": True,
                            "reason": "selected_by_preregistered_validation_rule",
                            "layer": 14,
                            "alpha": 0.05,
                            "validation_accuracy": 0.7,
                            "validation_delta": 0.1,
                        },
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="lacks frozen generation metadata"):
        load_locked_run_spec(selection, "cross")


def test_cpu_smoke_demo_builds_a_manifest(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "smoke"
    subprocess.run(
        [sys.executable, "scripts/revision_smoke_demo.py", "--out", str(out)],
        check=True,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        cwd=Path.cwd(),
    )
    manifest = json.loads((out / "SMOKE_OK.json").read_text())
    assert manifest["purpose"].startswith("installation")
    assert manifest["vectors"] == {
        "crosssteer_source": [3, 5],
        "target_calibrated": [3, 5],
    }


def test_all_revision_qwen_generation_paths_preserve_model_multi_eos():
    # Qwen2.5-Instruct finishes bare completions on a second EOS id.  Every
    # revision generation path must source the terminator set from the model,
    # rather than reduce it to tokenizer.eos_token_id.
    paths = (
        "scripts/revision_steering_validation_grid.py",
        "scripts/revision_generate_contrast_pool.py",
        "scripts/revision_crosssteer_smoke.py",
        "src/geoprobe/revision/pooled_trajectory.py",
        "src/geoprobe/extractors/hidden_states.py",
    )
    for relative in paths:
        text = Path(relative).read_text()
        assert "generation_config" in text, relative
        assert "eos_token_id=eos_ids" in text or '"eos_token_id": eos_ids' in text, relative


def test_generation_telemetry_distinguishes_eos_budget_and_other_stops():
    from geoprobe.revision.generation import summarize_generated_tokens

    eos = summarize_generated_tokens(torch.tensor([7, 2]), eos_token_id=2, max_new_tokens=2)
    assert eos.n_generated_tokens == 2
    assert eos.n_content_tokens == 1
    assert eos.stop_reason == "eos"
    assert not eos.truncated

    second_eos = summarize_generated_tokens(
        torch.tensor([7, 3]), eos_token_id=[2, 3], max_new_tokens=2
    )
    assert second_eos.n_content_tokens == 1
    assert second_eos.stop_reason == "eos"
    assert not second_eos.truncated

    budget = summarize_generated_tokens(torch.tensor([7, 8]), eos_token_id=2, max_new_tokens=2)
    assert budget.n_content_tokens == 2
    assert budget.stop_reason == "max_new_tokens"
    assert budget.truncated

    other = summarize_generated_tokens(torch.tensor([7]), eos_token_id=2, max_new_tokens=2)
    assert other.stop_reason == "other"
    assert not other.truncated

    empty = summarize_generated_tokens(
        torch.tensor([], dtype=torch.long), eos_token_id=2, max_new_tokens=2
    )
    assert empty.stop_reason == "empty"

    with pytest.raises(ValueError, match="positive"):
        summarize_generated_tokens(torch.tensor([1]), eos_token_id=2, max_new_tokens=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        summarize_generated_tokens(
            torch.ones((1, 1), dtype=torch.long), eos_token_id=2, max_new_tokens=2
        )


def test_generation_runtime_audit_records_full_model_eos_and_refuses_bad_telemetry(tmp_path: Path):
    from geoprobe.revision.runtime_generation_audit import (
        normalize_token_ids,
        write_generation_runtime_audit,
    )

    run = tmp_path / "run"
    run.mkdir()
    (run / "DONE").write_text("complete\n")
    (run / "resolved_manifest.json").write_text(
        json.dumps(
            {
                "generation": {
                    "eos_token_id": "model.generation_config.eos_token_id",
                }
            }
        )
    )
    rows = [
        {"stop_reason": "eos", "truncated": False},
        {"stop_reason": "max_new_tokens", "truncated": True},
        {"stop_reason": "other", "truncated": False},
    ]
    (run / "per_sample.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = tmp_path / "generation_config.json"
    config.write_text(json.dumps({"eos_token_id": [151645, 151643], "pad_token_id": 151643}))

    output = write_generation_runtime_audit(run, config)
    audit = json.loads(output.read_text())
    assert audit["model_generation_eos_token_ids"] == [151645, 151643]
    assert audit["model_generation_pad_token_id"] == 151643
    assert audit["row_count"] == 3
    assert audit["stop_reason_counts"] == {"eos": 1, "max_new_tokens": 1, "other": 1}
    assert audit["truncation_rate"] == pytest.approx(1 / 3)
    assert write_generation_runtime_audit(run, config) == output
    subprocess.run(
        [
            sys.executable,
            "scripts/revision_audit_generation_runtime.py",
            "--run",
            str(run),
            "--generation-config",
            str(config),
        ],
        check=True,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        cwd=Path.cwd(),
    )

    assert normalize_token_ids(3) == [3]
    with pytest.raises(ValueError, match="non-empty"):
        normalize_token_ids([])
    with pytest.raises(ValueError, match="integers only"):
        normalize_token_ids([3, "4"])

    rows[0]["stop_reason"] = "max_new_tokens"
    rows[0]["truncated"] = True
    (run / "per_sample.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="no EOS-completed rows"):
        write_generation_runtime_audit(run, config)


def test_validation_family_integrity_audit_requires_full_shared_grid_and_baseline(tmp_path: Path):
    from geoprobe.revision.validation_family_integrity import write_validation_family_audit

    def make_run(method: str) -> Path:
        run = tmp_path / method
        run.mkdir()
        (run / "DONE").write_text("complete\n")
        manifest = {
            "protocol": "tacl-11241-frozen-steering-grid-v1",
            "target_model": "target",
            "target_source": "local",
            "eval_start": 10,
            "eval_count": 2,
            "layers": [8],
            "alphas": [0.1, 0.2],
            "schedule": "constant",
            "schedule_parameter": 0.0,
            "position_mode": "decode_last",
            "injection_mode": "absolute",
            "normalization": "direction RMS=1 before alpha",
            "max_new_tokens": 64,
            "generation": {"do_sample": False},
            "prompt_template": "question={question}",
        }
        (run / "resolved_manifest.json").write_text(json.dumps(manifest))
        rows = []
        for sample_id in (10, 11):
            baseline = {
                "method": "baseline",
                "layer": 0,
                "alpha": 0.0,
                "sample_id": sample_id,
                "gold": sample_id,
                "pred": sample_id,
                "correct": True,
                "generated_text": f"baseline-{sample_id}",
                "n_generated_tokens": 3,
                "n_content_tokens": 2,
                "text_tokens": 2,
                "text_tokens_whitespace": 2,
                "stop_reason": "eos",
                "truncated": False,
                "repeated_4gram_fraction": 0.0,
                "answer_marker_position": 1,
                "answer_marker_relative_position": 0.5,
            }
            rows.append(baseline)
            for alpha in (0.1, 0.2):
                rows.append(
                    {
                        "method": method,
                        "layer": 8,
                        "alpha": alpha,
                        "sample_id": sample_id,
                    }
                )
        per_sample = run / "per_sample.jsonl"
        per_sample.write_text("".join(json.dumps(row) + "\n" for row in rows))
        runtime = {
            "row_count": len(rows),
            "per_sample_jsonl_sha256": hashlib.sha256(per_sample.read_bytes()).hexdigest(),
            "model_generation_eos_token_ids": [151645, 151643],
            "truncation_rate": 0.0,
        }
        (run / "generation_runtime_audit.json").write_text(json.dumps(runtime))
        return run

    cross = make_run("crosssteer_source")
    caa = make_run("caa_target_prompt_final")
    output = tmp_path / "integrity.json"
    assert (
        write_validation_family_audit(
            {"crosssteer_source": cross, "caa_target_prompt_final": caa}, output
        )
        == output
    )
    audit = json.loads(output.read_text())
    assert audit["status"] == "passed"
    assert audit["method_count"] == 2
    assert audit["shared_baseline_problem_count"] == 2
    assert audit["shared_generation_eos_token_ids"] == [151645, 151643]

    rows = [json.loads(line) for line in (caa / "per_sample.jsonl").read_text().splitlines()]
    rows[0]["generated_text"] = "mismatched"
    (caa / "per_sample.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="runtime audit hash"):
        write_validation_family_audit(
            {"crosssteer_source": cross, "caa_target_prompt_final": caa}, output
        )


def test_calibration_audit_requires_complete_non_degenerate_non_saturated_labels():
    from geoprobe.revision.capability_audit import audit_calibration_frame

    good = pd.DataFrame(
        {
            "sample_id": list(range(20)),
            "sample_idx": [0] * 20,
            "correct": [True] * 10 + [False] * 10,
            "n_gen_tokens": [32] * 20,
        }
    )
    audited = audit_calibration_frame(
        good,
        name="good",
        expected_ids=range(20),
        max_new_tokens=64,
        min_class_count=5,
        max_budget_hit_rate=0.25,
    )
    assert audited.eligible_for_direction
    assert audited.accuracy == pytest.approx(0.5)
    assert audited.budget_hit_rate == 0.0

    bad = good.copy()
    bad.loc[:, "n_gen_tokens"] = 64
    bad.loc[:, "correct"] = True
    audited_bad = audit_calibration_frame(
        bad,
        name="bad",
        expected_ids=range(20),
        max_new_tokens=64,
        min_class_count=5,
        max_budget_hit_rate=0.25,
    )
    assert not audited_bad.eligible_for_direction
    assert any("too_few_incorrect" in message for message in audited_bad.failures)
    assert any("budget_hit_rate" in message for message in audited_bad.failures)

    missing = good.drop(index=0)
    audited_missing = audit_calibration_frame(
        missing,
        name="missing",
        expected_ids=range(20),
        max_new_tokens=64,
        min_class_count=5,
        max_budget_hit_rate=0.25,
    )
    assert not audited_missing.eligible_for_direction
    assert any("problem_ids_mismatch" in message for message in audited_missing.failures)


def test_validation_selector_requires_identical_shared_target_baselines():
    from scripts.revision_select_validation import _assert_shared_target_protocol

    manifest = {
        "protocol": "tacl-11241-frozen-steering-grid-v1",
        "target_model": "target",
        "target_source": "huggingface",
        "eval_start": 100,
        "eval_count": 2,
        "layers": [8, 14],
        "alphas": [0.05],
        "schedule": "constant",
        "schedule_parameter": 64,
        "position_mode": "decode_last",
        "injection_mode": "absolute",
        "normalization": "direction RMS=1 before alpha",
        "max_new_tokens": 512,
        "prompt_template": "prompt",
    }
    baseline = pd.DataFrame(
        [
            {
                "method": "baseline",
                "sample_id": 100,
                "gold": 1.0,
                "pred": 1.0,
                "correct": True,
                "text_tokens": 3,
                "text_tokens_whitespace": 3,
                "distinct_2gram_ratio": 1.0,
                "distinct_4gram_ratio": 1.0,
                "repeated_4gram_fraction": 0.0,
                "answer_marker_position": 2,
                "answer_marker_relative_position": 2 / 3,
            },
            {
                "method": "baseline",
                "sample_id": 101,
                "gold": 2.0,
                "pred": 0.0,
                "correct": False,
                "text_tokens": 4,
                "text_tokens_whitespace": 4,
                "distinct_2gram_ratio": 1.0,
                "distinct_4gram_ratio": 1.0,
                "repeated_4gram_fraction": 0.0,
                "answer_marker_position": 3,
                "answer_marker_relative_position": 3 / 4,
            },
        ]
    )
    runs = [
        ("a", {"manifest": manifest, "rows": baseline}),
        ("b", {"manifest": manifest.copy(), "rows": baseline.copy()}),
    ]
    audit = _assert_shared_target_protocol(runs)
    assert audit["baseline_problem_ids"] == [100, 101]

    changed = baseline.copy()
    changed.loc[changed["sample_id"] == 101, "pred"] = 2.0
    with pytest.raises(ValueError, match="baseline differs"):
        _assert_shared_target_protocol(
            [runs[0], ("b", {"manifest": manifest.copy(), "rows": changed})]
        )

    incompatible = manifest.copy()
    incompatible["max_new_tokens"] = 1024
    with pytest.raises(ValueError, match="does not share"):
        _assert_shared_target_protocol(
            [runs[0], ("b", {"manifest": incompatible, "rows": baseline.copy()})]
        )


def test_holm_adjustment_is_monotone_and_preserves_original_order():
    from geoprobe.revision.stats import holm_adjust

    values = holm_adjust([0.01, 0.04, 0.03])
    assert values.tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([]).tolist() == []
    with pytest.raises(ValueError, match="one-dimensional"):
        holm_adjust([[0.1]])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjust([1.1])


def test_locked_report_loader_requires_selected_one_shot_artifact(tmp_path):
    from geoprobe.revision.locked_run import load_locked_run_spec, sha256_file
    from scripts.revision_build_locked_report import _load_locked

    method = "crosssteer_source"
    validation = tmp_path / "validation"
    validation.mkdir()
    vector = validation / "vector.pt"
    vector.write_bytes(b"frozen-vector")
    validation_manifest = {
        "protocol": "tacl-11241-frozen-steering-grid-v1",
        "position_mode": "decode_last",
        "injection_mode": "absolute",
        "schedule": "constant",
        "max_new_tokens": 512,
        "vectors": {method: str(vector)},
        "source_ids": list(range(100)),
    }
    (validation / "resolved_manifest.json").write_text(json.dumps(validation_manifest))
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "selection_status": "locked_before_locked_test_generation",
                "methods": [
                    {
                        "method": method,
                        "run_dir": str(validation),
                        "source_ids": list(range(100)),
                        "evaluation_ids": list(range(100, 200)),
                        "decision": {
                            "eligible": True,
                            "reason": "selected_by_preregistered_validation_rule",
                            "layer": 14,
                            "alpha": 0.1,
                            "validation_accuracy": 0.6,
                            "validation_delta": 0.1,
                        },
                    }
                ],
            }
        )
    )
    spec = load_locked_run_spec(selection, method)
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "DONE").write_text("complete\n")
    manifest = {
        "protocol": "tacl-11241-frozen-steering-grid-v1",
        "target_model": "target",
        "target_source": "huggingface",
        "eval_start": 200,
        "eval_count": 100,
        "layers": [14],
        "alphas": [0.1],
        "schedule": "constant",
        "schedule_parameter": 64,
        "position_mode": "decode_last",
        "injection_mode": "absolute",
        "normalization": "direction RMS=1 before alpha",
        "max_new_tokens": 512,
        "prompt_template": "prompt",
    }
    (locked / "resolved_manifest.json").write_text(json.dumps(manifest))
    (locked / "locked_launch_manifest.json").write_text(
        json.dumps({"selection_sha256": sha256_file(selection), "spec": spec.to_dict()})
    )
    rows = []
    for sample_id in range(200, 300):
        base = {
            "sample_id": sample_id,
            "gold": float(sample_id),
            "pred": float(sample_id),
            "correct": True,
            "text_tokens": 3,
            "n_generated_tokens": 3,
            "n_content_tokens": 2,
            "stop_reason": "eos",
            "truncated": False,
            "text_tokens_whitespace": 3,
            "distinct_2gram_ratio": 1.0,
            "distinct_4gram_ratio": 1.0,
            "repeated_4gram_fraction": 0.0,
            "answer_marker_position": 2,
            "answer_marker_relative_position": 2 / 3,
        }
        rows.append({**base, "method": "baseline", "layer": 0, "alpha": 0.0})
        rows.append({**base, "method": method, "layer": 14, "alpha": 0.1})
    pd.DataFrame(rows).to_parquet(locked / "per_sample.parquet", index=False)
    pd.DataFrame([{"method": method, "layer": 14, "alpha": 0.1}]).to_csv(
        locked / "summary.csv", index=False
    )

    loaded = _load_locked(selection, method, locked)
    assert loaded["summary"].n == 100
    assert loaded["summary"].delta == 0.0
    assert loaded["spec"].layer == 14


def test_long_context_launcher_keeps_frozen_vector_layer_and_alpha(tmp_path):
    from argparse import Namespace

    from geoprobe.revision.locked_run import load_locked_run_spec
    from scripts.revision_launch_long_context import _POLICIES, _command

    method = "crosssteer_source"
    validation = tmp_path / "validation"
    validation.mkdir()
    vector = validation / "vector.pt"
    vector.write_bytes(b"frozen")
    (validation / "resolved_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "position_mode": "decode_last",
                "injection_mode": "absolute",
                "schedule": "constant",
                "max_new_tokens": 512,
                "vectors": {method: str(vector)},
                "source_ids": list(range(100)),
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "selection_status": "locked_before_locked_test_generation",
                "methods": [
                    {
                        "method": method,
                        "run_dir": str(validation),
                        "evaluation_ids": list(range(100, 200)),
                        "decision": {
                            "eligible": True,
                            "reason": "selected_by_preregistered_validation_rule",
                            "layer": 14,
                            "alpha": 0.1,
                            "validation_accuracy": 0.6,
                            "validation_delta": 0.1,
                        },
                    }
                ],
            }
        )
    )
    spec = load_locked_run_spec(selection, method)
    args = Namespace(
        target_model="target", target_source="huggingface", target_device="cuda:0", budget=4096
    )
    relative = next(policy for policy in _POLICIES if policy.name == "relative-hidden-rms")
    command = _command(args, spec, relative, tmp_path / "out")
    assert command[command.index("--layers") + 1] == "14"
    assert command[command.index("--alphas") + 1] == "0.1"
    assert command[command.index("--max-new-tokens") + 1] == "4096"
    assert command[command.index("--injection-mode") + 1] == "relative_hidden_rms"
    assert "crosssteer_source=" in command[command.index("--vector") + 1]


def test_label_efficiency_subsets_are_hash_fixed_stratified_and_nested():
    from geoprobe.revision.label_efficiency import stratified_label_budget_ids

    labels = pd.DataFrame(
        {
            "sample_id": list(range(12)),
            "sample_idx": [0] * 12,
            "correct": [True] * 8 + [False] * 4,
        }
    )
    first = stratified_label_budget_ids(
        labels, budgets=[5, 10, 12], allowed_ids=range(12), salt="frozen-salt"
    )
    second = stratified_label_budget_ids(
        labels, budgets=[5, 10, 12], allowed_ids=range(12), salt="frozen-salt"
    )
    assert first == second
    assert len(first[5]) == 5 and len(first[10]) == 10 and len(first[12]) == 12
    assert set(first[5]) <= set(first[10]) <= set(first[12])
    assert 0 < sum(labels.set_index("sample_id").loc[item, "correct"] for item in first[5]) < 5
    with pytest.raises(ValueError, match="at least one correct"):
        stratified_label_budget_ids(
            labels.assign(correct=True), budgets=[5], allowed_ids=range(12), salt="frozen-salt"
        )


def test_label_efficiency_locked_launcher_freezes_target_selection_and_provenance(tmp_path):
    from argparse import Namespace

    from geoprobe.revision.locked_run import load_locked_run_spec, sha256_file
    from scripts.revision_launch_label_efficiency_locked import (
        _command,
        _label_vector_entry,
        _vector_entry,
    )

    method = "target_calibrated"
    validation = tmp_path / "validation"
    validation.mkdir()
    selected_vector = validation / "target_calibrated.pt"
    selected_vector.write_bytes(b"target-full")
    registry_manifest = {
        "protocol": "tacl-11241-vector-registry-v1",
        "source_ids": list(range(100)),
        "target_run_config_sha256": "target-config",
        "target_labels_sha256": "target-labels",
    }
    (validation / "manifest.json").write_text(json.dumps(registry_manifest))
    (validation / "resolved_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "position_mode": "decode_last",
                "injection_mode": "absolute",
                "schedule": "constant",
                "max_new_tokens": 512,
                "vectors": {method: str(selected_vector)},
                "source_ids": list(range(100)),
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "selection_status": "locked_before_locked_test_generation",
                "methods": [
                    {
                        "method": method,
                        "run_dir": str(validation),
                        "evaluation_ids": list(range(100, 200)),
                        "decision": {
                            "eligible": True,
                            "reason": "selected_by_preregistered_validation_rule",
                            "layer": 17,
                            "alpha": 0.15,
                            "validation_accuracy": 0.6,
                            "validation_delta": 0.1,
                        },
                    }
                ],
            }
        )
    )
    spec = load_locked_run_spec(selection, method)
    selected = _vector_entry(selected_vector)

    root = tmp_path / "label_vectors"
    directory = root / "k005"
    directory.mkdir(parents=True)
    label_vector = directory / "target_calibrated_k005.pt"
    label_vector.write_bytes(b"target-five")
    label_manifest = {
        "protocol": "tacl-11241-target-label-efficiency-v1",
        "budget": 5,
        "source_ids": [0, 1, 2, 3, 4],
        "target_run_config_sha256": "target-config",
        "target_labels_sha256": "target-labels",
        "vector": label_vector.name,
        "vector_sha256": sha256_file(label_vector),
    }
    (directory / "manifest.json").write_text(json.dumps(label_manifest))
    entry = _label_vector_entry(root, 5, target_registry=selected["manifest"])
    assert entry["budget"] == 5
    assert entry["source_ids"] == [0, 1, 2, 3, 4]

    args = Namespace(
        target_model="target",
        target_source="huggingface",
        target_device="cuda:1",
    )
    command = _command(
        args,
        spec,
        method="target_labels_k005",
        vector_path=str(entry["vector_path"]),
        out=tmp_path / "locked-k005",
    )
    assert command[command.index("--layers") + 1] == "17"
    assert command[command.index("--alphas") + 1] == "0.15"
    assert command[command.index("--eval-start") + 1] == "200"
    assert command[command.index("--eval-count") + 1] == "100"
    assert command[command.index("--max-new-tokens") + 1] == "512"
    assert command[command.index("--vector") + 1].startswith("target_labels_k005=")


def test_ood_scope_requires_explicit_cross_dataset_flag_and_symbolic_scoring():
    from scripts.revision_steering_validation_grid import (
        _score_completion,
        _validate_evaluation_scope,
    )

    with pytest.raises(ValueError, match="overlaps"):
        _validate_evaluation_scope(
            "gsm8k",
            source_ids={0, 1, 2},
            eval_start=0,
            eval_count=5,
            allow_cross_dataset_eval=False,
        )
    with pytest.raises(ValueError, match="allow-cross-dataset"):
        _validate_evaluation_scope(
            "math500",
            source_ids={0, 1, 2},
            eval_start=0,
            eval_count=5,
            allow_cross_dataset_eval=False,
        )
    _validate_evaluation_scope(
        "math500",
        source_ids={0, 1, 2},
        eval_start=0,
        eval_count=5,
        allow_cross_dataset_eval=True,
    )
    gold, prediction, correct = _score_completion("math500", "Final: \\boxed{0.5}", r"\frac{1}{2}")
    assert gold == r"\frac{1}{2}"
    assert prediction == "0.5"
    assert correct


def test_ood_launcher_uses_only_frozen_selected_intervention(tmp_path):
    from argparse import Namespace

    from geoprobe.revision.locked_run import load_locked_run_spec
    from scripts.revision_launch_ood_selected import _command

    method = "crosssteer_source"
    validation = tmp_path / "validation"
    validation.mkdir()
    vector = validation / "vector.pt"
    vector.write_bytes(b"frozen")
    (validation / "resolved_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "position_mode": "decode_last",
                "injection_mode": "absolute",
                "schedule": "constant",
                "max_new_tokens": 512,
                "vectors": {method: str(vector)},
                "source_ids": list(range(100)),
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "selection_status": "locked_before_locked_test_generation",
                "methods": [
                    {
                        "method": method,
                        "run_dir": str(validation),
                        "evaluation_ids": list(range(100, 200)),
                        "decision": {
                            "eligible": True,
                            "reason": "selected_by_preregistered_validation_rule",
                            "layer": 12,
                            "alpha": 0.2,
                            "validation_accuracy": 0.6,
                            "validation_delta": 0.1,
                        },
                    }
                ],
            }
        )
    )
    spec = load_locked_run_spec(selection, method)
    args = Namespace(target_model="target", target_source="huggingface", target_device="cuda:1")
    command = _command(args, spec, tmp_path / "ood")
    assert command[command.index("--dataset") + 1] == "math500"
    assert "--allow-cross-dataset-eval" in command
    assert command[command.index("--eval-count") + 1] == "500"
    assert command[command.index("--layers") + 1] == "12"
    assert command[command.index("--alphas") + 1] == "0.2"
    assert command[command.index("--max-new-tokens") + 1] == "512"


def test_label_efficiency_report_loader_requires_full_frozen_plan_and_shared_baseline(tmp_path):
    from scripts.revision_build_label_efficiency_report import (
        _assert_shared_baseline,
        _expected_methods,
        _load_plan,
        _load_run,
    )

    methods = [
        {"method": "target_labels_k000_source_transfer", "label_budget": 0},
        {"method": "target_labels_k005", "label_budget": 5},
    ]
    plan = {
        "protocol": "tacl-11241-target-label-efficiency-locked-v1",
        "target_model": "target",
        "target_source": "huggingface",
        "frozen_target_spec": {
            "locked_eval_start": 200,
            "locked_eval_count": 2,
            "layer": 14,
            "alpha": 0.1,
            "schedule": "constant",
            "injection_mode": "absolute",
            "max_new_tokens": 512,
        },
        "methods": methods,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    loaded_plan = _load_plan(plan_path)
    expected = _expected_methods(loaded_plan)
    assert set(expected) == {"target_labels_k000_source_transfer", "target_labels_k005"}

    common = {
        "protocol": "tacl-11241-frozen-steering-grid-v1",
        "target_model": "target",
        "target_source": "huggingface",
        "eval_start": 200,
        "eval_count": 2,
        "layers": [14],
        "alphas": [0.1],
        "schedule": "constant",
        "schedule_parameter": 64,
        "position_mode": "decode_last",
        "injection_mode": "absolute",
        "normalization": "direction RMS=1 before alpha",
        "max_new_tokens": 512,
        "prompt_template": "prompt",
    }

    def make_run(method: str) -> Path:
        directory = tmp_path / method
        directory.mkdir()
        (directory / "DONE").write_text("complete\n")
        (directory / "resolved_manifest.json").write_text(
            json.dumps({**common, "vectors": {method: str(directory / "vector.pt")}})
        )
        rows = []
        for sample_id in (200, 201):
            base = {
                "sample_id": sample_id,
                "gold": float(sample_id),
                "pred": float(sample_id),
                "correct": True,
                "text_tokens": 3,
                "n_generated_tokens": 3,
                "n_content_tokens": 2,
                "stop_reason": "eos",
                "truncated": False,
                "text_tokens_whitespace": 3,
                "distinct_2gram_ratio": 1.0,
                "distinct_4gram_ratio": 1.0,
                "repeated_4gram_fraction": 0.0,
                "answer_marker_position": 2,
                "answer_marker_relative_position": 2 / 3,
            }
            rows.append({**base, "method": "baseline", "layer": 0, "alpha": 0.0})
            rows.append({**base, "method": method, "layer": 14, "alpha": 0.1})
        pd.DataFrame(rows).to_parquet(directory / "per_sample.parquet", index=False)
        return directory

    runs = [
        _load_run(
            method=item["method"], path=make_run(item["method"]), expected=item, plan=loaded_plan
        )
        for item in methods
    ]
    _assert_shared_baseline(runs)
    assert [run["budget"] for run in runs] == [0, 5]


def test_ood_report_loader_requires_full_math500_and_frozen_spec(tmp_path):
    from geoprobe.revision.locked_run import sha256_file
    from scripts.revision_build_ood_report import _load

    method = "crosssteer_source"
    validation = tmp_path / "validation"
    validation.mkdir()
    vector = validation / "vector.pt"
    vector.write_bytes(b"frozen")
    (validation / "resolved_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "position_mode": "decode_last",
                "injection_mode": "absolute",
                "schedule": "constant",
                "max_new_tokens": 512,
                "vectors": {method: str(vector)},
                "source_ids": list(range(100)),
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "selection_status": "locked_before_locked_test_generation",
                "methods": [
                    {
                        "method": method,
                        "run_dir": str(validation),
                        "evaluation_ids": list(range(100, 200)),
                        "decision": {
                            "eligible": True,
                            "reason": "selected_by_preregistered_validation_rule",
                            "layer": 14,
                            "alpha": 0.1,
                            "validation_accuracy": 0.6,
                            "validation_delta": 0.1,
                        },
                    }
                ],
            }
        )
    )
    ood = tmp_path / "ood"
    ood.mkdir()
    (ood / "DONE").write_text("complete\n")
    (ood / "ood_launch_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-ood-launch-v1",
                "selection_sha256": sha256_file(selection),
                "frozen_target_spec": {
                    "method": method,
                    "vector_path": str(vector.resolve()),
                    "source_ids": list(range(100)),
                    "layer": 14,
                    "alpha": 0.1,
                    "validation_accuracy": 0.6,
                    "validation_delta": 0.1,
                    "validation_run_dir": str(validation.resolve()),
                    "locked_eval_start": 200,
                    "locked_eval_count": 100,
                    "schedule": "constant",
                    "position_mode": "decode_last",
                    "injection_mode": "absolute",
                    "max_new_tokens": 512,
                },
                "target_model": "target",
                "target_source": "huggingface",
            }
        )
    )
    manifest = {
        "protocol": "tacl-11241-frozen-steering-grid-v1",
        "dataset": "math500",
        "cross_dataset_eval": True,
        "evaluator": {
            "name": "math_verify_symbolic_full_completion",
            "version": "math-verify==0.9.0",
        },
        "target_model": "target",
        "target_source": "huggingface",
        "eval_start": 0,
        "eval_count": 500,
        "layers": [14],
        "alphas": [0.1],
        "schedule": "constant",
        "schedule_parameter": 64,
        "position_mode": "decode_last",
        "injection_mode": "absolute",
        "normalization": "direction RMS=1 before alpha",
        "max_new_tokens": 512,
        "prompt_template": "prompt",
    }
    (ood / "resolved_manifest.json").write_text(json.dumps(manifest))
    rows = []
    for sample_id in range(500):
        base = {
            "sample_id": sample_id,
            "gold": str(sample_id),
            "pred": str(sample_id),
            "correct": True,
            "text_tokens": 3,
            "n_generated_tokens": 3,
            "n_content_tokens": 2,
            "stop_reason": "eos",
            "truncated": False,
            "text_tokens_whitespace": 3,
            "distinct_2gram_ratio": 1.0,
            "distinct_4gram_ratio": 1.0,
            "repeated_4gram_fraction": 0.0,
            "answer_marker_position": 2,
            "answer_marker_relative_position": 2 / 3,
        }
        rows.append({**base, "method": "baseline", "layer": 0, "alpha": 0.0})
        rows.append({**base, "method": method, "layer": 14, "alpha": 0.1})
    pd.DataFrame(rows).to_parquet(ood / "per_sample.parquet", index=False)
    loaded = _load(selection, method, ood)
    assert loaded["summary"].n == 500
    assert loaded["summary"].delta == 0.0


def test_long_context_report_loader_requires_all_frozen_policies(tmp_path):
    from geoprobe.revision.locked_run import sha256_file
    from scripts.revision_build_long_context_report import (
        _assert_shared_baseline,
        _load_plan,
        _load_policy,
    )

    method = "crosssteer_source"
    validation = tmp_path / "validation"
    validation.mkdir()
    vector = validation / "vector.pt"
    vector.write_bytes(b"frozen")
    (validation / "resolved_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "position_mode": "decode_last",
                "injection_mode": "absolute",
                "schedule": "constant",
                "max_new_tokens": 512,
                "vectors": {method: str(vector)},
                "source_ids": list(range(100)),
            }
        )
    )
    selection = tmp_path / "selection.json"
    spec = {
        "method": method,
        "vector_path": str(vector.resolve()),
        "source_ids": list(range(100)),
        "layer": 14,
        "alpha": 0.1,
        "validation_accuracy": 0.6,
        "validation_delta": 0.1,
        "validation_run_dir": str(validation.resolve()),
        "locked_eval_start": 200,
        "locked_eval_count": 100,
        "schedule": "constant",
        "position_mode": "decode_last",
        "injection_mode": "absolute",
        "max_new_tokens": 512,
    }
    selection.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "selection_status": "locked_before_locked_test_generation",
                "methods": [
                    {
                        "method": method,
                        "run_dir": str(validation),
                        "evaluation_ids": list(range(100, 200)),
                        "decision": {
                            "eligible": True,
                            "reason": "selected_by_preregistered_validation_rule",
                            "layer": 14,
                            "alpha": 0.1,
                            "validation_accuracy": 0.6,
                            "validation_delta": 0.1,
                        },
                    }
                ],
            }
        )
    )
    root = tmp_path / "long"
    root.mkdir()
    policies = [
        {
            "name": "constant",
            "schedule": "constant",
            "schedule_parameter": 64.0,
            "injection_mode": "absolute",
        },
        {
            "name": "prefix-256",
            "schedule": "prefix",
            "schedule_parameter": 256.0,
            "injection_mode": "absolute",
        },
        {
            "name": "exponential-1024",
            "schedule": "exponential",
            "schedule_parameter": 1024.0,
            "injection_mode": "absolute",
        },
        {
            "name": "relative-hidden-rms",
            "schedule": "constant",
            "schedule_parameter": 64.0,
            "injection_mode": "relative_hidden_rms",
        },
    ]
    plan_path = root / "long_context_launch_manifest.json"
    plan_path.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-long-context-v1",
                "selection_sha256": sha256_file(selection),
                "spec": spec,
                "target_model": "target",
                "target_source": "huggingface",
                "target_device": "cuda:0",
                "budget": 4096,
                "policies": policies,
            }
        )
    )
    for policy in policies:
        directory = root / policy["name"]
        directory.mkdir()
        (directory / "DONE").write_text("complete\n")
        (directory / "resolved_manifest.json").write_text(
            json.dumps(
                {
                    "protocol": "tacl-11241-frozen-steering-grid-v1",
                    "target_model": "target",
                    "target_source": "huggingface",
                    "eval_start": 200,
                    "eval_count": 100,
                    "layers": [14],
                    "alphas": [0.1],
                    "schedule": policy["schedule"],
                    "schedule_parameter": policy["schedule_parameter"],
                    "position_mode": "decode_last",
                    "injection_mode": policy["injection_mode"],
                    "normalization": "direction RMS=1 before alpha",
                    "max_new_tokens": 4096,
                    "prompt_template": "prompt",
                    "vectors": {method: str(vector)},
                }
            )
        )
        rows = []
        for sample_id in range(200, 300):
            base = {
                "sample_id": sample_id,
                "gold": float(sample_id),
                "pred": float(sample_id),
                "correct": True,
                "text_tokens": 3,
                "n_generated_tokens": 3,
                "n_content_tokens": 2,
                "stop_reason": "eos",
                "truncated": False,
                "text_tokens_whitespace": 3,
                "distinct_2gram_ratio": 1.0,
                "distinct_4gram_ratio": 1.0,
                "repeated_4gram_fraction": 0.0,
                "answer_marker_position": 2,
                "answer_marker_relative_position": 2 / 3,
            }
            rows.append({**base, "method": "baseline", "layer": 0, "alpha": 0.0})
            rows.append({**base, "method": method, "layer": 14, "alpha": 0.1})
        pd.DataFrame(rows).to_parquet(directory / "per_sample.parquet", index=False)
    plan = _load_plan(plan_path, selection, method)
    items = [
        _load_policy(plan=plan, method=method, root=root, policy=policy) for policy in policies
    ]
    _assert_shared_baseline(items)
    assert {item["name"] for item in items} == {policy["name"] for policy in policies}


def test_validation_selector_only_backfills_documented_missing_absolute_injection_mode():
    from scripts.revision_select_validation import _canonical_target_manifest

    canonical, backfills = _canonical_target_manifest(
        {"protocol": "tacl-11241-frozen-steering-grid-v1"}
    )
    assert canonical["injection_mode"] == "absolute"
    assert backfills["injection_mode"]["value"] == "absolute"

    explicit, explicit_backfills = _canonical_target_manifest(
        {"protocol": "tacl-11241-frozen-steering-grid-v1", "injection_mode": "relative_hidden_rms"}
    )
    assert explicit["injection_mode"] == "relative_hidden_rms"
    assert explicit_backfills == {}


def test_validation_selector_canonicalizes_legacy_greedy_decoder_but_not_sampling():
    from scripts.revision_select_validation import _canonical_target_manifest

    legacy, _ = _canonical_target_manifest({"protocol": "tacl-11241-frozen-steering-grid-v1"})
    explicit, _ = _canonical_target_manifest(
        {
            "protocol": "tacl-11241-frozen-steering-grid-v1",
            "generation": {
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "base_seed": 11241,
                "seed_rule": None,
                "num_beams": 1,
                "use_cache": True,
                "eos_token_id": "model.generation_config.eos_token_id",
            },
        }
    )
    assert legacy["generation"] == explicit["generation"]

    with pytest.raises(ValueError, match="lacks frozen fields"):
        _canonical_target_manifest(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "generation": {"do_sample": True, "temperature": 0.6},
            }
        )


def test_revised_geovote_table_has_complete_same_pool_paired_provenance():
    audit = Path("revision/evidence/geovote-length-audit-n8-retrospective")
    comparison = pd.read_csv(audit / "paired_comparisons_vs_majority.csv").set_index("method")
    assert comparison.loc["historical_geovote", "bootstrap_ci_low"] == pytest.approx(-0.10)
    assert comparison.loc["historical_geovote", "bootstrap_ci_high"] == pytest.approx(0.0)
    assert comparison.loc["length_residual_geo_vote", "exact_sign_p_value"] == pytest.approx(1.0)
    assert comparison.loc["raw_geo_pick", "exact_sign_p_value"] == pytest.approx(0.0625)
    assert comparison.loc["logprob_weighted", "exact_sign_p_value"] == pytest.approx(0.03125)
    manuscript = Path("paper/sections/05_geovote.tex").read_text()
    assert "GSM8K candidate pool" in manuscript
    assert "held-out GSM8K questions" in manuscript
    assert "Shortest-output control" in manuscript
    assert "R/B denotes repairs/breaks" in manuscript
    assert "$-10,0$" in manuscript
    assert "0.0625" in manuscript


def test_matched_random_builder_preserves_reference_provenance(tmp_path: Path):
    from geoprobe.revision.vector_registry import sha256_file
    from scripts.revision_build_matched_random_vector import build_manifest

    registry = tmp_path / "source-registry"
    registry.mkdir()
    reference = registry / "crosssteer_source.pt"
    torch.save(torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1, reference)
    (registry / "manifest.json").write_text(json.dumps({"source_ids": [0, 1, 2]}))
    out = tmp_path / "random-control"
    vector = out / "matched_norm_random.pt"

    manifest = build_manifest(reference, vector, seed=11241)

    assert manifest["protocol"] == "tacl-11241-matched-random-vector-v1"
    assert manifest["source_ids"] == [0, 1, 2]
    assert manifest["reference_vector_sha256"] == sha256_file(reference)
    assert manifest["vector_sha256"] == sha256_file(vector)
    generated = torch.load(vector, map_location="cpu", weights_only=True)
    original = torch.load(reference, map_location="cpu", weights_only=True)
    assert torch.linalg.vector_norm(generated).item() == pytest.approx(
        torch.linalg.vector_norm(original).item()
    )


def test_sae_sparse_activation_baseline_is_train_split_only_and_deterministic(
    tmp_path: Path, monkeypatch
):
    import geoprobe.revision.sae_sparse_steering as sae_module
    from geoprobe.revision.sae_sparse_steering import (
        SAETrainingConfig,
        build_sae_sparse_activation_vectors,
        relative_progress_indices,
    )

    assert torch.equal(relative_progress_indices(3, 8), torch.tensor([0, 1, 2]))
    assert torch.equal(relative_progress_indices(9, 3), torch.tensor([0, 4, 8]))

    calibration = tmp_path / "calibration"
    trajectories = calibration / "trajectories"
    trajectories.mkdir(parents=True)
    (calibration / "DONE").write_text("complete\n")
    (calibration / "config.yaml").write_text("model: fake\n")
    calibration_rows = []
    for sample_id in (0, 1):
        hidden = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3) + sample_id
        save_trajectory(
            Trajectory(
                sample_id=sample_id,
                sample_idx=0,
                hidden_states=hidden,
                generated_token_ids=torch.arange(5),
                generated_text="synthetic",
                prompt="synthetic",
                prompt_len=1,
                sequence_logprob=0.0,
                model_id="synthetic",
                dtype="float32",
            ),
            trajectories / f"sample_{sample_id:04d}_idx_0.pt",
        )
        calibration_rows.append({"sample_id": sample_id, "sample_idx": 0, "correct": True})
    pd.DataFrame(calibration_rows).to_parquet(calibration / "labels.parquet", index=False)

    contrast = tmp_path / "contrast"
    contrast.mkdir()
    (contrast / "DONE").write_text("complete\n")
    (contrast / "config.yaml").write_text("model: fake\n")
    contrast_rows = []
    for sample_id in (0, 1):
        for sample_idx, correct in ((0, True), (1, False)):
            contrast_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_idx": sample_idx,
                    "correct": correct,
                    "n_gen_tokens": 10 + sample_idx,
                    "prompt": f"prompt-{sample_id}-",
                    "generated_text": str(sample_idx),
                    "prompt_len": 8,
                }
            )
    contrast_frame = pd.DataFrame(contrast_rows)
    contrast_frame.drop(columns=["prompt", "generated_text", "prompt_len"]).to_parquet(
        contrast / "labels.parquet", index=False
    )
    contrast_frame.to_parquet(contrast / "completions.parquet", index=False)
    monkeypatch.setattr(
        sae_module,
        "capture_prompt_final_states",
        lambda _model, _tokenizer, prompt: torch.full(
            (2, 3), 1.0 if prompt.endswith("0") else -1.0
        ),
    )

    split = RevisionSplit((0, 1), (2,), (3,), (4,))
    config = SAETrainingConfig(
        layers=(0, 1),
        relative_progress_positions=4,
        latent_multiplier=2,
        feature_keep_fraction=0.5,
        train_steps=4,
        batch_size=2,
        seed=7,
    )
    vectors, metadata = build_sae_sparse_activation_vectors(
        object(),
        object(),
        calibration_run=calibration,
        contrast_run=contrast,
        source_ids=split.source_train,
        split=split,
        config=config,
        device="cpu",
    )
    assert vectors["sae_sparse_activation"].shape == (2, 3)
    assert torch.count_nonzero(vectors["sae_sparse_activation"]) > 0
    assert metadata["source_ids"] == [0, 1]
    assert metadata["sae_training_uses_correctness_labels"] is False
    assert metadata["contrast_feature_selection_uses_same_question_correct_incorrect_pairs"]
    assert set(metadata["layer_audits"]) == {"0", "1"}


def test_negative_direction_builder_preserves_reference_provenance(tmp_path: Path):
    from geoprobe.revision.vector_registry import sha256_file
    from scripts.revision_build_negative_direction import build_manifest

    registry = tmp_path / "source-registry"
    registry.mkdir()
    reference = registry / "crosssteer_source.pt"
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1
    torch.save(original, reference)
    (registry / "manifest.json").write_text(json.dumps({"source_ids": [0, 1, 2]}))
    vector = tmp_path / "negative-control" / "negative_crosssteer_source.pt"

    manifest = build_manifest(reference, vector)

    assert manifest["protocol"] == "tacl-11241-negative-direction-control-v1"
    assert manifest["source_ids"] == [0, 1, 2]
    assert manifest["reference_vector_sha256"] == sha256_file(reference)
    assert manifest["vector_sha256"] == sha256_file(vector)
    assert torch.equal(torch.load(vector, map_location="cpu", weights_only=True), -original)


def test_svamp_loader_and_numeric_audit_are_pinned(tmp_path: Path, monkeypatch):
    import hashlib

    import geoprobe.datasets.svamp as svamp
    import scripts.revision_audit_svamp_grader as audit_module

    payload = [
        {
            "ID": f"item-{index}",
            "Body": "A value is 50.",
            "Question": "Add 1.",
            "Answer": 51.0,
            "Type": "Addition",
        }
        for index in range(1000)
    ]
    dataset = tmp_path / "SVAMP.json"
    dataset.write_text(json.dumps(payload))
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    monkeypatch.setattr(svamp, "SVAMP_SHA256", digest)
    rows = svamp.load_svamp(path=dataset)
    assert len(rows) == 1000
    assert rows[0].question == "A value is 50.\nAdd 1."
    assert rows[0].gold_answer == pytest.approx(51.0)
    different = tmp_path / "different.json"
    different.write_text("[]")
    with pytest.raises(ValueError, match="unpinned"):
        svamp.load_svamp(path=different)

    monkeypatch.setattr(audit_module, "SVAMP_SHA256", digest)
    monkeypatch.setattr(audit_module, "load_svamp", lambda: rows)
    audit = audit_module.audit_svamp()
    assert audit["passed"]
    assert audit["n_items"] == 1000


def test_svamp_ood_scope_and_frozen_launcher(tmp_path: Path):
    from argparse import Namespace

    from geoprobe.revision.locked_run import load_locked_run_spec
    from scripts.revision_launch_svamp_ood_selected import _command
    from scripts.revision_steering_validation_grid import (
        _score_completion,
        _validate_evaluation_scope,
    )

    _validate_evaluation_scope(
        "svamp", source_ids={0, 1}, eval_start=0, eval_count=1000, allow_cross_dataset_eval=True
    )
    with pytest.raises(ValueError, match="allow-cross-dataset"):
        _validate_evaluation_scope(
            "svamp",
            source_ids={0, 1},
            eval_start=0,
            eval_count=1000,
            allow_cross_dataset_eval=False,
        )
    gold, prediction, correct = _score_completion("svamp", "Final: \\boxed{51}", 51.0)
    assert gold == pytest.approx(51.0)
    assert prediction == pytest.approx(51.0)
    assert correct

    method = "crosssteer_source"
    validation = tmp_path / "validation"
    validation.mkdir()
    vector = validation / "vector.pt"
    vector.write_bytes(b"frozen")
    (validation / "resolved_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "position_mode": "decode_last",
                "injection_mode": "absolute",
                "schedule": "constant",
                "max_new_tokens": 512,
                "vectors": {method: str(vector)},
                "source_ids": list(range(100)),
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-frozen-steering-grid-v1",
                "selection_status": "locked_before_locked_test_generation",
                "methods": [
                    {
                        "method": method,
                        "run_dir": str(validation),
                        "evaluation_ids": list(range(100, 200)),
                        "decision": {
                            "eligible": True,
                            "reason": "selected_by_preregistered_validation_rule",
                            "layer": 12,
                            "alpha": 0.2,
                            "validation_accuracy": 0.6,
                            "validation_delta": 0.1,
                        },
                    }
                ],
            }
        )
    )
    spec = load_locked_run_spec(selection, method)
    args = Namespace(target_model="target", target_source="huggingface", target_device="cuda:1")
    command = _command(args, spec, tmp_path / "ood")
    assert command[command.index("--dataset") + 1] == "svamp"
    assert command[command.index("--eval-count") + 1] == "1000"
    assert command[command.index("--layers") + 1] == "12"


def test_anonymous_release_preflight_requires_current_overview_figure():
    from geoprobe.revision.release import REQUIRED_RELEASE_ITEMS

    assert "paper/figures/fig1_overview.png" in REQUIRED_RELEASE_ITEMS
    assert "paper/figures/fig1_revision_protocol.png" not in REQUIRED_RELEASE_ITEMS


def test_anonymous_release_preflight_requires_sources_evidence_and_no_identity_leaks(
    tmp_path: Path,
):
    from geoprobe.revision.release import (
        ANONYMOUS_PER_PROBLEM_TABLES,
        EXCLUDED_FROM_ANONYMOUS_RELEASE,
        REQUIRED_RELEASE_ITEMS,
        anonymous_release_preflight,
        sha256_file,
    )

    for relative in REQUIRED_RELEASE_ITEMS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("anonymous source\n")
    for csv_relative, manifest_relative in ANONYMOUS_PER_PROBLEM_TABLES:
        csv_path = tmp_path / csv_relative
        csv_path.write_text(
            "suite,comparison,arm,sample_id,correct\n"
            "fixture,method,baseline,0,True\n"
        )
        (tmp_path / manifest_relative).write_text(
            json.dumps(
                {
                    "output": {
                        "file": csv_path.name,
                        "sha256": sha256_file(csv_path),
                        "rows": 1,
                        "columns": [
                            "suite",
                            "comparison",
                            "arm",
                            "sample_id",
                            "correct",
                        ],
                    }
                }
            )
        )
    (tmp_path / "paper/sections").mkdir(parents=True, exist_ok=True)
    (tmp_path / "paper/sections/section.tex").write_text("clean section\n")
    (tmp_path / "revision").mkdir(parents=True, exist_ok=True)
    (tmp_path / "revision/EXECUTION_LOG.md").write_text("/" + "Users/private/log\n")

    pending = anonymous_release_preflight(tmp_path)
    assert pending["source_tree_ready"]
    assert not pending["ready_for_final_anonymous_release"]
    assert pending["evidence_state"] == "pending"
    assert pending["identifier_hits"] == []
    assert "revision/EXECUTION_LOG.md" in pending["excluded_from_anonymous_release"]
    assert {
        "paper/FIGURE_PROMPTS.md",
        "paper/FIGURE_PROMPTS_FINAL.md",
        "paper/figures/fig1.png",
        "paper/figures/fig1new.png",
        "paper/figures/fig1_overview.drawio",
        "paper/figures/fig1_overview.drawio.pdf",
        "paper/figures/fig1_overview.drawio.png",
        "paper/figures/fig1_overview.drawio.svg",
        "paper/figures/fig1_overview.vector.svg",
        "paper/figures/fig3_crosssteer_real.pdf",
        "paper/figures/fig3_crosssteer_real.png",
        "scripts/paper_figures.py",
        "scripts/sync_to_server.sh",
    } <= set(EXCLUDED_FROM_ANONYMOUS_RELEASE)

    evidence = tmp_path / "formal-report.json"
    evidence.write_text("{}\n")
    final = anonymous_release_preflight(tmp_path, evidence=[evidence], require_evidence=True)
    assert final["ready_for_final_anonymous_release"]
    assert final["evidence_state"] == "provided"
    assert final["evidence_entries"][0]["sha256"] is not None

    superseded_dir = tmp_path / "superseded-evidence"
    superseded_dir.mkdir()
    superseded = superseded_dir / "report.json"
    superseded.write_text("{}\n")
    (superseded_dir / "SUPERSEDED").write_text("invalid decoder envelope\n")
    invalid = anonymous_release_preflight(tmp_path, evidence=[superseded], require_evidence=True)
    assert not invalid["ready_for_final_anonymous_release"]
    assert invalid["superseded_evidence"] == ["superseded-evidence/report.json"]

    (tmp_path / "paper/sections/section.tex").write_text("/" + "Users/private/leak\n")
    leaked = anonymous_release_preflight(tmp_path, evidence=[evidence], require_evidence=True)
    assert not leaked["ready_for_final_anonymous_release"]
    assert leaked["identifier_hits"] == [
        {"path": "paper/sections/section.tex", "marker": "/Users/"}
    ]

    (tmp_path / "paper/sections/section.tex").write_text("clean section\n")
    (tmp_path / "paper/FIGURE_PROVENANCE.md").write_text("/" + "Users/private/figure-source\n")
    leaked_figure_doc = anonymous_release_preflight(
        tmp_path, evidence=[evidence], require_evidence=True
    )
    assert not leaked_figure_doc["ready_for_final_anonymous_release"]
    assert leaked_figure_doc["identifier_hits"] == [
        {"path": "paper/FIGURE_PROVENANCE.md", "marker": "/Users/"}
    ]

    (tmp_path / "paper/FIGURE_PROVENANCE.md").write_text("clean provenance\n")
    csv_relative, _manifest_relative = ANONYMOUS_PER_PROBLEM_TABLES[0]
    tampered_csv = tmp_path / csv_relative
    tampered_csv.write_text(
        "suite,comparison,arm,sample_id,correct,generated_text\n"
        "fixture,method,baseline,0,True,private\n"
    )
    unsafe = anonymous_release_preflight(
        tmp_path, evidence=[evidence], require_evidence=True
    )
    assert not unsafe["source_tree_ready"]
    assert "forbidden headers" in unsafe["per_problem_audit_errors"][0]["error"]


def test_crosssteer_direction_accepts_compact_pooled_hidden_states(tmp_path: Path):
    from geoprobe.revision.crosssteer import compute_crosssteer_direction
    from geoprobe.revision.protocol import RevisionSplit

    run = tmp_path / "pooled-run"
    trajectories = run / "trajectories"
    trajectories.mkdir(parents=True)
    (run / "DONE").write_text("done\n")
    (run / "config.yaml").write_text("synthetic: true\n")
    labels = pd.DataFrame(
        [
            {"sample_id": 0, "sample_idx": 0, "correct": True},
            {"sample_id": 1, "sample_idx": 0, "correct": False},
        ]
    )
    labels.to_parquet(run / "labels.parquet", index=False)
    for sample_id, value in [(0, 3.0), (1, -2.0)]:
        trajectory = Trajectory(
            sample_id=sample_id,
            sample_idx=0,
            hidden_states=torch.full((3, 5), value),
            generated_token_ids=torch.tensor([1, 2]),
            generated_text="synthetic",
            prompt="synthetic",
            prompt_len=1,
            sequence_logprob=float("nan"),
            model_id="synthetic",
            dtype="float32",
        )
        save_trajectory(trajectory, trajectories / f"sample_{sample_id:04d}_idx_0.pt")
    direction, metadata = compute_crosssteer_direction(
        run, (0, 1), split=RevisionSplit((0, 1), (2,), (3,), (4,))
    )
    assert direction.shape == (3, 5)
    assert torch.allclose(direction, torch.full((3, 5), 5.0))
    assert metadata["aggregation"] == "mean_over_tokens_per_problem_then_mean_over_class"


def test_formal_chain_scripts_prefer_checked_out_revision_source():
    expected = 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"'
    scripts = (
        "scripts/revision_run_baseline_chain.sh",
        "scripts/revision_run_r1_primary_chain.sh",
        "scripts/revision_run_r1_official_context_chain.sh",
        "scripts/revision_run_r1_formal_comparison_chain.sh",
        "scripts/revision_run_r1_long_context_chain.sh",
        "scripts/revision_run_frozen_locked_chain.sh",
        "scripts/revision_run_frozen_ood_chain.sh",
        "scripts/revision_run_frozen_long_context_chain.sh",
        "scripts/revision_run_qwen_validation_v2.sh",
        "scripts/revision_run_qwen_ood_chain_v2.sh",
        "scripts/revision_run_qwen_long_context_chain_v2.sh",
        "scripts/revision_run_qwen_label_efficiency_chain_v2.sh",
    )
    for relative in scripts:
        assert expected in Path(relative).read_text(), relative
    baseline = Path("scripts/revision_run_baseline_chain.sh").read_text()
    assert "unexpected Qwen baseline-chain failure" in baseline
    assert "FAILURE_REASON" in baseline
    # Formal calibration artifacts live under the revision run root.  The
    # baseline chain must not accidentally look in the legacy shared run root
    # after the source/target grids have completed.
    assert '--calibration-run "$ROOT/revision_gsm8k_qwen_instruct_calibration_100"' in baseline
    assert (
        '--calibration-run "$RUNS_ROOT/revision_gsm8k_qwen_instruct_calibration_100"'
        not in baseline
    )


def test_pooled_generation_uses_local_seed_without_unsupported_generator_kwarg():
    from geoprobe.revision.pooled_trajectory import _generate_with_local_seed

    class _SamplingModel:
        def __init__(self):
            self.kwargs: list[dict[str, object]] = []
            self.draws: list[torch.Tensor] = []

        def generate(self, **kwargs):
            self.kwargs.append(kwargs)
            self.draws.append(torch.rand(4))
            return object()

    model = _SamplingModel()
    torch.manual_seed(991)
    expected_next = torch.rand(1)
    torch.manual_seed(991)
    _generate_with_local_seed(
        model,
        {"input_ids": torch.tensor([[1, 2]])},
        generation_seed=11241,
        device=torch.device("cpu"),
    )
    observed_next = torch.rand(1)
    _generate_with_local_seed(
        model,
        {"input_ids": torch.tensor([[1, 2]])},
        generation_seed=11241,
        device=torch.device("cpu"),
    )

    assert torch.equal(observed_next, expected_next)
    assert torch.equal(model.draws[0], model.draws[1])
    assert all("generator" not in kwargs for kwargs in model.kwargs)


def test_pooled_readiness_is_correctness_blind_and_detects_budget_or_loop_failures():
    from geoprobe.revision.pooled_readiness import audit_pooled_readiness_frame

    frame = pd.DataFrame(
        [
            {
                "sample_id": 0,
                "sample_idx": 0,
                "correct": True,
                "n_gen_tokens": 5,
                "trajectory_representation": "pooled_generated_state_mean_v1",
            },
            {
                "sample_id": 1,
                "sample_idx": 0,
                "correct": False,
                "n_gen_tokens": 7,
                "trajectory_representation": "pooled_generated_state_mean_v1",
            },
        ]
    )
    healthy = audit_pooled_readiness_frame(
        frame,
        generated_text_by_id={0: "a b c d e", 1: "w x y z q"},
        expected_ids=(0, 1),
        max_new_tokens=10,
    )
    assert healthy.eligible

    pathological = frame.copy()
    pathological.loc[0, "n_gen_tokens"] = 10
    failed = audit_pooled_readiness_frame(
        pathological,
        generated_text_by_id={0: " ".join(["loop"] * 100), 1: "w x y z q"},
        expected_ids=(0, 1),
        max_new_tokens=10,
        max_budget_hit_rate=0.25,
        max_severe_repetition_rate=0.05,
    )
    assert not failed.eligible
    assert any("budget_hit_rate" in failure for failure in failed.failures)
    assert any("severe_repetition_rate" in failure for failure in failed.failures)


def test_r1_official_context_chain_is_resource_gated_and_correctness_blind_before_calibration():
    script = Path("scripts/revision_run_r1_official_context_chain.sh").read_text()
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert "wait_r1_gpu_free" in script
    assert "wait_qwen_locked_chain" not in script
    assert 'nvidia-smi -i "$R1_VISIBLE_DEVICE"' in script
    assert "--expected-id-range 0:20 --max-new-tokens 32768" in script
    assert "revision_audit_pooled_readiness.py" in script
    assert "--max-severe-repetition-rate 0.05" in script
    assert "--min-class-count 10" in script


def test_sampling_grid_signature_freezes_paired_problem_seed_rule():
    from scripts.revision_steering_validation_grid import _problem_seed, _run_signature

    args = Namespace(
        target_model="r1",
        target_source="modelscope",
        target_device="cuda:0",
        eval_start=100,
        eval_count=100,
        layers=[8],
        alphas=[0.1],
        schedule="constant",
        schedule_parameter=64.0,
        injection_mode="absolute",
        max_new_tokens=32768,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        seed=11241,
    )
    signature = _run_signature(args, {"cross": Path("/tmp/cross.pt")})
    assert signature["generation"] == {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "base_seed": 11241,
        "seed_rule": "seed * 100003 + sample_id * 1009",
        "num_beams": 1,
        "use_cache": True,
        "eos_token_id": "model.generation_config.eos_token_id",
    }
    assert _problem_seed(11241, 100) == _problem_seed(11241, 100)
    assert _problem_seed(11241, 100) != _problem_seed(11241, 101)


def test_qwen_locked_chain_is_complete_family_and_r1_is_independent():
    qwen = Path("scripts/revision_run_qwen_locked_chain.sh").read_text()
    r1 = Path("scripts/revision_run_r1_official_context_chain.sh").read_text()
    assert "revision_select_validation.py" in qwen
    for method in (
        "crosssteer_source",
        "target_calibrated",
        "caa_target_prompt_final",
        "actadd_target_prompt_final",
        "sparse_caa_coordinate_10pct",
        "matched_norm_random",
        "sae_sparse_activation",
        "negative_crosssteer_source",
    ):
        assert method in qwen
    assert "revision_build_locked_report.py" in qwen
    assert "revision_build_label_efficiency_report.py" in qwen
    # ROOT is RUNS_ROOT/tacl-revision; using RUNS_ROOT here skips the revision
    # namespace and makes the post-locked target-label vector build fail.
    assert '--target-run "$ROOT/revision_gsm8k_qwen_instruct_calibration_100"' in qwen
    assert '--target-run "$RUNS_ROOT/revision_gsm8k_qwen_instruct_calibration_100"' not in qwen
    assert "wait_qwen_locked_chain" not in r1
    assert "wait_r1_gpu_free" in r1


def test_qwen_validation_v2_regenerates_eos_contaminated_control_vectors():
    script = Path("scripts/revision_run_qwen_validation_v2.sh").read_text()
    config = Path("configs/revision_gsm8k_qwen_instruct_caa_source_k4_v2.yaml").read_text()

    assert "prepare_eos_corrected_baselines" in script
    assert "revision_generate_contrast_pool.py" in script
    assert "revision_build_prompt_contrast_registry.py" in script
    assert "revision_build_sae_sparse_registry.py" in script
    assert 'CONTRAST_POOL="$RUNS_ROOT/revision_gsm8k_qwen_instruct_caa_source_k4_v2"' in script
    assert 'CONTRAST_REGISTRY="$ROOT/prompt-contrast-registry-gsm8k-ids0-99-v2"' in script
    assert 'SAE_REGISTRY="$ROOT/sae-sparse-registry-qwen-gsm8k-ids0-99-v2"' in script
    assert "v1 pool" in config
    assert "100% truncated" in config


def test_qwen_v2_label_efficiency_depends_only_on_corrected_locked_evidence():
    script = Path("scripts/revision_run_qwen_label_efficiency_chain_v2.sh").read_text()

    assert 'SELECTION="$ROOT/selection-qwen-instruct-v2.json"' in script
    assert 'LOCKED_ROOT="$ROOT/locked-gsm8k-qwen-instruct-v2"' in script
    assert 'LOCKED_REPORT="$ROOT/locked-report-qwen-instruct-v2"' in script
    assert 'wait_done "$ROOT/qwen-validation-v2-chain"' in script
    assert 'wait_done "$ROOT/qwen-ood-chain-v2"' in script
    assert "revision_build_label_budget_vectors.py" in script
    assert "revision_launch_label_efficiency_locked.py" in script
    assert "revision_build_label_efficiency_report.py" in script
    assert "selection-qwen-instruct-v1" not in script
    assert "qwen-locked-chain-v1" not in script


def test_qwen_v2_ood_and_long_context_depend_only_on_corrected_v2_artifacts():
    ood = Path("scripts/revision_run_qwen_ood_chain_v2.sh").read_text()
    long_context = Path("scripts/revision_run_qwen_long_context_chain_v2.sh").read_text()

    assert 'SELECTION="$ROOT/selection-qwen-instruct-v2.json"' in ood
    assert 'LOCKED_ROOT="$ROOT/locked-gsm8k-qwen-instruct-v2"' in ood
    assert 'wait_done "$ROOT/qwen-validation-v2-chain"' in ood
    assert "selection-qwen-instruct-v1" not in ood
    assert "qwen-locked-chain-v1" not in ood
    assert 'wait_done "$ROOT/qwen-label-efficiency-chain-v2"' in long_context
    assert 'SELECTION="$ROOT/selection-qwen-instruct-v2.json"' in long_context
    assert "selection-qwen-instruct-v1" not in long_context
    assert "for budget in 4096 32768" in long_context
    assert "constant prefix-256 exponential-1024 relative-hidden-rms" in long_context


def test_qwen_ood_and_long_context_chains_are_frozen_and_gpu_isolated():
    ood = Path("scripts/revision_run_qwen_ood_chain.sh").read_text()
    long_context = Path("scripts/revision_run_qwen_long_context_chain.sh").read_text()
    r1 = Path("scripts/revision_run_r1_official_context_chain.sh").read_text()
    assert "QWEN_DEVICE=${QWEN_DEVICE:-cuda:0}" in ood
    assert "math500" in ood and "svamp" in ood
    assert "revision_build_ood_report.py" in ood
    assert 'wait_done "$ROOT/qwen-locked-chain-v1"' in ood
    assert 'wait_done "$ROOT/qwen-ood-chain-v1"' in long_context
    assert "for budget in 4096 32768" in long_context
    assert "constant prefix-256 exponential-1024 relative-hidden-rms" in long_context
    assert "R1_VISIBLE_DEVICE=${R1_VISIBLE_DEVICE:-1}" in r1
    assert 'export CUDA_VISIBLE_DEVICES="$R1_VISIBLE_DEVICE"' in r1


def test_r1_long_context_chain_reuses_immutable_official_selection():
    script = Path("scripts/revision_run_r1_long_context_chain.sh").read_text()
    protocol = Path("revision/LONG_CONTEXT_PROTOCOL.md").read_text()
    execution = Path("revision/R1_FORMAL_EXECUTION_CHAIN.md").read_text()

    assert 'FORMAL_CHAIN="$ROOT/r1-formal-comparison-chain-v2"' in script
    assert 'SELECTION="$ROOT/frozen-selection-r1official32k-v2.json"' in script
    assert 'allowed = {"crosssteer_source", "target_calibrated"}' in script
    assert "for budget in 4096 32768" in script
    assert "constant prefix-256 exponential-1024 relative-hidden-rms" in script
    assert 'export CUDA_VISIBLE_DEVICES="$R1_VISIBLE_DEVICE"' in script
    assert "--do-sample" not in script  # copied only through the frozen launcher spec
    assert "selected 32,768-token sampled decoder" in protocol
    assert "4,096 and 32,768 maximum new tokens" in execution


def test_revision_docs_do_not_treat_short_r1_diagnostic_as_current_capability_evidence():
    target_matrix = Path("revision/TARGET_MATRIX_V1.md").read_text()
    experiment_matrix = Path("revision/EXPERIMENT_MATRIX.md").read_text()
    crosssteer = Path("paper/sections/06_crosssteer.tex").read_text()
    assert "32,768" in target_matrix
    assert "no 512-token fallback" in target_matrix
    assert "temperature 0.6" in target_matrix
    assert "R1 official context" in experiment_matrix
    assert "We evaluated the R1-Distill target separately" in crosssteer
    assert "severe repetition in two outputs" in crosssteer
    assert "stopped that branch before calibration or intervention" in crosssteer
    assert "model-valid 32,768-token sampled" in crosssteer
    assert "primary decoder is greedy with the same prompt" not in crosssteer


def test_historical_figure_prompts_are_quarantined_from_revision_evidence():
    prompt_paths = [
        Path("paper/FIGURE_PROMPTS.md"),
        Path("paper/FIGURE_PROMPTS_FINAL.md"),
    ]
    existing = [path for path in prompt_paths if path.is_file()]
    # The private author worktree retains both files as explicit deprecated
    # history; the anonymous release intentionally excludes both.  A partial
    # state would indicate a broken quarantine boundary.
    assert len(existing) in {0, len(prompt_paths)}
    for path in existing:
        prompt = path.read_text()
        assert prompt.startswith("# Deprecated submitted-version figure prompts")
        assert "must not be used" in prompt
    crosssteer = Path("paper/sections/06_crosssteer.tex").read_text()
    assert "causal performance claim" not in crosssteer
    assert "tested transfer hypothesis rather than a performance" in crosssteer
    assert "does not establish an advantage" in crosssteer


def test_body_figure_projection_coordinates_are_complete_and_frozen():
    """The supervised correctness-density figure must be reproducible from a
    compact, tracked coordinate artifact rather than from an opaque PDF only.
    """

    path = Path("paper/data/hidden_projection_coords.csv")
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "7514cf779f0d84fbb59f2a3e6a108e3006b489afd357b323231decba3d7510b6"
    )
    frame = pd.read_csv(path)
    assert set(frame.columns) == {
        "model",
        "sample_id",
        "correct",
        "layer",
        "x_correctness",
        "y_residual_pc",
    }
    assert set(frame["model"]) == {"Base", "Instruct", "Math-Instruct", "R1-Distill"}
    assert frame.shape == (11_600, 6)
    assert not frame.duplicated(["model", "sample_id", "layer"]).any()
    assert set(frame.groupby("model")["sample_id"].nunique()) == {100}
    assert set(frame.groupby(["model", "sample_id"]).size()) == {29}
    assert set(frame["layer"]) == set(range(29))

    generator = Path("scripts/make_phenomenon_hidden_figures.py").read_text()
    assert '"separation"' in generator
    assert "plot_correctness_separation(args.coords)" in generator


def test_contrast_pool_sampling_uses_forked_problem_seed_without_global_rng_leakage():
    from scripts.revision_generate_contrast_pool import _contrast_seed, _generate

    class _Batch(dict):
        def to(self, _device):
            return self

    class _Tokenizer:
        eos_token_id = 0

        def __call__(self, _prompt, return_tensors):
            assert return_tensors == "pt"
            return _Batch(input_ids=torch.tensor([[1, 2]]))

        def decode(self, ids, skip_special_tokens):
            assert skip_special_tokens
            return " ".join(str(value) for value in ids.tolist())

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.kwargs: list[dict[str, object]] = []

        def generate(self, **kwargs):
            self.kwargs.append(kwargs)
            token = torch.randint(3, 100, (1, 1))
            return torch.cat([kwargs["input_ids"], token], dim=1)

    model = _Model()
    tokenizer = _Tokenizer()
    torch.manual_seed(7331)
    expected_next = torch.rand(1)
    torch.manual_seed(7331)
    first = _generate(
        model,
        tokenizer,
        "prompt",
        max_new_tokens=16,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        generation_seed=_contrast_seed(11241, 5, 2),
    )
    observed_next = torch.rand(1)
    second = _generate(
        model,
        tokenizer,
        "prompt",
        max_new_tokens=16,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        generation_seed=_contrast_seed(11241, 5, 2),
    )
    assert first == second
    assert torch.equal(observed_next, expected_next)
    assert all("generator" not in kwargs for kwargs in model.kwargs)
    assert _contrast_seed(11241, 5, 2) != _contrast_seed(11241, 5, 3)


def test_r1_official_contrast_config_and_protocol_reject_legacy_512_fallback():
    config = Path("configs/revision_gsm8k_r1_official32k_caa_source_k4.yaml").read_text()
    protocol = Path("revision/R1_FORMAL_COMPARISON_PROTOCOL.md").read_text()
    legacy = Path("configs/revision_gsm8k_r1_caa_source_k4.yaml").read_text()
    assert "max_new_tokens: 32768" in config
    assert "temperature: 0.6" in config
    assert "top_p: 0.95" in config
    assert "historical 512-token R1 pool/configuration is inadmissible" in protocol
    assert "LEGACY SHORT-BUDGET ARTIFACT" in legacy


def test_paper_declares_matched_online_steering_budget_and_discloses_offline_inputs():
    body = Path("paper/sections/06_crosssteer.tex").read_text()
    appendix = Path("paper/sections/09_appendix.tex").read_text()
    response = Path("revision/RESPONSE_LETTER_DRAFT.md").read_text()
    response_builder = Path("scripts/build_revision_response_letter.py").read_text()
    response_header = Path("revision/response_letter_header.tex").read_text()
    assert "same online form" in body
    assert "tab:steering-budget" in appendix
    assert "preloaded residual vector add" in appendix
    assert "no wall-clock ranking" in appendix
    assert "direction-construction/online-overhead" in response
    assert "direct token-count-only control" in response
    assert "512-token stress envelope" in response
    assert "package for post-acceptance release" in response
    assert "no supplementary material or external anonymous link" in response
    assert "supplementary package" not in response
    assert "``" not in response
    assert "''" not in response
    assert "Candidate-final" not in response
    point_by_point_headings = (
        "## Summary of the revision",
        "## Response to the Action Editor",
        "### AE-1. Scale-dependent signature stability",
        "## Response to Reviewer A",
        "### A-1. Consolidate the fragmented methods",
        "## Response to Reviewer B",
        "### B-1. Stability across model scale and sample size",
        "### B-2. GeoVote versus completion length and significance",
        "### B-3. Source transfer, target calibration, and conventional steering baselines",
        "### B-4. Long-context repetition loops and safety",
        "### B-5. Correctness versus text style, length, and formatting",
        "### B-6c. Ambiguous notation",
        "## Response to Reviewer C",
        "### C-1. OOD generalization",
        "### C-2. Comparison with conventional steering",
        "### C-4. Reproducibility and software",
        "## Transparency note on the corrected generation configuration",
        "## Closing response",
    )
    positions = [response.index(heading) for heading in point_by_point_headings]
    assert positions == sorted(positions)
    assert "Section 3 (PDF pp. 4--6)" in response
    assert "AUTHOR_INPUT_NEEDED" in response_builder
    assert "bookmarks=false" in response_builder
    assert "Anonymous authors" in response_header


def test_contrast_pool_audit_requires_complete_pairs_and_bounded_budget_hits():
    from geoprobe.revision.contrast_pool_audit import audit_contrast_pool_frame

    frame = pd.DataFrame(
        [
            {"sample_id": 0, "sample_idx": 0, "correct": True, "n_gen_tokens": 30},
            {"sample_id": 0, "sample_idx": 1, "correct": False, "n_gen_tokens": 31},
            {"sample_id": 1, "sample_idx": 0, "correct": True, "n_gen_tokens": 32},
            {"sample_id": 1, "sample_idx": 1, "correct": False, "n_gen_tokens": 33},
        ]
    )
    good = audit_contrast_pool_frame(
        frame,
        expected_ids=(0, 1),
        n_samples_per_problem=2,
        max_new_tokens=64,
        min_class_count=2,
        min_same_question_pairs=2,
    )
    assert good.eligible_for_prompt_contrast
    assert good.n_same_question_pairs == 2

    bad = audit_contrast_pool_frame(
        frame.assign(n_gen_tokens=64),
        expected_ids=(0, 1),
        n_samples_per_problem=2,
        max_new_tokens=64,
        min_class_count=2,
        min_same_question_pairs=2,
        max_budget_hit_rate=0.25,
    )
    assert not bad.eligible_for_prompt_contrast
    assert any("budget_hit_rate" in failure for failure in bad.failures)


def test_locked_report_triage_requires_complete_unsaturated_locked_evidence(tmp_path: Path):
    methods = [
        "crosssteer_source",
        "target_calibrated",
        "caa_target_prompt_final",
        "actadd_target_prompt_final",
        "sae_sparse_activation",
        "sparse_caa_coordinate_10pct",
        "matched_norm_random",
        "negative_crosssteer_source",
    ]
    rows = []
    for method in methods:
        delta = 2.0 if method == "crosssteer_source" else 1.0
        rows.append(
            {
                "method": method,
                "layer": 8,
                "alpha": 0.05,
                "n": 100,
                "baseline_accuracy": 0.60,
                "method_accuracy": 0.60 + delta / 100,
                "delta_pp": delta,
                "bootstrap_ci_low_pp": 0.1 if method == "crosssteer_source" else -1.0,
                "bootstrap_ci_high_pp": 4.0,
                "repairs": 5,
                "breaks": 3,
                "exact_sign_p_value": 0.01,
                "holm_adjusted_p_value": 0.04,
                "baseline_mean_generated_tokens": 200.0,
                "method_mean_generated_tokens": 201.0,
                "generated_token_delta": 1.0,
                "baseline_truncation_rate": 0.01,
                "method_truncation_rate": 0.02,
                "baseline_repetition": 0.01,
                "method_repetition": 0.01,
            }
        )
    report = tmp_path / "locked_summary.csv"
    pd.DataFrame(rows).to_csv(report, index=False)
    out = tmp_path / "triage.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/revision_triage_locked_report.py",
            "--report",
            str(report),
            "--out",
            str(out),
            "--source-report-required",
        ],
        check=True,
        cwd=Path.cwd(),
    )
    result = json.loads(out.read_text())
    assert result["narrative"] == "METHOD_CLAIM_SUPPORTED"
    assert result["decoding_envelope_valid"]

    # A tie is not superiority, and a saturated locked report is invalid even
    # if its numerical delta looks favorable.
    tied = pd.DataFrame(rows)
    tied.loc[tied["method"] == "caa_target_prompt_final", "delta_pp"] = 2.0
    tied.loc[tied["method"] == "caa_target_prompt_final", "method_truncation_rate"] = 1.0
    tied.to_csv(report, index=False)
    subprocess.run(
        [
            sys.executable,
            "scripts/revision_triage_locked_report.py",
            "--report",
            str(report),
            "--out",
            str(out),
        ],
        check=True,
        cwd=Path.cwd(),
    )
    result = json.loads(out.read_text())
    assert result["narrative"] == "INVALID_DECODING_ENVELOPE"
    assert "caa_target_prompt_final" in result["crosssteer"]["beaten_by"]


def test_behavior_columns_reports_paired_length_and_repair_break_signatures(tmp_path: Path):
    rows = [
        {
            "method": "baseline",
            "sample_id": 0,
            "correct": False,
            "text_tokens": 10,
            "text_tokens_whitespace": 10,
            "n_generated_tokens": 11,
            "truncated": False,
            "repeated_4gram_fraction": 0.01,
            "distinct_4gram_ratio": 0.9,
            "answer_marker_relative_position": 0.8,
        },
        {
            "method": "baseline",
            "sample_id": 1,
            "correct": True,
            "text_tokens": 20,
            "text_tokens_whitespace": 20,
            "n_generated_tokens": 21,
            "truncated": False,
            "repeated_4gram_fraction": 0.02,
            "distinct_4gram_ratio": 0.8,
            "answer_marker_relative_position": 0.7,
        },
        {
            "method": "crosssteer_source",
            "sample_id": 0,
            "correct": True,
            "text_tokens": 12,
            "text_tokens_whitespace": 12,
            "n_generated_tokens": 13,
            "truncated": False,
            "repeated_4gram_fraction": 0.02,
            "distinct_4gram_ratio": 0.88,
            "answer_marker_relative_position": 0.75,
        },
        {
            "method": "crosssteer_source",
            "sample_id": 1,
            "correct": False,
            "text_tokens": 18,
            "text_tokens_whitespace": 18,
            "n_generated_tokens": 19,
            "truncated": False,
            "repeated_4gram_fraction": 0.03,
            "distinct_4gram_ratio": 0.78,
            "answer_marker_relative_position": 0.65,
        },
    ]
    per_sample = tmp_path / "per_sample.jsonl"
    per_sample.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "behavior.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/revision_behavior_columns.py",
            "--per-sample",
            str(per_sample),
            "--out",
            str(out),
        ],
        check=True,
        cwd=Path.cwd(),
    )
    summary = pd.read_csv(out).iloc[0]
    assert summary["method"] == "crosssteer_source"
    assert summary["repairs"] == 1
    assert summary["breaks"] == 1
    assert summary["delta_text_tokens"] == pytest.approx(0.0)
    assert summary["repair_delta_text_tokens"] == pytest.approx(2.0)
    assert summary["break_delta_text_tokens"] == pytest.approx(-2.0)


def test_selection_preflight_materializes_score_blind_runtime_and_family_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from scripts import revision_select_validation as selection

    model = tmp_path / "target-model"
    model.mkdir()
    run = tmp_path / "crosssteer"
    run.mkdir()
    (run / "resolved_manifest.json").write_text(json.dumps({"target_model": str(model)}))
    calls: dict[str, object] = {}

    def fake_runtime(run_dir: Path, generation_config: Path) -> Path:
        calls["runtime"] = (Path(run_dir), Path(generation_config))
        return Path(run_dir) / "generation_runtime_audit.json"

    def fake_family(run_dirs: dict[str, Path], output: Path) -> Path:
        calls["family"] = {name: Path(path) for name, path in run_dirs.items()}
        output.write_text('{"status":"passed"}\n')
        return output

    monkeypatch.setattr(selection, "write_generation_runtime_audit", fake_runtime)
    monkeypatch.setattr(selection, "write_validation_family_audit", fake_family)
    selection_out = tmp_path / "selection.json"
    audit_path, digest = selection._write_family_integrity_audit(
        [("crosssteer_source", run)],
        output=selection_out,
    )

    assert audit_path == tmp_path / "selection_family_integrity.json"
    assert calls["runtime"] == (run.resolve(), model / "generation_config.json")
    assert calls["family"] == {"crosssteer_source": run.resolve()}
    assert digest == hashlib.sha256(audit_path.read_bytes()).hexdigest()


def test_calibration_generation_provenance_requires_pre_generation_manifest(tmp_path: Path):
    from geoprobe.revision.calibration_provenance import audit_calibration_generation_provenance

    run = tmp_path / "calibration"
    run.mkdir()
    (run / "DONE").write_text("complete\n")
    config = run / "config.yaml"
    config.write_text("exp_id: test\n")
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    (run / "resolved_generation_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-calibration-generation-provenance-v1",
                "config_sha256": config_sha,
                "model_generation_eos_token_ids": [10, 11],
                "generation": {
                    "max_new_tokens": 8,
                    "eos_token_id": "model.generation_config.eos_token_id",
                },
            }
        )
    )
    labels = pd.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "sample_idx": [0, 0, 0],
            "n_gen_tokens": [3, 8, 8],
            "stop_reason": ["eos", "max_new_tokens", "eos"],
            "truncated": [False, True, False],
        }
    )
    labels.to_parquet(run / "labels.parquet", index=False)
    audit = audit_calibration_generation_provenance(run, max_new_tokens=8)
    assert audit["model_generation_eos_token_ids"] == [10, 11]
    assert audit["stop_reason_counts"] == {"eos": 2, "max_new_tokens": 1}
    assert audit["truncation_rate"] == pytest.approx(1 / 3)

    labels.loc[0, "truncated"] = True
    labels.to_parquet(run / "labels.parquet", index=False)
    with pytest.raises(ValueError, match="inconsistent truncation"):
        audit_calibration_generation_provenance(run, max_new_tokens=8)


def test_vector_registry_binds_audited_generation_provenance(tmp_path: Path):
    from geoprobe.revision.protocol import RevisionSplit
    from geoprobe.revision.vector_registry import build_vector_registry

    def make_run(name: str) -> Path:
        run = tmp_path / name
        (run / "trajectories").mkdir(parents=True)
        (run / "DONE").write_text("done\n")
        (run / "config.yaml").write_text(f"name: {name}\n")
        pd.DataFrame(
            {
                "sample_id": [0, 1],
                "sample_idx": [0, 0],
                "correct": [True, False],
                "n_gen_tokens": [3, 3],
            }
        ).to_parquet(run / "labels.parquet", index=False)
        for sample_id in (0, 1):
            torch.save(
                {
                    "sample_id": sample_id,
                    "sample_idx": 0,
                    "hidden_states": torch.ones(3, 3, 4),
                    "generated_token_ids": torch.tensor([1, 2, 3]),
                    "generated_text": "x",
                    "prompt": "p",
                    "prompt_len": 1,
                    "sequence_logprob": 0.0,
                    "model_id": name,
                    "dtype": "bfloat16",
                },
                run / "trajectories" / f"sample_{sample_id:04d}_idx_0.pt",
            )
        return run

    source, target = make_run("source"), make_run("target")

    def make_audit(run: Path) -> Path:
        path = tmp_path / f"{run.name}-audit.json"
        entry = {
            "run": str(run.resolve()),
            "eligible_for_direction": True,
            "labels_sha256": hashlib.sha256((run / "labels.parquet").read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((run / "config.yaml").read_bytes()).hexdigest(),
            "generation_provenance": {"truncation_rate": 0.0},
        }
        path.write_text(
            json.dumps(
                {
                    "protocol": "tacl-11241-calibration-capability-audit-v1",
                    "max_budget_hit_rate": 0.25,
                    "runs": [entry],
                }
            )
        )
        return path

    source_audit, target_audit = make_audit(source), make_audit(target)
    vectors, metadata = build_vector_registry(
        source,
        target,
        [0, 1],
        split=RevisionSplit(
            source_train=(0, 1), validation=(2,), locked_test=(3,), replication_reserve=(4,)
        ),
        source_calibration_audit=source_audit,
        target_calibration_audit=target_audit,
    )
    assert set(vectors) == {"crosssteer_source", "target_calibrated"}
    assert metadata["source_calibration_audit"]["path"] == str(source_audit.resolve())
    assert metadata["target_calibration_audit"]["run"]["run"] == str(target.resolve())

    bad = json.loads(source_audit.read_text())
    bad["max_budget_hit_rate"] = 0.5
    source_audit.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="relaxes the 25%"):
        build_vector_registry(
            source,
            target,
            [0, 1],
            split=RevisionSplit(
                source_train=(0, 1), validation=(2,), locked_test=(3,), replication_reserve=(4,)
            ),
            source_calibration_audit=source_audit,
            target_calibration_audit=target_audit,
        )


def test_qwen_correction_chain_accepts_only_logged_expected_provenance_gate():
    script = Path("scripts/revision_run_qwen_calibration_correction_v3.sh").read_text()
    assert 'V2_SELECTION_LOG="$V2_CHAIN/freeze-selection.log"' in script
    assert "lacks a bound source_calibration_audit" in script
    assert 'grep -Eq "$EXPECTED_GATE" "$V2_SELECTION_LOG"' in script
    assert "failure_reason_path=%s" in script
    assert 'export GEOPROBE_RUNS="$ROOT"' in script


def test_signature_confirmation_waits_for_successful_steering_correction():
    script = Path("scripts/revision_run_signature_confirmation_v1.sh").read_text()
    assert "PAUSED_FOR_CORRECTION_FAILURE" in script
    assert "signature confirmation paused pending correction restart" in script
    assert "diagnostic confirmation remains independent and will proceed" not in script


def test_signature_restart_uses_isolated_chain_names():
    confirmation = Path("scripts/revision_run_signature_confirmation_v1.sh").read_text()
    figures = Path("scripts/revision_run_signature_figure_postprocess_v1.sh").read_text()
    assert (
        "SIGNATURE_CHAIN_NAME=${SIGNATURE_CHAIN_NAME:-qwen-signature-confirmation-v1}"
        in confirmation
    )
    assert 'CHAIN="$ROOT/$SIGNATURE_CHAIN_NAME"' in confirmation
    assert "SIGNATURE_CHAIN_NAME=${SIGNATURE_CHAIN_NAME:-qwen-signature-confirmation-v1}" in figures
    assert (
        "SIGNATURE_FIGURE_CHAIN_NAME=${SIGNATURE_FIGURE_CHAIN_NAME:-qwen-signature-figure-postprocess-v1}"
        in figures
    )
    assert 'PARENT="$ROOT/$SIGNATURE_CHAIN_NAME"' in figures


def test_signature_audit_binds_dependent_local_values_separately():
    script = Path("scripts/revision_run_signature_confirmation_v1.sh").read_text()
    assert (
        'local family=$1\n  local budget=$2\n  local out="$ROOT/signature-confirmation-${family}-gsm8k-b${budget}-v1"'
        in script
    )
    assert "local family=$1 budget=$2 out=" not in script


def test_v3_downstream_launcher_does_not_require_untracked_script_execute_bits():
    launcher = Path("scripts/revision_launch_v3_downstream.sh").read_text()
    assert 'bash "$REPO/scripts/revision_run_qwen_ood_chain_v3.sh"' in launcher
    assert 'bash "$REPO/scripts/revision_run_qwen_label_efficiency_chain_v3.sh"' in launcher
    assert 'bash "$REPO/scripts/revision_run_qwen_long_context_chain_v3.sh"' in launcher


def test_v3_downstream_waits_for_signature_chain_before_using_gpu():
    launcher = Path("scripts/revision_launch_v3_downstream.sh").read_text()
    assert 'SIGNATURE_CHAIN="${SIGNATURE_CHAIN:-$ROOT/qwen-signature-confirmation-v3}"' in launcher
    assert "waiting for signature confirmation chain" in launcher
    assert "signature confirmation chain complete" in launcher


def test_active_body_figures_follow_one_visual_grammar_contract():
    """Prevent heterogeneous result panels from being recombined for space.

    The visual content inside a PDF remains a human QA responsibility, but the
    manuscript-level contract is machine checked here: every numbered figure
    has one included artifact, the active quantitative artifacts are exactly
    the audited homogeneous set, and accuracy and token-cost results remain in
    separate figure environments.
    """
    section_paths = sorted(Path("paper/sections").glob("*.tex"))
    manuscript = "\n".join(path.read_text() for path in section_paths)
    blocks = re.findall(
        r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",
        manuscript,
        flags=re.DOTALL,
    )
    included = []
    for block in blocks:
        artifacts = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", block)
        assert len(artifacts) == 1, block
        included.extend(artifacts)

    expected = {
        "figures/fig1_overview.png",
        "figures/fig_curvature_schematic.pdf",
        "figures/fig2_signature_heatmaps.pdf",
        "figures/fig_signature_distance_stability.pdf",
        "figures/fig3_correctness_separation.pdf",
        "figures/fig_locked_comparison_forest.pdf",
    }
    assert set(included) == expected
    assert not any("fig3_crosssteer_real" in artifact for artifact in included)
    assert not any("fig_long_context_" in artifact for artifact in included)

    audit = Path("revision/FIGURE_LAYOUT_AUDIT.md").read_text()
    assert "Do not use heterogeneous quantitative composites" in audit
    assert "accuracy with token-cost marks" in audit
