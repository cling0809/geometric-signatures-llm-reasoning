from __future__ import annotations

import numpy as np
import pytest

from geoprobe.revision.diagnostic_generalization import (
    align_relative_depth,
    balanced_partition_selections,
    heldout_model_retrieval,
    stable_partitions,
)
from geoprobe.revision.signature_stability import SignatureData


def _synthetic_items(*, with_lengths: bool = True) -> list[SignatureData]:
    ids = np.arange(180, dtype=int)
    incorrect = (ids % 2 == 0)
    rng = np.random.default_rng(7)
    effects = {
        "base": np.array([2.4, -2.2, 0.8, -0.7]),
        "math": np.array([-2.1, 2.5, 1.2, -1.3]),
        "coder": np.array([0.7, 1.0, -2.6, 2.2]),
    }
    items = []
    for name, effect in effects.items():
        scores = incorrect[:, None] * effect[None, :] + rng.normal(0.0, 0.14, size=(ids.size, 4))
        items.append(
            SignatureData(
                name=name,
                sample_ids=ids,
                incorrect=incorrect,
                scores=scores,
                cells=(("shape", 0), ("shape", 1), ("turn", 0), ("turn", 1)),
                n_gen_tokens=(20 + ids % 17).astype(float) if with_lengths else None,
            )
        )
    return items


def test_hash_partitions_are_deterministic_and_disjoint():
    ids = np.arange(90, dtype=int)
    first = stable_partitions(ids, salt="diagnostic-generalization-v1")
    second = stable_partitions(ids, salt="diagnostic-generalization-v1")
    assert {name: values.tolist() for name, values in first.items()} == {
        name: values.tolist() for name, values in second.items()
    }
    assert set().union(*(set(values) for values in first.values())) == set(ids)
    assert sum(len(values) for values in first.values()) == len(ids)
    assert all(len(values) >= 6 for values in first.values())


def test_balanced_selection_uses_common_class_counts_without_scores():
    items = _synthetic_items()
    partition = stable_partitions(items[0].sample_ids, salt="balance-v1")["A"]
    selected = balanced_partition_selections(items, partition, salt="balance-v1:A")
    incorrect_counts = {name: value.n_incorrect for name, value in selected.items()}
    correct_counts = {name: value.n_correct for name, value in selected.items()}
    assert min(incorrect_counts.values()) >= 10
    assert min(correct_counts.values()) >= 10
    assert len(set(incorrect_counts.values())) == 1
    assert len(set(correct_counts.values())) == 1
    for item in items:
        choice = selected[item.name]
        assert item.incorrect[choice.incorrect_rows].all()
        assert (~item.incorrect[choice.correct_rows]).all()



def test_relative_depth_alignment_supports_different_model_depths():
    items = _synthetic_items()
    original = items[0]
    expanded = SignatureData(
        name=original.name,
        sample_ids=original.sample_ids,
        incorrect=original.incorrect,
        scores=np.column_stack(
            [
                original.scores[:, 0],
                0.5 * (original.scores[:, 0] + original.scores[:, 1]),
                original.scores[:, 1],
                original.scores[:, 2],
                0.5 * (original.scores[:, 2] + original.scores[:, 3]),
                original.scores[:, 3],
            ]
        ),
        cells=(("shape", 0), ("shape", 1), ("shape", 2), ("turn", 0), ("turn", 1), ("turn", 2)),
        n_gen_tokens=original.n_gen_tokens,
    )
    aligned = align_relative_depth([expanded, *items[1:]], n_depth_bins=5)
    assert all(item.scores.shape == (180, 10) for item in aligned)
    assert all(item.cells == aligned[0].cells for item in aligned)


def test_heldout_retrieval_recovers_synthetic_model_identity():
    retrieval, distances, bootstrap, summary = heldout_model_retrieval(
        _synthetic_items(),
        salt="diagnostic-generalization-v1",
        n_bootstrap=40,
        n_permutations=500,
        seed=11241,
    )
    assert len(retrieval) == 3 * 2 * 3  # all ordered partition pairs x models
    assert retrieval["correct_retrieval"].all()
    assert summary["point_macro_top1"] == 1.0
    assert summary["chance_top1"] == pytest.approx(1 / 3)
    assert summary["bootstrap_ci_low"] > summary["chance_top1"]
    assert summary["permutation_p_value"] < 0.05
    assert set(distances["partition"]) == {"A", "B", "C"}
    assert bootstrap["macro_top1"].between(0.0, 1.0).all()


def test_heldout_retrieval_refuses_missing_length_control():
    with pytest.raises(ValueError, match="n_gen_tokens"):
        heldout_model_retrieval(
            _synthetic_items(with_lengths=False),
            salt="diagnostic-generalization-v1",
            n_bootstrap=5,
            n_permutations=10,
            seed=1,
        )


def test_heldout_retrieval_refuses_underpowered_correctness_partition():
    items = _synthetic_items()
    sparse = [
        SignatureData(
            name=item.name,
            sample_ids=np.arange(36, dtype=int),
            incorrect=item.incorrect[:36],
            scores=item.scores[:36],
            cells=item.cells,
            n_gen_tokens=item.n_gen_tokens[:36],
        )
        for item in items
    ]
    with pytest.raises(ValueError, match="at least 10 correct and 10 incorrect"):
        heldout_model_retrieval(
            sparse,
            salt="underpowered-v1",
            n_bootstrap=5,
            n_permutations=10,
            seed=1,
        )


def test_partition_signature_cells_materializes_all_frozen_cells():
    from geoprobe.revision.diagnostic_generalization import partition_signature_cells

    rows = partition_signature_cells(
        _synthetic_items(),
        salt="diagnostic-generalization-v1",
        n_depth_bins=5,
        min_per_class=10,
    )
    assert set(rows["partition"]) == {"A", "B", "C"}
    assert set(rows["model"]) == {"base", "coder", "math"}
    assert rows.groupby(["partition", "model"]).size().eq(10).all()
    assert rows["n_incorrect"].ge(10).all()
    assert rows["n_correct"].ge(10).all()
