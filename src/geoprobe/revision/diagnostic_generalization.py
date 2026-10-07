"""Held-out diagnostic retrieval for post-training trajectory signatures.

This module deliberately treats a complete metric-by-layer signature as the
object of inference.  It never selects a single layer or metric after seeing a
result.  It is separate from steering: its output cannot select a direction,
layer, alpha, prompt, or decoder.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from geoprobe.revision.signature_stability import SignatureData

_PARTITIONS = ("A", "B", "C")


@dataclass(frozen=True)
class PartitionSelection:
    """Balanced correct/incorrect row indices for one model/partition."""

    incorrect_rows: np.ndarray
    correct_rows: np.ndarray

    @property
    def n_incorrect(self) -> int:
        return int(self.incorrect_rows.size)

    @property
    def n_correct(self) -> int:
        return int(self.correct_rows.size)


def align_relative_depth(items: list[SignatureData], *, n_depth_bins: int) -> list[SignatureData]:
    """Align common metrics and problem IDs on a fixed relative-depth grid.

    This makes a matched-lineage test exact when depths already agree and makes
    a cross-lineage replication possible without choosing a favorable physical
    layer.  Interpolation operates per sample and metric before AUC computation.
    """
    if len(items) < 2:
        raise ValueError("at least two models are required")
    if n_depth_bins < 2:
        raise ValueError("n_depth_bins must be at least two")
    common_ids = set(items[0].sample_ids)
    common_metrics = {metric for metric, _ in items[0].cells}
    for item in items[1:]:
        common_ids &= set(item.sample_ids)
        common_metrics &= {metric for metric, _ in item.cells}
    if not common_ids or not common_metrics:
        raise ValueError("models have no common problem IDs or metric names")
    ordered_ids = np.asarray(sorted(common_ids), dtype=int)
    ordered_metrics = tuple(sorted(common_metrics))
    grid = np.linspace(0.0, 1.0, n_depth_bins)
    aligned: list[SignatureData] = []
    for item in items:
        id_order = {int(sample_id): row for row, sample_id in enumerate(item.sample_ids)}
        rows = np.asarray([id_order[int(sample_id)] for sample_id in ordered_ids], dtype=int)
        columns_by_metric: dict[str, list[tuple[int, int]]] = {metric: [] for metric in ordered_metrics}
        for column, (metric, layer) in enumerate(item.cells):
            if metric in columns_by_metric:
                columns_by_metric[metric].append((int(layer), column))
        interpolated: list[np.ndarray] = []
        for metric in ordered_metrics:
            pairs = sorted(columns_by_metric[metric])
            if len(pairs) < 2:
                raise ValueError(f"{item.name}: metric {metric!r} has fewer than two depth points")
            layers = np.asarray([layer for layer, _ in pairs], dtype=float)
            if np.unique(layers).size != layers.size:
                raise ValueError(f"{item.name}: metric {metric!r} has duplicate layer entries")
            relative_depth = layers / layers.max() if layers.max() > 0 else np.zeros_like(layers)
            columns = np.asarray([column for _, column in pairs], dtype=int)
            values = item.scores[np.ix_(rows, columns)]
            interpolated.append(
                np.vstack([np.interp(grid, relative_depth, row) for row in values])
            )
        aligned.append(
            SignatureData(
                name=item.name,
                sample_ids=ordered_ids,
                incorrect=item.incorrect[rows],
                scores=np.concatenate(interpolated, axis=1),
                cells=tuple(
                    (metric, depth_bin)
                    for metric in ordered_metrics
                    for depth_bin in range(n_depth_bins)
                ),
                n_gen_tokens=(item.n_gen_tokens[rows] if item.n_gen_tokens is not None else None),
            )
        )
    return aligned


def stable_partitions(sample_ids: np.ndarray, *, salt: str) -> dict[str, np.ndarray]:
    """Assign IDs to immutable A/B/C partitions using SHA-256, not row order."""
    if not salt:
        raise ValueError("salt must be non-empty")
    buckets: dict[str, list[int]] = {name: [] for name in _PARTITIONS}
    for sample_id in np.asarray(sample_ids, dtype=int):
        digest = hashlib.sha256(f"{salt}:{int(sample_id)}".encode()).digest()
        buckets[_PARTITIONS[int.from_bytes(digest[:8], "big") % len(_PARTITIONS)]].append(
            int(sample_id)
        )
    return {name: np.asarray(sorted(ids), dtype=int) for name, ids in buckets.items()}


def _hash_choose(ids: np.ndarray, *, count: int, salt: str) -> np.ndarray:
    """Choose a fixed subset without using observed trajectory scores."""
    ids = np.asarray(ids, dtype=int)
    if count < 1 or count > ids.size:
        raise ValueError("invalid fixed class-balance count")
    ranked = sorted(
        (hashlib.sha256(f"{salt}:{int(sample_id)}".encode()).digest(), int(sample_id))
        for sample_id in ids
    )
    return np.asarray([sample_id for _, sample_id in ranked[:count]], dtype=int)


def _rows_for_ids(data: SignatureData, sample_ids: np.ndarray) -> np.ndarray:
    index = {int(sample_id): row for row, sample_id in enumerate(data.sample_ids)}
    try:
        return np.asarray([index[int(sample_id)] for sample_id in sample_ids], dtype=int)
    except KeyError as error:  # pragma: no cover - guarded by alignment
        raise ValueError(f"{data.name}: requested a non-aligned sample ID") from error


def balanced_partition_selections(
    items: list[SignatureData],
    sample_ids: np.ndarray,
    *,
    salt: str,
    min_per_class: int = 10,
) -> dict[str, PartitionSelection]:
    """Balance outcome counts across models using only labels and stable hashes.

    A model with a different raw accuracy must not receive a larger AUC signal
    merely because it contributes more examples of one correctness class.
    """
    if not items:
        raise ValueError("at least one model is required")
    if min_per_class < 2:
        raise ValueError("min_per_class must be at least two")
    available: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for item in items:
        rows = _rows_for_ids(item, sample_ids)
        incorrect_ids = item.sample_ids[rows][item.incorrect[rows]]
        correct_ids = item.sample_ids[rows][~item.incorrect[rows]]
        available[item.name] = (incorrect_ids, correct_ids)
    n_incorrect = min(ids[0].size for ids in available.values())
    n_correct = min(ids[1].size for ids in available.values())
    if n_incorrect < min_per_class or n_correct < min_per_class:
        counts = {name: (int(wrong.size), int(right.size)) for name, (wrong, right) in available.items()}
        raise ValueError(
            "every model/partition needs at least "
            f"{min_per_class} correct and {min_per_class} incorrect examples after alignment; "
            f"observed counts={counts}"
        )
    selections: dict[str, PartitionSelection] = {}
    for item in items:
        wrong_ids, right_ids = available[item.name]
        chosen_wrong = _hash_choose(
            wrong_ids,
            count=n_incorrect,
            salt=f"{salt}:{item.name}:incorrect",
        )
        chosen_right = _hash_choose(
            right_ids,
            count=n_correct,
            salt=f"{salt}:{item.name}:correct",
        )
        selections[item.name] = PartitionSelection(
            incorrect_rows=_rows_for_ids(item, chosen_wrong),
            correct_rows=_rows_for_ids(item, chosen_right),
        )
    return selections


def residualized_signature(
    data: SignatureData,
    selection: PartitionSelection,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """AUC vector after residualizing each score against generated-token length."""
    if data.n_gen_tokens is None:
        raise ValueError(f"{data.name}: diagnostic generalization requires n_gen_tokens")
    wrong_rows = selection.incorrect_rows
    right_rows = selection.correct_rows
    if rng is not None:
        wrong_rows = rng.choice(wrong_rows, size=wrong_rows.size, replace=True)
        right_rows = rng.choice(right_rows, size=right_rows.size, replace=True)
    rows = np.concatenate([wrong_rows, right_rows])
    labels = data.incorrect[rows].astype(int)
    lengths = np.log1p(np.asarray(data.n_gen_tokens[rows], dtype=float))
    design = np.column_stack([lengths, np.ones_like(lengths)])
    residuals = data.scores[rows].astype(float) - design @ np.linalg.lstsq(
        design, data.scores[rows].astype(float), rcond=None
    )[0]
    return np.asarray(
        [roc_auc_score(labels, residuals[:, column]) for column in range(residuals.shape[1])],
        dtype=float,
    )


def _signature_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm((left - 0.5) - (right - 0.5)))


def _partition_signatures(
    items: list[SignatureData],
    partitions: dict[str, np.ndarray],
    *,
    salt: str,
    min_per_class: int,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, PartitionSelection]]]:
    signatures: dict[str, dict[str, np.ndarray]] = {}
    selections: dict[str, dict[str, PartitionSelection]] = {}
    for partition, sample_ids in partitions.items():
        selected = balanced_partition_selections(
            items,
            sample_ids,
            salt=f"{salt}:{partition}",
            min_per_class=min_per_class,
        )
        selections[partition] = selected
        signatures[partition] = {
            item.name: residualized_signature(item, selected[item.name], rng=rng) for item in items
        }
    return signatures, selections


def partition_signature_cells(
    items: list[SignatureData],
    *,
    salt: str,
    n_depth_bins: int = 29,
    min_per_class: int = 10,
) -> pd.DataFrame:
    """Return every fixed, length-residualized signature cell by hash partition.

    This is the numeric source for the revised heatmap.  It intentionally
    materializes *all* predeclared metric--depth cells and records balanced
    class support, so a later figure cannot silently retain only visually
    favorable layers or metrics.
    """
    if min_per_class < 2:
        raise ValueError("min_per_class must be at least two")
    items = align_relative_depth(items, n_depth_bins=n_depth_bins)
    if any(item.n_gen_tokens is None for item in items):
        missing = [item.name for item in items if item.n_gen_tokens is None]
        raise ValueError(f"n_gen_tokens is required for every run; missing {missing}")
    partitions = stable_partitions(items[0].sample_ids, salt=salt)
    if any(ids.size < 6 for ids in partitions.values()):
        sizes = {name: int(ids.size) for name, ids in partitions.items()}
        raise ValueError(f"each hash partition needs at least six problems; observed {sizes}")
    signatures, selections = _partition_signatures(
        items,
        partitions,
        salt=salt,
        min_per_class=min_per_class,
    )
    rows: list[dict[str, object]] = []
    for partition in _PARTITIONS:
        for item in items:
            selection = selections[partition][item.name]
            for (metric, depth_bin), auc in zip(
                item.cells, signatures[partition][item.name], strict=True
            ):
                rows.append(
                    {
                        "partition": partition,
                        "model": item.name,
                        "metric": metric,
                        "relative_depth_bin": int(depth_bin),
                        "auc_incorrect_vs_correct": float(auc),
                        "n_incorrect": selection.n_incorrect,
                        "n_correct": selection.n_correct,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["partition", "model", "metric", "relative_depth_bin"]
    ).reset_index(drop=True)


def _retrieval_rows(
    signatures: dict[str, dict[str, np.ndarray]],
    selections: dict[str, dict[str, PartitionSelection]],
) -> pd.DataFrame:
    names = sorted(next(iter(signatures.values())))
    rows: list[dict[str, object]] = []
    for query_partition, gallery_partition in permutations(_PARTITIONS, 2):
        query = signatures[query_partition]
        gallery = signatures[gallery_partition]
        for name in names:
            distances = {other: _signature_distance(query[name], gallery[other]) for other in names}
            predicted = min(names, key=lambda other: (distances[other], other))
            rows.append(
                {
                    "query_partition": query_partition,
                    "gallery_partition": gallery_partition,
                    "query_model": name,
                    "predicted_model": predicted,
                    "correct_retrieval": bool(predicted == name),
                    "predicted_distance": distances[predicted],
                    "self_distance": distances[name],
                    "query_n_incorrect": selections[query_partition][name].n_incorrect,
                    "query_n_correct": selections[query_partition][name].n_correct,
                    "gallery_n_incorrect": selections[gallery_partition][name].n_incorrect,
                    "gallery_n_correct": selections[gallery_partition][name].n_correct,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["query_partition", "gallery_partition", "query_model"]
    ).reset_index(drop=True)


def _pairwise_partition_distances(signatures: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    names = sorted(next(iter(signatures.values())))
    rows: list[dict[str, object]] = []
    for partition, vectors in signatures.items():
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                rows.append(
                    {
                        "partition": partition,
                        "left": left,
                        "right": right,
                        "distance": _signature_distance(vectors[left], vectors[right]),
                    }
                )
    return pd.DataFrame(rows)


def permutation_p_value(
    retrieval: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> float:
    """Permutation null for model identity, preserving every retrieval distance."""
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    required = {"query_partition", "gallery_partition", "query_model", "predicted_model", "correct_retrieval"}
    if missing := required - set(retrieval.columns):
        raise ValueError(f"retrieval missing {sorted(missing)}")
    observed = float(retrieval["correct_retrieval"].mean())
    names = sorted(retrieval["query_model"].unique())
    rng = np.random.default_rng(seed)
    null_values = np.empty(n_permutations, dtype=float)
    groups = list(retrieval.groupby(["query_partition", "gallery_partition"], sort=True))
    for draw in range(n_permutations):
        outcomes: list[bool] = []
        for _, group in groups:
            assigned = dict(zip(names, rng.permutation(names), strict=True))
            outcomes.extend(assigned[predicted] == query for predicted, query in zip(
                group["predicted_model"], group["query_model"], strict=True
            ))
        null_values[draw] = float(np.mean(outcomes))
    return float((1 + np.count_nonzero(null_values >= observed)) / (n_permutations + 1))


def heldout_model_retrieval(
    items: list[SignatureData],
    *,
    salt: str,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
    n_depth_bins: int = 29,
    min_per_class: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run frozen split retrieval with balanced length-residualized signatures."""
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if min_per_class < 2:
        raise ValueError("min_per_class must be at least two")
    items = align_relative_depth(items, n_depth_bins=n_depth_bins)
    if any(item.n_gen_tokens is None for item in items):
        missing = [item.name for item in items if item.n_gen_tokens is None]
        raise ValueError(f"n_gen_tokens is required for every run; missing {missing}")
    partitions = stable_partitions(items[0].sample_ids, salt=salt)
    if any(ids.size < 6 for ids in partitions.values()):
        sizes = {name: int(ids.size) for name, ids in partitions.items()}
        raise ValueError(f"each hash partition needs at least six problems; observed {sizes}")
    point_signatures, point_selections = _partition_signatures(
        items,
        partitions,
        salt=salt,
        min_per_class=min_per_class,
    )
    retrieval = _retrieval_rows(point_signatures, point_selections)
    distances = _pairwise_partition_distances(point_signatures)

    rng = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, object]] = []
    for draw in range(n_bootstrap):
        signatures, selections = _partition_signatures(
            items,
            partitions,
            salt=salt,
            min_per_class=min_per_class,
            rng=rng,
        )
        frame = _retrieval_rows(signatures, selections)
        bootstrap_rows.append(
            {"bootstrap_id": draw, "macro_top1": float(frame["correct_retrieval"].mean())}
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    summary: dict[str, object] = {
        "protocol": "tacl-11241-diagnostic-generalization-v1",
        "models": [item.name for item in items],
        "n_models": len(items),
        "chance_top1": 1.0 / len(items),
        "n_common_problem_ids": int(items[0].sample_ids.size),
        "partition_sizes": {name: int(ids.size) for name, ids in partitions.items()},
        "point_macro_top1": float(retrieval["correct_retrieval"].mean()),
        "bootstrap_ci_low": float(bootstrap["macro_top1"].quantile(0.025)),
        "bootstrap_ci_high": float(bootstrap["macro_top1"].quantile(0.975)),
        "permutation_p_value": permutation_p_value(
            retrieval,
            n_permutations=n_permutations,
            seed=seed + 1,
        ),
        "n_bootstrap": n_bootstrap,
        "n_permutations": n_permutations,
        "seed": seed,
        "partition_salt": salt,
        "length_control": "per-cell OLS residual against log1p(n_gen_tokens)",
        "class_balance": "minimum correct/incorrect count across models within each hash partition",
        "minimum_per_class": min_per_class,
        "selection": "stable SHA-256 by model/class/sample_id; no metric/layer/result selection",
        "relative_depth_bins": n_depth_bins,
    }
    return retrieval, distances, bootstrap, summary
