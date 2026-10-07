"""Bootstrap stability audit for submitted-paper signature topology."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class SignatureData:
    name: str
    sample_ids: np.ndarray
    incorrect: np.ndarray
    scores: np.ndarray  # [n_problems, n_metric_layer_cells]
    cells: tuple[tuple[str, int], ...]
    n_gen_tokens: np.ndarray | None = None


def _auc_matrix(incorrect: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Compute AUC(incorrect, score) independently for every feature column."""
    incorrect = np.asarray(incorrect, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != incorrect.size:
        raise ValueError("scores must have shape [n_problems, n_features]")
    positive = int(incorrect.sum())
    negative = int((~incorrect).sum())
    if positive == 0 or negative == 0:
        return np.full(scores.shape[1], np.nan)
    ranks = rankdata(scores, axis=0, method="average")
    sum_positive = ranks[incorrect].sum(axis=0)
    return (sum_positive - positive * (positive + 1) / 2.0) / (positive * negative)


def load_signature_data(name: str, run_dir: str) -> SignatureData:
    """Load one greedy run into aligned per-problem metric feature vectors."""
    run = pd.io.common.stringify_path(run_dir)
    labels = pd.read_parquet(f"{run}/labels.parquet")
    metrics = pd.read_parquet(f"{run}/metrics.parquet")
    if "sample_idx" in labels.columns:
        labels = labels[labels["sample_idx"] == 0]
    if "sample_idx" in metrics.columns:
        metrics = metrics[metrics["sample_idx"] == 0]
    required_labels = {"sample_id", "correct"}
    required_metrics = {"sample_id", "metric", "layer", "value"}
    if missing := required_labels - set(labels.columns):
        raise ValueError(f"{name}: labels missing {sorted(missing)}")
    if missing := required_metrics - set(metrics.columns):
        raise ValueError(f"{name}: metrics missing {sorted(missing)}")
    if labels["sample_id"].duplicated().any():
        raise ValueError(f"{name}: greedy labels are not unique per sample_id")
    wide = metrics.pivot(index="sample_id", columns=["metric", "layer"], values="value").sort_index()
    labels = labels.set_index("sample_id").sort_index()
    common = labels.index.intersection(wide.index)
    if common.empty:
        raise ValueError(f"{name}: no common sample IDs between labels and metrics")
    labels = labels.loc[common]
    wide = wide.loc[common]
    if wide.isna().any().any():
        raise ValueError(f"{name}: metric pivot contains missing values")
    length_column = next((column for column in ("n_gen_tokens", "text_tokens") if column in labels), None)
    n_gen_tokens = (
        labels[length_column].to_numpy(dtype=float) if length_column is not None else None
    )
    if n_gen_tokens is not None and (
        ~np.isfinite(n_gen_tokens) | (n_gen_tokens < 0)
    ).any():
        raise ValueError(f"{name}: {length_column} must contain finite non-negative lengths")
    cells = tuple((str(metric), int(layer)) for metric, layer in wide.columns)
    return SignatureData(
        name=name,
        sample_ids=common.to_numpy(dtype=int),
        incorrect=(~labels["correct"].astype(bool)).to_numpy(),
        scores=wide.to_numpy(dtype=float),
        cells=cells,
        n_gen_tokens=n_gen_tokens,
    )


def align_signature_data(items: list[SignatureData]) -> list[SignatureData]:
    """Require common problems and metric/layer cells across a comparison family."""
    if len(items) < 2:
        raise ValueError("at least two models are required")
    common_ids = set(items[0].sample_ids)
    common_cells = set(items[0].cells)
    for item in items[1:]:
        common_ids &= set(item.sample_ids)
        common_cells &= set(item.cells)
    if not common_ids or not common_cells:
        raise ValueError("models have no common problem IDs or metric/layer cells")
    ordered_ids = np.array(sorted(common_ids), dtype=int)
    ordered_cells = tuple(sorted(common_cells))
    aligned: list[SignatureData] = []
    for item in items:
        id_order = {int(sample_id): index for index, sample_id in enumerate(item.sample_ids)}
        cell_order = {cell: index for index, cell in enumerate(item.cells)}
        rows = [id_order[int(sample_id)] for sample_id in ordered_ids]
        cols = [cell_order[cell] for cell in ordered_cells]
        aligned.append(
            SignatureData(
                name=item.name,
                sample_ids=ordered_ids,
                incorrect=item.incorrect[rows],
                scores=item.scores[np.ix_(rows, cols)],
                cells=ordered_cells,
                n_gen_tokens=(item.n_gen_tokens[rows] if item.n_gen_tokens is not None else None),
            )
        )
    return aligned


def signature_point(data: SignatureData) -> np.ndarray:
    return _auc_matrix(data.incorrect, data.scores)


def pairwise_distances(signatures: dict[str, np.ndarray]) -> pd.DataFrame:
    """Euclidean distances between AUC signatures after centering at chance."""
    rows: list[dict[str, object]] = []
    for left, right in combinations(sorted(signatures), 2):
        a = signatures[left]
        b = signatures[right]
        if np.isnan(a).any() or np.isnan(b).any():
            distance = np.nan
        else:
            distance = float(np.linalg.norm((a - 0.5) - (b - 0.5)))
        rows.append({"left": left, "right": right, "distance": distance})
    return pd.DataFrame(rows)


def bootstrap_topology(
    items: list[SignatureData], *, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bootstrap pair distances and nearest-neighbor topology at problem level."""
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    items = align_signature_data(items)
    n = len(items[0].sample_ids)
    rng = np.random.default_rng(seed)
    names = [item.name for item in items]
    point = {item.name: signature_point(item) for item in items}
    point_distances = pairwise_distances(point)
    pair_values: dict[tuple[str, str], list[float]] = {
        (row.left, row.right): [] for row in point_distances.itertuples(index=False)
    }
    neighbor_counts: dict[tuple[str, str], int] = {(name, other): 0 for name in names for other in names if other != name}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_signatures = {
            item.name: _auc_matrix(item.incorrect[indices], item.scores[indices]) for item in items
        }
        distances = pairwise_distances(sample_signatures)
        for row in distances.itertuples(index=False):
            pair_values[(row.left, row.right)].append(float(row.distance))
        matrix = np.full((len(names), len(names)), np.inf)
        for row in distances.itertuples(index=False):
            left = names.index(row.left)
            right = names.index(row.right)
            matrix[left, right] = matrix[right, left] = float(row.distance)
        for index, name in enumerate(names):
            nearest = names[int(np.argmin(matrix[index]))]
            neighbor_counts[(name, nearest)] += 1
    distance_rows = []
    for row in point_distances.itertuples(index=False):
        values = np.asarray(pair_values[(row.left, row.right)])
        distance_rows.append(
            {
                "left": row.left,
                "right": row.right,
                "point_distance": row.distance,
                "bootstrap_mean": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
        )
    nearest_rows = []
    point_lookup = {
        tuple(sorted((row.left, row.right))): float(row.distance) for row in point_distances.itertuples(index=False)
    }
    for name in names:
        point_nearest = min(
            (other for other in names if other != name),
            key=lambda other: point_lookup[tuple(sorted((name, other)))],
        )
        for other in names:
            if other == name:
                continue
            nearest_rows.append(
                {
                    "model": name,
                    "candidate_neighbor": other,
                    "point_nearest_neighbor": point_nearest,
                    "bootstrap_probability": neighbor_counts[(name, other)] / n_bootstrap,
                }
            )
    point_rows = []
    for item in items:
        signature = point[item.name]
        point_rows.extend(
            {
                "model": item.name,
                "metric": metric,
                "layer": layer,
                "auc_incorrect_vs_correct": value,
            }
            for (metric, layer), value in zip(item.cells, signature, strict=True)
        )
    return pd.DataFrame(distance_rows), pd.DataFrame(nearest_rows), pd.DataFrame(point_rows)
