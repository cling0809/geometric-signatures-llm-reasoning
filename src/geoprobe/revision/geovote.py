"""Same-pool, length-controlled GeoVote audit utilities.

This module is intentionally retrospective when used on the submitted-paper
candidate pools: those pools were already examined during the original work.
It is useful to quantify the reviewer-raised length confound, but cannot turn
such a pool into a new locked result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearLengthResidualizer:
    """A train-only linear nuisance model for geometry confidence."""

    intercept: float
    log_length_coefficient: float

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> LinearLengthResidualizer:
        required = {"geo_conf", "n_gen_tokens"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"missing required candidate columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("cannot fit a residualizer on an empty frame")
        x = np.log1p(frame["n_gen_tokens"].astype(float).to_numpy())
        y = frame["geo_conf"].astype(float).to_numpy()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("length and geometry confidence must be finite")
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        return cls(intercept=float(coef[0]), log_length_coefficient=float(coef[1]))

    def predict(self, lengths: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(lengths, dtype=float)
        if np.any(values < 0) or not np.isfinite(values).all():
            raise ValueError("lengths must be finite and non-negative")
        return self.intercept + self.log_length_coefficient * np.log1p(values)

    def residualize(self, frame: pd.DataFrame) -> pd.Series:
        return frame["geo_conf"].astype(float) - self.predict(frame["n_gen_tokens"])


def _valid_prediction(value: object) -> bool:
    return value is not None and not pd.isna(value)


def _stable_majority_prediction(group: pd.DataFrame) -> object | None:
    counts = Counter(value for value in group["pred"].tolist() if _valid_prediction(value))
    if not counts:
        return None
    best_count = max(counts.values())
    for value in group.sort_values("sample_idx")["pred"].tolist():
        if _valid_prediction(value) and counts[value] == best_count:
            return value
    raise AssertionError("unreachable")


def _prediction_correct(group: pd.DataFrame, prediction: object | None) -> bool:
    if not _valid_prediction(prediction):
        return False
    matched = group[group["pred"] == prediction]
    if matched.empty:
        return False
    values = matched["correct"].astype(bool).unique()
    if len(values) != 1:
        raise ValueError("the same parsed answer has inconsistent correctness labels")
    return bool(values[0])


def _weighted_vote(group: pd.DataFrame, weights: np.ndarray) -> bool:
    if len(group) != len(weights):
        raise ValueError("weights must align with group rows")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    totals: defaultdict[object, float] = defaultdict(float)
    ordered = group.sort_values("sample_idx")
    for (_, row), weight in zip(ordered.iterrows(), weights, strict=True):
        prediction = row["pred"]
        if _valid_prediction(prediction):
            totals[prediction] += float(weight)
    if not totals:
        return False
    best_weight = max(totals.values())
    for prediction in ordered["pred"].tolist():
        if _valid_prediction(prediction) and totals[prediction] == best_weight:
            return _prediction_correct(group, prediction)
    raise AssertionError("unreachable")


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores)
    values = np.exp(shifted)
    return values / values.sum()


def same_pool_outcomes(
    frame: pd.DataFrame,
    *,
    score_column: str,
    historical_sign: str = "min",
) -> pd.DataFrame:
    """Return problem-level outcomes for fixed same-pool voting baselines.

    ``historical_geovote`` reproduces the submitted paper's frozen vote
    (Eq.~1 of the submitted GeoVote section): each candidate is weighted by
    its historical metric value, with ``w = 1/(mu + eps)`` when the frozen
    sign is ``min`` and ``w = max(mu, 0)`` when it is ``max``; the answer
    with the largest total weight wins.  The locked GSM8K Table 4 row used
    MeanStepNorm at layer 20 with the frozen sign ``min`` (inverse weights
    on ``geo_value``), matching the generating independence run and the
    locked paired CSV.  ``residual_score_vote`` instead re-runs a softmax
    weighted vote on the (possibly length-residualized) ``score_column``.
    """
    if historical_sign not in {"min", "max"}:
        raise ValueError(f"historical_sign must be 'min' or 'max', got {historical_sign!r}")
    required = {
        "sample_id",
        "sample_idx",
        "pred",
        "correct",
        "n_gen_tokens",
        "sequence_logprob",
        score_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required candidate columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for sample_id, group in frame.groupby("sample_id", sort=True):
        group = group.sort_values("sample_idx").reset_index(drop=True)
        score = group[score_column].astype(float).to_numpy()
        if not np.isfinite(score).all():
            raise ValueError(f"non-finite {score_column} for sample_id={sample_id}")
        geo_value = group.get("geo_value")
        if geo_value is None:
            geo_weights = _softmax(score)
        else:
            raw = geo_value.astype(float).to_numpy()
            if np.any(raw < 0) or not np.isfinite(raw).all():
                raise ValueError("geo_value must be finite and non-negative for historical GeoVote")
            if historical_sign == "min":
                geo_weights = 1.0 / (raw + 1e-8)
            else:
                geo_weights = np.maximum(raw, 0.0)
            geo_weights = geo_weights / geo_weights.sum()
        residual_weights = _softmax(score)
        logprob_weights = _softmax(group["sequence_logprob"].astype(float).to_numpy())
        min_length = group["n_gen_tokens"].astype(float).to_numpy()
        shortest_index = int(np.argmin(min_length))
        longest_index = int(np.argmax(min_length))
        score_index = int(np.argmax(score))
        rows.append(
            {
                "sample_id": int(sample_id),
                "n_candidates": int(len(group)),
                "first": bool(group.loc[0, "correct"]),
                "majority": _prediction_correct(group, _stable_majority_prediction(group)),
                "shortest": bool(group.loc[shortest_index, "correct"]),
                "longest": bool(group.loc[longest_index, "correct"]),
                "score_pick": bool(group.loc[score_index, "correct"]),
                "historical_geovote": _weighted_vote(group, geo_weights),
                "residual_score_vote": _weighted_vote(group, residual_weights),
                "logprob_weighted": _weighted_vote(group, logprob_weights),
                "oracle": bool(group["correct"].astype(bool).any()),
            }
        )
    return pd.DataFrame(rows)


def correlation_with_length(frame: pd.DataFrame, score_column: str) -> float:
    """Return candidate-level Spearman correlation between score and token length."""
    if frame.empty:
        raise ValueError("cannot compute a correlation on an empty frame")
    return float(frame[[score_column, "n_gen_tokens"]].corr(method="spearman").iloc[0, 1])
