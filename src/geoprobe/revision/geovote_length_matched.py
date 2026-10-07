"""Length-matched null controls for the retrospective GeoVote audit.

The control preserves the original candidate pool and the complete multiset of
geometry scores for every problem.  It only permutes scores within adjacent
pairs after sorting candidates by generated length.  This destroys score
information that is not stable under a fine-grained length match while keeping
the candidate answers, lengths, and majority outcome unchanged.

This module is an audit control, not a new selector and not a replacement for a
locked benchmark result.  It must be fit and evaluated on the same retrospective
candidate pool used by the existing GeoVote length audit.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from geoprobe.revision.geovote import same_pool_outcomes


def _adjacent_pairs(group: pd.DataFrame) -> Iterator[np.ndarray]:
    """Yield stable adjacent index pairs after sorting by generated length."""
    ordered = group.sort_values(
        ["n_gen_tokens", "sample_idx"], kind="stable"
    )
    indices = ordered.index.to_numpy()
    for start in range(0, len(indices) - 1, 2):
        yield indices[start : start + 2]


def permute_scores_within_length_pairs(
    frame: pd.DataFrame,
    *,
    score_column: str,
    rng: np.random.Generator,
    output_column: str = "length_matched_score",
) -> pd.DataFrame:
    """Create one length-matched score permutation without using labels.

    Candidates are sorted within each problem by ``n_gen_tokens`` and then by
    ``sample_idx``.  Geometry scores are independently shuffled inside each
    adjacent pair.  An odd final candidate, if present, is left unchanged.
    The operation preserves every problem's candidate pool and score multiset.
    """
    required = {"sample_id", "sample_idx", "n_gen_tokens", score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")
    if not np.isfinite(frame[score_column].astype(float)).all():
        raise ValueError(f"{score_column} must be finite")
    if (frame["n_gen_tokens"].astype(float) < 0).any():
        raise ValueError("n_gen_tokens must be non-negative")

    out = frame.copy()
    out[output_column] = out[score_column].astype(float)
    for _, group in frame.groupby("sample_id", sort=True):
        for pair in _adjacent_pairs(group):
            values = frame.loc[pair, score_column].astype(float).to_numpy()
            out.loc[pair, output_column] = rng.permutation(values)
    return out


def run_length_matched_null(
    frame: pd.DataFrame,
    *,
    score_column: str,
    permutations: int = 1000,
    seed: int = 20260806,
) -> pd.DataFrame:
    """Evaluate the length-matched null as a distribution of paired deltas."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    baseline = same_pool_outcomes(frame, score_column=score_column)["majority"].astype(int)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for permutation in range(permutations):
        permuted = permute_scores_within_length_pairs(
            frame,
            score_column=score_column,
            rng=rng,
        )
        outcomes = same_pool_outcomes(
            permuted,
            score_column="length_matched_score",
        )
        method = outcomes["residual_score_vote"].astype(int)
        paired = method.to_numpy() - baseline.to_numpy()
        rows.append(
            {
                "permutation": permutation,
                "baseline_accuracy": float(baseline.mean()),
                "method_accuracy": float(method.mean()),
                "delta_pp": float(100.0 * paired.mean()),
                "repairs": int(((baseline == 0) & (method == 1)).sum()),
                "breaks": int(((baseline == 1) & (method == 0)).sum()),
            }
        )
    return pd.DataFrame(rows)
