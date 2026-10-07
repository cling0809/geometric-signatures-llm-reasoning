"""Paired, problem-level uncertainty utilities for revision experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


@dataclass(frozen=True)
class PairedBinarySummary:
    n: int
    baseline_accuracy: float
    method_accuracy: float
    delta: float
    repairs: int
    breaks: int
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    exact_sign_p_value: float


def _validate_binary(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isin(array, [0, 1, False, True]).all():
        raise ValueError(f"{name} must contain only binary outcomes")
    return array.astype(np.int8)


def paired_binary_summary(
    baseline_correct: np.ndarray,
    method_correct: np.ndarray,
    *,
    bootstrap_reps: int = 10_000,
    seed: int = 0,
) -> PairedBinarySummary:
    """Return paired accuracy effect, percentile bootstrap CI and exact sign test.

    A positive delta denotes a method improvement.  The exact test operates only
    on discordant problem pairs (repairs versus breaks), which is appropriate for
    deterministic paired decoding outcomes.
    """
    baseline = _validate_binary(baseline_correct, "baseline_correct")
    method = _validate_binary(method_correct, "method_correct")
    if baseline.shape != method.shape:
        raise ValueError("baseline_correct and method_correct must have equal shape")
    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be positive")

    differences = method - baseline
    repairs = int(np.sum(differences == 1))
    breaks = int(np.sum(differences == -1))
    discordant = repairs + breaks
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(repairs, n=discordant, p=0.5, alternative="two-sided").pvalue)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, size=(bootstrap_reps, baseline.size))
    bootstrap_deltas = differences[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_deltas, [0.025, 0.975])

    return PairedBinarySummary(
        n=int(baseline.size),
        baseline_accuracy=float(baseline.mean()),
        method_accuracy=float(method.mean()),
        delta=float(differences.mean()),
        repairs=repairs,
        breaks=breaks,
        bootstrap_ci_low=float(ci_low),
        bootstrap_ci_high=float(ci_high),
        exact_sign_p_value=p_value,
    )


def holm_adjust(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Return Holm--Bonferroni adjusted p-values in original order.

    The result is monotone in ordered raw p-values and clipped to ``[0, 1]``.
    It is deterministic and intentionally does not decide which comparisons are
    a family; callers must declare that family in their protocol artifact.
    """
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if values.size == 0:
        return values.copy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p_values must be finite values in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    m = len(ordered)
    adjusted_ordered = np.maximum.accumulate((m - np.arange(m)) * ordered)
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_ordered
    return adjusted
