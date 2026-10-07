"""Deterministic selection for the preregistered steering validation grid.

This module deliberately contains no correctness recomputation and no adjustable
thresholds.  It applies the rule declared in ``revision/VALIDATION_GRID_V1.md``
to a fully completed validation run and emits an auditable decision record
before any locked-test generation is allowed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ValidationSelection:
    """One method's fixed validation decision."""

    method: str
    eligible: bool
    reason: str
    baseline_mean_text_tokens: float
    baseline_mean_repeated_4gram_fraction: float
    cells_total: int
    cells_passing_behavior_guard: int
    layer: int | None = None
    alpha: float | None = None
    validation_accuracy: float | None = None
    validation_delta: float | None = None
    mean_text_tokens: float | None = None
    mean_repeated_4gram_fraction: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def baseline_behavior(rows: pd.DataFrame) -> tuple[float, float]:
    """Return the preregistered baseline token/repetition reference values."""
    required = {"method", "text_tokens", "repeated_4gram_fraction"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"per-sample rows are missing required columns: {sorted(missing)}")
    baseline = rows[rows["method"] == "baseline"]
    if baseline.empty:
        raise ValueError("per-sample rows contain no baseline")
    return (
        float(baseline["text_tokens"].mean()),
        float(baseline["repeated_4gram_fraction"].mean()),
    )


def select_validation_cell(
    summary: pd.DataFrame,
    *,
    method: str,
    baseline_mean_text_tokens: float,
    baseline_mean_repeated_4gram_fraction: float,
    max_repetition_increase: float = 0.05,
    max_length_multiple: float = 1.5,
) -> ValidationSelection:
    """Apply the frozen guard and lexicographic tie rule to one method.

    Ranking is exactly: maximum accuracy, then smaller alpha, then lower
    repetition, then shorter output, then lower layer index.  A cell that
    breaches either behaviour guard cannot be selected regardless of accuracy.
    """
    required = {
        "method",
        "layer",
        "alpha",
        "method_accuracy",
        "delta",
        "mean_text_tokens",
        "mean_repeated_4gram_fraction",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"summary is missing required columns: {sorted(missing)}")
    if max_repetition_increase < 0 or max_length_multiple <= 0:
        raise ValueError("behaviour thresholds must be non-negative/positive")

    cells = summary[summary["method"] == method].copy()
    if cells.empty:
        raise ValueError(f"summary contains no rows for method={method!r}")
    if cells.duplicated(["layer", "alpha"]).any():
        raise ValueError(f"summary contains duplicate layer/alpha cells for {method!r}")

    repetition_limit = baseline_mean_repeated_4gram_fraction + max_repetition_increase
    length_limit = baseline_mean_text_tokens * max_length_multiple
    eligible = cells[
        (cells["mean_repeated_4gram_fraction"] <= repetition_limit)
        & (cells["mean_text_tokens"] <= length_limit)
    ].copy()
    common = {
        "method": method,
        "baseline_mean_text_tokens": baseline_mean_text_tokens,
        "baseline_mean_repeated_4gram_fraction": baseline_mean_repeated_4gram_fraction,
        "cells_total": int(len(cells)),
        "cells_passing_behavior_guard": int(len(eligible)),
    }
    if eligible.empty:
        return ValidationSelection(
            eligible=False,
            reason="no_cell_passed_preregistered_behavior_guard",
            **common,
        )

    selected = eligible.sort_values(
        [
            "method_accuracy",
            "alpha",
            "mean_repeated_4gram_fraction",
            "mean_text_tokens",
            "layer",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).iloc[0]
    return ValidationSelection(
        eligible=True,
        reason="selected_by_preregistered_validation_rule",
        layer=int(selected["layer"]),
        alpha=float(selected["alpha"]),
        validation_accuracy=float(selected["method_accuracy"]),
        validation_delta=float(selected["delta"]),
        mean_text_tokens=float(selected["mean_text_tokens"]),
        mean_repeated_4gram_fraction=float(selected["mean_repeated_4gram_fraction"]),
        **common,
    )
