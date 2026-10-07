"""Structural capability and token-budget checks for revision calibration runs.

This is intentionally not an ``official-score`` comparator.  It prevents a
steering vector from being built from a degenerate or heavily budget-saturated
calibration run, while preserving the separate requirement to compare task
performance with an official/reference protocol before making a capability
claim.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import pandas as pd

_REQUIRED_COLUMNS = {
    "sample_id",
    "sample_idx",
    "correct",
    "n_gen_tokens",
}


@dataclass(frozen=True)
class CalibrationAudit:
    """Audit result for one train-only calibration artifact."""

    name: str
    n_rows: int
    n_problem_ids: int
    accuracy: float
    n_correct: int
    n_incorrect: int
    budget_hit_rate: float
    eligible_for_direction: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_calibration_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    expected_ids: Iterable[int],
    max_new_tokens: int,
    min_class_count: int = 10,
    max_budget_hit_rate: float = 0.25,
) -> CalibrationAudit:
    """Check calibration completeness before computing correct-minus-wrong means.

    The check is deliberately structural: it verifies one completion per
    expected training problem, both labels in useful quantity, and a bounded
    fraction of outputs that consume the configured generation budget.  It does
    not compare different models' accuracies or use any validation/locked label.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if min_class_count <= 0:
        raise ValueError("min_class_count must be positive")
    if not 0.0 <= max_budget_hit_rate <= 1.0:
        raise ValueError("max_budget_hit_rate must be in [0, 1]")
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing required label columns: {missing}")

    expected = {int(value) for value in expected_ids}
    observed = {int(value) for value in frame["sample_id"].tolist()}
    failures: list[str] = []
    if observed != expected:
        missing_ids = sorted(expected - observed)
        unexpected_ids = sorted(observed - expected)
        failures.append(
            f"problem_ids_mismatch: missing={missing_ids[:8]} unexpected={unexpected_ids[:8]}"
        )
    grouped = frame.groupby("sample_id", sort=False).size()
    if not grouped.empty and not grouped.eq(1).all():
        failures.append("expected_exactly_one_completion_per_problem")
    if not frame.empty and not frame["sample_idx"].eq(0).all():
        failures.append("expected_sample_idx_zero_for_calibration")

    correct = frame["correct"].astype(bool)
    n_rows = int(len(frame))
    n_correct = int(correct.sum())
    n_incorrect = n_rows - n_correct
    accuracy = float(correct.mean()) if n_rows else 0.0
    budget_hit_rate = (
        float((frame["n_gen_tokens"].astype(int) >= max_new_tokens).mean()) if n_rows else 1.0
    )
    if n_correct < min_class_count:
        failures.append(f"too_few_correct={n_correct}<min_class_count={min_class_count}")
    if n_incorrect < min_class_count:
        failures.append(f"too_few_incorrect={n_incorrect}<min_class_count={min_class_count}")
    if budget_hit_rate > max_budget_hit_rate:
        failures.append(
            f"budget_hit_rate={budget_hit_rate:.3f}>max_budget_hit_rate={max_budget_hit_rate:.3f}"
        )

    return CalibrationAudit(
        name=name,
        n_rows=n_rows,
        n_problem_ids=len(observed),
        accuracy=accuracy,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        budget_hit_rate=budget_hit_rate,
        eligible_for_direction=not failures,
        failures=tuple(failures),
    )
