"""Frozen, stratified target-label subsets for the TACL revision efficiency curve."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import pandas as pd


def _hash_order(ids: Iterable[int], *, salt: str) -> list[int]:
    return sorted(
        (int(value) for value in ids),
        key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest(),
    )


def stratified_label_budget_ids(
    labels: pd.DataFrame,
    *,
    budgets: Iterable[int],
    allowed_ids: Iterable[int],
    salt: str,
) -> dict[int, list[int]]:
    """Choose deterministic, nested target-calibration subsets using labels only.

    Each budget is a total number of labeled target trajectories.  We reserve
    approximately half for each correctness class when available; any shortage
    is filled from the other class.  The class-wise ordering is a stable hash of
    problem IDs, so no geometry metric, validation result, or locked outcome can
    influence the selected subset.  Returned subsets are nested by budget.
    """
    requested = sorted({int(value) for value in budgets})
    if not requested or requested[0] <= 1:
        raise ValueError("all label budgets must be integers greater than one")
    if not salt:
        raise ValueError("salt must be non-empty")
    required = {"sample_id", "correct"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels are missing columns: {sorted(missing)}")
    work = labels.copy()
    if "sample_idx" in work.columns:
        work = work[work["sample_idx"] == 0]
    allowed = {int(value) for value in allowed_ids}
    work = work[work["sample_id"].astype(int).isin(allowed)]
    if work["sample_id"].duplicated().any():
        raise ValueError("expected exactly one target calibration row per problem")
    observed = {int(value) for value in work["sample_id"]}
    if observed != allowed:
        raise ValueError("target calibration labels do not exactly cover allowed IDs")
    correct_ids = _hash_order(work.loc[work["correct"].astype(bool), "sample_id"], salt=f"{salt}:correct")
    incorrect_ids = _hash_order(work.loc[~work["correct"].astype(bool), "sample_id"], salt=f"{salt}:incorrect")
    if not correct_ids or not incorrect_ids:
        raise ValueError("target calibration needs at least one correct and one incorrect example")
    result: dict[int, list[int]] = {}
    for budget in requested:
        if budget > len(allowed):
            raise ValueError(f"budget={budget} exceeds {len(allowed)} allowed target IDs")
        want_correct = min((budget + 1) // 2, len(correct_ids))
        want_incorrect = min(budget // 2, len(incorrect_ids))
        remaining = budget - want_correct - want_incorrect
        add_correct = min(remaining, len(correct_ids) - want_correct)
        want_correct += add_correct
        remaining -= add_correct
        want_incorrect += min(remaining, len(incorrect_ids) - want_incorrect)
        selected = correct_ids[:want_correct] + incorrect_ids[:want_incorrect]
        if len(selected) != budget:
            raise ValueError(f"could not fill target label budget={budget}")
        result[budget] = sorted(selected)
    previous: set[int] = set()
    for budget in requested:
        current = set(result[budget])
        if not previous <= current:
            raise AssertionError("hash-selected target-label budgets must be nested")
        previous = current
    return result
