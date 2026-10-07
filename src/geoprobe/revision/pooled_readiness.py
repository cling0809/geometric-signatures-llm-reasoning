"""Non-result-dependent technical audit for official-context pooled extraction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import pandas as pd

from geoprobe.revision.behavior import repeated_ngram_fraction


@dataclass(frozen=True)
class PooledReadinessAudit:
    """Technical readiness of a compact long-context extraction artifact.

    This deliberately does not inspect correctness or compare model scores.  It
    verifies only that a frozen long-context extraction is complete, has the
    expected compact representation, is not budget-saturated, and does not
    exhibit gross repetition loops before it is allowed to supply calibration
    trajectories.
    """

    n_rows: int
    n_problem_ids: int
    budget_hit_rate: float
    severe_repetition_rate: float
    eligible: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_pooled_readiness_frame(
    frame: pd.DataFrame,
    *,
    generated_text_by_id: Mapping[int, str],
    expected_ids: Iterable[int],
    max_new_tokens: int,
    expected_representation: str = "pooled_generated_state_mean_v1",
    max_budget_hit_rate: float = 0.25,
    max_severe_repetition_rate: float = 0.05,
    severe_repetition_fraction: float = 0.95,
) -> PooledReadinessAudit:
    """Audit frozen long-context extraction without looking at correctness.

    A severe loop is a completion with at least four whitespace tokens and a
    repeated 4-gram fraction at or above ``severe_repetition_fraction``.  All
    thresholds are passed by the launcher and recorded before results are read.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    for name, value in (
        ("max_budget_hit_rate", max_budget_hit_rate),
        ("max_severe_repetition_rate", max_severe_repetition_rate),
        ("severe_repetition_fraction", severe_repetition_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    required = {"sample_id", "sample_idx", "n_gen_tokens", "trajectory_representation"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required label columns: {missing}")
    expected = {int(value) for value in expected_ids}
    observed = {int(value) for value in frame["sample_id"].tolist()}
    failures: list[str] = []
    if observed != expected:
        failures.append("problem_ids_mismatch")
    counts = frame.groupby("sample_id", sort=False).size()
    if not counts.empty and not counts.eq(1).all():
        failures.append("expected_exactly_one_completion_per_problem")
    if not frame.empty and not frame["sample_idx"].eq(0).all():
        failures.append("expected_sample_idx_zero")
    if not frame.empty and not frame["trajectory_representation"].eq(expected_representation).all():
        failures.append("unexpected_trajectory_representation")
    if any(sample_id not in generated_text_by_id for sample_id in observed):
        failures.append("missing_generated_text")

    n_rows = int(len(frame))
    budget_hit_rate = (
        float((frame["n_gen_tokens"].astype(int) >= max_new_tokens).mean()) if n_rows else 1.0
    )
    if budget_hit_rate > max_budget_hit_rate:
        failures.append(
            f"budget_hit_rate={budget_hit_rate:.3f}>max_budget_hit_rate={max_budget_hit_rate:.3f}"
        )
    severe = 0
    for sample_id in observed:
        tokens = generated_text_by_id.get(sample_id, "").split()
        if len(tokens) >= 4 and repeated_ngram_fraction(tokens, 4) >= severe_repetition_fraction:
            severe += 1
    severe_rate = severe / n_rows if n_rows else 1.0
    if severe_rate > max_severe_repetition_rate:
        failures.append(
            "severe_repetition_rate="
            f"{severe_rate:.3f}>max_severe_repetition_rate={max_severe_repetition_rate:.3f}"
        )
    return PooledReadinessAudit(
        n_rows=n_rows,
        n_problem_ids=len(observed),
        budget_hit_rate=budget_hit_rate,
        severe_repetition_rate=severe_rate,
        eligible=not failures,
        failures=tuple(failures),
    )
