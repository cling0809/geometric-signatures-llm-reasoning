"""Structural audit for the sampled source-only CAA/ActAdd contrast pool.

This audit is deliberately outcome-blind with respect to validation and locked
sets.  It validates only the predeclared source pool before its completed
correct/incorrect continuations can support CAA, ActAdd, sparse-CAA or the
SAE-space control fit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import pandas as pd

_REQUIRED_COLUMNS = {"sample_id", "sample_idx", "correct", "n_gen_tokens"}


@dataclass(frozen=True)
class ContrastPoolAudit:
    """Audit facts for a fixed multi-sample source contrast pool."""

    n_rows: int
    n_problem_ids: int
    n_samples_per_problem: int
    n_correct: int
    n_incorrect: int
    budget_hit_rate: float
    n_same_question_pairs: int
    eligible_for_prompt_contrast: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_contrast_pool_frame(
    frame: pd.DataFrame,
    *,
    expected_ids: Iterable[int],
    n_samples_per_problem: int,
    max_new_tokens: int,
    min_class_count: int = 10,
    min_same_question_pairs: int = 2,
    max_budget_hit_rate: float = 0.25,
) -> ContrastPoolAudit:
    """Verify coverage and decode-budget health before prompt contrast fitting.

    A pair is counted only when a single source question has both a correct and
    an incorrect sampled completion.  This matches the deterministic
    same-question pairing rule used by the CAA/ActAdd registry builder.
    """
    if n_samples_per_problem < 2:
        raise ValueError("n_samples_per_problem must be at least two")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if min_class_count <= 0:
        raise ValueError("min_class_count must be positive")
    if min_same_question_pairs <= 0:
        raise ValueError("min_same_question_pairs must be positive")
    if not 0.0 <= max_budget_hit_rate <= 1.0:
        raise ValueError("max_budget_hit_rate must be in [0, 1]")

    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing contrast-pool label columns: {missing}")

    expected = {int(value) for value in expected_ids}
    observed = {int(value) for value in frame["sample_id"].tolist()}
    failures: list[str] = []
    if observed != expected:
        failures.append(
            "problem_ids_mismatch: "
            f"missing={sorted(expected - observed)[:8]} "
            f"unexpected={sorted(observed - expected)[:8]}"
        )
    if frame.duplicated(["sample_id", "sample_idx"]).any():
        failures.append("duplicate_problem_sample_indices")

    expected_indices = set(range(n_samples_per_problem))
    for sample_id, group in frame.groupby("sample_id", sort=False):
        indices = {int(value) for value in group["sample_idx"].tolist()}
        if indices != expected_indices:
            failures.append(
                f"sample_indices_mismatch_for_id={int(sample_id)}: "
                f"observed={sorted(indices)} expected={sorted(expected_indices)}"
            )
            break

    correct = frame["correct"].astype(bool)
    n_rows = int(len(frame))
    n_correct = int(correct.sum())
    n_incorrect = n_rows - n_correct
    budget_hit_rate = (
        float((frame["n_gen_tokens"].astype(int) >= max_new_tokens).mean()) if n_rows else 1.0
    )
    grouped = frame.assign(_correct=correct).groupby("sample_id", sort=False)["_correct"]
    n_pairs = int(sum(bool(values.any()) and bool((~values).any()) for _, values in grouped))

    if n_correct < min_class_count:
        failures.append(f"too_few_correct={n_correct}<min_class_count={min_class_count}")
    if n_incorrect < min_class_count:
        failures.append(f"too_few_incorrect={n_incorrect}<min_class_count={min_class_count}")
    if budget_hit_rate > max_budget_hit_rate:
        failures.append(
            f"budget_hit_rate={budget_hit_rate:.3f}>max_budget_hit_rate={max_budget_hit_rate:.3f}"
        )
    if n_pairs < min_same_question_pairs:
        failures.append(
            f"same_question_pairs={n_pairs}<min_same_question_pairs={min_same_question_pairs}"
        )

    return ContrastPoolAudit(
        n_rows=n_rows,
        n_problem_ids=len(observed),
        n_samples_per_problem=n_samples_per_problem,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        budget_hit_rate=budget_hit_rate,
        n_same_question_pairs=n_pairs,
        eligible_for_prompt_contrast=not failures,
        failures=tuple(failures),
    )
