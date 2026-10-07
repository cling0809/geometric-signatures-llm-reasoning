from __future__ import annotations

import numpy as np
import pandas as pd

from geoprobe.revision.geovote_length_matched import (
    permute_scores_within_length_pairs,
    run_length_matched_null,
)


def make_frame() -> pd.DataFrame:
    rows = []
    for sample_id in range(3):
        for sample_idx in range(8):
            rows.append(
                {
                    "sample_id": sample_id,
                    "sample_idx": sample_idx,
                    "pred": f"answer-{sample_id}-{sample_idx}",
                    "correct": sample_idx == 0,
                    "n_gen_tokens": 100 + 10 * sample_idx,
                    "geo_conf": float(sample_idx + sample_id / 10),
                    "sequence_logprob": -float(sample_idx + 1),
                }
            )
    return pd.DataFrame(rows)


def test_length_matched_permutation_preserves_problem_score_multisets() -> None:
    frame = make_frame()
    permuted = permute_scores_within_length_pairs(
        frame,
        score_column="geo_conf",
        rng=np.random.default_rng(7),
    )
    for sample_id, group in frame.groupby("sample_id"):
        expected = sorted(group["geo_conf"].tolist())
        observed = sorted(
            permuted.loc[permuted["sample_id"] == sample_id, "length_matched_score"].tolist()
        )
        assert observed == expected
    assert np.array_equal(frame["sample_id"].to_numpy(), permuted["sample_id"].to_numpy())
    assert np.array_equal(frame["n_gen_tokens"].to_numpy(), permuted["n_gen_tokens"].to_numpy())


def test_length_matched_permutation_is_deterministic_for_fixed_seed() -> None:
    frame = make_frame()
    first = permute_scores_within_length_pairs(
        frame,
        score_column="geo_conf",
        rng=np.random.default_rng(11),
    )
    second = permute_scores_within_length_pairs(
        frame,
        score_column="geo_conf",
        rng=np.random.default_rng(11),
    )
    assert first["length_matched_score"].equals(second["length_matched_score"])


def test_length_matched_null_reports_finite_distribution() -> None:
    rows = run_length_matched_null(
        make_frame(),
        score_column="geo_conf",
        permutations=17,
        seed=13,
    )
    assert len(rows) == 17
    assert rows["permutation"].tolist() == list(range(17))
    assert np.isfinite(rows[["baseline_accuracy", "method_accuracy", "delta_pp"]]).all().all()
    assert (rows["repairs"] >= 0).all()
    assert (rows["breaks"] >= 0).all()
