import pandas as pd

from geoprobe.analysis.selection import evaluate_strategies


def _build_run(records):
    """Build labels_df + metrics_df from a list of per-(sid, idx) records.

    record keys: sample_id, sample_idx, gold, pred, correct, sequence_logprob,
                 (optional) metric_values: dict {(metric, layer): value}
    """
    labels_rows = []
    metric_rows = []
    for r in records:
        labels_rows.append({
            "sample_id": r["sample_id"],
            "sample_idx": r["sample_idx"],
            "gold": r["gold"],
            "pred": r["pred"],
            "correct": r["correct"],
            "n_gen_tokens": r.get("n_gen_tokens", 100),
            "sequence_logprob": r["sequence_logprob"],
        })
        for (m, lyr), v in r.get("metric_values", {}).items():
            metric_rows.append({
                "sample_id": r["sample_id"],
                "sample_idx": r["sample_idx"],
                "metric": m,
                "layer": lyr,
                "value": v,
            })
    return pd.DataFrame(labels_rows), pd.DataFrame(metric_rows)


def test_oracle_picks_any_correct():
    # 2 questions, 3 samples each. One has all wrong, other has at least one right.
    recs = []
    for sid in (0, 1):
        for idx in range(3):
            correct = (sid == 1 and idx == 0)
            recs.append({
                "sample_id": sid, "sample_idx": idx,
                "gold": 7.0, "pred": 7.0 if correct else 9.0,
                "correct": correct, "sequence_logprob": -10.0 - idx,
            })
    L, M = _build_run(recs)
    res = evaluate_strategies(L, M, geo_metric=None, geo_layer=None)
    oracle = res[res["strategy"] == "oracle"]["accuracy"].iloc[0]
    # 1 of 2 questions has a correct sample => oracle = 0.5
    assert oracle == 0.5


def test_majority_vote_picks_most_common():
    # 1 question, 3 samples: preds [7, 7, 9]; correct only if pred == 7
    recs = [
        {"sample_id": 0, "sample_idx": 0, "gold": 7.0, "pred": 7.0, "correct": True,  "sequence_logprob": -10.0},
        {"sample_id": 0, "sample_idx": 1, "gold": 7.0, "pred": 7.0, "correct": True,  "sequence_logprob": -11.0},
        {"sample_id": 0, "sample_idx": 2, "gold": 7.0, "pred": 9.0, "correct": False, "sequence_logprob": -9.0},
    ]
    L, M = _build_run(recs)
    res = evaluate_strategies(L, M, geo_metric=None, geo_layer=None)
    mv = res[res["strategy"] == "majority_vote"]["accuracy"].iloc[0]
    assert mv == 1.0


def test_geo_min_strategy_uses_smallest_value():
    # 1 question. Two samples with same logprob but different metric values.
    # Sample with smaller metric is the correct one.
    recs = [
        {
            "sample_id": 0, "sample_idx": 0, "gold": 5.0, "pred": 5.0, "correct": True,
            "sequence_logprob": -10.0,
            "metric_values": {("mean_step_norm", 20): 0.1},
        },
        {
            "sample_id": 0, "sample_idx": 1, "gold": 5.0, "pred": 3.0, "correct": False,
            "sequence_logprob": -10.0,
            "metric_values": {("mean_step_norm", 20): 0.9},
        },
    ]
    L, M = _build_run(recs)
    res = evaluate_strategies(L, M, geo_metric="mean_step_norm", geo_layer=20, geo_sign="min")
    geo = res[res["strategy"].str.startswith("geo_min_mean_step_norm")]
    assert (geo["accuracy"] == 1.0).all()


def test_geo_max_strategy_flips_sign():
    recs = [
        {
            "sample_id": 0, "sample_idx": 0, "gold": 5.0, "pred": 5.0, "correct": True,
            "sequence_logprob": -10.0,
            "metric_values": {("curvature_var", 15): 0.9},
        },
        {
            "sample_id": 0, "sample_idx": 1, "gold": 5.0, "pred": 3.0, "correct": False,
            "sequence_logprob": -10.0,
            "metric_values": {("curvature_var", 15): 0.1},
        },
    ]
    L, M = _build_run(recs)
    res = evaluate_strategies(L, M, geo_metric="curvature_var", geo_layer=15, geo_sign="max")
    geo = res[res["strategy"].str.startswith("geo_max_curvature_var")]
    assert (geo["accuracy"] == 1.0).all()
