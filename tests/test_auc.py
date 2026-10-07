import numpy as np
import pandas as pd

from geoprobe.analysis.auc import compute_auc_table, trivial_baseline


def _make_metrics_labels(n=20, seed=0):
    rng = np.random.default_rng(seed)
    correct = rng.integers(0, 2, size=n).astype(bool)
    # "perfect" metric: value 1.0 when incorrect, 0.0 when correct
    perfect_vals = np.where(correct, 0.0, 1.0)
    # "useless" metric: random
    useless_vals = rng.random(size=n)

    rows = []
    for layer in [0, 1]:
        for sid, v in enumerate(perfect_vals):
            rows.append({"sample_id": sid, "metric": "perfect", "layer": layer, "value": float(v)})
        for sid, v in enumerate(useless_vals):
            rows.append({"sample_id": sid, "metric": "useless", "layer": layer, "value": float(v)})
    metrics_df = pd.DataFrame(rows)
    labels_df = pd.DataFrame(
        {"sample_id": np.arange(n), "correct": correct, "n_gen_tokens": rng.integers(50, 500, size=n)}
    )
    return metrics_df, labels_df


def test_perfect_predictor_has_auc_1():
    m, l = _make_metrics_labels(n=20)
    auc = compute_auc_table(m, l, positive_label="incorrect")
    perfect = auc[auc["metric"] == "perfect"]
    assert (perfect["auc"] == 1.0).all()


def test_useless_predictor_has_auc_near_0_5():
    m, l = _make_metrics_labels(n=200, seed=1)
    auc = compute_auc_table(m, l, positive_label="incorrect")
    useless = auc[auc["metric"] == "useless"]
    # within 0.1 of chance for n=200
    assert ((useless["auc"] - 0.5).abs() < 0.1).all()


def test_trivial_baseline_returns_dict():
    _, l = _make_metrics_labels(n=20)
    b = trivial_baseline(l, column="n_gen_tokens")
    assert {"feature", "auc", "ap"} <= set(b.keys())
