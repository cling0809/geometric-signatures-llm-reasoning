import numpy as np
import pandas as pd

from geoprobe.analysis.signatures import (
    pairwise_signature_distance,
    signature_matrix,
)


def _toy_metrics_labels(n_samples=30, n_layers=4, metrics=("a", "b"), seed=0):
    rng = np.random.default_rng(seed)
    correct = rng.integers(0, 2, size=n_samples).astype(bool)
    rows = []
    for m in metrics:
        for layer in range(n_layers):
            for sid in range(n_samples):
                rows.append({"sample_id": sid, "metric": m, "layer": layer,
                             "value": rng.normal()})
    metrics_df = pd.DataFrame(rows)
    labels_df = pd.DataFrame({"sample_id": np.arange(n_samples), "correct": correct})
    return metrics_df, labels_df


def test_signature_matrix_shape():
    m, l = _toy_metrics_labels(n_samples=20, n_layers=3, metrics=("x", "y", "z"))
    sig = signature_matrix(m, l)
    assert sig.shape == (3, 3)
    assert list(sig.index) == ["x", "y", "z"]
    assert list(sig.columns) == [0, 1, 2]


def test_pairwise_distance_diagonal_is_zero():
    sigs = {
        f"r{i}": pd.DataFrame(np.random.RandomState(i).rand(3, 4) * 0.4 + 0.3,
                              index=["a", "b", "c"], columns=[0, 1, 2, 3])
        for i in range(3)
    }
    dist = pairwise_signature_distance(sigs)
    assert dist.shape == (3, 3)
    assert (np.diag(dist.values) == 0).all()
    # symmetric
    assert np.allclose(dist.values, dist.values.T)


def test_pairwise_distance_identical_sigs_gives_zero():
    sig = pd.DataFrame([[0.6, 0.4], [0.5, 0.7]], index=["a", "b"], columns=[0, 1])
    sigs = {"x": sig, "y": sig.copy()}
    dist = pairwise_signature_distance(sigs)
    assert dist.loc["x", "y"] == 0.0
