"""Input-provenance gate for fresh trajectory-signature confirmation runs.

The submitted heatmaps were produced before the revision had complete generation
telemetry.  This gate prevents legacy directories from being silently treated as
confirmation evidence: a diagnostic retrieval can only use a completed run whose
resolved decoder envelope, labels, per-problem metrics and trajectory artifacts
agree exactly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geoprobe.revision.calibration_provenance import audit_calibration_generation_provenance
from geoprobe.revision.vector_registry import require_completed_run, sha256_file


def audit_diagnostic_signature_run(
    run_dir: Path,
    *,
    max_new_tokens: int,
    max_truncation_rate: float = 0.25,
) -> dict[str, object]:
    """Validate one newly generated greedy run before signature analysis.

    This is deliberately an *input* gate, not an outcome filter.  It checks
    provenance and structural completeness only; it does not inspect a model's
    retrieval score, pairwise signature distance, or selected metric/layer.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not 0.0 <= max_truncation_rate <= 1.0:
        raise ValueError("max_truncation_rate must be in [0, 1]")

    run = require_completed_run(Path(run_dir))
    provenance = audit_calibration_generation_provenance(run, max_new_tokens=max_new_tokens)
    if float(provenance["truncation_rate"]) > max_truncation_rate:
        raise ValueError(
            f"{run}: truncation_rate={float(provenance['truncation_rate']):.3f} "
            f"> max_truncation_rate={max_truncation_rate:.3f}"
        )

    labels_path = run / "labels.parquet"
    metrics_path = run / "metrics.parquet"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"{run}: metrics.parquet is required before signature analysis")
    labels = pd.read_parquet(labels_path)
    metrics = pd.read_parquet(metrics_path)

    greedy = labels.loc[labels["sample_idx"].eq(0)].copy()
    if greedy.empty:
        raise ValueError(f"{run}: no greedy labels")
    if greedy["sample_id"].duplicated().any():
        raise ValueError(f"{run}: greedy labels are not unique per sample_id")
    expected_ids = {int(value) for value in greedy["sample_id"]}

    required_metric_columns = {"sample_id", "sample_idx", "metric", "layer", "value"}
    if missing := required_metric_columns - set(metrics.columns):
        raise ValueError(f"{run}: metrics missing {sorted(missing)}")
    metric_greedy = metrics.loc[metrics["sample_idx"].eq(0)].copy()
    if metric_greedy.empty:
        raise ValueError(f"{run}: no greedy metric rows")
    observed_metric_ids = {int(value) for value in metric_greedy["sample_id"]}
    if observed_metric_ids != expected_ids:
        raise ValueError(
            f"{run}: labels/metrics problem-id mismatch: "
            f"missing={sorted(expected_ids - observed_metric_ids)[:8]} "
            f"unexpected={sorted(observed_metric_ids - expected_ids)[:8]}"
        )
    if metric_greedy.duplicated(["sample_id", "metric", "layer"]).any():
        raise ValueError(f"{run}: duplicate greedy metric cells")
    if not pd.api.types.is_numeric_dtype(metric_greedy["value"]):
        raise ValueError(f"{run}: metric values are not numeric")
    if not metric_greedy["value"].map(float).map(pd.notna).all():
        raise ValueError(f"{run}: metric values contain NaN")

    trajectory_dir = run / "trajectories"
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(f"{run}: trajectories directory is required")
    missing_trajectories = [
        int(sample_id)
        for sample_id in sorted(expected_ids)
        if not (trajectory_dir / f"sample_{int(sample_id):04d}_idx_0.pt").is_file()
    ]
    if missing_trajectories:
        raise FileNotFoundError(
            f"{run}: missing greedy trajectory artifacts for IDs {missing_trajectories[:8]}"
        )

    manifest = {
        "protocol": "tacl-11241-diagnostic-signature-input-audit-v1",
        "run_dir": str(run.resolve()),
        "max_new_tokens": max_new_tokens,
        "max_truncation_rate": max_truncation_rate,
        "n_greedy_problem_ids": len(expected_ids),
        "n_greedy_metric_rows": int(len(metric_greedy)),
        "config_sha256": sha256_file(run / "config.yaml"),
        "labels_sha256": sha256_file(labels_path),
        "metrics_sha256": sha256_file(metrics_path),
        "generation_provenance": provenance,
    }
    return manifest
