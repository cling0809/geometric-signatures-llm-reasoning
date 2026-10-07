from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from geoprobe.revision.diagnostic_provenance import audit_diagnostic_signature_run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fresh_run(tmp_path: Path, *, truncated: bool = False) -> Path:
    run = tmp_path / "fresh"
    trajectories = run / "trajectories"
    trajectories.mkdir(parents=True)
    (run / "DONE").write_text("\n")
    config = run / "config.yaml"
    config.write_text("exp_id: fresh\n")
    (run / "resolved_generation_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-calibration-generation-provenance-v1",
                "config_sha256": _sha256(config),
                "model": "example/model",
                "model_generation_eos_token_ids": [1, 2],
                "generation": {
                    "max_new_tokens": 8,
                    "eos_token_id": "model.generation_config.eos_token_id",
                },
            }
        )
    )
    labels = pd.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "sample_idx": [0, 0, 0],
            "correct": [True, False, True],
            "n_gen_tokens": [8 if truncated else 3, 4, 5],
            "stop_reason": ["max_new_tokens" if truncated else "eos", "eos", "other"],
            "truncated": [truncated, False, False],
        }
    )
    labels.to_parquet(run / "labels.parquet", index=False)
    pd.DataFrame(
        {
            "sample_id": [0, 0, 1, 1, 2, 2],
            "sample_idx": [0] * 6,
            "metric": ["shape", "shape", "shape", "shape", "shape", "shape"],
            "layer": [0, 1, 0, 1, 0, 1],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    ).to_parquet(run / "metrics.parquet", index=False)
    for sample_id in range(3):
        (trajectories / f"sample_{sample_id:04d}_idx_0.pt").write_bytes(b"fixture")
    return run


def test_diagnostic_input_audit_binds_fresh_generation_metrics_and_trajectories(tmp_path: Path):
    run = _write_fresh_run(tmp_path)
    report = audit_diagnostic_signature_run(run, max_new_tokens=8)
    assert report["n_greedy_problem_ids"] == 3
    assert report["n_greedy_metric_rows"] == 6
    assert report["generation_provenance"]["truncation_rate"] == 0.0
    assert report["metrics_sha256"] == _sha256(run / "metrics.parquet")


def test_diagnostic_input_audit_rejects_excessive_truncation(tmp_path: Path):
    run = _write_fresh_run(tmp_path, truncated=True)
    with pytest.raises(ValueError, match="truncation_rate"):
        audit_diagnostic_signature_run(run, max_new_tokens=8, max_truncation_rate=0.25)


def test_diagnostic_input_audit_rejects_missing_trajectory(tmp_path: Path):
    run = _write_fresh_run(tmp_path)
    (run / "trajectories" / "sample_0001_idx_0.pt").unlink()
    with pytest.raises(FileNotFoundError, match="missing greedy trajectory"):
        audit_diagnostic_signature_run(run, max_new_tokens=8)
