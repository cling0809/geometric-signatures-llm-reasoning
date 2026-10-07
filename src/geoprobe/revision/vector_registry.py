"""Build and fingerprint train-only steering-vector artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import torch

from geoprobe.revision.crosssteer import compute_crosssteer_direction
from geoprobe.revision.protocol import RevisionSplit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_completed_run(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    required = [run / "DONE", run / "config.yaml", run / "labels.parquet", run / "trajectories"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete calibration run; missing {missing}")
    return run


def load_calibration_audit_reference(
    audit_path: str | Path,
    *,
    run: Path,
    role: str,
) -> dict[str, object]:
    """Bind a vector input to a successful, provenance-aware capability audit."""
    path = Path(audit_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {role} calibration audit: {path}")
    payload = json.loads(path.read_text())
    if payload.get("protocol") != "tacl-11241-calibration-capability-audit-v1":
        raise ValueError(f"{role} calibration audit has an unexpected protocol")
    threshold = payload.get("max_budget_hit_rate")
    if not isinstance(threshold, (int, float)) or float(threshold) > 0.25:
        raise ValueError(f"{role} calibration audit relaxes the 25% budget-hit ceiling")
    matching = [entry for entry in payload.get("runs", []) if entry.get("run") == str(run.resolve())]
    if len(matching) != 1:
        raise ValueError(f"{role} calibration audit does not uniquely bind run {run}")
    entry = matching[0]
    if entry.get("eligible_for_direction") is not True:
        raise ValueError(f"{role} calibration audit did not approve direction construction")
    if entry.get("labels_sha256") != sha256_file(run / "labels.parquet"):
        raise ValueError(f"{role} calibration audit labels hash mismatch")
    if entry.get("config_sha256") != sha256_file(run / "config.yaml"):
        raise ValueError(f"{role} calibration audit config hash mismatch")
    provenance = entry.get("generation_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{role} calibration audit lacks generation provenance")
    if provenance.get("truncation_rate") is None:
        raise ValueError(f"{role} calibration audit lacks truncation telemetry")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "run": entry,
    }


def build_vector_registry(
    source_run: str | Path,
    target_run: str | Path,
    source_ids: Iterable[int],
    *,
    split: RevisionSplit,
    source_calibration_audit: str | Path | None = None,
    target_calibration_audit: str | Path | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Construct source-transfer and target-calibrated directions without leakage."""
    source = require_completed_run(source_run)
    target = require_completed_run(target_run)
    ids = tuple(int(value) for value in source_ids)
    split.assert_direction_ids(ids)
    source_direction, source_metadata = compute_crosssteer_direction(source, ids, split=split)
    target_direction, target_metadata = compute_crosssteer_direction(target, ids, split=split)
    if source_direction.shape != target_direction.shape:
        raise ValueError(
            "source/target vectors have incompatible [layers, hidden] shapes: "
            f"{tuple(source_direction.shape)} vs {tuple(target_direction.shape)}"
        )
    metadata: dict[str, object] = {
        "protocol": "tacl-11241-vector-registry-v1",
        "source_ids": list(ids),
        "source_run": str(source.resolve()),
        "target_run": str(target.resolve()),
        "source_run_config_sha256": sha256_file(source / "config.yaml"),
        "target_run_config_sha256": sha256_file(target / "config.yaml"),
        "source_labels_sha256": sha256_file(source / "labels.parquet"),
        "target_labels_sha256": sha256_file(target / "labels.parquet"),
        "source_direction": source_metadata,
        "target_direction": target_metadata,
        "shape": list(source_direction.shape),
    }
    if (source_calibration_audit is None) != (target_calibration_audit is None):
        raise ValueError("source and target calibration audits must be supplied together")
    if source_calibration_audit is not None and target_calibration_audit is not None:
        metadata["source_calibration_audit"] = load_calibration_audit_reference(
            source_calibration_audit,
            run=source,
            role="source",
        )
        metadata["target_calibration_audit"] = load_calibration_audit_reference(
            target_calibration_audit,
            run=target,
            role="target",
        )
    return {"crosssteer_source": source_direction, "target_calibrated": target_direction}, metadata


def save_vector_registry(
    vectors: dict[str, torch.Tensor], metadata: dict[str, object], out_dir: str | Path
) -> None:
    """Write a compact, fingerprinted vector registry and JSON provenance manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, vector in vectors.items():
        if vector.ndim != 2:
            raise ValueError(f"{name} must have shape [layers, hidden]")
        torch.save(vector.cpu(), out / f"{name}.pt")
    artifact_hashes = {name: sha256_file(out / f"{name}.pt") for name in vectors}
    payload = {**metadata, "vector_sha256": artifact_hashes}
    (out / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
