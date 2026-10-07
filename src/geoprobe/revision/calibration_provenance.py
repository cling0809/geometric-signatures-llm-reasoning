"""Audit calibrated trajectory generations before they define a steering vector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit_calibration_generation_provenance(
    run_dir: Path,
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    """Validate a newly extracted calibration run without recomputing its labels.

    The extractor writes its generation envelope before producing trajectories.
    This audit ties that envelope to the completed labels and rejects legacy
    calibration directories that cannot establish their exact EOS telemetry.
    """
    run = Path(run_dir).resolve()
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    required = {
        "DONE": run / "DONE",
        "config": run / "config.yaml",
        "labels": run / "labels.parquet",
        "manifest": run / "resolved_generation_manifest.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"calibration run lacks required generation provenance: {missing}")
    manifest = _load_object(required["manifest"])
    if manifest.get("protocol") != "tacl-11241-calibration-generation-provenance-v1":
        raise ValueError("unexpected calibration generation provenance protocol")
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("generation provenance lacks a generation mapping")
    if generation.get("max_new_tokens") != max_new_tokens:
        raise ValueError("generation provenance max_new_tokens disagrees with audit")
    if generation.get("eos_token_id") != "model.generation_config.eos_token_id":
        raise ValueError("calibration generation did not delegate EOS to model generation config")
    eos_ids = manifest.get("model_generation_eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in eos_ids)
    ):
        raise ValueError("calibration generation provenance lacks a valid EOS set")
    if manifest.get("config_sha256") != sha256_file(required["config"]):
        raise ValueError("calibration generation provenance config hash mismatch")

    labels = pd.read_parquet(required["labels"])
    required_columns = {"sample_id", "sample_idx", "n_gen_tokens", "stop_reason", "truncated"}
    if missing_columns := required_columns - set(labels.columns):
        raise ValueError(f"labels lack generation telemetry: {sorted(missing_columns)}")
    if labels.empty:
        raise ValueError("labels are empty")
    lengths = labels["n_gen_tokens"].astype(int)
    if (lengths < 1).any() or (lengths > max_new_tokens).any():
        raise ValueError("labels contain an invalid generated-token count")
    allowed_reasons = {"eos", "max_new_tokens", "other"}
    reasons = labels["stop_reason"]
    if not reasons.isin(allowed_reasons).all():
        raise ValueError("labels contain an invalid stop reason")
    truncated = labels["truncated"]
    if not truncated.map(lambda value: isinstance(value, bool)).all():
        raise ValueError("labels lack boolean truncation telemetry")
    expected_truncated = reasons.eq("max_new_tokens")
    if not (truncated.astype(bool) == expected_truncated).all():
        raise ValueError("labels have inconsistent truncation telemetry")
    if (reasons.eq("max_new_tokens") & lengths.lt(max_new_tokens)).any():
        raise ValueError("labels mark a pre-budget completion as truncated")
    if not reasons.eq("eos").any():
        raise ValueError("calibration run has no EOS-completed trajectories")
    labels_sha = sha256_file(required["labels"])
    return {
        "protocol": "tacl-11241-calibration-generation-audit-v1",
        "run_dir": str(run),
        "config_sha256": sha256_file(required["config"]),
        "labels_sha256": labels_sha,
        "generation_provenance_sha256": sha256_file(required["manifest"]),
        "max_new_tokens": max_new_tokens,
        "model_generation_eos_token_ids": eos_ids,
        "row_count": int(len(labels)),
        "stop_reason_counts": {
            str(name): int(count) for name, count in reasons.value_counts().sort_index().items()
        },
        "truncation_rate": float(expected_truncated.mean()),
    }
