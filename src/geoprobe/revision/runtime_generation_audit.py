"""Immutable audit for the effective model generation terminators of a completed run.

A model config can define multiple EOS token ids.  Recording only the symbolic
string ``model.generation_config.eos_token_id`` in a run manifest is not enough
to make a multi-EOS decoding decision independently inspectable, especially
when a prior experiment accidentally supplied only ``tokenizer.eos_token_id``.
This module writes a sidecar audit; it never mutates a run's resolved manifest
or its generated rows.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

_AUDIT_NAME = "generation_runtime_audit.json"
_ALLOWED_STOP_REASONS = {"eos", "max_new_tokens", "other"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_token_ids(value: object) -> list[int]:
    """Normalize a generation-config token-id field without silently guessing."""
    if isinstance(value, bool):
        raise ValueError("token id must be an integer or a non-empty sequence of integers")
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("token-id sequence must be non-empty")
        ids: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("token-id sequence must contain integers only")
            ids.append(int(item))
        if len(set(ids)) != len(ids):
            raise ValueError("token-id sequence must not contain duplicates")
        return ids
    raise ValueError("token id must be an integer or a non-empty sequence of integers")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows_path = path / "per_sample.jsonl"
    if not rows_path.exists():
        raise FileNotFoundError(f"missing generated-row artifact: {rows_path}")
    rows: list[dict[str, Any]] = []
    with rows_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"row {line_number} is not a JSON object")
            rows.append(item)
    if not rows:
        raise ValueError("run has no generated rows")
    return rows


def build_generation_runtime_audit(
    run_dir: Path,
    generation_config_path: Path,
) -> dict[str, object]:
    """Create an inspectable EOS/termination summary for a completed run.

    This validates the effective model-level generation configuration and the
    run's recorded stopping telemetry.  It deliberately does not infer a token
    terminator from decoded text, because special EOS tokens are removed before
    text is stored.
    """
    run_dir = Path(run_dir).resolve()
    generation_config_path = Path(generation_config_path).resolve()
    manifest_path = run_dir / "resolved_manifest.json"
    done_path = run_dir / "DONE"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing resolved manifest: {manifest_path}")
    if not done_path.exists():
        raise ValueError("runtime audit requires a completed run (missing DONE)")
    if not generation_config_path.exists():
        raise FileNotFoundError(f"missing generation config: {generation_config_path}")

    manifest = json.loads(manifest_path.read_text())
    config = json.loads(generation_config_path.read_text())
    eos_ids = normalize_token_ids(config.get("eos_token_id"))
    pad_token_id = config.get("pad_token_id")
    if pad_token_id is not None and (isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int)):
        raise ValueError("generation config pad_token_id must be an integer when present")

    rows = _load_rows(run_dir)
    stop_reasons = Counter()
    truncated = 0
    for index, row in enumerate(rows):
        reason = row.get("stop_reason")
        if reason not in _ALLOWED_STOP_REASONS:
            raise ValueError(f"row {index} has invalid stop_reason: {reason!r}")
        stop_reasons[str(reason)] += 1
        is_truncated = row.get("truncated")
        if not isinstance(is_truncated, bool):
            raise ValueError(f"row {index} lacks boolean truncation telemetry")
        if is_truncated != (reason == "max_new_tokens"):
            raise ValueError(f"row {index} has inconsistent stop/truncation telemetry")
        truncated += int(is_truncated)
    if stop_reasons["eos"] == 0:
        raise ValueError("run has no EOS-completed rows; cannot validate an EOS decoding envelope")

    generation = manifest.get("generation")
    if not isinstance(generation, dict) or generation.get("eos_token_id") != "model.generation_config.eos_token_id":
        raise ValueError("manifest does not declare model.generation_config.eos_token_id")

    return {
        "protocol": "tacl-11241-generation-runtime-audit-v1",
        "run_dir": str(run_dir),
        "resolved_manifest_sha256": _sha256(manifest_path),
        "per_sample_jsonl_sha256": _sha256(run_dir / "per_sample.jsonl"),
        "generation_config_path": str(generation_config_path),
        "generation_config_sha256": _sha256(generation_config_path),
        "model_generation_eos_token_ids": eos_ids,
        "model_generation_pad_token_id": None if pad_token_id is None else int(pad_token_id),
        "row_count": len(rows),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "truncation_rate": truncated / len(rows),
        "interpretation": (
            "The resolved manifest delegates EOS to the model generation config; "
            "this sidecar records that config's exact token ids and validates the "
            "run's explicit termination telemetry."
        ),
    }


def write_generation_runtime_audit(
    run_dir: Path,
    generation_config_path: Path,
) -> Path:
    """Write an immutable sidecar audit or reject conflicting re-generation."""
    payload = build_generation_runtime_audit(run_dir, generation_config_path)
    output = Path(run_dir).resolve() / _AUDIT_NAME
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != encoded:
        raise ValueError(f"refuse to overwrite different runtime audit: {output}")
    output.write_text(encoded)
    return output
