"""Read-only integrity audit for a completed frozen steering validation family."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_BASELINE_COLUMNS = (
    "sample_id",
    "gold",
    "pred",
    "correct",
    "generated_text",
    "n_generated_tokens",
    "n_content_tokens",
    "text_tokens",
    "text_tokens_whitespace",
    "stop_reason",
    "truncated",
    "repeated_4gram_fraction",
    "answer_marker_position",
    "answer_marker_relative_position",
)
_SHARED_MANIFEST_FIELDS = (
    "protocol",
    "target_model",
    "target_source",
    "eval_start",
    "eval_count",
    "layers",
    "alphas",
    "schedule",
    "schedule_parameter",
    "position_mode",
    "injection_mode",
    "normalization",
    "max_new_tokens",
    "generation",
    "prompt_template",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows_path = path / "per_sample.jsonl"
    if not rows_path.is_file():
        raise FileNotFoundError(f"missing per-sample rows: {rows_path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(rows_path.read_text().splitlines(), start=1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{rows_path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _float_set(values: list[object], *, field: str) -> set[float]:
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"manifest {field} must contain numeric values")
    return {float(value) for value in values}


def audit_validation_family(run_dirs: Mapping[str, Path]) -> dict[str, object]:
    """Validate completed frozen grid coverage and shared baseline equality.

    The audit intentionally ignores accuracy aggregates and selection outcomes.
    It verifies only data completeness, decoder-side telemetry provenance and
    deterministic no-steering comparability before scores may be interpreted.
    """
    if len(run_dirs) < 2:
        raise ValueError("at least two completed method runs are required")
    if len(set(run_dirs)) != len(run_dirs):  # defensive for non-dict callers
        raise ValueError("method names must be unique")

    reference_manifest: dict[str, Any] | None = None
    reference_baseline: dict[int, dict[str, Any]] | None = None
    run_entries: list[dict[str, object]] = []
    shared_eos: list[int] | None = None

    for method, raw_path in sorted(run_dirs.items()):
        path = Path(raw_path).resolve()
        if not (path / "DONE").is_file():
            raise ValueError(f"run is incomplete (missing DONE): {path}")
        manifest_path = path / "resolved_manifest.json"
        runtime_path = path / "generation_runtime_audit.json"
        if not manifest_path.is_file() or not runtime_path.is_file():
            raise FileNotFoundError(f"run lacks manifest or runtime audit: {path}")
        manifest = _load_json(manifest_path)
        runtime = _load_json(runtime_path)
        rows = _load_rows(path)

        manifest_method = None
        for row in rows:
            row_method = row.get("method")
            if row_method != "baseline":
                manifest_method = str(row_method)
                break
        if manifest_method != method:
            raise ValueError(f"{path} contains method {manifest_method!r}, expected {method!r}")
        eval_start = manifest.get("eval_start")
        eval_count = manifest.get("eval_count")
        layers = manifest.get("layers")
        alphas = manifest.get("alphas")
        if isinstance(eval_start, bool) or not isinstance(eval_start, int):
            raise ValueError(f"{path} manifest eval_start must be an integer")
        if isinstance(eval_count, bool) or not isinstance(eval_count, int) or eval_count <= 0:
            raise ValueError(f"{path} manifest eval_count must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise ValueError(f"{path} manifest layers must be a non-empty list")
        if not isinstance(alphas, list) or not alphas:
            raise ValueError(f"{path} manifest alphas must be a non-empty list")
        layer_set = {int(layer) for layer in layers}
        alpha_set = _float_set(alphas, field="alphas")
        expected_ids = set(range(eval_start, eval_start + eval_count))
        baseline = [row for row in rows if row.get("method") == "baseline"]
        active = [row for row in rows if row.get("method") == method]
        if len(rows) != eval_count * (1 + len(layer_set) * len(alpha_set)):
            raise ValueError(f"{path} has incomplete or extra rows: {len(rows)}")
        if len(baseline) != eval_count or len(active) != eval_count * len(layer_set) * len(alpha_set):
            raise ValueError(f"{path} has incorrect baseline/method row counts")
        baseline_ids = [int(row["sample_id"]) for row in baseline]
        if set(baseline_ids) != expected_ids or len(set(baseline_ids)) != len(baseline_ids):
            raise ValueError(f"{path} baseline problem ids do not exactly cover the manifest range")
        cell_counts = Counter((int(row["layer"]), float(row["alpha"])) for row in active)
        expected_cells = {(layer, alpha) for layer in layer_set for alpha in alpha_set}
        if set(cell_counts) != expected_cells or set(cell_counts.values()) != {eval_count}:
            raise ValueError(f"{path} does not exactly cover its frozen layer/alpha grid")
        row_keys = [
            (str(row["method"]), int(row["layer"]), float(row["alpha"]), int(row["sample_id"]))
            for row in rows
        ]
        if len(row_keys) != len(set(row_keys)):
            raise ValueError(f"{path} contains duplicate generated-row keys")
        for row in baseline:
            missing = [column for column in _BASELINE_COLUMNS if column not in row]
            if missing:
                raise ValueError(f"{path} baseline lacks fields: {missing}")

        if runtime.get("row_count") != len(rows):
            raise ValueError(f"{path} runtime audit row count disagrees with generated rows")
        if runtime.get("per_sample_jsonl_sha256") != _sha256(path / "per_sample.jsonl"):
            raise ValueError(f"{path} runtime audit hash disagrees with generated rows")
        eos = runtime.get("model_generation_eos_token_ids")
        if not isinstance(eos, list) or not eos or any(not isinstance(value, int) for value in eos):
            raise ValueError(f"{path} runtime audit lacks a valid EOS set")
        if shared_eos is None:
            shared_eos = eos
        elif eos != shared_eos:
            raise ValueError(f"{path} uses a different model EOS set from the family")
        truncation_rate = runtime.get("truncation_rate")
        if not isinstance(truncation_rate, (int, float)) or float(truncation_rate) > 0.25:
            raise ValueError(f"{path} fails the frozen 25% decoding-envelope ceiling")

        if reference_manifest is None:
            reference_manifest = manifest
        else:
            differences = {
                field: (reference_manifest.get(field), manifest.get(field))
                for field in _SHARED_MANIFEST_FIELDS
                if reference_manifest.get(field) != manifest.get(field)
            }
            if differences:
                raise ValueError(f"{path} differs from the shared frozen target protocol: {differences}")
        canonical_baseline = {
            int(row["sample_id"]): {column: row[column] for column in _BASELINE_COLUMNS}
            for row in baseline
        }
        if reference_baseline is None:
            reference_baseline = canonical_baseline
        elif canonical_baseline != reference_baseline:
            mismatches = sorted(
                sample_id
                for sample_id in expected_ids
                if canonical_baseline.get(sample_id) != reference_baseline.get(sample_id)
            )
            raise ValueError(f"{path} baseline differs from the reference on ids: {mismatches[:10]}")

        run_entries.append(
            {
                "method": method,
                "run_dir": str(path),
                "per_sample_jsonl_sha256": _sha256(path / "per_sample.jsonl"),
                "resolved_manifest_sha256": _sha256(manifest_path),
                "runtime_audit_sha256": _sha256(runtime_path),
                "row_count": len(rows),
                "baseline_row_count": len(baseline),
                "frozen_cell_count": len(expected_cells),
                "truncation_rate": float(truncation_rate),
            }
        )

    assert reference_manifest is not None and reference_baseline is not None and shared_eos is not None
    return {
        "protocol": "tacl-11241-validation-family-integrity-audit-v1",
        "status": "passed",
        "method_count": len(run_entries),
        "methods": run_entries,
        "shared_generation_eos_token_ids": shared_eos,
        "shared_baseline_problem_count": len(reference_baseline),
        "claim_boundary": (
            "This read-only audit establishes completed grid/provenance integrity only; "
            "it does not compute or interpret a method score, select a cell, or read locked labels."
        ),
    }


def write_validation_family_audit(run_dirs: Mapping[str, Path], output: Path) -> Path:
    """Write an immutable integrity artifact, refusing conflicting replacement."""
    payload = audit_validation_family(run_dirs)
    output = Path(output).resolve()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != encoded:
        raise ValueError(f"refuse to overwrite different validation-family audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded)
    return output
