#!/usr/bin/env python3
"""Freeze one steering configuration from completed validation-grid artifacts.

The command must run before generating any locked GSM8K steering candidates. It
only reads completed validation artifacts, verifies their fixed protocol, and
writes the selection plus SHA-256 evidence.  It never opens locked-set labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from geoprobe.revision.runtime_generation_audit import write_generation_runtime_audit
from geoprobe.revision.validation_family_integrity import write_validation_family_audit
from geoprobe.revision.validation_selection import baseline_behavior, select_validation_cell


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_BASELINE_CORE_COLUMNS = (
    "sample_id",
    "gold",
    "pred",
    "correct",
    "text_tokens",
    "text_tokens_whitespace",
    "distinct_2gram_ratio",
    "distinct_4gram_ratio",
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


def _canonical_target_manifest(
    manifest: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Canonicalize documented legacy greedy defaults without mutating artifacts.

    The two initial Qwen validation jobs predate complete generation metadata;
    two later baseline adapters also omitted fields that are irrelevant under
    greedy decoding.  We retain every original artifact/hash and compare only
    a normalized decoder record.  Explicit non-default values remain a hard
    mismatch, so this compatibility rule cannot mask a sampled or altered run.
    """
    canonical = dict(manifest)
    backfills: dict[str, object] = {}
    if "injection_mode" not in canonical:
        canonical["injection_mode"] = "absolute"
        backfills["injection_mode"] = {
            "value": "absolute",
            "reason": "legacy runner CLI default; original manifest retained",
        }

    raw_generation = canonical.get("generation", {})
    if raw_generation is None:
        raw_generation = {}
    if not isinstance(raw_generation, dict):
        raise ValueError("generation metadata must be a mapping")
    do_sample = bool(raw_generation.get("do_sample", False))
    if do_sample:
        required = {"temperature", "top_p", "base_seed", "seed_rule"}
        missing = sorted(required - set(raw_generation))
        if missing:
            raise ValueError(
                "sampled generation metadata lacks frozen fields: "
                f"{missing}"
            )
    normalized_generation = {
        "do_sample": do_sample,
        "temperature": float(raw_generation.get("temperature", 1.0)),
        "top_p": float(raw_generation.get("top_p", 1.0)),
        "base_seed": int(raw_generation.get("base_seed", 11241)),
        "seed_rule": raw_generation.get("seed_rule"),
        "num_beams": int(raw_generation.get("num_beams", 1)),
        "use_cache": bool(raw_generation.get("use_cache", True)),
        "eos_token_id": raw_generation.get("eos_token_id", "model.generation_config.eos_token_id"),
    }
    if not 0.0 < normalized_generation["top_p"] <= 1.0:
        raise ValueError("generation metadata has invalid top_p")
    if normalized_generation["do_sample"] and normalized_generation["temperature"] <= 0.0:
        raise ValueError("sampled generation metadata has non-positive temperature")
    canonical["generation"] = normalized_generation
    return canonical, backfills


def _assert_shared_target_protocol(runs: list[tuple[str, dict[str, object]]]) -> dict[str, object]:
    """Require comparable target settings and byte-equivalent baseline rows.

    Each method runner regenerates the deterministic no-steering baseline.  A
    fair multi-method selection must reject a comparison if those baselines are
    not the same on the same validation problem IDs.
    """
    if not runs:
        raise ValueError("at least one completed run is required")
    reference_name, reference = runs[0]
    reference_manifest, reference_backfills = _canonical_target_manifest(reference["manifest"])
    reference_rows = reference["rows"]
    baseline = reference_rows[reference_rows["method"] == "baseline"].copy()
    missing = sorted(set(_BASELINE_CORE_COLUMNS) - set(baseline.columns))
    if missing:
        raise ValueError(f"{reference_name} baseline lacks core fields: {missing}")
    baseline = (
        baseline.loc[:, _BASELINE_CORE_COLUMNS].sort_values("sample_id").reset_index(drop=True)
    )
    if baseline["sample_id"].duplicated().any():
        raise ValueError(f"{reference_name} has duplicate baseline problem IDs")

    for name, run in runs[1:]:
        manifest, backfills = _canonical_target_manifest(run["manifest"])
        mismatches = {
            field: (reference_manifest.get(field), manifest.get(field))
            for field in _SHARED_MANIFEST_FIELDS
            if reference_manifest.get(field) != manifest.get(field)
        }
        if mismatches:
            raise ValueError(
                f"{name} does not share the target evaluation protocol with {reference_name}: {mismatches}"
            )
        candidate = run["rows"][run["rows"]["method"] == "baseline"].copy()
        missing = sorted(set(_BASELINE_CORE_COLUMNS) - set(candidate.columns))
        if missing:
            raise ValueError(f"{name} baseline lacks core fields: {missing}")
        candidate = (
            candidate.loc[:, _BASELINE_CORE_COLUMNS].sort_values("sample_id").reset_index(drop=True)
        )
        try:
            assert_frame_equal(baseline, candidate, check_dtype=False, check_exact=True)
        except AssertionError as exc:
            raise ValueError(
                f"{name} baseline differs from {reference_name}; refuse unfair comparison"
            ) from exc
    return {
        "reference_run": reference_name,
        "baseline_problem_ids": [int(value) for value in baseline["sample_id"].tolist()],
        "core_fields": list(_BASELINE_CORE_COLUMNS),
        "manifest_fields": list(_SHARED_MANIFEST_FIELDS),
        "legacy_manifest_backfills": {
            reference_name: reference_backfills,
            **{name: _canonical_target_manifest(run["manifest"])[1] for name, run in runs[1:]},
        },
    }






def _assert_vector_calibration_audits(parsed: list[tuple[str, Path]]) -> dict[str, object]:
    """Reject calibration-derived directions without a bound 25%-ceiling audit.

    This consumes vector provenance only, before any validation summary is
    loaded.  It prevents an output-side fair-comparison audit from accidentally
    legitimizing a direction built from a saturated source trajectory pool.
    """
    required_methods = {
        "crosssteer_source": "source_calibration_audit",
        "target_calibrated": "target_calibration_audit",
        "sae_sparse_activation": "calibration_audit",
    }
    records: dict[str, object] = {}
    for method, directory in parsed:
        if method not in required_methods:
            continue
        manifest = json.loads((directory.resolve() / "resolved_manifest.json").read_text())
        vectors = manifest.get("vectors")
        if not isinstance(vectors, dict) or not isinstance(vectors.get(method), str):
            raise ValueError(f"{directory} lacks the declared {method} vector path")
        registry_manifest_path = Path(vectors[method]).resolve().parent / "manifest.json"
        if not registry_manifest_path.is_file():
            raise FileNotFoundError(f"{method} vector registry lacks manifest: {registry_manifest_path}")
        registry = json.loads(registry_manifest_path.read_text())
        role_key = required_methods[method]
        record = registry.get(role_key)
        if not isinstance(record, dict):
            raise ValueError(f"{method} vector registry lacks a bound {role_key}")
        audit_path = Path(str(record.get("path", ""))).resolve()
        if not audit_path.is_file() or record.get("sha256") != _sha256(audit_path):
            raise ValueError(f"{method} vector registry calibration-audit hash mismatch")
        audit = json.loads(audit_path.read_text())
        if audit.get("max_budget_hit_rate", 1.0) > 0.25:
            raise ValueError(f"{method} vector registry uses a relaxed source-quality ceiling")
        runs = audit.get("runs")
        if not isinstance(runs, list) or not any(entry == record.get("run") for entry in runs):
            raise ValueError(f"{method} vector registry audit record is not present in its audit file")
        run_entry = record["run"]
        provenance = run_entry.get("generation_provenance") if isinstance(run_entry, dict) else None
        if not isinstance(provenance, dict) or provenance.get("truncation_rate") is None:
            raise ValueError(f"{method} vector registry lacks generation-provenance telemetry")
        records[method] = {
            "registry_manifest": str(registry_manifest_path),
            "registry_manifest_sha256": _sha256(registry_manifest_path),
            "calibration_audit": record,
        }
    missing = set(required_methods) - set(records)
    if missing:
        raise ValueError(
            "formal selection requires audited CrossSteer, target-calibrated and SAE vectors; "
            f"missing {sorted(missing)}"
        )
    return records


def _write_family_integrity_audit(
    parsed: list[tuple[str, Path]],
    *,
    output: Path,
) -> tuple[Path, str]:
    """Materialize decoder telemetry and a score-blind family audit before selection.

    The selection program is intentionally the first consumer of validation
    scores.  It must therefore reject a grid family whose exact multi-EOS
    envelope, row manifests, frozen cells, or no-steering baselines are not
    auditable.  Runtime sidecars are derived only from completed rows plus the
    declared target model's generation config; no score or locked label is read.
    """
    run_dirs: dict[str, Path] = {}
    for method, directory in parsed:
        resolved = directory.resolve()
        manifest_path = resolved / "resolved_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing resolved manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        target_model = manifest.get("target_model")
        if not isinstance(target_model, str) or not target_model:
            raise ValueError(f"{resolved} lacks a concrete target_model for runtime EOS audit")
        generation_config = Path(target_model) / "generation_config.json"
        write_generation_runtime_audit(resolved, generation_config)
        run_dirs[method] = resolved

    audit_path = output.resolve().with_name(f"{output.stem}_family_integrity.json")
    write_validation_family_audit(run_dirs, audit_path)
    return audit_path, _sha256(audit_path)


def _load_completed_run(
    path: Path, *, expected_method: str, expected_cells: int
) -> dict[str, object]:
    done = path / "DONE"
    summary_path = path / "summary.csv"
    rows_path = path / "per_sample.parquet"
    manifest_path = path / "resolved_manifest.json"
    missing = [
        item.name for item in (done, summary_path, rows_path, manifest_path) if not item.exists()
    ]
    if missing:
        raise FileNotFoundError(f"{path} is not a completed validation run; missing {missing}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != "tacl-11241-frozen-steering-grid-v1":
        raise ValueError(f"{path} has unexpected protocol")
    if manifest.get("position_mode") != "decode_last":
        raise ValueError(f"{path} did not use post-prompt decode_last steering")
    if manifest.get("normalization") != "direction RMS=1 before alpha":
        raise ValueError(f"{path} did not use declared RMS normalization")
    vectors = manifest.get("vectors", {})
    if set(vectors) != {expected_method}:
        raise ValueError(f"{path} has methods {sorted(vectors)}; expected only {expected_method!r}")

    summary = pd.read_csv(summary_path)
    cells = summary[summary["method"] == expected_method]
    if len(cells) != expected_cells:
        raise ValueError(
            f"{path} has {len(cells)} completed {expected_method!r} cells; expected {expected_cells}"
        )
    rows = pd.read_parquet(rows_path)
    source_ids = {int(value) for value in manifest["source_ids"]}
    evaluation_ids = {int(value) for value in rows["sample_id"].unique()}
    overlap = source_ids & evaluation_ids
    if overlap:
        raise ValueError(f"{path} evaluation overlaps vector source IDs: {sorted(overlap)[:10]}")
    return {
        "manifest": manifest,
        "summary": summary,
        "rows": rows,
        "hashes": {
            "resolved_manifest.json": _sha256(manifest_path),
            "summary.csv": _sha256(summary_path),
            "per_sample.parquet": _sha256(rows_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="METHOD=DIR")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-cells", type=int, default=16)
    args = parser.parse_args()
    if args.expected_cells <= 0:
        raise SystemExit("--expected-cells must be positive")

    parsed: list[tuple[str, Path]] = []
    for raw in args.run:
        if "=" not in raw:
            raise SystemExit("--run must be METHOD=DIR")
        method, directory = raw.split("=", 1)
        if not method or not directory:
            raise SystemExit("--run must be METHOD=DIR")
        parsed.append((method, Path(directory)))
    if len({method for method, _ in parsed}) != len(parsed):
        raise SystemExit("each method can be selected only once")

    vector_calibration_audits = _assert_vector_calibration_audits(parsed)
    family_integrity_path, family_integrity_sha256 = _write_family_integrity_audit(
        parsed,
        output=args.out,
    )
    completed = [
        (
            method,
            _load_completed_run(
                directory, expected_method=method, expected_cells=args.expected_cells
            ),
        )
        for method, directory in parsed
    ]
    shared_target_audit = _assert_shared_target_protocol(completed)

    decisions: list[dict[str, object]] = []
    for method, run in completed:
        directory = next(path for candidate_method, path in parsed if candidate_method == method)
        baseline_tokens, baseline_repetition = baseline_behavior(run["rows"])
        decision = select_validation_cell(
            run["summary"],
            method=method,
            baseline_mean_text_tokens=baseline_tokens,
            baseline_mean_repeated_4gram_fraction=baseline_repetition,
        )
        decisions.append(
            {
                "method": method,
                "run_dir": str(directory.resolve()),
                "decision": decision.to_dict(),
                "evidence_hashes": run["hashes"],
                "run_signature_sha256": run["manifest"].get("signature_sha256"),
                "source_ids": run["manifest"]["source_ids"],
                "evaluation_ids": sorted(int(value) for value in run["rows"]["sample_id"].unique()),
            }
        )

    payload = {
        "protocol": "tacl-11241-frozen-steering-grid-v1",
        "selection_rule": "revision/VALIDATION_GRID_V1.md",
        "selection_status": "locked_before_locked_test_generation",
        "shared_target_audit": shared_target_audit,
        "validation_family_integrity_audit": {
            "path": str(family_integrity_path),
            "sha256": family_integrity_sha256,
        },
        "vector_calibration_audits": vector_calibration_audits,
        "methods": decisions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if existing != payload:
            raise SystemExit("refuse to overwrite a different validation selection")
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
