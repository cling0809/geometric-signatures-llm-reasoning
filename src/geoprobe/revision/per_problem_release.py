"""Build anonymized per-problem audit tables from formal revision runs.

The formal ``per_sample.parquet`` artifacts contain decoded text and benchmark
answers that should not be redistributed in the anonymous review package.  This
module verifies each source artifact against the frozen report manifest and
exports only correctness and behavior telemetry needed to reproduce paired
accuracy and safety analyses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

SAFE_SOURCE_COLUMNS = (
    "method",
    "layer",
    "alpha",
    "sample_id",
    "correct",
    "text_tokens",
    "n_generated_tokens",
    "n_content_tokens",
    "stop_reason",
    "truncated",
    "text_tokens_whitespace",
    "distinct_2gram_ratio",
    "distinct_4gram_ratio",
    "repeated_4gram_fraction",
    "answer_marker_position",
    "answer_marker_relative_position",
)

FORBIDDEN_RELEASE_COLUMNS = frozenset(
    {
        "gold",
        "pred",
        "generated_text",
        "generation_seed",
        "prompt",
        "problem",
        "question",
        "answer",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report_entries(report_manifest: Path) -> list[dict[str, Any]]:
    report = json.loads(report_manifest.read_text())
    entries = report.get("methods")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"report manifest has no methods: {report_manifest}")
    return entries


def export_anonymous_per_problem(
    *,
    suite: str,
    suite_root: Path,
    report_manifest: Path,
    out_csv: Path,
    out_manifest: Path,
) -> dict[str, Any]:
    """Verify formal parquets and export a content-free per-problem table."""

    suite_root = suite_root.resolve()
    report_manifest = report_manifest.resolve()
    if not suite or not suite.strip():
        raise ValueError("suite must be a non-empty identifier")
    if not suite_root.is_dir():
        raise FileNotFoundError(suite_root)
    if not report_manifest.is_file():
        raise FileNotFoundError(report_manifest)

    frames: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    for entry in _load_report_entries(report_manifest):
        run_dir = Path(str(entry["run_dir"]))
        comparison = run_dir.name
        source = suite_root / comparison / "per_sample.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)

        expected_hash = entry.get("hashes", {}).get("per_sample.parquet")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"missing per_sample.parquet hash for {comparison}")
        observed_hash = sha256_file(source)
        if observed_hash != expected_hash:
            raise ValueError(
                f"source hash mismatch for {comparison}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )

        raw = pd.read_parquet(source)
        missing = sorted(set(SAFE_SOURCE_COLUMNS) - set(raw.columns))
        if missing:
            raise ValueError(f"{comparison} missing release columns: {missing}")

        sanitized = raw.loc[:, SAFE_SOURCE_COLUMNS].copy()
        sanitized.insert(0, "comparison", comparison)
        sanitized.insert(0, "suite", suite)
        sanitized = sanitized.rename(columns={"method": "arm"})
        if FORBIDDEN_RELEASE_COLUMNS & set(sanitized.columns):
            raise AssertionError("forbidden content column reached the release table")
        if sanitized.duplicated(["comparison", "arm", "sample_id"]).any():
            raise ValueError(f"duplicate comparison/arm/sample_id rows in {comparison}")

        frames.append(sanitized)
        sources.append(
            {
                "comparison": comparison,
                "relative_source": f"{comparison}/per_sample.parquet",
                "sha256": observed_hash,
                "rows": int(len(raw)),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["comparison", "arm", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False, float_format="%.10g", lineterminator="\n")

    manifest: dict[str, Any] = {
        "protocol": "tacl-11241-anonymous-per-problem-audit-v1",
        "suite": suite,
        "report_manifest_sha256": sha256_file(report_manifest),
        "source_artifacts": sources,
        "output": {
            "file": out_csv.name,
            "sha256": sha256_file(out_csv),
            "rows": int(len(combined)),
            "columns": list(combined.columns),
        },
        "excluded_content_columns": sorted(FORBIDDEN_RELEASE_COLUMNS),
        "privacy_boundary": (
            "No prompt, problem text, gold answer, prediction, generated text, "
            "or generation seed is included."
        ),
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
