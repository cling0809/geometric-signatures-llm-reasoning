#!/usr/bin/env python3
"""Build the audited anonymous source/evidence archive for TACL-11241."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

from geoprobe.revision.release import (
    EXCLUDED_FROM_ANONYMOUS_RELEASE,
    anonymous_release_preflight,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = "geoprobe-tacl-11241-anonymous"
DEFAULT_ARTIFACT_DIR = ROOT.parent / "geoprobe-release-artifacts" / "internal-audit"
# Prefer the newest existing verified archive.  20260813 is historical and is
# no longer on disk; 20260816 is the last verified baseline before the 20260819
# Fig. 1 print-prep package.
_BASELINE_STAMPS = ("20260819", "20260816", "20260813")


def _archive_path(stamp: str) -> Path:
    return (
        DEFAULT_ARTIFACT_DIR
        / f"TACL-11241-major-revision-anonymous-final-{stamp}.tar.gz"
    )


def resolve_baseline_archive(explicit: Path | None = None) -> Path:
    """Return a verified baseline archive path, fail-closed if none exists."""
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"verified baseline archive not found: {explicit}")
        return explicit
    for stamp in _BASELINE_STAMPS:
        candidate = _archive_path(stamp)
        if candidate.is_file():
            return candidate
    expected = ", ".join(str(_archive_path(stamp)) for stamp in _BASELINE_STAMPS)
    raise SystemExit(f"verified baseline archive not found; expected one of: {expected}")


def _is_excluded_release_member(relative: str) -> bool:
    for item in EXCLUDED_FROM_ANONYMOUS_RELEASE:
        if item.endswith("/"):
            if relative.startswith(item):
                return True
        elif relative == item:
            return True
    return False


def _current_evidence() -> list[Path]:
    manifest = ROOT / "revision" / "ANONYMOUS_RELEASE_PREFLIGHT.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    evidence = [ROOT / entry["path"] for entry in payload["evidence_entries"]]
    for relative in (
        "revision/evidence/r1-readiness-32k-v1/README.md",
        "revision/evidence/r1-readiness-32k-v1/capability20-readiness.json",
        "revision/evidence/r1-readiness-32k-v1/STOPPED_BY_R1_READINESS_GATE",
        "revision/evidence/r1-corrected-short-v1/README.md",
        "revision/evidence/r1-corrected-short-v1/r1-corrected-selection-v1.json",
        "revision/evidence/r1-corrected-short-v1/locked_report_manifest.json",
        "revision/evidence/r1-corrected-short-v1/locked_target_calibrated_summary.csv",
        "revision/evidence/r1-corrected-short-v1/locked_crosssteer_source_summary.csv",
        "revision/evidence/ood-4096-math500-qwen-instruct-v1/README.md",
        "revision/evidence/ood-4096-math500-qwen-instruct-v1/ood_4096_summary.csv",
        "revision/evidence/ood-4096-math500-qwen-instruct-v1/ood_4096_report_manifest.json",
    ):
        candidate = ROOT / relative
        if candidate not in evidence:
            evidence.append(candidate)
    return evidence


def _baseline_files(archive: Path) -> list[str]:
    if not archive.is_file():
        raise SystemExit(f"verified baseline archive not found: {archive}")
    manifest_member = f"{PACKAGE_DIR}/RELEASE_PACKAGE_MANIFEST.json"
    with tarfile.open(archive, "r:gz") as handle:
        source_manifest = json.load(handle.extractfile(manifest_member))
    files = list(source_manifest["files"])
    replacements = {
        "paper/figures/fig1_revision_protocol.png": "paper/figures/fig1_overview.png",
    }
    files = [replacements.get(relative, relative) for relative in files]
    # The current Fig. 1 is author supplied.  Do not ship the obsolete generator
    # that produced the superseded protocol overview.  Outcome-contingent
    # decision-tree drafts are also excluded: the final manuscript and response
    # are already frozen and must be the only active narrative.
    files = [
        relative
        for relative in files
        if relative not in {
            "scripts/make_revision_method_figure.py",
            "revision/V2_RESULT_DECISION_TREE.md",
            "scripts/sync_to_server.sh",
        }
        and not _is_excluded_release_member(relative)
    ]
    # The audited package builder was added after the verified baseline archive;
    # include it so the final archive can be reproduced from its own source.
    if "scripts/build_anonymous_release.py" not in files:
        files.append("scripts/build_anonymous_release.py")
    for relative in (
        "revision/FINAL_UPLOAD.md",
        "revision/FINAL_AUTHOR_CHECKLIST.md",
        "revision/ORIGINAL_DECISION_AND_REVIEWS_ANONYMIZED.md",
        "revision/decision_letter_header.tex",
        "scripts/build_anonymized_decision.py",
        "scripts/build_tacl_resubmission_bundle.py",
        "revision/evidence/signature-stability-retrospective/7b_pairwise_signature_distance.csv",
        "revision/evidence/r1-readiness-32k-v1/README.md",
        "revision/evidence/r1-readiness-32k-v1/capability20-readiness.json",
        "revision/evidence/r1-readiness-32k-v1/STOPPED_BY_R1_READINESS_GATE",
        "revision/evidence/r1-corrected-short-v1/README.md",
        "revision/evidence/r1-corrected-short-v1/r1-corrected-selection-v1.json",
        "revision/evidence/r1-corrected-short-v1/locked_report_manifest.json",
        "revision/evidence/r1-corrected-short-v1/locked_target_calibrated_summary.csv",
        "revision/evidence/r1-corrected-short-v1/locked_crosssteer_source_summary.csv",
        "revision/evidence/ood-4096-math500-qwen-instruct-v1/README.md",
        "revision/evidence/ood-4096-math500-qwen-instruct-v1/ood_4096_summary.csv",
        "revision/evidence/ood-4096-math500-qwen-instruct-v1/ood_4096_report_manifest.json",
    ):
        if relative not in files:
            files.append(relative)
    if len(files) != len(set(files)):
        raise SystemExit("release file list contains duplicate members")
    return sorted(files)


def _write_manifest(stage: Path, files: list[str]) -> None:
    payload = {
        "excluded": list(EXCLUDED_FROM_ANONYMOUS_RELEASE),
        "file_count": len(files),
        "files": files,
        "sha256": {relative: sha256_file(stage / relative) for relative in files},
    }
    (stage / "RELEASE_PACKAGE_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build(*, baseline_archive: Path, output_dir: Path, stamp: str) -> tuple[Path, Path]:
    evidence = _current_evidence()
    preflight = anonymous_release_preflight(ROOT, evidence=evidence, require_evidence=True)
    if not preflight["ready_for_final_anonymous_release"]:
        raise SystemExit(json.dumps(preflight, indent=2, sort_keys=True))
    # The preflight snapshot is written beside the archive, not back into the
    # tracked tree, so it cannot go stale relative to the commit it describes.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ANONYMOUS_RELEASE_PREFLIGHT.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    files = _baseline_files(baseline_archive)
    missing = [relative for relative in files if not (ROOT / relative).is_file()]
    if missing:
        raise SystemExit(f"release members missing from current tree: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"TACL-11241-major-revision-anonymous-final-{stamp}"
    archive = output_dir / f"{name}.tar.gz"
    external_manifest = output_dir / f"{name}.manifest.json"

    with tempfile.TemporaryDirectory(prefix="tacl11241-release-") as tmp:
        stage = Path(tmp) / PACKAGE_DIR
        stage.mkdir(parents=True)
        for relative in files:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        _write_manifest(stage, files)
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(stage, arcname=PACKAGE_DIR)
        shutil.copy2(stage / "RELEASE_PACKAGE_MANIFEST.json", external_manifest)

    return archive, external_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-archive", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    baseline = (
        args.baseline_archive.resolve() if args.baseline_archive is not None else None
    )
    archive, manifest = build(
        baseline_archive=resolve_baseline_archive(baseline),
        output_dir=args.output_dir.resolve(),
        stamp=args.stamp,
    )
    print(archive)
    print(manifest)
    print(f"sha256={sha256_file(archive)}")


if __name__ == "__main__":
    main()
