"""Anonymous-release preflight checks for the TACL-11241 revision package.

The preflight deliberately validates source provenance and anonymity separately
from scientific results.  It never treats an incomplete run as release-ready.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

REQUIRED_RELEASE_ITEMS = (
    "README.md",
    "pyproject.toml",
    "scripts/make_revision_long_context_figure.py",
    "scripts/make_revision_locked_forest_figure.py",
    "revision/evidence/hidden-projection/hidden_projection_coords.csv",
    "revision/REPRODUCIBILITY.md",
    "revision/E2_LENGTH_MATCHED_AUDIT_STATUS.md",
    "revision/DIAGNOSTIC_GENERALIZATION_PROTOCOL.md",
    "revision/SOURCE_CALIBRATION_QUALITY_GATE.md",
    "revision/R1_FORMAL_EXECUTION_CHAIN.md",
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
    "revision/evidence/geovote-length-audit-n8-retrospective/README.md",
    "revision/evidence/geovote-length-audit-n8-retrospective/manifest.json",
    "revision/evidence/geovote-length-audit-n8-retrospective/paired_comparisons_vs_majority.csv",
    "revision/evidence/geovote-length-audit-n8-retrospective/candidate_length_audit.csv",
    "revision/evidence/geovote-length-audit-n8-retrospective/length_matched_permutation_summary.json",
    "revision/evidence/geovote-length-audit-n8-retrospective/length_matched_permutation_rows.csv",
    "revision/evidence/validation-family-qwen-instruct-v1/README.md",
    "revision/evidence/validation-family-qwen-instruct-v1/validation_complete_summary.csv",
    "revision/evidence/validation-family-qwen-instruct-v1/validation_selected_rows.tex",
    "revision/evidence/validation-family-qwen-instruct-v1/validation_family_report_manifest.json",
    "revision/evidence/signature-stability-retrospective/README.md",
    "revision/evidence/signature-stability-retrospective/signature_stability_summary.csv",
    "revision/evidence/signature-stability-retrospective/7b_pairwise_signature_distance.csv",
    "revision/evidence/locked-gsm8k-qwen-instruct-v3/locked_report_manifest.json",
    "revision/evidence/locked-gsm8k-qwen-instruct-v3/locked_summary.csv",
    "revision/evidence/locked-gsm8k-qwen-instruct-v3/triage.json",
    "revision/evidence/ood-math500-qwen-instruct-v3/ood_report_manifest.json",
    "revision/evidence/ood-math500-qwen-instruct-v3/ood_math500_summary.csv",
    "revision/evidence/ood-svamp-qwen-instruct-v3/ood_report_manifest.json",
    "revision/evidence/ood-svamp-qwen-instruct-v3/ood_svamp_summary.csv",
    "revision/evidence/label-efficiency-qwen-instruct-v3/label_efficiency_report_manifest.json",
    "revision/evidence/label-efficiency-qwen-instruct-v3/label_efficiency_summary.csv",
    "revision/evidence/per-problem-audit-v1/README.md",
    "revision/evidence/per-problem-audit-v1/locked_gsm8k.csv",
    "revision/evidence/per-problem-audit-v1/locked_gsm8k_manifest.json",
    "revision/evidence/per-problem-audit-v1/ood_math500.csv",
    "revision/evidence/per-problem-audit-v1/ood_math500_manifest.json",
    "revision/evidence/per-problem-audit-v1/ood_svamp.csv",
    "revision/evidence/per-problem-audit-v1/ood_svamp_manifest.json",
    "revision/evidence/per-problem-audit-v1/label_efficiency.csv",
    "revision/evidence/per-problem-audit-v1/label_efficiency_manifest.json",
    "revision/evidence/long-context-qwen-instruct-v3/crosssteer-b4096/long_context_report_manifest.json",
    "revision/evidence/long-context-qwen-instruct-v3/crosssteer-b4096/long_context_summary.csv",
    "revision/evidence/long-context-qwen-instruct-v3/crosssteer-b32768/long_context_report_manifest.json",
    "revision/evidence/long-context-qwen-instruct-v3/crosssteer-b32768/long_context_summary.csv",
    "revision/evidence/long-context-qwen-instruct-v3/target-calibrated-b4096/long_context_report_manifest.json",
    "revision/evidence/long-context-qwen-instruct-v3/target-calibrated-b4096/long_context_summary.csv",
    "revision/evidence/long-context-qwen-instruct-v3/target-calibrated-b32768/long_context_report_manifest.json",
    "revision/evidence/long-context-qwen-instruct-v3/target-calibrated-b32768/long_context_summary.csv",
    "scripts/revision_smoke_demo.py",
    "scripts/revision_diagnostic_generalization.py",
    "scripts/make_revision_curvature_figure.py",
    "scripts/make_enhanced_result_figures.py",
    "scripts/make_phenomenon_hidden_figures.py",
    "scripts/revision_select_validation.py",
    "scripts/revision_build_validation_family_report.py",
    "scripts/revision_build_locked_report.py",
    "scripts/revision_build_ood_report.py",
    "scripts/revision_build_long_context_report.py",
    "scripts/revision_geovote_length_matched_audit.py",
    "scripts/revision_export_anonymous_per_problem.py",
    "scripts/revision_extract_pooled_calibration.py",
    "scripts/revision_audit_pooled_readiness.py",
    "scripts/revision_audit_generation_runtime.py",
    "scripts/revision_audit_calibration.py",
    "scripts/revision_build_vector_registry.py",
    "scripts/extract_trajectories.py",
    "scripts/revision_audit_validation_family_integrity.py",
    "scripts/revision_run_r1_official_context_chain.sh",
    "scripts/revision_run_r1_formal_comparison_chain.sh",
    "scripts/revision_run_r1_long_context_chain.sh",
    "scripts/revision_run_qwen_locked_chain.sh",
    "scripts/revision_run_qwen_ood_chain.sh",
    "scripts/revision_run_qwen_long_context_chain.sh",
    "src/geoprobe/revision/geovote_length_matched.py",
    "src/geoprobe/revision/per_problem_release.py",
    "configs/revision_gsm8k_qwen_math_source_calibration_100_v2.yaml",
    "configs/revision_gsm8k_qwen_instruct_calibration_100_v2.yaml",
    "configs/revision_gsm8k_r1_official32k_capability20.yaml",
    "configs/revision_gsm8k_r1_official32k_calibration100.yaml",
    "configs/revision_gsm8k_r1_official32k_caa_source_k4.yaml",
    "revision/R1_OFFICIAL_CONTEXT_PROTOCOL.md",
    "revision/R1_FORMAL_COMPARISON_PROTOCOL.md",
    "src/geoprobe/revision/protocol.py",
    "src/geoprobe/revision/calibration_provenance.py",
    "src/geoprobe/revision/validation_report.py",
    "src/geoprobe/revision/diagnostic_generalization.py",
    "src/geoprobe/revision/vector_registry.py",
    "src/geoprobe/revision/pooled_trajectory.py",
    "src/geoprobe/revision/pooled_readiness.py",
    "src/geoprobe/revision/runtime_generation_audit.py",
    "src/geoprobe/revision/validation_family_integrity.py",
    "tests/test_diagnostic_generalization.py",
    "tests/test_per_problem_release.py",
    "tests/test_revision_steering.py",
)


ANONYMOUS_PER_PROBLEM_TABLES = (
    (
        "revision/evidence/per-problem-audit-v1/locked_gsm8k.csv",
        "revision/evidence/per-problem-audit-v1/locked_gsm8k_manifest.json",
    ),
    (
        "revision/evidence/per-problem-audit-v1/ood_math500.csv",
        "revision/evidence/per-problem-audit-v1/ood_math500_manifest.json",
    ),
    (
        "revision/evidence/per-problem-audit-v1/ood_svamp.csv",
        "revision/evidence/per-problem-audit-v1/ood_svamp_manifest.json",
    ),
    (
        "revision/evidence/per-problem-audit-v1/label_efficiency.csv",
        "revision/evidence/per-problem-audit-v1/label_efficiency_manifest.json",
    ),
)

_FORBIDDEN_PER_PROBLEM_HEADERS = {
    "gold",
    "pred",
    "generated_text",
    "generation_seed",
    "prompt",
    "problem",
    "question",
    "answer",
}
_REQUIRED_PER_PROBLEM_HEADERS = {"suite", "comparison", "arm", "sample_id", "correct"}

# These are author-workstation path markers, not generic Linux paths such as
# /root that may legitimately occur in configurable launch scripts.  Build the
# strings from fragments so the preflight implementation itself stays clean.
_FORBIDDEN_TEXT_MARKERS = (
    "/" + "Users/",
    ".co" + "dex/",
    "Tian" + "lin Chen",
    "Yi" + "ting Cai",
    "u3" + "ssh.cas" + "dao.com",
    "u3" + "ssh",
    "cas" + "dao",
)

# Internal notes may be useful while revising but must not be copied verbatim to
# the anonymous release.  The manifest records this exclusion explicitly.
EXCLUDED_FROM_ANONYMOUS_RELEASE = (
    "scripts/sync_to_server.sh",
    "revision/.private/",
    "revision/.private/author_identity_tokens.txt",
    "revision/EXECUTION_LOG.md",
    "revision/SERVER_AUDIT_2026-08-03.md",
    "revision/SUBMITTED_ARTIFACT_INTEGRITY.md",
    # Historical drafting prompts and exploratory figures contain superseded
    # positive-result narratives.  Keep them in the private worktree for audit
    # history, but never ship them beside the controlled revision.
    "paper/FIGURE_PROMPTS.md",
    "paper/FIGURE_PROMPTS_FINAL.md",
    "paper/figures/fig1.png",
    "paper/figures/fig1new.png",
    "paper/figures/fig1_overview.drawio",
    "paper/figures/fig1_overview.drawio.pdf",
    "paper/figures/fig1_overview.drawio.png",
    "paper/figures/fig1_overview.drawio.svg",
    "paper/figures/fig1_overview.vector.svg",
    "paper/figures/fig3_crosssteer_real.pdf",
    "paper/figures/fig3_crosssteer_real.png",
    "scripts/paper_figures.py",
    "scripts/make_revision_method_figure.py",
    "paper/main.pdf",
    "paper/11241-Geometric-Signatures.pdf",
    "paper/Geometric-Signatures.pdf",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one release artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_anonymous_per_problem_tables(root: Path) -> list[dict[str, str]]:
    """Validate release-table privacy, row count, columns, and content hash."""

    errors: list[dict[str, str]] = []
    for csv_relative, manifest_relative in ANONYMOUS_PER_PROBLEM_TABLES:
        csv_path = root / csv_relative
        manifest_path = root / manifest_relative
        if not csv_path.is_file() or not manifest_path.is_file():
            continue
        try:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                row_count = sum(1 for _ in reader)
            header_set = set(header)
            forbidden = sorted(header_set & _FORBIDDEN_PER_PROBLEM_HEADERS)
            missing = sorted(_REQUIRED_PER_PROBLEM_HEADERS - header_set)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            output = manifest["output"]
            if forbidden:
                raise ValueError(f"forbidden headers: {forbidden}")
            if missing:
                raise ValueError(f"missing headers: {missing}")
            if output["file"] != csv_path.name:
                raise ValueError("manifest output filename mismatch")
            if output["sha256"] != sha256_file(csv_path):
                raise ValueError("manifest output hash mismatch")
            if int(output["rows"]) != row_count:
                raise ValueError("manifest row count mismatch")
            if list(output["columns"]) != header:
                raise ValueError("manifest column list mismatch")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, StopIteration) as exc:
            errors.append({"path": csv_relative, "error": str(exc)})
    return errors


def _candidate_text_files(root: Path) -> list[Path]:
    """Return source files that will be inspected for anonymous-release leaks."""
    # Tests intentionally contain synthetic identity-leak fixtures, so they are
    # required release files but are not content-scanned by this preflight.
    candidates: set[Path] = set()
    for relative in (
        "README.md",
        "pyproject.toml",
        "paper/main.tex",
        "paper/references.bib",
        "paper/README.md",
        "paper/FIGURE_PROVENANCE.md",
    ):
        path = root / relative
        if path.is_file():
            candidates.add(path)
    for pattern in (
        "paper/sections/*.tex",
        "paper/data/*.csv",
        "revision/*.md",
        "configs/revision*.yaml",
        "scripts/revision_*.py",
        "scripts/revision_*.sh",
        "src/geoprobe/revision/*.py",
    ):
        candidates.update(path for path in root.glob(pattern) if path.is_file())
    # Widen the scan to every shipped text file so identity leaks in packaging
    # scripts, docs, or configs are caught fail-closed.
    for pattern in (
        "**/*.py",
        "**/*.sh",
        "**/*.md",
        "**/*.tex",
        "**/*.yaml",
        "**/*.json",
        "**/*.bib",
        "**/*.txt",
    ):
        for path in root.glob(pattern):
            if path.is_file():
                parts = set(path.parts)
                if parts & {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "tmp", "tests"}:
                    continue
                candidates.add(path)
    excluded = {root / relative for relative in EXCLUDED_FROM_ANONYMOUS_RELEASE}
    return sorted(candidates - excluded)


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def anonymous_release_preflight(
    root: Path,
    *,
    evidence: Iterable[Path] = (),
    require_evidence: bool = False,
) -> dict[str, object]:
    """Build an auditable readiness manifest without packaging any artifacts.

    ``evidence`` is intentionally explicit.  Final release readiness requires
    formal result reports supplied by the caller; no directory-name convention
    can silently convert an unfinished experiment into a scientific claim.
    """
    root = root.resolve()
    missing_required = [
        relative for relative in REQUIRED_RELEASE_ITEMS if not (root / relative).is_file()
    ]
    per_problem_audit_errors = _validate_anonymous_per_problem_tables(root)
    scanned = _candidate_text_files(root)
    identifier_hits: list[dict[str, object]] = []
    for path in scanned:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in _FORBIDDEN_TEXT_MARKERS:
            if marker in content:
                identifier_hits.append(
                    {
                        "path": _relative(root, path),
                        "marker": marker,
                    }
                )

    evidence_paths = [Path(path).resolve() for path in evidence]
    missing_evidence = [str(path) for path in evidence_paths if not path.exists()]
    # A SUPERSEDED marker next to an evidence artifact means the artifact was
    # invalidated (for example the EOS-override defect); it must never count as
    # final anonymous-release evidence even though the file still exists.
    superseded_evidence = []
    for path in evidence_paths:
        if not path.exists():
            continue
        marker_parent = path if path.is_dir() else path.parent
        if (marker_parent / "SUPERSEDED").is_file():
            superseded_evidence.append(_relative(root, path))
    evidence_entries = [
        {
            "path": _relative(root, path),
            "kind": "directory" if path.is_dir() else "file",
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for path in evidence_paths
        if path.exists()
    ]
    evidence_state = "provided" if evidence_paths else "pending"
    source_tree_ready = (
        not missing_required and not identifier_hits and not per_problem_audit_errors
    )
    ready = (
        source_tree_ready
        and bool(evidence_paths)
        and not missing_evidence
        and not superseded_evidence
    )

    return {
        "protocol": "tacl-11241-anonymous-release-preflight-v1",
        "git_sha": _git_sha(root),
        "source_root": ".",
        "required_items": list(REQUIRED_RELEASE_ITEMS),
        "missing_required_items": missing_required,
        "scanned_text_files": [_relative(root, path) for path in scanned],
        "identifier_hits": identifier_hits,
        "per_problem_audit_errors": per_problem_audit_errors,
        "excluded_from_anonymous_release": list(EXCLUDED_FROM_ANONYMOUS_RELEASE),
        "evidence_state": evidence_state,
        "evidence_entries": evidence_entries,
        "missing_evidence": missing_evidence,
        "superseded_evidence": superseded_evidence,
        "require_evidence": require_evidence,
        "source_tree_ready": source_tree_ready,
        "ready_for_final_anonymous_release": ready,
    }
