"""Audit and render the complete frozen steering-validation family.

The selection JSON records only one behavior-eligible cell per method.  This
module retains the full fixed validation landscape so a revision cannot present
only selected peaks after locked generation has begun.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from geoprobe.revision.validation_selection import select_validation_cell

_PROTOCOL = "tacl-11241-frozen-validation-family-report-v1"
_REQUIRED_SUMMARY_COLUMNS = {
    "method",
    "layer",
    "alpha",
    "mean_text_tokens",
    "mean_repeated_4gram_fraction",
    "n",
    "baseline_accuracy",
    "method_accuracy",
    "delta",
    "repairs",
    "breaks",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "exact_sign_p_value",
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one audit artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_entries(selection_path: Path) -> list[dict[str, object]]:
    payload = json.loads(selection_path.read_text())
    if payload.get("protocol") != "tacl-11241-frozen-steering-grid-v1":
        raise ValueError("selection has an unexpected protocol")
    if payload.get("selection_status") != "locked_before_locked_test_generation":
        raise ValueError("selection was not frozen before locked generation")
    entries = payload.get("methods")
    if not isinstance(entries, list) or not entries:
        raise ValueError("selection has no method decisions")
    if any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("selection contains a non-object method entry")
    methods = [entry.get("method") for entry in entries]
    if any(not isinstance(method, str) or not method for method in methods):
        raise ValueError("selection contains an invalid method name")
    if len(set(methods)) != len(methods):
        raise ValueError("selection contains duplicate method decisions")
    return entries


def _same_decision(expected: dict[str, object], observed: dict[str, object]) -> bool:
    """Compare JSON-decoded decision records without CSV float round-off noise."""
    if set(expected) != set(observed):
        return False
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if isinstance(expected_value, bool) or isinstance(observed_value, bool):
            if expected_value is not observed_value:
                return False
        elif isinstance(expected_value, (int, float)) and isinstance(observed_value, (int, float)):
            if not math.isclose(float(expected_value), float(observed_value), rel_tol=0.0, abs_tol=1e-12):
                return False
        elif expected_value != observed_value:
            return False
    return True


def build_validation_family_report(selection_path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load all selected-run summaries and verify the frozen decision lineage.

    The resulting frame has every layer/alpha cell for every selected method,
    including cells that were not selected or did not pass the behavior guard.
    """
    selection = Path(selection_path)
    entries = _selection_entries(selection)
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, object]] = []

    for entry in entries:
        method = str(entry["method"])
        run_dir_raw = entry.get("run_dir")
        decision = entry.get("decision")
        hashes = entry.get("evidence_hashes")
        if not isinstance(run_dir_raw, str) or not isinstance(decision, dict) or not isinstance(hashes, dict):
            raise ValueError(f"{method} selection entry lacks audited run metadata")
        run_dir = Path(run_dir_raw)
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        observed_hash = sha256_file(summary_path)
        if observed_hash != hashes.get("summary.csv"):
            raise ValueError(f"{method} summary hash differs from frozen selection")

        summary = pd.read_csv(summary_path)
        missing = sorted(_REQUIRED_SUMMARY_COLUMNS - set(summary.columns))
        if missing:
            raise ValueError(f"{method} summary is missing columns: {missing}")
        if set(summary["method"]) != {method}:
            raise ValueError(f"{method} summary contains mismatched method rows")
        if summary.duplicated(["layer", "alpha"]).any():
            raise ValueError(f"{method} summary contains duplicate layer/alpha cells")

        baseline_tokens = float(decision.get("baseline_mean_text_tokens", float("nan")))
        baseline_repetition = float(decision.get("baseline_mean_repeated_4gram_fraction", float("nan")))
        if not pd.notna(baseline_tokens) or not pd.notna(baseline_repetition):
            raise ValueError(f"{method} decision lacks behavior baseline values")
        recomputed = select_validation_cell(
            summary,
            method=method,
            baseline_mean_text_tokens=baseline_tokens,
            baseline_mean_repeated_4gram_fraction=baseline_repetition,
        ).to_dict()
        if not _same_decision(decision, recomputed):
            raise ValueError(f"{method} selection decision does not match its frozen summary")

        report = summary.copy()
        report["behavior_guard_passed"] = (
            report["mean_repeated_4gram_fraction"] <= baseline_repetition + 0.05
        ) & (report["mean_text_tokens"] <= baseline_tokens * 1.5)
        report["selected"] = False
        if bool(decision["eligible"]):
            selected = (report["layer"] == int(decision["layer"])) & (
                report["alpha"] == float(decision["alpha"])
            )
            if int(selected.sum()) != 1:
                raise ValueError(f"{method} frozen selected cell is absent or ambiguous")
            report.loc[selected, "selected"] = True
        report["selection_eligible"] = bool(decision["eligible"])
        report["selection_reason"] = str(decision["reason"])
        report["validation_run_dir"] = str(run_dir)
        frames.append(report)
        inputs.append(
            {
                "method": method,
                "run_dir": str(run_dir),
                "summary_sha256": observed_hash,
                "decision": decision,
            }
        )

    report = pd.concat(frames, ignore_index=True).sort_values(
        ["method", "layer", "alpha"], kind="mergesort"
    ).reset_index(drop=True)
    manifest = {
        "protocol": _PROTOCOL,
        "selection_path": str(selection),
        "selection_sha256": sha256_file(selection),
        "methods": inputs,
        "n_methods": len(inputs),
        "n_cells": int(len(report)),
        "selected_cells": int(report["selected"].sum()),
        "outputs": {},
    }
    return report, manifest


def tex_selected_rows(report: pd.DataFrame) -> str:
    """Render the one frozen choice per method for an appendix-facing table."""
    selected = report[report["selected"]].sort_values("method", kind="mergesort")
    header = (
        "Method & $\\ell$ & $\\alpha$ & Val. acc. & $\\Delta$ (95\\% CI) & "
        "Repair/Break & $p$ \\\\"
    )
    rows = [header, "\\midrule"]
    for row in selected.to_dict("records"):
        method = str(row["method"]).replace("_", "\\_")
        rows.append(
            f"{method} & {int(row['layer'])} & {float(row['alpha']):.3g} & "
            f"{float(row['method_accuracy']):.3f} & "
            f"{100 * float(row['delta']):+.1f} ["
            f"{100 * float(row['bootstrap_ci_low']):+.1f}, "
            f"{100 * float(row['bootstrap_ci_high']):+.1f}] & "
            f"{int(row['repairs'])}/{int(row['breaks'])} & "
            f"{float(row['exact_sign_p_value']):.4f} \\\\"
        )
    return "\n".join(rows) + "\n"


def write_validation_family_report(
    selection_path: str | Path, out: str | Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Write full CSV, selected-cell TeX rows, and a hash-linked manifest."""
    report, manifest = build_validation_family_report(selection_path)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "validation_complete_summary.csv"
    tex_path = output / "validation_selected_rows.tex"
    manifest_path = output / "validation_family_report_manifest.json"
    report.to_csv(summary_path, index=False)
    tex_path.write_text(tex_selected_rows(report))
    manifest["outputs"] = {
        summary_path.name: sha256_file(summary_path),
        tex_path.name: sha256_file(tex_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return report, manifest
