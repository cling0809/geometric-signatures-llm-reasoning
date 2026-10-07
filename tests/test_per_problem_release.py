from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from geoprobe.revision.per_problem_release import (
    FORBIDDEN_RELEASE_COLUMNS,
    export_anonymous_per_problem,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_rows() -> pd.DataFrame:
    rows = []
    for arm in ("baseline", "crosssteer_source"):
        for sample_id in (200, 201):
            rows.append(
                {
                    "method": arm,
                    "layer": 14,
                    "alpha": 0.025,
                    "sample_id": sample_id,
                    "gold": 7.0,
                    "pred": 7.0 if sample_id == 200 else 8.0,
                    "correct": sample_id == 200,
                    "generated_text": "private completion",
                    "text_tokens": 10,
                    "generation_seed": None,
                    "n_generated_tokens": 11,
                    "n_content_tokens": 10,
                    "stop_reason": "eos",
                    "truncated": False,
                    "text_tokens_whitespace": 10,
                    "distinct_2gram_ratio": 0.9,
                    "distinct_4gram_ratio": 0.8,
                    "repeated_4gram_fraction": 0.0,
                    "answer_marker_position": 8.0,
                    "answer_marker_relative_position": 0.8,
                }
            )
    return pd.DataFrame(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    suite_root = tmp_path / "locked"
    run = suite_root / "crosssteer_source"
    run.mkdir(parents=True)
    source = run / "per_sample.parquet"
    _formal_rows().to_parquet(source, index=False)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "methods": [
                    {
                        "method": "crosssteer_source",
                        "run_dir": "/remote/crosssteer_source",
                        "hashes": {"per_sample.parquet": _sha256(source)},
                    }
                ]
            }
        )
    )
    return suite_root, report


def test_export_anonymous_per_problem_verifies_and_removes_content(tmp_path: Path):
    suite_root, report = _fixture(tmp_path)
    out = tmp_path / "audit.csv"
    manifest_path = tmp_path / "audit_manifest.json"

    manifest = export_anonymous_per_problem(
        suite="locked_gsm8k",
        suite_root=suite_root,
        report_manifest=report,
        out_csv=out,
        out_manifest=manifest_path,
    )

    released = pd.read_csv(out)
    assert len(released) == 4
    assert set(released["arm"]) == {"baseline", "crosssteer_source"}
    assert not (FORBIDDEN_RELEASE_COLUMNS & set(released.columns))
    assert "generated_text" not in out.read_text()
    assert manifest["output"]["sha256"] == _sha256(out)
    assert json.loads(manifest_path.read_text()) == manifest


def test_export_anonymous_per_problem_rejects_source_hash_mismatch(tmp_path: Path):
    suite_root, report = _fixture(tmp_path)
    payload = json.loads(report.read_text())
    payload["methods"][0]["hashes"]["per_sample.parquet"] = "0" * 64
    report.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="source hash mismatch"):
        export_anonymous_per_problem(
            suite="locked_gsm8k",
            suite_root=suite_root,
            report_manifest=report,
            out_csv=tmp_path / "audit.csv",
            out_manifest=tmp_path / "audit_manifest.json",
        )


def test_released_per_problem_tables_match_frozen_summaries():
    root = Path("revision/evidence/per-problem-audit-v1")
    suites = {
        "locked_gsm8k": (
            root / "locked_gsm8k.csv",
            Path("revision/evidence/locked-gsm8k-qwen-instruct-v3/locked_summary.csv"),
        ),
        "ood_math500": (
            root / "ood_math500.csv",
            Path("revision/evidence/ood-math500-qwen-instruct-v3/ood_math500_summary.csv"),
        ),
        "ood_svamp": (
            root / "ood_svamp.csv",
            Path("revision/evidence/ood-svamp-qwen-instruct-v3/ood_svamp_summary.csv"),
        ),
        "label_efficiency": (
            root / "label_efficiency.csv",
            Path(
                "revision/evidence/label-efficiency-qwen-instruct-v3/"
                "label_efficiency_summary.csv"
            ),
        ),
    }
    expected_rows = {
        "locked_gsm8k": 1600,
        "ood_math500": 8000,
        "ood_svamp": 16000,
        "label_efficiency": 1200,
    }

    for suite, (csv_path, summary_path) in suites.items():
        frame = pd.read_csv(csv_path)
        summary = pd.read_csv(summary_path).set_index("method")
        manifest = json.loads((root / f"{suite}_manifest.json").read_text())
        assert len(frame) == expected_rows[suite] == manifest["output"]["rows"]
        assert _sha256(csv_path) == manifest["output"]["sha256"]
        assert not (FORBIDDEN_RELEASE_COLUMNS & set(frame.columns))
        report_path = {
            "locked_gsm8k": Path(
                "revision/evidence/locked-gsm8k-qwen-instruct-v3/"
                "locked_report_manifest.json"
            ),
            "ood_math500": Path(
                "revision/evidence/ood-math500-qwen-instruct-v3/ood_report_manifest.json"
            ),
            "ood_svamp": Path(
                "revision/evidence/ood-svamp-qwen-instruct-v3/ood_report_manifest.json"
            ),
            "label_efficiency": Path(
                "revision/evidence/label-efficiency-qwen-instruct-v3/"
                "label_efficiency_report_manifest.json"
            ),
        }[suite]
        assert _sha256(report_path) == manifest["report_manifest_sha256"]

        for _comparison, group in frame.groupby("comparison", sort=False):
            method_arms = sorted(set(group["arm"]) - {"baseline"})
            assert len(method_arms) == 1
            method = method_arms[0]
            expected = summary.loc[method]
            baseline = group[group["arm"] == "baseline"].set_index("sample_id")
            treated = group[group["arm"] == method].set_index("sample_id")
            assert baseline.index.equals(treated.index)
            assert len(baseline) == int(expected["n"])
            assert baseline["correct"].mean() == pytest.approx(
                expected["baseline_accuracy"]
            )
            assert treated["correct"].mean() == pytest.approx(expected["method_accuracy"])
            repairs = ((~baseline["correct"]) & treated["correct"]).sum()
            breaks = (baseline["correct"] & (~treated["correct"])).sum()
            assert repairs == int(expected["repairs"])
            assert breaks == int(expected["breaks"])
            assert baseline["n_generated_tokens"].mean() == pytest.approx(
                expected["baseline_mean_generated_tokens"]
            )
            assert treated["n_generated_tokens"].mean() == pytest.approx(
                expected["method_mean_generated_tokens"]
            )
