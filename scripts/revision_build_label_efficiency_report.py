#!/usr/bin/env python3
"""Build the full frozen target-label-efficiency report from locked artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from geoprobe.revision.stats import holm_adjust, paired_binary_summary

_PROTOCOL = "tacl-11241-target-label-efficiency-locked-v1"
_BASELINE_COLUMNS = (
    "sample_id",
    "gold",
    "pred",
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
_SHARED_FIELDS = (
    "target_model",
    "target_source",
    "eval_start",
    "eval_count",
    "schedule",
    "schedule_parameter",
    "position_mode",
    "injection_mode",
    "normalization",
    "max_new_tokens",
    "prompt_template",
)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be METHOD=/path/to/run")
    method, raw_path = value.split("=", 1)
    if not method or not raw_path:
        raise argparse.ArgumentTypeError("--run must have non-empty method and path")
    return method, Path(raw_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_plan(path: Path) -> dict[str, object]:
    plan = json.loads(path.read_text())
    if plan.get("protocol") != _PROTOCOL:
        raise ValueError("unexpected target-label-efficiency launch protocol")
    spec = plan.get("frozen_target_spec")
    if not isinstance(spec, dict):
        raise ValueError("launch plan lacks frozen target specification")
    methods = plan.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("launch plan lacks methods")
    return plan


def _expected_methods(plan: dict[str, object]) -> dict[str, dict[str, object]]:
    expected: dict[str, dict[str, object]] = {}
    for item in plan["methods"]:
        if not isinstance(item, dict):
            raise ValueError("launch plan method entry must be an object")
        method = item.get("method")
        budget = item.get("label_budget")
        if not isinstance(method, str) or not isinstance(budget, int) or budget < 0:
            raise ValueError("launch plan method lacks a nonnegative label budget")
        if method in expected:
            raise ValueError(f"launch plan repeats {method}")
        expected[method] = item
    return expected


def _load_run(
    *,
    method: str,
    path: Path,
    expected: dict[str, object],
    plan: dict[str, object],
) -> dict[str, object]:
    required = ("DONE", "resolved_manifest.json", "per_sample.parquet")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is incomplete: {missing}")
    manifest = json.loads((path / "resolved_manifest.json").read_text())
    if manifest.get("protocol") != "tacl-11241-frozen-steering-grid-v1":
        raise ValueError(f"{method} has an unexpected grid protocol")
    spec = plan["frozen_target_spec"]
    expected_pairs = {
        "target_model": plan["target_model"],
        "target_source": plan["target_source"],
        "eval_start": spec["locked_eval_start"],
        "eval_count": spec["locked_eval_count"],
        "layers": [spec["layer"]],
        "alphas": [spec["alpha"]],
        "schedule": spec["schedule"],
        "position_mode": "decode_last",
        "injection_mode": spec["injection_mode"],
        "max_new_tokens": spec["max_new_tokens"],
    }
    mismatches = {
        key: (expected_value, manifest.get(key))
        for key, expected_value in expected_pairs.items()
        if manifest.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"{method} differs from the frozen target settings: {mismatches}")
    if set(manifest.get("vectors", {})) != {method}:
        raise ValueError(f"{method} run has unexpected vectors")

    rows = pd.read_parquet(path / "per_sample.parquet")
    baseline = rows[rows["method"] == "baseline"].copy()
    intervention = rows[rows["method"] == method].copy()
    required_fields = set(_BASELINE_COLUMNS)
    missing_fields = sorted(
        required_fields - set(baseline.columns) | required_fields - set(intervention.columns)
    )
    if missing_fields:
        raise ValueError(f"{method} lacks fields: {missing_fields}")
    layer, alpha = int(spec["layer"]), float(spec["alpha"])
    intervention = intervention[(intervention["layer"] == layer) & (intervention["alpha"] == alpha)]
    expected_ids = set(
        range(
            int(spec["locked_eval_start"]),
            int(spec["locked_eval_start"]) + int(spec["locked_eval_count"]),
        )
    )
    for name, frame in (("baseline", baseline), ("intervention", intervention)):
        if set(frame["sample_id"]) != expected_ids or frame["sample_id"].duplicated().any():
            raise ValueError(f"{method} {name} does not cover exactly the frozen locked IDs")
    baseline = baseline.sort_values("sample_id").reset_index(drop=True)
    intervention = intervention.sort_values("sample_id").reset_index(drop=True)
    return {
        "method": method,
        "budget": int(expected["label_budget"]),
        "path": path,
        "manifest": manifest,
        "baseline": baseline,
        "intervention": intervention,
        "summary": paired_binary_summary(
            baseline["correct"].astype(int).to_numpy(),
            intervention["correct"].astype(int).to_numpy(),
        ),
        "hashes": {
            "resolved_manifest.json": _sha256(path / "resolved_manifest.json"),
            "per_sample.parquet": _sha256(path / "per_sample.parquet"),
        },
    }


def _assert_shared_baseline(loaded: list[dict[str, object]]) -> None:
    reference = loaded[0]
    ref_manifest = reference["manifest"]
    ref_baseline = reference["baseline"].loc[:, _BASELINE_COLUMNS]
    for item in loaded[1:]:
        mismatches = {
            field: (ref_manifest.get(field), item["manifest"].get(field))
            for field in _SHARED_FIELDS
            if ref_manifest.get(field) != item["manifest"].get(field)
        }
        if mismatches:
            raise ValueError(f"{item['method']} differs from shared target protocol: {mismatches}")
        try:
            assert_frame_equal(
                ref_baseline,
                item["baseline"].loc[:, _BASELINE_COLUMNS],
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as error:
            raise ValueError(
                f"{item['method']} baseline differs from {reference['method']}"
            ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != len({method for method, _ in args.run}):
        raise SystemExit("each method may appear only once")
    plan = _load_plan(args.plan)
    expected = _expected_methods(plan)
    supplied = {method for method, _ in args.run}
    if supplied != set(expected):
        raise SystemExit(
            f"provided methods {sorted(supplied)} do not equal frozen plan {sorted(expected)}"
        )
    loaded = [
        _load_run(method=method, path=path, expected=expected[method], plan=plan)
        for method, path in args.run
    ]
    _assert_shared_baseline(loaded)
    adjusted = holm_adjust([item["summary"].exact_sign_p_value for item in loaded])
    report_rows: list[dict[str, object]] = []
    for item, holm_p in zip(loaded, adjusted, strict=True):
        baseline = item["baseline"]
        intervention = item["intervention"]
        summary = item["summary"]
        report_rows.append(
            {
                "label_budget": item["budget"],
                "method": item["method"],
                "n": summary.n,
                "baseline_accuracy": summary.baseline_accuracy,
                "method_accuracy": summary.method_accuracy,
                "delta_pp": 100 * summary.delta,
                "bootstrap_ci_low_pp": 100 * summary.bootstrap_ci_low,
                "bootstrap_ci_high_pp": 100 * summary.bootstrap_ci_high,
                "repairs": summary.repairs,
                "breaks": summary.breaks,
                "exact_sign_p_value": summary.exact_sign_p_value,
                "holm_adjusted_p_value": float(holm_p),
                "baseline_mean_generated_tokens": float(baseline["n_generated_tokens"].mean()),
                "method_mean_generated_tokens": float(intervention["n_generated_tokens"].mean()),
                "generated_token_delta": float(
                    intervention["n_generated_tokens"].mean()
                    - baseline["n_generated_tokens"].mean()
                ),
                "baseline_repetition": float(baseline["repeated_4gram_fraction"].mean()),
                "method_repetition": float(intervention["repeated_4gram_fraction"].mean()),
            }
        )
    report = pd.DataFrame(report_rows).sort_values("label_budget").reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out / "label_efficiency_summary.csv", index=False)
    tex = [
        "Labels & Base & Method & $\\Delta$ (95\\% CI) & Repair/Break & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for row in report.to_dict("records"):
        tex.append(
            f"{row['label_budget']} & {row['baseline_accuracy']:.3f} & {row['method_accuracy']:.3f} & "
            f"{row['delta_pp']:+.1f} [{row['bootstrap_ci_low_pp']:+.1f}, {row['bootstrap_ci_high_pp']:+.1f}] & "
            f"{row['repairs']}/{row['breaks']} & {row['holm_adjusted_p_value']:.4f} \\\\"
        )
    (args.out / "label_efficiency_rows.tex").write_text("\n".join(tex) + "\n")
    manifest = {
        "protocol": "tacl-11241-target-label-efficiency-report-v1",
        "plan_sha256": _sha256(args.plan),
        "methods": [
            {
                "method": item["method"],
                "label_budget": item["budget"],
                "run_dir": str(item["path"].resolve()),
                "hashes": item["hashes"],
            }
            for item in loaded
        ],
        "multiple_comparison_family": [item["method"] for item in loaded],
        "outputs": {
            "label_efficiency_summary.csv": _sha256(args.out / "label_efficiency_summary.csv"),
            "label_efficiency_rows.tex": _sha256(args.out / "label_efficiency_rows.tex"),
        },
    }
    (args.out / "label_efficiency_report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
