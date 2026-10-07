#!/usr/bin/env python3
"""Build an all-method frozen OOD report from audited MATH-500 or SVAMP artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from geoprobe.datasets.svamp import SVAMP_EXPECTED_COUNT, SVAMP_SHA256
from geoprobe.revision.locked_run import load_locked_run_spec, sha256_file
from geoprobe.revision.stats import holm_adjust, paired_binary_summary

_MATH_PROTOCOL = "tacl-11241-frozen-ood-launch-v1"
_SVAMP_PROTOCOL = "tacl-11241-frozen-svamp-ood-launch-v1"
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
    "dataset",
    "cross_dataset_eval",
    "evaluator",
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
        raise argparse.ArgumentTypeError("--run must be METHOD=/path/to/OOD-run")
    method, raw_path = value.split("=", 1)
    if not method or not raw_path:
        raise argparse.ArgumentTypeError("--run must have non-empty method/path")
    return method, Path(raw_path)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(
    selection: Path, method: str, path: Path, *, dataset: str = "math500"
) -> dict[str, object]:
    required = ("DONE", "resolved_manifest.json", "per_sample.parquet", "ood_launch_manifest.json")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path} incomplete: {missing}")

    spec = load_locked_run_spec(selection, method)
    launch = json.loads((path / "ood_launch_manifest.json").read_text())
    expected_launch_protocol = {"math500": _MATH_PROTOCOL, "svamp": _SVAMP_PROTOCOL}.get(dataset)
    if expected_launch_protocol is None:
        raise ValueError(f"unsupported OOD dataset: {dataset}")
    if launch.get("protocol") != expected_launch_protocol or launch.get("selection_sha256") != sha256_file(
        selection
    ):
        raise ValueError(f"{method} is not tied to the frozen selection")
    if launch.get("frozen_target_spec") != spec.to_dict():
        raise ValueError(f"{method} OOD specification differs from frozen selection")

    manifest = json.loads((path / "resolved_manifest.json").read_text())
    expected_count = {"math500": 500, "svamp": SVAMP_EXPECTED_COUNT}[dataset]
    expected_manifest = {
        "protocol": "tacl-11241-frozen-steering-grid-v1",
        "dataset": dataset,
        "cross_dataset_eval": True,
        "eval_start": 0,
        "eval_count": expected_count,
        "layers": [spec.layer],
        "alphas": [spec.alpha],
        "target_model": launch["target_model"],
        "target_source": launch["target_source"],
        "schedule": spec.schedule,
        "position_mode": "decode_last",
        "injection_mode": spec.injection_mode,
        "max_new_tokens": spec.max_new_tokens,
    }
    mismatches = {
        field: (expected_value, manifest.get(field))
        for field, expected_value in expected_manifest.items()
        if manifest.get(field) != expected_value
    }
    if mismatches:
        raise ValueError(f"{method} OOD grid differs from frozen {dataset} protocol: {mismatches}")
    evaluator = manifest.get("evaluator")
    expected_evaluator = {
        "math500": ("math_verify_symbolic_full_completion", "math-verify==0.9.0"),
        "svamp": ("svamp_numeric_boxed_last_number", f"sha256:{SVAMP_SHA256}"),
    }[dataset]
    if (
        not isinstance(evaluator, dict)
        or evaluator.get("name") != expected_evaluator[0]
        or evaluator.get("version") != expected_evaluator[1]
    ):
        raise ValueError(f"{method} does not use the pinned {dataset} evaluator")

    rows = pd.read_parquet(path / "per_sample.parquet")
    baseline = rows[rows["method"] == "baseline"].copy()
    intervention = rows[
        (rows["method"] == method) & (rows["layer"] == spec.layer) & (rows["alpha"] == spec.alpha)
    ].copy()
    missing_fields = sorted(
        set(_BASELINE_COLUMNS) - set(baseline.columns)
        | set(_BASELINE_COLUMNS) - set(intervention.columns)
    )
    if missing_fields:
        raise ValueError(f"{method} missing OOD fields: {missing_fields}")
    expected_ids = set(range(expected_count))
    for name, frame in (("baseline", baseline), ("intervention", intervention)):
        if set(frame["sample_id"]) != expected_ids or frame["sample_id"].duplicated().any():
            raise ValueError(f"{method} {name} does not cover the frozen OOD dataset exactly once")
    baseline = baseline.sort_values("sample_id").reset_index(drop=True)
    intervention = intervention.sort_values("sample_id").reset_index(drop=True)
    return {
        "method": method,
        "path": path,
        "spec": spec,
        "manifest": manifest,
        "baseline": baseline,
        "intervention": intervention,
        "summary": paired_binary_summary(
            baseline["correct"].astype(int).to_numpy(),
            intervention["correct"].astype(int).to_numpy(),
        ),
        "hashes": {
            name: _hash(path / name)
            for name in ("resolved_manifest.json", "per_sample.parquet", "ood_launch_manifest.json")
        },
    }


def _assert_shared_target(loaded: list[dict[str, object]]) -> None:
    if not loaded:
        raise ValueError("at least one OOD run is required")
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
            raise ValueError(f"{item['method']} has incompatible OOD target protocol: {mismatches}")
        try:
            assert_frame_equal(
                ref_baseline,
                item["baseline"].loc[:, _BASELINE_COLUMNS],
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as error:
            raise ValueError(
                f"{item['method']} OOD baseline differs from {reference['method']}"
            ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--dataset", choices=["math500", "svamp"], default="math500")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != len({method for method, _ in args.run}):
        raise SystemExit("each method may appear once")

    selection_payload = json.loads(args.selection.read_text())
    expected = {
        item["method"]
        for item in selection_payload.get("methods", [])
        if isinstance(item, dict) and bool(item.get("decision", {}).get("eligible"))
    }
    supplied = {method for method, _ in args.run}
    if supplied != expected:
        raise SystemExit(
            f"provided runs {sorted(supplied)} do not equal eligible selection methods {sorted(expected)}"
        )
    loaded = [_load(args.selection, method, path, dataset=args.dataset) for method, path in args.run]
    _assert_shared_target(loaded)

    adjusted = holm_adjust([item["summary"].exact_sign_p_value for item in loaded])
    report_rows: list[dict[str, object]] = []
    for item, holm_p in zip(loaded, adjusted, strict=True):
        baseline = item["baseline"]
        intervention = item["intervention"]
        summary = item["summary"]
        report_rows.append(
            {
                "method": item["method"],
                "layer": item["spec"].layer,
                "alpha": item["spec"].alpha,
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
                "baseline_truncation_rate": float(baseline["truncated"].astype(float).mean()),
                "method_truncation_rate": float(intervention["truncated"].astype(float).mean()),
                "baseline_repetition": float(baseline["repeated_4gram_fraction"].mean()),
                "method_repetition": float(intervention["repeated_4gram_fraction"].mean()),
            }
        )
    report = pd.DataFrame(report_rows).sort_values("method").reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary_name = f"ood_{args.dataset}_summary.csv"
    rows_name = f"ood_{args.dataset}_rows.tex"
    report.to_csv(args.out / summary_name, index=False)
    tex_header = (
        "Method & $\\ell$ & $\\alpha$ & Base & Method & $\\Delta$ (95\\% CI) "
        "& Repair/Break & $p_{\\mathrm{Holm}}$ " + "\\\\"
    )
    tex_rows = [tex_header, "\\midrule"]
    for row in report.to_dict("records"):
        method_tex = str(row["method"]).replace("_", "\\_")
        tex_rows.append(
            f"{method_tex} & {row['layer']} & {row['alpha']:.3g} & "
            f"{row['baseline_accuracy']:.3f} & {row['method_accuracy']:.3f} & "
            f"{row['delta_pp']:+.1f} [{row['bootstrap_ci_low_pp']:+.1f}, "
            f"{row['bootstrap_ci_high_pp']:+.1f}] & {row['repairs']}/{row['breaks']} & "
            f"{row['holm_adjusted_p_value']:.4f} \\\\"
        )
    (args.out / rows_name).write_text("\n".join(tex_rows) + "\n")
    report_manifest = {
        "protocol": "tacl-11241-ood-report-v1",
        "dataset": args.dataset,
        "selection_sha256": _hash(args.selection),
        "methods": [
            {
                "method": item["method"],
                "run_dir": str(item["path"].resolve()),
                "spec": item["spec"].to_dict(),
                "hashes": item["hashes"],
            }
            for item in loaded
        ],
        "multiple_comparison_family": [item["method"] for item in loaded],
        "outputs": {
            summary_name: _hash(args.out / summary_name),
            rows_name: _hash(args.out / rows_name),
        },
    }
    (args.out / "ood_report_manifest.json").write_text(
        json.dumps(report_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
