#!/usr/bin/env python3
"""Build the final auditable locked steering report from frozen run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from geoprobe.revision.locked_run import load_locked_run_spec, sha256_file
from geoprobe.revision.stats import holm_adjust, paired_binary_summary

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
        raise argparse.ArgumentTypeError("--run must be METHOD=/path/to/locked-run")
    method, raw_path = value.split("=", 1)
    if not method or not raw_path:
        raise argparse.ArgumentTypeError("--run must have non-empty method and path")
    return method, Path(raw_path)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(path: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is not a completed locked run; missing {missing}")


def _table(rows: pd.DataFrame, *, method: str, layer: int, alpha: float, ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = rows[rows["method"] == "baseline"].copy()
    intervention = rows[
        (rows["method"] == method) & (rows["layer"] == layer) & (rows["alpha"] == alpha)
    ].copy()
    if set(baseline["sample_id"]) != set(ids) or set(intervention["sample_id"]) != set(ids):
        raise ValueError(f"{method} locked rows do not exactly cover the frozen IDs")
    if baseline["sample_id"].duplicated().any() or intervention["sample_id"].duplicated().any():
        raise ValueError(f"{method} locked rows duplicate a problem ID")
    missing = sorted(set(_BASELINE_COLUMNS) - set(baseline.columns) | set(_BASELINE_COLUMNS) - set(intervention.columns))
    if missing:
        raise ValueError(f"{method} lacks locked behavior/termination fields: {missing}")
    return (
        baseline.sort_values("sample_id").reset_index(drop=True),
        intervention.sort_values("sample_id").reset_index(drop=True),
    )


def _load_locked(selection: Path, method: str, path: Path) -> dict[str, object]:
    _require(path, ("DONE", "resolved_manifest.json", "per_sample.parquet", "summary.csv", "locked_launch_manifest.json"))
    spec = load_locked_run_spec(selection, method)
    launch = json.loads((path / "locked_launch_manifest.json").read_text())
    if launch.get("selection_sha256") != sha256_file(selection):
        raise ValueError(f"{method} locked launch was not tied to this frozen selection")
    if launch.get("spec") != spec.to_dict():
        raise ValueError(f"{method} locked launch spec differs from selection")
    manifest = json.loads((path / "resolved_manifest.json").read_text())
    if manifest.get("protocol") != "tacl-11241-frozen-steering-grid-v1":
        raise ValueError(f"{method} has unexpected locked-run protocol")
    if manifest.get("eval_start") != spec.locked_eval_start or manifest.get("eval_count") != spec.locked_eval_count:
        raise ValueError(f"{method} does not use selected locked ID range")
    if manifest.get("layers") != [spec.layer] or manifest.get("alphas") != [spec.alpha]:
        raise ValueError(f"{method} did not use exactly the frozen layer/alpha")
    rows = pd.read_parquet(path / "per_sample.parquet")
    ids = list(range(spec.locked_eval_start, spec.locked_eval_start + spec.locked_eval_count))
    baseline, intervention = _table(rows, method=method, layer=spec.layer, alpha=spec.alpha, ids=ids)
    summary = paired_binary_summary(
        baseline["correct"].astype(int).to_numpy(), intervention["correct"].astype(int).to_numpy()
    )
    return {
        "method": method,
        "path": path,
        "spec": spec,
        "manifest": manifest,
        "baseline": baseline,
        "intervention": intervention,
        "summary": summary,
        "hashes": {
            "resolved_manifest.json": _hash(path / "resolved_manifest.json"),
            "per_sample.parquet": _hash(path / "per_sample.parquet"),
            "summary.csv": _hash(path / "summary.csv"),
            "locked_launch_manifest.json": _hash(path / "locked_launch_manifest.json"),
        },
    }


def _assert_shared_target(loaded: list[dict[str, object]]) -> None:
    if not loaded:
        raise ValueError("at least one locked method run is required")
    reference = loaded[0]
    ref_manifest = reference["manifest"]
    ref_baseline = reference["baseline"].loc[:, _BASELINE_COLUMNS]
    for item in loaded[1:]:
        manifest = item["manifest"]
        mismatches = {
            field: (ref_manifest.get(field), manifest.get(field))
            for field in _SHARED_FIELDS
            if ref_manifest.get(field) != manifest.get(field)
        }
        if mismatches:
            raise ValueError(f"{item['method']} has incompatible locked target protocol: {mismatches}")
        try:
            assert_frame_equal(ref_baseline, item["baseline"].loc[:, _BASELINE_COLUMNS], check_dtype=False, check_exact=True)
        except AssertionError as exc:
            raise ValueError(f"{item['method']} locked baseline differs from {reference['method']}") from exc


def _tex(value: object) -> str:
    return str(value).replace("_", "\\_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != len({method for method, _ in args.run}):
        raise SystemExit("each method may appear only once")
    selection = json.loads(args.selection.read_text())
    selected_methods = {
        item["method"]
        for item in selection.get("methods", [])
        if isinstance(item, dict) and bool(item.get("decision", {}).get("eligible"))
    }
    supplied = {method for method, _ in args.run}
    if supplied != selected_methods:
        raise SystemExit(f"provided runs {sorted(supplied)} do not equal eligible selection methods {sorted(selected_methods)}")

    loaded = [_load_locked(args.selection, method, path) for method, path in args.run]
    _assert_shared_target(loaded)
    raw_p = [item["summary"].exact_sign_p_value for item in loaded]
    adjusted = holm_adjust(raw_p)
    report_rows: list[dict[str, object]] = []
    for item, p_holm in zip(loaded, adjusted, strict=True):
        baseline = item["baseline"]
        method = item["intervention"]
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
                "holm_adjusted_p_value": float(p_holm),
                "baseline_mean_generated_tokens": float(baseline["n_generated_tokens"].mean()),
                "method_mean_generated_tokens": float(method["n_generated_tokens"].mean()),
                "generated_token_delta": float(method["n_generated_tokens"].mean() - baseline["n_generated_tokens"].mean()),
                "baseline_truncation_rate": float(baseline["truncated"].astype(float).mean()),
                "method_truncation_rate": float(method["truncated"].astype(float).mean()),
                "baseline_repetition": float(baseline["repeated_4gram_fraction"].mean()),
                "method_repetition": float(method["repeated_4gram_fraction"].mean()),
            }
        )
    report = pd.DataFrame(report_rows).sort_values("method").reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out / "locked_summary.csv", index=False)
    tex_rows = [
        "Method & $\\ell$ & $\\alpha$ & Base & Method & $\\Delta$ (95\\% CI) & Repair/Break & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for row in report.to_dict("records"):
        tex_rows.append(
            f"{_tex(row['method'])} & {row['layer']} & {row['alpha']:.3g} & "
            f"{row['baseline_accuracy']:.3f} & {row['method_accuracy']:.3f} & "
            f"{row['delta_pp']:+.1f} [{row['bootstrap_ci_low_pp']:+.1f}, {row['bootstrap_ci_high_pp']:+.1f}] & "
            f"{row['repairs']}/{row['breaks']} & {row['holm_adjusted_p_value']:.4f} \\\\"
        )
    (args.out / "locked_summary_rows.tex").write_text("\n".join(tex_rows) + "\n")
    manifest = {
        "protocol": "tacl-11241-locked-report-v1",
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
        "outputs": {"locked_summary.csv": _hash(args.out / "locked_summary.csv"), "locked_summary_rows.tex": _hash(args.out / "locked_summary_rows.tex")},
    }
    (args.out / "locked_report_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
