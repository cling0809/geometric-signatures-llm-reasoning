#!/usr/bin/env python3
"""Build a frozen long-context stability report for one selected method/budget."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from geoprobe.revision.locked_run import load_locked_run_spec, sha256_file
from geoprobe.revision.stats import holm_adjust, paired_binary_summary

_PROTOCOL = "tacl-11241-long-context-v1"
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
    "max_new_tokens",
    "position_mode",
    "normalization",
    "prompt_template",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_plan(path: Path, selection: Path, method: str) -> dict[str, object]:
    plan = json.loads(path.read_text())
    if plan.get("protocol") != _PROTOCOL or plan.get("selection_sha256") != sha256_file(selection):
        raise ValueError("long-context plan is not tied to the supplied frozen selection")
    spec = load_locked_run_spec(selection, method)
    if plan.get("spec") != spec.to_dict():
        raise ValueError("long-context plan differs from frozen selection")
    budget = plan.get("budget")
    if budget not in (4096, 32768):
        raise ValueError("long-context plan has unsupported budget")
    policies = plan.get("policies")
    if not isinstance(policies, list) or {
        item.get("name") for item in policies if isinstance(item, dict)
    } != {"constant", "prefix-256", "exponential-1024", "relative-hidden-rms"}:
        raise ValueError("long-context plan lacks exactly the four frozen policies")
    return plan


def _load_policy(
    *, plan: dict[str, object], method: str, root: Path, policy: dict[str, object]
) -> dict[str, object]:
    name = str(policy["name"])
    path = root / name
    required = ("DONE", "resolved_manifest.json", "per_sample.parquet")
    missing = [item for item in required if not (path / item).exists()]
    if missing:
        raise FileNotFoundError(f"{name} run incomplete: {missing}")
    manifest = json.loads((path / "resolved_manifest.json").read_text())
    spec = plan["spec"]
    expected = {
        "protocol": "tacl-11241-frozen-steering-grid-v1",
        "target_model": plan["target_model"],
        "target_source": plan["target_source"],
        "eval_start": spec["locked_eval_start"],
        "eval_count": spec["locked_eval_count"],
        "layers": [spec["layer"]],
        "alphas": [spec["alpha"]],
        "schedule": policy["schedule"],
        "schedule_parameter": policy["schedule_parameter"],
        "position_mode": "decode_last",
        "injection_mode": policy["injection_mode"],
        "max_new_tokens": plan["budget"],
    }
    mismatches = {
        key: (value, manifest.get(key))
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{name} does not match frozen long-context policy: {mismatches}")
    if set(manifest.get("vectors", {})) != {method}:
        raise ValueError(f"{name} has unexpected vector methods")
    rows = pd.read_parquet(path / "per_sample.parquet")
    baseline = rows[rows["method"] == "baseline"].copy()
    intervention = rows[
        (rows["method"] == method)
        & (rows["layer"] == spec["layer"])
        & (rows["alpha"] == spec["alpha"])
    ].copy()
    missing_fields = sorted(
        set(_BASELINE_COLUMNS) - set(baseline.columns)
        | set(_BASELINE_COLUMNS) - set(intervention.columns)
    )
    if missing_fields:
        raise ValueError(f"{name} missing behavior fields: {missing_fields}")
    expected_ids = set(
        range(
            int(spec["locked_eval_start"]),
            int(spec["locked_eval_start"]) + int(spec["locked_eval_count"]),
        )
    )
    for label, frame in (("baseline", baseline), ("intervention", intervention)):
        if set(frame["sample_id"]) != expected_ids or frame["sample_id"].duplicated().any():
            raise ValueError(f"{name} {label} does not cover frozen IDs exactly once")
    baseline = baseline.sort_values("sample_id").reset_index(drop=True)
    intervention = intervention.sort_values("sample_id").reset_index(drop=True)
    return {
        "name": name,
        "path": path,
        "policy": policy,
        "manifest": manifest,
        "baseline": baseline,
        "intervention": intervention,
        "summary": paired_binary_summary(
            baseline["correct"].astype(int).to_numpy(),
            intervention["correct"].astype(int).to_numpy(),
        ),
        "hashes": {
            name: _hash(path / name) for name in ("resolved_manifest.json", "per_sample.parquet")
        },
    }


def _assert_shared_baseline(items: list[dict[str, object]]) -> None:
    reference = items[0]
    ref_manifest = reference["manifest"]
    ref_baseline = reference["baseline"].loc[:, _BASELINE_COLUMNS]
    for item in items[1:]:
        mismatches = {
            field: (ref_manifest.get(field), item["manifest"].get(field))
            for field in _SHARED_FIELDS
            if ref_manifest.get(field) != item["manifest"].get(field)
        }
        if mismatches:
            raise ValueError(
                f"{item['name']} does not share long-context baseline protocol: {mismatches}"
            )
        try:
            assert_frame_equal(
                ref_baseline,
                item["baseline"].loc[:, _BASELINE_COLUMNS],
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as error:
            raise ValueError(f"{item['name']} baseline differs from {reference['name']}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = _load_plan(args.plan, args.selection, args.method)
    policies = sorted(plan["policies"], key=lambda item: str(item["name"]))
    items = [
        _load_policy(plan=plan, method=args.method, root=args.plan.parent, policy=policy)
        for policy in policies
    ]
    _assert_shared_baseline(items)
    adjusted = holm_adjust([item["summary"].exact_sign_p_value for item in items])
    report_rows: list[dict[str, object]] = []
    for item, holm_p in zip(items, adjusted, strict=True):
        baseline = item["baseline"]
        intervention = item["intervention"]
        summary = item["summary"]
        report_rows.append(
            {
                "policy": item["name"],
                "budget": plan["budget"],
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
    report = pd.DataFrame(report_rows).sort_values("policy").reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out / "long_context_summary.csv", index=False)
    header = (
        "Policy & Base & Method & $\\Delta$ (95\\% CI) & Repair/Break & $p_{\\mathrm{Holm}}$ "
        + "\\\\"
    )
    tex_rows = [header, "\\midrule"]
    for row in report.to_dict("records"):
        tex_rows.append(
            f"{row['policy']} & {row['baseline_accuracy']:.3f} & {row['method_accuracy']:.3f} & "
            f"{row['delta_pp']:+.1f} [{row['bootstrap_ci_low_pp']:+.1f}, "
            f"{row['bootstrap_ci_high_pp']:+.1f}] & {row['repairs']}/{row['breaks']} & "
            f"{row['holm_adjusted_p_value']:.4f} \\\\"
        )
    (args.out / "long_context_rows.tex").write_text("\n".join(tex_rows) + "\n")
    manifest = {
        "protocol": "tacl-11241-long-context-report-v1",
        "selection_sha256": _hash(args.selection),
        "plan_sha256": _hash(args.plan),
        "method": args.method,
        "budget": plan["budget"],
        "policies": [
            {"name": item["name"], "run_dir": str(item["path"].resolve()), "hashes": item["hashes"]}
            for item in items
        ],
        "outputs": {
            "long_context_summary.csv": _hash(args.out / "long_context_summary.csv"),
            "long_context_rows.tex": _hash(args.out / "long_context_rows.tex"),
        },
    }
    (args.out / "long_context_report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
