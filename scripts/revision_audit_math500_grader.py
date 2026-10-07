#!/usr/bin/env python3
"""Audit the pinned symbolic MATH-500 evaluator before a formal OOD launch."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from geoprobe.datasets import is_correct_math500_symbolic, load_math500

_PROTOCOL = "tacl-11241-math500-symbolic-grader-audit-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        version = importlib.metadata.version("math-verify")
    except importlib.metadata.PackageNotFoundError as error:
        raise SystemExit(
            "math-verify==0.9.0 is required for the formal MATH-500 grader audit"
        ) from error
    if version != "0.9.0":
        raise SystemExit(f"expected math-verify==0.9.0, found {version}")

    samples = load_math500()
    parse_failures = [
        sample.id
        for sample in samples
        if not is_correct_math500_symbolic(
            r"\boxed{" + sample.gold_answer + "}", sample.gold_answer
        )
    ]
    synthetic_cases = [
        ("Final: \\boxed{0.5}", r"\frac{1}{2}", True),
        ("Final: \\boxed{x^2 + 2x + 1}", r"(x+1)^2", True),
        ("Final: \\boxed{0.6}", r"\frac{1}{2}", False),
        ("No mathematical answer is given.", "42", False),
    ]
    synthetic = [
        {
            "completion": completion,
            "gold": gold,
            "expected": expected,
            "observed": is_correct_math500_symbolic(completion, gold),
        }
        for completion, gold, expected in synthetic_cases
    ]
    synthetic_failures = [
        index for index, result in enumerate(synthetic) if result["observed"] != result["expected"]
    ]
    payload = {
        "protocol": _PROTOCOL,
        "math_verify_version": version,
        "dataset": "HuggingFaceH4/MATH-500",
        "n_gold_answers": len(samples),
        "gold_self_equivalence_failures": parse_failures,
        "synthetic_cases": synthetic,
        "synthetic_failure_indices": synthetic_failures,
        "passed": not parse_failures and not synthetic_failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and json.loads(args.out.read_text()) != payload:
        raise SystemExit("refuse to overwrite a different MATH-500 grader audit")
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit("MATH-500 symbolic grader audit failed")


if __name__ == "__main__":
    main()
