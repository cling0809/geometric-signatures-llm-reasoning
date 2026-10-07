#!/usr/bin/env python3
"""Run the frozen held-out trajectory-signature generalization audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geoprobe.revision.diagnostic_generalization import (
    heldout_model_retrieval,
    partition_signature_cells,
)
from geoprobe.revision.diagnostic_provenance import audit_diagnostic_signature_run
from geoprobe.revision.signature_stability import load_signature_data


def _parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must have NAME=/absolute/run/path")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--run must have non-empty NAME and path")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=_parse_run)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--partition-salt", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-truncation-rate", type=float, default=0.25)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=11241)
    parser.add_argument("--relative-depth-bins", type=int, default=29)
    parser.add_argument("--min-per-class", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    input_audit = [
        audit_diagnostic_signature_run(
            Path(path),
            max_new_tokens=args.max_new_tokens,
            max_truncation_rate=args.max_truncation_rate,
        )
        for _, path in args.run
    ]
    (args.out / "input_provenance_audit.json").write_text(
        json.dumps(input_audit, indent=2, sort_keys=True) + "\n"
    )
    runs = [load_signature_data(name, path) for name, path in args.run]
    signature_cells = partition_signature_cells(
        runs,
        salt=args.partition_salt,
        n_depth_bins=args.relative_depth_bins,
        min_per_class=args.min_per_class,
    )
    retrieval, distances, bootstrap, summary = heldout_model_retrieval(
        runs,
        salt=args.partition_salt,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
        n_depth_bins=args.relative_depth_bins,
        min_per_class=args.min_per_class,
    )
    signature_cells.to_csv(args.out / "partition_signature_cells.csv", index=False)
    retrieval.to_csv(args.out / "heldout_model_retrieval.csv", index=False)
    distances.to_csv(args.out / "partition_pairwise_distances.csv", index=False)
    bootstrap.to_csv(args.out / "retrieval_bootstrap.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": summary["protocol"],
                "runs": [{"name": name, "path": str(Path(path).resolve())} for name, path in args.run],
                "partition_salt": args.partition_salt,
                "max_new_tokens": args.max_new_tokens,
                "max_truncation_rate": args.max_truncation_rate,
                "input_provenance_audit": "input_provenance_audit.json",
                "n_bootstrap": args.n_bootstrap,
                "n_permutations": args.n_permutations,
                "seed": args.seed,
                "relative_depth_bins": args.relative_depth_bins,
                "minimum_per_class": args.min_per_class,
                "output_files": [
                    "partition_signature_cells.csv",
                    "input_provenance_audit.json",
                    "heldout_model_retrieval.csv",
                    "partition_pairwise_distances.csv",
                    "retrieval_bootstrap.csv",
                    "summary.json",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
