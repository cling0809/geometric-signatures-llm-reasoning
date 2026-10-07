#!/usr/bin/env python3
"""Run an explicit retrospective signature bootstrap audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geoprobe.revision.signature_stability import bootstrap_topology, load_signature_data


def _parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must have NAME=/absolute/run/path")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--run must have a non-empty name and path")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = [load_signature_data(name, path) for name, path in args.run]
    distances, neighbors, cells = bootstrap_topology(
        items, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    distances.to_csv(args.out / "distance_bootstrap.csv", index=False)
    neighbors.to_csv(args.out / "nearest_neighbor_stability.csv", index=False)
    cells.to_csv(args.out / "signature_cells.csv", index=False)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "tacl-11241-retrospective-signature-bootstrap-v1",
                "retrospective_only": True,
                "runs": [{"name": name, "path": path} for name, path in args.run],
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n"
    )
    print(distances.to_string(index=False))
    print(neighbors.to_string(index=False))


if __name__ == "__main__":
    main()
