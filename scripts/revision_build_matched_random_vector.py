#!/usr/bin/env python3
"""Build one deterministic, matched-norm random-direction control artifact.

The control is intentionally a single predeclared seeded isotropic direction,
not a random-seed search.  It inherits the reference vector's source-training
IDs and proves that an apparent intervention effect is not explained merely by
injecting an arbitrary perturbation with the same raw L2 magnitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from geoprobe.revision.vector_registry import sha256_file
from geoprobe.steering import matched_norm_random_direction

_PROTOCOL = "tacl-11241-matched-random-vector-v1"
_DEFAULT_SEED = 11241


def build_manifest(reference: Path, out_vector: Path, *, seed: int) -> dict[str, object]:
    """Construct immutable provenance for one matched-norm random vector."""
    reference = reference.resolve()
    reference_manifest = reference.parent / "manifest.json"
    if not reference.is_file():
        raise FileNotFoundError(f"reference vector does not exist: {reference}")
    if not reference_manifest.is_file():
        raise FileNotFoundError(f"reference vector lacks sibling manifest: {reference_manifest}")

    source_manifest = json.loads(reference_manifest.read_text())
    source_ids = [int(value) for value in source_manifest.get("source_ids", [])]
    if not source_ids:
        raise ValueError(f"reference manifest does not record non-empty source_ids: {reference_manifest}")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("reference manifest contains duplicate source IDs")

    reference_vector = torch.load(reference, map_location="cpu", weights_only=True)
    if not isinstance(reference_vector, torch.Tensor) or reference_vector.ndim != 2:
        raise ValueError("reference vector must be a rank-2 [layers, hidden] torch tensor")
    if not torch.isfinite(reference_vector).all():
        raise ValueError("reference vector contains non-finite values")
    reference_l2 = float(torch.linalg.vector_norm(reference_vector.float()).item())
    if reference_l2 <= 0:
        raise ValueError("reference vector must have non-zero L2 norm")

    random_vector = matched_norm_random_direction(reference_vector, seed=seed)
    out_vector.parent.mkdir(parents=True, exist_ok=True)
    torch.save(random_vector.cpu(), out_vector)
    random_l2 = float(torch.linalg.vector_norm(random_vector).item())
    return {
        "protocol": _PROTOCOL,
        "construction": "fixed_seeded_isotropic_gaussian_direction_matched_to_reference_raw_l2",
        "seed": seed,
        "reference_vector": str(reference),
        "reference_vector_sha256": sha256_file(reference),
        "reference_manifest": str(reference_manifest),
        "reference_manifest_sha256": sha256_file(reference_manifest),
        "source_ids": source_ids,
        "shape": list(reference_vector.shape),
        "reference_raw_l2_norm": reference_l2,
        "random_raw_l2_norm": random_l2,
        "vector": out_vector.name,
        "vector_sha256": sha256_file(out_vector),
        "normalization_at_evaluation": "direction RMS=1 before alpha",
        "selection_rule": "one fixed control seed; no seed selection or rerun on score",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    args = parser.parse_args()

    out_dir = args.out
    vector_path = out_dir / "matched_norm_random.pt"
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        expected_reference = str(args.reference.resolve())
        existing_vector = out_dir / str(existing.get("vector", ""))
        if (
            existing.get("protocol") != _PROTOCOL
            or existing.get("seed") != args.seed
            or existing.get("reference_vector") != expected_reference
            or existing.get("reference_vector_sha256") != sha256_file(args.reference)
            or not existing_vector.is_file()
            or existing.get("vector_sha256") != sha256_file(existing_vector)
        ):
            raise SystemExit(
                "refuse to overwrite a different matched-norm random artifact; "
                "choose a new output directory under a separately declared protocol"
            )
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    manifest = build_manifest(args.reference, vector_path, seed=args.seed)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
