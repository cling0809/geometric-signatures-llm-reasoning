#!/usr/bin/env python3
"""Build a fingerprinted sign-reversed steering direction control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from geoprobe.revision.vector_registry import sha256_file
from geoprobe.steering import opposite_direction

_PROTOCOL = "tacl-11241-negative-direction-control-v1"


def build_manifest(reference: Path, out_vector: Path) -> dict[str, object]:
    """Create the exact negative of a tracked reference direction."""
    reference = reference.resolve()
    source_manifest_path = reference.parent / "manifest.json"
    if not reference.is_file():
        raise FileNotFoundError(f"reference vector does not exist: {reference}")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"reference vector lacks sibling manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_ids = [int(value) for value in source_manifest.get("source_ids", [])]
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("reference manifest must record non-empty unique source_ids")
    reference_vector = torch.load(reference, map_location="cpu", weights_only=True)
    if not isinstance(reference_vector, torch.Tensor) or reference_vector.ndim != 2:
        raise ValueError("reference vector must be a rank-2 [layers, hidden] torch tensor")
    negative = opposite_direction(reference_vector)
    out_vector.parent.mkdir(parents=True, exist_ok=True)
    torch.save(negative.cpu(), out_vector)
    return {
        "protocol": _PROTOCOL,
        "construction": "exact_sign_reversal_of_crosssteer_source_direction",
        "reference_vector": str(reference),
        "reference_vector_sha256": sha256_file(reference),
        "reference_manifest": str(source_manifest_path),
        "reference_manifest_sha256": sha256_file(source_manifest_path),
        "source_ids": source_ids,
        "shape": list(reference_vector.shape),
        "reference_raw_l2_norm": float(torch.linalg.vector_norm(reference_vector.float()).item()),
        "negative_raw_l2_norm": float(torch.linalg.vector_norm(negative).item()),
        "vector": out_vector.name,
        "vector_sha256": sha256_file(out_vector),
        "normalization_at_evaluation": "direction RMS=1 before alpha",
        "selection_rule": "one exact opposite direction; no sign search after validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    vector_path = args.out / "negative_crosssteer_source.pt"
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_vector = args.out / str(existing.get("vector", ""))
        if (
            existing.get("protocol") != _PROTOCOL
            or existing.get("reference_vector") != str(args.reference.resolve())
            or existing.get("reference_vector_sha256") != sha256_file(args.reference)
            or not existing_vector.is_file()
            or existing.get("vector_sha256") != sha256_file(existing_vector)
        ):
            raise SystemExit("refuse to overwrite a different negative-direction control")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return
    manifest = build_manifest(args.reference, vector_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
