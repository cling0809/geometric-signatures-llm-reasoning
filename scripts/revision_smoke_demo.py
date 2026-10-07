#!/usr/bin/env python3
"""CPU-only reproducibility smoke demo for the TACL-11241 revision package.

This command requires no model download or GPU. It creates two tiny synthetic,
completed calibration runs; verifies source/target split isolation; constructs
train-only CrossSteer and target-calibrated vectors; and writes an auditable
manifest. It is an installation test, not a scientific experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from geoprobe.extractors import Trajectory, save_trajectory
from geoprobe.revision.protocol import RevisionSplit
from geoprobe.revision.vector_registry import (
    build_vector_registry,
    save_vector_registry,
    sha256_file,
)


def _write_synthetic_run(path: Path, *, correct_value: float, incorrect_value: float) -> None:
    (path / "trajectories").mkdir(parents=True, exist_ok=True)
    (path / "DONE").write_text("synthetic-complete\n")
    (path / "config.yaml").write_text("synthetic: true\n")
    labels: list[dict[str, object]] = []
    for sample_id, correct in enumerate((True, False, True, False)):
        value = correct_value if correct else incorrect_value
        trajectory = Trajectory(
            sample_id=sample_id,
            sample_idx=0,
            hidden_states=torch.full((3, 3, 5), value, dtype=torch.float32),
            generated_token_ids=torch.zeros(3, dtype=torch.int64),
            generated_text="synthetic",
            prompt="synthetic",
            prompt_len=1,
            sequence_logprob=0.0,
            model_id="synthetic",
            dtype="float32",
        )
        save_trajectory(trajectory, path / "trajectories" / f"sample_{sample_id:04d}_idx_0.pt")
        labels.append({"sample_id": sample_id, "sample_idx": 0, "correct": correct})
    pd.DataFrame(labels).to_parquet(path / "labels.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("--out must be absent or empty; smoke artifacts are immutable")
    source = args.out / "source_run"
    target = args.out / "target_run"
    _write_synthetic_run(source, correct_value=2.0, incorrect_value=-1.0)
    _write_synthetic_run(target, correct_value=4.0, incorrect_value=-2.0)
    split = RevisionSplit((0, 1, 2, 3), (4,), (5,), (6,))
    vectors, metadata = build_vector_registry(source, target, split.source_train, split=split)
    registry = args.out / "vector_registry"
    save_vector_registry(vectors, metadata, registry)
    manifest = {
        "protocol": "tacl-11241-cpu-smoke-v1",
        "purpose": "installation and split-isolation smoke test; not scientific evidence",
        "source_ids": list(split.source_train),
        "vectors": {name: list(vector.shape) for name, vector in vectors.items()},
        "vector_registry_manifest_sha256": sha256_file(registry / "manifest.json"),
    }
    (args.out / "SMOKE_OK.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
