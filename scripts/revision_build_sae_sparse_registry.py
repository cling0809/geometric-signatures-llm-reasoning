#!/usr/bin/env python3
"""Build an auditable SAE-space sparse-activation steering registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from geoprobe.models import load_model_and_tokenizer
from geoprobe.revision.protocol import RevisionSplit
from geoprobe.revision.sae_sparse_steering import (
    SAETrainingConfig,
    build_sae_sparse_activation_vectors,
)
from geoprobe.revision.vector_registry import (
    load_calibration_audit_reference,
    require_completed_run,
    sha256_file,
)

_PROTOCOL = "tacl-11241-sae-sparse-activation-v1"



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--contrast-run", type=Path, required=True)
    parser.add_argument("--calibration-audit", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-source", default="huggingface")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    split = RevisionSplit.gsm8k_formal()
    # This CLI is deliberately frozen: changing an SAE width, sparsity penalty,
    # progress rule or seed after validation would be another unreported method
    # search.  Exploratory variants require a separately versioned protocol.
    config = SAETrainingConfig()
    calibration_run = require_completed_run(args.calibration_run)
    calibration_audit = None
    if args.calibration_audit is not None:
        calibration_audit = load_calibration_audit_reference(
            args.calibration_audit,
            run=calibration_run,
            role="SAE target",
        )
    vector_path = args.out / "sae_sparse_activation.pt"
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_vector = args.out / str(existing.get("vector", ""))
        expected = {
            "protocol": _PROTOCOL,
            "calibration_run": str(args.calibration_run.resolve()),
            "contrast_run": str(args.contrast_run.resolve()),
            "training": config.to_dict(),
        }
        if calibration_audit is not None:
            expected["calibration_audit"] = calibration_audit
        if (
            any(existing.get(key) != value for key, value in expected.items())
            or not existing_vector.is_file()
            or existing.get("vector_sha256") != sha256_file(existing_vector)
        ):
            raise SystemExit("refuse to overwrite a different SAE sparse activation registry")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model, source=args.model_source, dtype="bfloat16", device=args.device
    )
    vectors, metadata = build_sae_sparse_activation_vectors(
        model,
        tokenizer,
        calibration_run=args.calibration_run,
        contrast_run=args.contrast_run,
        source_ids=split.source_train,
        split=split,
        config=config,
        device=args.device,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    vector = vectors["sae_sparse_activation"]
    torch.save(vector.cpu(), vector_path)
    payload = {
        **metadata,
        "vector": vector_path.name,
        "vector_sha256": sha256_file(vector_path),
    }
    if calibration_audit is not None:
        payload["calibration_audit"] = calibration_audit
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
