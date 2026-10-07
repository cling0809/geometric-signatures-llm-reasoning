#!/usr/bin/env python3
"""Build faithful prompt-final CAA/ActAdd vectors from same-question source contrasts."""

from __future__ import annotations

import argparse
from pathlib import Path

from geoprobe.models import load_model_and_tokenizer
from geoprobe.revision.prompt_contrast_registry import (
    build_prompt_contrast_vectors,
    save_prompt_contrast_registry,
)
from geoprobe.revision.protocol import RevisionSplit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-source", choices=["modelscope", "huggingface"], default="modelscope")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sparse-keep-fraction", type=float, default=0.10)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--source-count", type=int, default=100)
    args = parser.parse_args()
    if args.source_start != 0 or args.source_count != 100:
        raise SystemExit("formal CAA/ActAdd registry is frozen to GSM8K source IDs 0--99")
    if args.sparse_keep_fraction != 0.10:
        raise SystemExit("the first fair sparse ablation is frozen to keep_fraction=0.10")

    split = RevisionSplit.gsm8k_formal()
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model,
        source=args.model_source,
        device=args.device,
        dtype=args.dtype,
    )
    vectors, metadata = build_prompt_contrast_vectors(
        model,
        tokenizer,
        target_run=args.target_run,
        source_ids=split.source_train,
        split=split,
        sparse_keep_fraction=args.sparse_keep_fraction,
    )
    metadata["target_model"] = args.model
    metadata["target_model_source"] = args.model_source
    metadata["target_model_dtype"] = args.dtype
    save_prompt_contrast_registry(vectors, metadata, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
