"""Model + tokenizer loading. ModelScope is the default source (see ADR-002)."""

from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ModelSource = Literal["modelscope", "huggingface"]

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model_and_tokenizer(
    model_id: str,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    source: ModelSource = "modelscope",
):
    """Load a causal LM + tokenizer. Returns (model, tokenizer).

    For source="modelscope", we pre-fetch the snapshot via modelscope's API
    and then load it from the local path using transformers — this keeps the
    rest of the codebase HF-API-pure.
    """
    if source == "modelscope":
        from modelscope import snapshot_download  # type: ignore

        local_path = snapshot_download(model_id)
        load_target = local_path
    elif source == "huggingface":
        load_target = model_id
    else:
        raise ValueError(f"unknown source: {source}")

    torch_dtype = _DTYPE_MAP[dtype]
    tokenizer = AutoTokenizer.from_pretrained(load_target, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        load_target,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer
