"""Generation telemetry used by the TACL-11241 revision runners.

The formal steering comparison needs to distinguish an EOS-completed answer
from an answer that merely reached ``max_new_tokens``.  This small, pure helper
keeps that classification independent of model-specific ``generate`` return
objects and makes it auditable in per-problem artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GenerationTelemetry:
    """Token-level termination facts for one decoded completion."""

    generated_token_ids: torch.Tensor
    n_generated_tokens: int
    n_content_tokens: int
    stop_reason: str
    truncated: bool


def summarize_generated_tokens(
    generated_token_ids: torch.Tensor,
    *,
    eos_token_id: int | Iterable[int] | None,
    max_new_tokens: int,
) -> GenerationTelemetry:
    """Classify an already generated continuation without text heuristics.

    ``generate`` emits the EOS token in the returned sequence when it is the
    terminator.  A completion ending in EOS is therefore recorded as ``eos``
    even if it happens to use the final allowed position.  A non-EOS completion
    that consumes the full token budget is ``max_new_tokens`` / truncated.
    Any shorter non-EOS completion is retained as ``other`` so that callers do
    not overclaim an EOS cause that was not observed.

    ``eos_token_id`` may be a single id or an iterable of ids.  Multi-eos models
    (for example Qwen2.5-Instruct, whose generation config declares both
    ``<|im_end|>`` and ``<|endoftext|>``) must pass the full set; a single id
    would misclassify completions that terminate on the other EOS token.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if generated_token_ids.ndim != 1:
        raise ValueError("generated_token_ids must be one-dimensional")

    eos_set: set[int] | None = None
    if eos_token_id is not None:
        if isinstance(eos_token_id, int):
            eos_set = {int(eos_token_id)}
        else:
            eos_set = {int(value) for value in eos_token_id}

    ids = generated_token_ids.detach().to(device="cpu", dtype=torch.long).clone()
    n_generated = int(ids.numel())
    ends_with_eos = bool(n_generated > 0 and eos_set is not None and int(ids[-1].item()) in eos_set)
    n_content = n_generated - int(ends_with_eos)
    if ends_with_eos:
        reason = "eos"
        truncated = False
    elif n_generated >= max_new_tokens:
        reason = "max_new_tokens"
        truncated = True
    elif n_generated == 0:
        reason = "empty"
        truncated = False
    else:
        reason = "other"
        truncated = False
    return GenerationTelemetry(
        generated_token_ids=ids,
        n_generated_tokens=n_generated,
        n_content_tokens=n_content,
        stop_reason=reason,
        truncated=truncated,
    )
