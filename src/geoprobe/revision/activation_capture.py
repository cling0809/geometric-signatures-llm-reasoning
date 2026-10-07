"""Prompt-level activation capture for fair CAA and ActAdd baselines.

Historical steering scripts saved hidden states only while generating a
completion.  That made a first-generated-token proxy look like a prompt-final
activation, which is not an auditable CAA/ActAdd implementation.  These
helpers run an explicit forward pass and extract the final non-padding prompt
token at every residual-stream depth.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


@torch.inference_mode()
def capture_prompt_final_states(model, tokenizer, prompt: str) -> torch.Tensor:
    """Return prompt-final residual states with shape ``[layers, hidden]``.

    The function captures a genuine prompt forward pass (not a generated-token
    proxy).  It does not apply any steering hook, so contrastive examples can
    be recorded before vector construction.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    elif isinstance(inputs, dict):
        inputs = {key: value.to(device) for key, value in inputs.items()}
    else:
        raise TypeError("tokenizer output must be a mapping or support .to(device)")
    outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) == 0:
        raise ValueError("model did not return hidden states")
    attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else inputs["attention_mask"]
    if attention_mask is None:
        position = -1
    else:
        if attention_mask.shape[0] != 1:
            raise ValueError("capture_prompt_final_states expects one prompt at a time")
        position = int(attention_mask[0].sum().item()) - 1
    if position < 0:
        raise ValueError("prompt must contain at least one non-padding token")
    states = torch.stack([state[0, position, :].detach().cpu() for state in hidden_states], dim=0)
    return states.float()


def capture_prompt_pairs(
    model,
    tokenizer,
    positive_prompts: Iterable[str],
    negative_prompts: Iterable[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture aligned positive/negative prompt-final states for CAA/ActAdd."""
    positive = list(positive_prompts)
    negative = list(negative_prompts)
    if len(positive) != len(negative):
        raise ValueError("positive and negative prompt lists must have equal length")
    if not positive:
        raise ValueError("at least one contrast pair is required")
    positive_states = [capture_prompt_final_states(model, tokenizer, prompt) for prompt in positive]
    negative_states = [capture_prompt_final_states(model, tokenizer, prompt) for prompt in negative]
    if any(state.shape != positive_states[0].shape for state in positive_states + negative_states):
        raise ValueError("all contrast prompts must yield the same [layers, hidden] shape")
    return torch.stack(positive_states), torch.stack(negative_states)
