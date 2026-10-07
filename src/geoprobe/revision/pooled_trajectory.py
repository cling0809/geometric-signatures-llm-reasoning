"""Memory-bounded extraction of generated-state trajectory means.

CrossSteer uses a per-problem mean over generated-state hidden vectors.  For
long R1 completions, retaining the full ``[T, L, H]`` tensor is unnecessary and
can be infeasible.  This module reproduces that mean with a generation pass and
KV-cached teacher-forced replay while retaining only ``[L, H]`` accumulators.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import GenerationConfig

from geoprobe.extractors.hidden_states import Trajectory


@dataclass(frozen=True)
class PooledReplayMetadata:
    """Auditable details of one compact generated-state replay."""

    state_count: int
    chunk_size: int
    state_position_rule: str = "prompt_final_plus_generated_prefix"
    aggregation: str = "mean_over_generated_state_positions"


def _stack_last_states(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Return `[layers, hidden]` last-position states as float32."""
    return torch.stack([state[0, -1, :].float() for state in hidden_states], dim=0)


def _stack_chunk_sums(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Return `[layers, hidden]` sums over all sequence positions in one replay chunk."""
    return torch.stack([state[0].float().sum(dim=0) for state in hidden_states], dim=0)


@torch.inference_mode()
def replay_generated_state_mean(
    model,
    *,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor | None,
    generated_token_ids: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, PooledReplayMetadata]:
    """Replay one completion and return the exact generated-state mean.

    The legacy extractor records one state per generated token: the prompt-final
    state for the first generated token, then the state after each generated
    prefix token except the final completion token.  This function uses the same
    positions and therefore matches ``legacy_hidden_states.mean(dim=0)`` up to
    numerical replay precision.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != 1:
        raise ValueError("prompt_input_ids must have shape [1, prompt_tokens]")
    generated = generated_token_ids.to(prompt_input_ids.device, dtype=torch.long).flatten()
    if generated.numel() == 0:
        raise ValueError("cannot pool an empty generation")

    prompt_kwargs: dict[str, object] = {
        "input_ids": prompt_input_ids,
        "use_cache": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if prompt_attention_mask is not None:
        prompt_kwargs["attention_mask"] = prompt_attention_mask
    prompt_output = model(**prompt_kwargs)
    total = _stack_last_states(prompt_output.hidden_states)
    state_count = 1
    cache = prompt_output.past_key_values

    # The final generated token has no corresponding legacy generation-step
    # state: it is sampled from the final state already counted before it.
    replay_tokens = generated[:-1]
    for start in range(0, replay_tokens.numel(), chunk_size):
        chunk = replay_tokens[start : start + chunk_size].unsqueeze(0)
        output = model(
            input_ids=chunk,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        total = total + _stack_chunk_sums(output.hidden_states)
        state_count += int(chunk.shape[1])
        cache = output.past_key_values

    if state_count != int(generated.numel()):
        raise AssertionError(f"state count {state_count} != generated token count {generated.numel()}")
    return total / state_count, PooledReplayMetadata(state_count=state_count, chunk_size=chunk_size)


def _generate_with_local_seed(
    model,
    generation_kwargs: dict[str, object],
    *,
    generation_seed: int | None,
    device: torch.device,
):
    """Call ``generate`` with a reproducible, local RNG state.

    Some supported Transformers versions reject a ``generator=`` model kwarg.
    Sampling therefore uses ``fork_rng`` rather than forwarding that unsupported
    kwarg.  The caller's CPU/CUDA RNG state is restored afterwards, so the
    per-problem seed rule is auditable and independent of extraction order.
    """
    if generation_seed is None:
        return model.generate(**generation_kwargs)

    cuda_devices: list[int] = []
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("CUDA generation requires an explicit device index")
        cuda_devices = [device.index]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(generation_seed)
        return model.generate(**generation_kwargs)


@torch.inference_mode()
def generate_pooled_trajectory(
    model,
    tokenizer,
    *,
    prompt: str,
    sample_id: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    generation_seed: int | None,
    replay_chunk_size: int,
    model_id: str,
    dtype: str,
) -> tuple[Trajectory, PooledReplayMetadata]:
    """Generate once, replay compactly, and return a 2D pooled ``Trajectory``."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(inputs["input_ids"].shape[1])
    # Preserve all model-declared terminators.  Qwen2.5-Instruct has two EOS
    # ids; passing tokenizer.eos_token_id alone would silently drop one.
    eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_ids,
    )
    generation_kwargs: dict[str, object] = {
        **inputs,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_hidden_states": False,
        "output_scores": False,
    }
    outputs = _generate_with_local_seed(
        model,
        generation_kwargs,
        generation_seed=generation_seed if do_sample else None,
        device=device,
    )
    sequence = outputs.sequences[0]
    generated = sequence[prompt_len:].detach().cpu().to(torch.int64)
    pooled, replay = replay_generated_state_mean(
        model,
        prompt_input_ids=inputs["input_ids"],
        prompt_attention_mask=inputs.get("attention_mask"),
        generated_token_ids=generated,
        chunk_size=replay_chunk_size,
    )
    trajectory = Trajectory(
        sample_id=sample_id,
        sample_idx=0,
        hidden_states=pooled.cpu().to(torch.bfloat16),
        generated_token_ids=generated,
        generated_text=tokenizer.decode(generated, skip_special_tokens=True),
        prompt=prompt,
        prompt_len=prompt_len,
        sequence_logprob=float("nan"),
        model_id=model_id,
        dtype=dtype,
    )
    return trajectory, replay
