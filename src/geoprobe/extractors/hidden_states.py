"""Per-token hidden-state extraction during generation.

We use `model.generate(output_hidden_states=True, output_scores=True,
return_dict_in_generate=True)`.

`outputs.hidden_states` is a tuple of length T (one per generated token).
Each entry is itself a tuple of (L+1) tensors (embedding + each transformer layer).
Shapes:
  - step 0:    [B, prompt_len, hidden]   (forward pass over prompt)
  - step t>0:  [B, 1, hidden]              (incremental decode)

For each step t we take the *last* position. The resulting trajectory has
shape [T, L+1, hidden].

`outputs.scores` is a tuple of length T, each [B, vocab_size]; the logits at
each generation step. We compute the per-step log-prob of the chosen token
and sum to get the sequence log-prob.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import GenerationConfig


@dataclass
class Trajectory:
    sample_id: int            # GSM8K problem index
    sample_idx: int           # which sample for this problem (0 for greedy / first)
    hidden_states: torch.Tensor  # [T, L+1, hidden], bf16 on CPU
    generated_token_ids: torch.Tensor  # [T], int64 on CPU
    generated_text: str
    prompt: str
    prompt_len: int
    sequence_logprob: float   # sum of log p(token | context) over generated tokens
    model_id: str
    dtype: str


@torch.inference_mode()
def extract_trajectory(
    model,
    tokenizer,
    prompt: str,
    sample_id: int,
    *,
    sample_idx: int = 0,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    model_id: str = "",
    dtype: str = "bfloat16",
) -> Trajectory:
    """Generate (greedy or sampled) and return the per-token hidden-state trajectory.

    Args:
        do_sample: if False, greedy. If True, use temperature + top_p sampling.
        sample_idx: passthrough into the returned Trajectory; useful when
            this function is called N times per question.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(inputs["input_ids"].shape[1])

    # Respect the complete model generation configuration.  In particular,
    # Qwen2.5-Instruct exposes multiple EOS ids while tokenizer.eos_token_id
    # contains only one of them.
    eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_ids,
    )

    outputs = model.generate(
        **inputs,
        generation_config=gen_cfg,
        output_hidden_states=True,
        output_scores=True,
        return_dict_in_generate=True,
    )

    seq = outputs.sequences[0]
    gen_token_ids = seq[prompt_len:].detach().cpu()
    generated_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

    # Sequence log-prob: sum log softmax(scores[t])[gen_token_ids[t]]
    # outputs.scores is a tuple of T tensors [B, vocab_size]
    seq_lp = 0.0
    for t, logits in enumerate(outputs.scores):
        lp = F.log_softmax(logits[0], dim=-1)
        seq_lp += float(lp[gen_token_ids[t]].item())

    # Hidden states: per-step last position, stacked
    per_step = []
    for step_hs in outputs.hidden_states:
        layer_vecs = torch.stack([h[0, -1, :].detach().to("cpu") for h in step_hs], dim=0)
        per_step.append(layer_vecs)
    hidden = torch.stack(per_step, dim=0).to(torch.bfloat16)

    return Trajectory(
        sample_id=sample_id,
        sample_idx=sample_idx,
        hidden_states=hidden,
        generated_token_ids=gen_token_ids.to(torch.int64),
        generated_text=generated_text,
        prompt=prompt,
        prompt_len=prompt_len,
        sequence_logprob=seq_lp,
        model_id=model_id,
        dtype=dtype,
    )


def save_trajectory(traj: Trajectory, path) -> None:
    torch.save(
        {
            "sample_id": traj.sample_id,
            "sample_idx": traj.sample_idx,
            "hidden_states": traj.hidden_states,
            "generated_token_ids": traj.generated_token_ids,
            "generated_text": traj.generated_text,
            "prompt": traj.prompt,
            "prompt_len": traj.prompt_len,
            "sequence_logprob": traj.sequence_logprob,
            "model_id": traj.model_id,
            "dtype": traj.dtype,
        },
        path,
    )


def load_trajectory(path) -> Trajectory:
    d = torch.load(path, map_location="cpu", weights_only=False)
    # Back-compat: older runs lack sample_idx / sequence_logprob
    d.setdefault("sample_idx", 0)
    d.setdefault("sequence_logprob", 0.0)
    return Trajectory(**d)
