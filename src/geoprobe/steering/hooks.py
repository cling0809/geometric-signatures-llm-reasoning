"""Forward-hook based activation steering for HuggingFace causal LMs.

The hook supports a declared schedule and either all-position or last-position
application.  Formal autoregressive steering runs should use ``position_mode=
"decode_last"`` so a prompt prefill forward pass is not perturbed at all; legacy
all-position behavior remains the default for backward compatibility and is
never reused as formal revision evidence without an explicit manifest.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from geoprobe.steering.schedules import ConstantSchedule, SteeringSchedule

PositionMode = Literal["all", "last", "decode_last"]
InjectionMode = Literal["absolute", "relative_hidden_rms"]


def _get_layer_module(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise ValueError(f"could not find transformer layers on {type(model).__name__}")


class SteeringHook:
    """Add a scheduled steering vector to a selected transformer-layer output.

    ``layer_idx`` follows trajectory indexing: 0 is embedding output, 1 is the
    first transformer block output.  ``position_mode='decode_last'`` skips a
    multi-token prompt-prefill call, then changes only the final position of each
    decode forward.  This is the formal-revision mode.  ``last`` includes the
    prompt-final state and remains available for controlled ablations.
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        vector: torch.Tensor,
        alpha: float = 1.0,
        *,
        schedule: SteeringSchedule | Callable[[int], float] | None = None,
        position_mode: PositionMode = "all",
        injection_mode: InjectionMode = "absolute",
    ):
        if layer_idx < 1:
            raise ValueError("layer_idx must be >= 1 (0 is the embedding)")
        if position_mode not in ("all", "last", "decode_last"):
            raise ValueError("position_mode must be 'all', 'last', or 'decode_last'")
        if injection_mode not in ("absolute", "relative_hidden_rms"):
            raise ValueError("injection_mode must be 'absolute' or 'relative_hidden_rms'")
        self.model = model
        self.layer_idx = layer_idx
        self.alpha = float(alpha)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.vector = vector.to(device=device, dtype=dtype)
        self.schedule = schedule or ConstantSchedule()
        self.position_mode = position_mode
        self.injection_mode = injection_mode
        self._handle = None
        self._generation_step = 0

    @property
    def generation_step(self) -> int:
        """Number of hooked forwards applied during this context-manager use."""
        return self._generation_step

    def _addition(self, hidden: torch.Tensor, scale: float) -> torch.Tensor:
        """Compute the perturbation under the declared absolute/relative budget."""
        delta = scale * self.vector
        if self.injection_mode == "absolute":
            return delta
        # Vector RMS is normalized to one before alpha selection.  This mode
        # rescales it so the injected coordinate RMS is a fixed fraction of the
        # residual coordinate RMS at the actually changed position.
        rms = hidden.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        return rms.to(dtype=hidden.dtype) * delta

    def _steer_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim < 2:
            raise ValueError("steering requires a sequence dimension")
        # With HF generation, the initial prompt prefill has sequence length >1
        # and later cached decode forwards have length 1.  Skipping prefill is
        # necessary for a fair post-prompt steering protocol.
        if self.position_mode == "decode_last" and hidden.shape[-2] > 1:
            return hidden
        scale = self.alpha * float(self.schedule(self._generation_step))
        self._generation_step += 1
        if scale == 0.0:
            return hidden
        if self.position_mode == "all":
            return hidden + self._addition(hidden, scale)
        steered = hidden.clone()
        position = steered[..., -1:, :]
        addition = self._addition(position, scale)
        if addition.ndim == 1:
            steered[..., -1, :] = steered[..., -1, :] + addition
        else:
            steered[..., -1, :] = steered[..., -1, :] + addition[..., 0, :]
        return steered

    def __enter__(self):
        block = _get_layer_module(self.model, self.layer_idx - 1)

        def hook(module, inputs, output):
            del module, inputs
            if isinstance(output, tuple):
                hidden = output[0]
                return (self._steer_hidden(hidden), *output[1:])
            return self._steer_hidden(output)

        self._handle = block.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
