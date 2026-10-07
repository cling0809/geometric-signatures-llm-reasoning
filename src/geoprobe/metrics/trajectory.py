"""Per-layer trajectory geometry: path length + 3-point discrete curvature.

A trajectory is a tensor of shape [T, L, H] where:
  T = number of generated tokens
  L = number of layers (we include the embedding layer at index 0)
  H = hidden dimension

All computations are performed in float32 even if input is bf16, because the
norms / dot products underflow badly in bf16 for high-dim vectors.

Conventions:
  - `trajectory_length(hs)` returns a tensor of shape [L]: total path length
    per layer = sum_t ||h_{t+1} - h_t||_2.
  - `per_step_curvature(hs)` returns [T-2, L]: the discrete turning angle (rad)
    at each interior step, per layer. 0 = straight line, pi = U-turn.
  - `mean_curvature(hs)` returns [L]: mean of per_step_curvature over t.
  - For T < 2 we return zeros (length) or empty (curvature).
"""

from __future__ import annotations

import torch


def _as_float32(hs: torch.Tensor) -> torch.Tensor:
    if hs.dtype != torch.float32:
        return hs.float()
    return hs


def trajectory_length(hs: torch.Tensor) -> torch.Tensor:
    """[T, L, H] -> [L] total path length per layer."""
    if hs.shape[0] < 2:
        return torch.zeros(hs.shape[1], dtype=torch.float32, device=hs.device)
    x = _as_float32(hs)
    diffs = x[1:] - x[:-1]                # [T-1, L, H]
    seg = diffs.norm(dim=-1)              # [T-1, L]
    return seg.sum(dim=0)                 # [L]


def per_step_curvature(hs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """[T, L, H] -> [T-2, L] discrete turning angle (radians) per layer.

    Angle is between consecutive *displacement* vectors (h_t - h_{t-1}) and
    (h_{t+1} - h_t). Returns an empty tensor [0, L] when T < 3.
    """
    L = hs.shape[1]
    if hs.shape[0] < 3:
        return torch.zeros((0, L), dtype=torch.float32, device=hs.device)
    x = _as_float32(hs)
    v = x[1:] - x[:-1]                    # [T-1, L, H]
    v_prev = v[:-1]                        # [T-2, L, H]
    v_next = v[1:]                         # [T-2, L, H]
    num = (v_prev * v_next).sum(dim=-1)    # [T-2, L]
    den = v_prev.norm(dim=-1) * v_next.norm(dim=-1) + eps
    # Clamp to exact [-1, 1] — pytorch arccos handles the endpoints fine
    # and an artificial narrower clamp would inject ~0.001 rad on straight lines.
    cos = (num / den).clamp(-1.0, 1.0)
    return torch.arccos(cos)               # [T-2, L]


def mean_curvature(hs: torch.Tensor) -> torch.Tensor:
    """[T, L, H] -> [L] mean turning angle (rad) over interior steps."""
    psc = per_step_curvature(hs)
    if psc.shape[0] == 0:
        return torch.zeros(hs.shape[1], dtype=torch.float32, device=hs.device)
    return psc.mean(dim=0)


# ---------- Length-residualized / per-step metrics (Phase 2 P2) ----------
#
# Phase 1 showed `trajectory_length` is essentially n_gen_tokens * mean_step_norm,
# so raw length is a length proxy. These per-step / dispersion summaries try to
# capture geometry without the length confound.


def mean_step_norm(hs: torch.Tensor) -> torch.Tensor:
    """[L] average displacement per step = trajectory_length / (T-1).

    Length-invariant: doesn't grow with T.
    """
    T = hs.shape[0]
    if T < 2:
        return torch.zeros(hs.shape[1], dtype=torch.float32, device=hs.device)
    return trajectory_length(hs) / float(T - 1)


def curvature_stats(hs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-layer summary statistics of per_step_curvature.

    Returns dict with keys: mean, var, max, p90, p99 — each [L].

    Why beyond the mean: a sign-flip in mean_curvature was found in Phase 1.
    But "errors are straighter on average" could be driven either by uniform
    flattening, or by absence of a few sharp turns (peaks). Variance / max /
    high quantiles disentangle these.
    """
    L = hs.shape[1]
    psc = per_step_curvature(hs)
    if psc.shape[0] == 0:
        z = torch.zeros(L, dtype=torch.float32, device=hs.device)
        return {"mean": z, "var": z, "max": z, "p90": z, "p99": z}
    return {
        "mean": psc.mean(dim=0),
        "var": psc.var(dim=0, unbiased=False),
        "max": psc.max(dim=0).values,
        "p90": psc.quantile(0.90, dim=0),
        "p99": psc.quantile(0.99, dim=0),
    }


def trajectory_metrics(hs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute all metrics in one pass. Returns dict of [L] tensors.

    Includes:
      - trajectory_length              (length-confounded)
      - mean_step_norm                 (length-invariant: avg |displacement|)
      - mean_curvature, curvature_var, curvature_max, curvature_p90, curvature_p99
        (turning-angle distribution per layer; all length-invariant)
    """
    out: dict[str, torch.Tensor] = {
        "trajectory_length": trajectory_length(hs),
        "mean_step_norm": mean_step_norm(hs),
    }
    for stat_name, vec in curvature_stats(hs).items():
        out[f"curvature_{stat_name}"] = vec
    return out
