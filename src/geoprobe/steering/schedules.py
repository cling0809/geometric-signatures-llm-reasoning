"""Predeclared steering-strength schedules for long-context stability studies.

The schedules are intentionally small and explicit.  A formal experiment must
select a schedule on validation data and freeze it before the locked test.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Protocol


class SteeringSchedule(Protocol):
    """Return a multiplicative scale for a zero-indexed generation step."""

    def __call__(self, generation_step: int) -> float: ...


@dataclass(frozen=True)
class ConstantSchedule:
    """Apply the same steering scale at every generation step."""

    def __call__(self, generation_step: int) -> float:
        if generation_step < 0:
            raise ValueError("generation_step must be non-negative")
        return 1.0


@dataclass(frozen=True)
class PrefixSchedule:
    """Apply steering only during the first ``n_steps`` generated tokens."""

    n_steps: int

    def __post_init__(self) -> None:
        if self.n_steps < 0:
            raise ValueError("n_steps must be non-negative")

    def __call__(self, generation_step: int) -> float:
        if generation_step < 0:
            raise ValueError("generation_step must be non-negative")
        return 1.0 if generation_step < self.n_steps else 0.0


@dataclass(frozen=True)
class LinearDecaySchedule:
    """Linearly decay from 1 at step 0 to 0 at ``decay_steps``."""

    decay_steps: int

    def __post_init__(self) -> None:
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")

    def __call__(self, generation_step: int) -> float:
        if generation_step < 0:
            raise ValueError("generation_step must be non-negative")
        return max(0.0, 1.0 - generation_step / self.decay_steps)


@dataclass(frozen=True)
class ExponentialDecaySchedule:
    """Exponentially decay with positive time constant ``tau``."""

    tau: float

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")

    def __call__(self, generation_step: int) -> float:
        if generation_step < 0:
            raise ValueError("generation_step must be non-negative")
        return exp(-generation_step / self.tau)
