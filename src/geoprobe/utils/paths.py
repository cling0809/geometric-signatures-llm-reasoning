"""Path resolution. All code reads paths via these helpers — see ADR-002."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(os.environ.get("GEOPROBE_HOME", Path.home() / "AI" / "geoprobe"))


def cache_dir() -> Path:
    return Path(os.environ.get("GEOPROBE_CACHE", Path.home() / "AI" / "cache"))


def runs_root() -> Path:
    return Path(os.environ.get("GEOPROBE_RUNS", Path.home() / "AI" / "runs"))


def exp_dir(exp_id: str, create: bool = True) -> Path:
    """Return ~/AI/runs/<exp_id>/, creating the standard subdir layout if requested."""
    base = runs_root() / exp_id
    if create:
        for sub in ("logs", "trajectories", "metrics", "plots"):
            (base / sub).mkdir(parents=True, exist_ok=True)
    return base
