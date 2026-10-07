"""Lightweight YAML config loader. Validates only the universal required keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REQUIRED_TOP_KEYS = ("exp_id", "model", "dataset")


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    missing = [k for k in _REQUIRED_TOP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config missing required keys {missing}: {path}")
    return cfg


def dump_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Write a config snapshot to disk (used to freeze the exact config used in a run)."""
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
