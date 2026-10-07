"""Compute per-layer metrics for every trajectory in a run directory.

Reads:
  <run>/trajectories/sample_*.pt   (geoprobe.extractors.Trajectory)
  <run>/labels.parquet             (sample_id, gold, pred, correct, n_gen_tokens)

Writes:
  <run>/metrics.parquet            long format: sample_id, metric, layer, value

`metrics.parquet` is intentionally long-form so adding new metrics later is a
pure append; no schema migration. Downstream analysis pivots as needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geoprobe.extractors import load_trajectory
from geoprobe.metrics import trajectory_metrics


def compute_run_metrics(run_dir: str | Path) -> pd.DataFrame:
    run = Path(run_dir)
    traj_dir = run / "trajectories"
    files = sorted(traj_dir.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"no sample_*.pt in {traj_dir}")

    rows: list[dict] = []
    for f in files:
        t = load_trajectory(f)
        metrics = trajectory_metrics(t.hidden_states)
        for name, vec in metrics.items():
            for layer_idx, val in enumerate(vec.tolist()):
                rows.append(
                    {
                        "sample_id": int(t.sample_id),
                        "sample_idx": int(t.sample_idx),
                        "metric": name,
                        "layer": int(layer_idx),
                        "value": float(val),
                    }
                )
    df = pd.DataFrame(rows)
    return df


def write_run_metrics(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    df = compute_run_metrics(run)
    out = run / "metrics.parquet"
    df.to_parquet(out, index=False)
    return out
