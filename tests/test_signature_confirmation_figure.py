from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.revision_plot_signature_confirmation import (
    load_partition_cells,
    render_all_cell_heatmaps,
    summarize_cells,
)


def _write_audit(path: Path) -> Path:
    path.mkdir()
    (path / "DONE").write_text("\n")
    (path / "manifest.json").write_text(
        json.dumps(
            {"protocol": "tacl-11241-diagnostic-generalization-v1", "minimum_per_class": 10}
        )
    )
    (path / "input_provenance_audit.json").write_text("[]\n")
    rows = []
    for partition in ("A", "B", "C"):
        for model in ("Base", "Math"):
            for metric in ("shape", "turn"):
                for depth in range(3):
                    rows.append(
                        {
                            "partition": partition,
                            "model": model,
                            "metric": metric,
                            "relative_depth_bin": depth,
                            "auc_incorrect_vs_correct": 0.45 + 0.01 * depth,
                            "n_incorrect": 10,
                            "n_correct": 11,
                        }
                    )
    pd.DataFrame(rows).to_csv(path / "partition_signature_cells.csv", index=False)
    return path


def test_signature_figure_uses_all_partitions_and_fixed_matrix(tmp_path: Path):
    audit = _write_audit(tmp_path / "audit")
    cells = load_partition_cells(audit)
    mean, sd = summarize_cells(cells)
    assert len(mean) == 12
    assert sd["partition_sd_auc"].eq(0.0).all()
    output = tmp_path / "figure.png"
    render_all_cell_heatmaps(mean, sd, output, title="fixture")
    assert output.is_file()
    assert output.stat().st_size > 0


def test_signature_figure_refuses_incomplete_partition_grid(tmp_path: Path):
    audit = _write_audit(tmp_path / "audit")
    cells = pd.read_csv(audit / "partition_signature_cells.csv")
    cells = cells.iloc[:-1]
    cells.to_csv(audit / "partition_signature_cells.csv", index=False)
    with pytest.raises(ValueError, match="incomplete metric-depth grid"):
        load_partition_cells(audit)
