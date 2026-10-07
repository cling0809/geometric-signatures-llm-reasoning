# Anonymous per-problem audit tables

These tables expose the problem-level correctness and behavior fields needed to
recompute the paper's paired analyses without redistributing benchmark questions,
gold answers, model predictions, or generated text.

## Files

| Suite | CSV | Rows | Comparisons |
|---|---|---:|---:|
| Locked GSM8K | `locked_gsm8k.csv` | 1,600 | 8 methods, each paired with its baseline on 100 IDs |
| OOD MATH-500 | `ood_math500.csv` | 8,000 | 8 methods, each paired with its baseline on 500 IDs |
| OOD SVAMP | `ood_svamp.csv` | 16,000 | 8 methods, each paired with its baseline on 1,000 IDs |
| Label efficiency | `label_efficiency.csv` | 1,200 | 6 label budgets, each paired with its baseline on 100 IDs |

Each CSV contains `suite`, `comparison`, `arm`, `sample_id`, `correct`, the
selected layer/strength, and length, stopping, truncation, repetition, diversity,
and answer-marker telemetry.  It deliberately excludes prompt/problem text,
gold answers, predictions, generated text, and generation seeds.

Each sibling `*_manifest.json` verifies every source `per_sample.parquet` against
the SHA-256 recorded by the corresponding frozen report.  It also records the
released CSV hash, row count, columns, and privacy boundary.  Regenerate a table
with:

```bash
PYTHONPATH=src python scripts/revision_export_anonymous_per_problem.py \
  --suite locked_gsm8k \
  --suite-root /path/to/formal/locked-gsm8k-qwen-instruct-v3 \
  --report-manifest revision/evidence/locked-gsm8k-qwen-instruct-v3/locked_report_manifest.json \
  --out-csv revision/evidence/per-problem-audit-v1/locked_gsm8k.csv \
  --out-manifest revision/evidence/per-problem-audit-v1/locked_gsm8k_manifest.json
```

## Minimal paired recomputation

```python
import pandas as pd

df = pd.read_csv("locked_gsm8k.csv")
one = df[df["comparison"] == "crosssteer_source"]
wide = one.pivot(index="sample_id", columns="arm", values="correct")
method = wide["crosssteer_source"].astype(bool)
baseline = wide["baseline"].astype(bool)
print("baseline accuracy", baseline.mean())
print("method accuracy", method.mean())
print("repairs", ((~baseline) & method).sum())
print("breaks", (baseline & (~method)).sum())
```

The full report-building scripts reproduce bootstrap intervals, exact paired
tests, Holm correction, and aggregate behavior summaries.
