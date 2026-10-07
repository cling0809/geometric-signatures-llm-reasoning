#!/usr/bin/env bash
set -euo pipefail

cd /root/AI/geoprobe
source /root/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe
export GEOPROBE_RUNS=/root/AI/runs

python scripts/extract_metrics_streaming.py \
  --config configs/2026-05-21_e5r-llama32-1b-base-gsm8k-100-2048-metriconly.yaml \
  --flush-every 25

python scripts/extract_metrics_streaming.py \
  --config configs/2026-05-21_e5r-llama32-1b-instruct-gsm8k-100-2048-metriconly.yaml \
  --flush-every 25

python scripts/phase3_signature_matrix.py \
  --runs \
    /root/AI/runs/2026-05-21_e5r-llama32-1b-base-gsm8k-100-2048-metriconly \
    /root/AI/runs/2026-05-21_e5r-llama32-1b-instruct-gsm8k-100-2048-metriconly \
  --labels Llama-Base-2048 Llama-Instruct-2048 \
  --out /root/AI/runs/2026-05-21_e5r-llama32-signature-gsm8k-100-2048
