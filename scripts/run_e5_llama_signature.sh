#!/usr/bin/env bash
set -euo pipefail

cd /root/AI/geoprobe
source /root/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe
export GEOPROBE_RUNS=/root/AI/runs

python scripts/extract_metrics_streaming.py \
  --config configs/2026-05-21_e5-llama32-1b-base-gsm8k-100-metriconly.yaml \
  --flush-every 25

python scripts/extract_metrics_streaming.py \
  --config configs/2026-05-21_e5-llama32-1b-instruct-gsm8k-100-metriconly.yaml \
  --flush-every 25

python scripts/phase3_signature_matrix.py \
  --runs \
    /root/AI/runs/2026-05-21_e5-llama32-1b-base-gsm8k-100-metriconly \
    /root/AI/runs/2026-05-21_e5-llama32-1b-instruct-gsm8k-100-metriconly \
  --labels Llama-Base Llama-Instruct \
  --out /root/AI/runs/2026-05-21_e5-llama32-signature-gsm8k-100
