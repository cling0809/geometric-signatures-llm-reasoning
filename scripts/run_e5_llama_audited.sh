#!/usr/bin/env bash
set -euo pipefail

cd /root/AI/geoprobe
source /root/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe
export GEOPROBE_RUNS=/root/AI/runs

configs=(
  configs/2026-05-21_e5a-llama32-1b-base-gsm8k-100-512-audited.yaml
  configs/2026-05-21_e5a-llama32-1b-instruct-gsm8k-100-512-audited.yaml
  configs/2026-05-21_e5a-llama32-1b-base-gsm8k-100-2048-audited.yaml
  configs/2026-05-21_e5a-llama32-1b-instruct-gsm8k-100-2048-audited.yaml
)

for cfg in "${configs[@]}"; do
  python scripts/extract_metrics_streaming.py --config "$cfg" --flush-every 25
done

python scripts/phase3_signature_matrix.py \
  --runs \
    /root/AI/runs/2026-05-21_e5a-llama32-1b-base-gsm8k-100-512-audited \
    /root/AI/runs/2026-05-21_e5a-llama32-1b-instruct-gsm8k-100-512-audited \
  --labels Llama-Base-512 Llama-Instruct-512 \
  --out /root/AI/runs/2026-05-21_e5a-llama32-signature-gsm8k-100-512-audited

python scripts/phase3_signature_matrix.py \
  --runs \
    /root/AI/runs/2026-05-21_e5a-llama32-1b-base-gsm8k-100-2048-audited \
    /root/AI/runs/2026-05-21_e5a-llama32-1b-instruct-gsm8k-100-2048-audited \
  --labels Llama-Base-2048 Llama-Instruct-2048 \
  --out /root/AI/runs/2026-05-21_e5a-llama32-signature-gsm8k-100-2048-audited
