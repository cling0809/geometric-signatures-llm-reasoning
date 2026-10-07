#!/usr/bin/env bash
set -euo pipefail

cd /root/AI/geoprobe
source /root/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe
export GEOPROBE_RUNS=/root/AI/runs

configs=(
  configs/2026-05-20_robust2048-qwen-base-gsm8k-100.yaml
  configs/2026-05-20_robust2048-qwen-instruct-gsm8k-100.yaml
  configs/2026-05-20_robust2048-qwen-math-gsm8k-100.yaml
  configs/2026-05-20_robust2048-r1-distill-gsm8k-100.yaml
)

for cfg in "${configs[@]}"; do
  python scripts/extract_trajectories.py --config "$cfg"
done

python scripts/phase3_signature_matrix.py \
  --runs \
    /root/AI/runs/2026-05-20_robust2048-qwen-base-gsm8k-100 \
    /root/AI/runs/2026-05-20_robust2048-qwen-instruct-gsm8k-100 \
    /root/AI/runs/2026-05-20_robust2048-qwen-math-gsm8k-100 \
    /root/AI/runs/2026-05-20_robust2048-r1-distill-gsm8k-100 \
  --labels Base Instruct Math-Instruct R1-Distill \
  --out /root/AI/runs/2026-05-20_phase3-robust2048-distance
