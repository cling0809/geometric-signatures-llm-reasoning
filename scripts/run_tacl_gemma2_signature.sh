#!/usr/bin/env bash
set -euo pipefail

cd /root/AI/geoprobe
source /root/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe
export GEOPROBE_RUNS=/root/AI/runs

LOG_DIR=/root/AI/runs/2026-05-27_tacl-gemma2-signature-gsm8k-100/logs
mkdir -p "$LOG_DIR"

run_metric() {
  local config="$1"
  local log_name="$2"
  python scripts/extract_metrics_streaming.py \
    --config "$config" \
    --flush-every 10 \
    > "$LOG_DIR/$log_name" 2>&1
}

run_metric configs/2026-05-27_tacl-gemma2-2b-base-gsm8k-100.yaml gemma2_base.log &
pid_base=$!
run_metric configs/2026-05-27_tacl-gemma2-2b-it-gsm8k-100.yaml gemma2_it.log &
pid_it=$!

wait "$pid_base"
wait "$pid_it"

python scripts/phase3_signature_matrix.py \
  --runs \
    /root/AI/runs/2026-05-27_tacl-gemma2-2b-base-gsm8k-100 \
    /root/AI/runs/2026-05-27_tacl-gemma2-2b-it-gsm8k-100 \
  --labels Gemma2-Base Gemma2-IT \
  --out /root/AI/runs/2026-05-27_tacl-gemma2-signature-gsm8k-100 \
  > "$LOG_DIR/aggregate.log" 2>&1

date > /root/AI/runs/2026-05-27_tacl-gemma2-signature-gsm8k-100/DONE
