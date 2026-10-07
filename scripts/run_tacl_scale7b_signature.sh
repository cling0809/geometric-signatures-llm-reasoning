#!/usr/bin/env bash
set -euo pipefail

cd /root/AI/geoprobe
source /root/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe
export GEOPROBE_RUNS=/root/AI/runs

LOG_DIR=/root/AI/runs/2026-05-26_tacl-scale7b-signature/logs
mkdir -p "$LOG_DIR"

run_metric() {
  local config="$1"
  local log_name="$2"
  python scripts/extract_metrics_streaming.py \
    --config "$config" \
    --flush-every 10 \
    > "$LOG_DIR/$log_name" 2>&1
}

run_metric configs/2026-05-26_tacl-scale7b-qwen-base-gsm8k-100.yaml qwen_base.log &
pid_base=$!
run_metric configs/2026-05-26_tacl-scale7b-qwen-instruct-gsm8k-100.yaml qwen_instruct.log &
pid_inst=$!

wait "$pid_base"
run_metric configs/2026-05-26_tacl-scale7b-qwen-math-gsm8k-100.yaml qwen_math.log &
pid_math=$!

wait "$pid_inst"
run_metric configs/2026-05-26_tacl-scale7b-r1-distill-gsm8k-100.yaml r1_distill.log &
pid_r1=$!

wait "$pid_math"
wait "$pid_r1"

python scripts/phase3_signature_matrix.py \
  --runs \
    /root/AI/runs/2026-05-26_tacl-scale7b-qwen-base-gsm8k-100 \
    /root/AI/runs/2026-05-26_tacl-scale7b-qwen-instruct-gsm8k-100 \
    /root/AI/runs/2026-05-26_tacl-scale7b-qwen-math-gsm8k-100 \
    /root/AI/runs/2026-05-26_tacl-scale7b-r1-distill-gsm8k-100 \
  --labels Qwen7B-Base Qwen7B-Instruct Qwen7B-Math R1Distill-Qwen7B \
  --out /root/AI/runs/2026-05-26_tacl-scale7b-signature \
  > "$LOG_DIR/aggregate.log" 2>&1

date > /root/AI/runs/2026-05-26_tacl-scale7b-signature/DONE
