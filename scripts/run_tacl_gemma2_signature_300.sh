#!/usr/bin/env bash
set -euo pipefail

cd /root/AI/geoprobe
source /root/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe
export GEOPROBE_RUNS=/root/AI/runs

AGG_DIR=/root/AI/runs/2026-05-27_tacl-gemma2-signature-gsm8k-300
LOG_DIR="$AGG_DIR/logs"
mkdir -p "$LOG_DIR"

seed_from_100() {
  local src="$1"
  local dst="$2"
  if [[ -d "$dst" ]]; then
    return 0
  fi
  mkdir -p "$dst"
  for f in completed.jsonl labels.jsonl metrics.jsonl generations.jsonl labels.parquet metrics.parquet; do
    if [[ -f "$src/$f" ]]; then
      cp "$src/$f" "$dst/$f"
    fi
  done
  mkdir -p "$dst/logs" "$dst/metrics" "$dst/plots" "$dst/trajectories"
}

seed_from_100 \
  /root/AI/runs/2026-05-27_tacl-gemma2-2b-base-gsm8k-100 \
  /root/AI/runs/2026-05-27_tacl-gemma2-2b-base-gsm8k-300
seed_from_100 \
  /root/AI/runs/2026-05-27_tacl-gemma2-2b-it-gsm8k-100 \
  /root/AI/runs/2026-05-27_tacl-gemma2-2b-it-gsm8k-300

run_metric() {
  local config="$1"
  local log_name="$2"
  python scripts/extract_metrics_streaming.py \
    --config "$config" \
    --flush-every 10 \
    > "$LOG_DIR/$log_name" 2>&1
}

run_metric configs/2026-05-27_tacl-gemma2-2b-base-gsm8k-300.yaml gemma2_base_300.log &
pid_base=$!
run_metric configs/2026-05-27_tacl-gemma2-2b-it-gsm8k-300.yaml gemma2_it_300.log &
pid_it=$!

wait "$pid_base"
wait "$pid_it"

python scripts/phase3_signature_matrix.py \
  --runs \
    /root/AI/runs/2026-05-27_tacl-gemma2-2b-base-gsm8k-300 \
    /root/AI/runs/2026-05-27_tacl-gemma2-2b-it-gsm8k-300 \
  --labels Gemma2-Base Gemma2-IT \
  --out "$AGG_DIR" \
  > "$LOG_DIR/aggregate.log" 2>&1

date > "$AGG_DIR/DONE"
