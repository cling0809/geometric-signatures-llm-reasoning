#!/usr/bin/env bash
# Phase 5 — 32k showcase: full max_new_tokens=32768 greedy runs for R1-Distill.
# Run two datasets in parallel on the two GPUs after current longthink runs finish.
#
# Each call sweeps alpha={0, 2} on n=100 questions. Expect 3-6 hours per dataset.

set -euo pipefail
cd /root/AI/geoprobe
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe

# GSM8K on GPU 0
GSM_OUT=/root/AI/runs/2026-05-18_c4-gsm8k-r1-32k-showcase
GSM_LOG=/root/AI/runs/2026-05-18_c4-gsm8k-r1-32k-showcase_console.log
python scripts/c4_steer_r1.py \
    --steering-vector /root/AI/runs/2026-05-17_p1-source-math-L14-a2/steering_vector.pt \
    --target-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --target-source modelscope \
    --target-device cuda:0 \
    --layer 14 \
    --alphas 0.0 2.0 \
    --n-questions 100 \
    --dataset gsm8k \
    --max-new-tokens 32768 \
    --out "$GSM_OUT" \
    > "$GSM_LOG" 2>&1 &
GSM_PID=$!
echo "GSM8K 32k showcase PID=$GSM_PID"

# MATH-500 on GPU 1
M500_OUT=/root/AI/runs/2026-05-18_c4-math500-r1-32k-showcase
M500_LOG=/root/AI/runs/2026-05-18_c4-math500-r1-32k-showcase_console.log
python scripts/c4_steer_r1.py \
    --steering-vector /root/AI/runs/2026-05-18_c4-math500-source-math500-qwenmath/steering_vector.pt \
    --target-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --target-source modelscope \
    --target-device cuda:1 \
    --layer 14 \
    --alphas 0.0 2.0 \
    --n-questions 100 \
    --dataset math500 \
    --max-new-tokens 32768 \
    --out "$M500_OUT" \
    > "$M500_LOG" 2>&1 &
M500_PID=$!
echo "MATH-500 32k showcase PID=$M500_PID"

# Don't wait — exit and let them run in background
echo "Both launched. Logs:"
echo "  $GSM_LOG"
echo "  $M500_LOG"
