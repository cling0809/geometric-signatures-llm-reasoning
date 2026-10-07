#!/usr/bin/env bash
# cons@8 showcase — approximating DeepSeek-R1 paper's cons@64 setup but at
# cons@8 (compute-feasible).
#
# Runs R1-Distill with:
#   - cons-n = 8 (8 samples per question, majority vote)
#   - temperature 0.6, top-p 0.95 (matching DeepSeek-R1 paper)
#   - max_new_tokens = 8192
#
# Two datasets in parallel on the two GPUs after 32k showcase finishes.
# Each call sweeps alpha={0, 2} on n=100 questions.

set -euo pipefail
cd /root/AI/geoprobe
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe

GSM_OUT=/root/AI/runs/2026-05-19_c4-gsm8k-r1-cons8-8k
GSM_LOG=/root/AI/runs/2026-05-19_c4-gsm8k-r1-cons8-8k_console.log
python scripts/c4_steer_r1.py \
    --steering-vector /root/AI/runs/2026-05-17_p1-source-math-L14-a2/steering_vector.pt \
    --target-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --target-source modelscope \
    --target-device cuda:0 \
    --layer 14 \
    --alphas 0.0 2.0 \
    --n-questions 100 \
    --dataset gsm8k \
    --max-new-tokens 8192 \
    --cons-n 8 \
    --temperature 0.6 \
    --top-p 0.95 \
    --out "$GSM_OUT" \
    > "$GSM_LOG" 2>&1 &
GSM_PID=$!
echo "GSM8K cons@8 PID=$GSM_PID"

M500_OUT=/root/AI/runs/2026-05-19_c4-math500-r1-cons8-8k
M500_LOG=/root/AI/runs/2026-05-19_c4-math500-r1-cons8-8k_console.log
python scripts/c4_steer_r1.py \
    --steering-vector /root/AI/runs/2026-05-18_c4-math500-source-math500-qwenmath/steering_vector.pt \
    --target-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --target-source modelscope \
    --target-device cuda:1 \
    --layer 14 \
    --alphas 0.0 2.0 \
    --n-questions 100 \
    --dataset math500 \
    --max-new-tokens 8192 \
    --cons-n 8 \
    --temperature 0.6 \
    --top-p 0.95 \
    --out "$M500_OUT" \
    > "$M500_LOG" 2>&1 &
M500_PID=$!
echo "MATH-500 cons@8 PID=$M500_PID"

echo "Both launched. Logs:"
echo "  $GSM_LOG"
echo "  $M500_LOG"
