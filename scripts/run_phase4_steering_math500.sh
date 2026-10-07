#!/usr/bin/env bash
# Phase 4 A steering on MATH-500. Two source vectors:
#   (1) cross-task: Qwen-Math trajectories from GSM8K (best source from P1)
#   (2) same-task : Qwen-Instruct (non-math) trajectories from MATH-500
#
# After Qwen-Math on MATH-500 finishes, a third run can be added (same script):
#   (3) Qwen-Math trajectories from MATH-500 (best source + same task)

set -euo pipefail
cd /root/AI/geoprobe
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe

# (1) cross-task: GSM8K-derived Qwen-Math vector → MATH-500
python scripts/c4_steer_r1.py \
    --source-run /root/AI/runs/2026-05-17_extract-pilot-gsm8k-100 \
    --target-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --target-source modelscope \
    --target-device cuda:1 \
    --layer 14 \
    --alphas 0.0 1.0 2.0 \
    --n-questions 100 \
    --dataset math500 \
    --out /root/AI/runs/2026-05-17_c4-math500-source-gsm8k-qwenmath \
    > /root/AI/runs/2026-05-17_c4-math500-source-gsm8k-qwenmath_console.log 2>&1
echo "==> cross-task done"

# (2) same-task: MATH-500-derived Qwen-Instruct vector → MATH-500
python scripts/c4_steer_r1.py \
    --source-run /root/AI/runs/2026-05-17_extract-qwen-instruct-math500-100 \
    --target-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --target-source modelscope \
    --target-device cuda:1 \
    --layer 14 \
    --alphas 0.0 1.0 2.0 \
    --n-questions 100 \
    --dataset math500 \
    --out /root/AI/runs/2026-05-17_c4-math500-source-math500-qwenInstruct \
    > /root/AI/runs/2026-05-17_c4-math500-source-math500-qwenInstruct_console.log 2>&1
echo "==> same-task done"

echo "==> Phase 4 A R1-Distill steering done"
