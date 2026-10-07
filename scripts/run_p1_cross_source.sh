#!/usr/bin/env bash
# P1.1: C4 cross-source comparison. Same target (R1-Distill), same layer (14),
# same alpha (2.0), but steering vector computed from 3 different source models.
#
# Pairs with the already-run Qwen-Instruct source (0.45) — this script adds
# Qwen-Base and Qwen-Math-Instruct.

set -euo pipefail
cd /root/AI/geoprobe
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe

RUNS=/root/AI/runs
LOG_ROOT=/root/AI/runs

for source_name in qwen-1.5b-base qwen-math-1.5b; do
    case "$source_name" in
        qwen-1.5b-base)
            src_run="$RUNS/2026-05-17_extract-qwen-1.5b-base-100"
            tag="base"
            ;;
        qwen-math-1.5b)
            src_run="$RUNS/2026-05-17_extract-pilot-gsm8k-100"
            tag="math"
            ;;
    esac
    out_dir="$RUNS/2026-05-17_p1-source-${tag}-L14-a2"
    log="$LOG_ROOT/2026-05-17_p1-source-${tag}-L14-a2_console.log"
    echo "==> cross-source ${tag}"
    python scripts/c4_steer_r1.py \
        --source-run "$src_run" \
        --target-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --target-source modelscope \
        --target-device cuda:1 \
        --layer 14 \
        --alphas 0.0 2.0 \
        --n-questions 100 \
        --out "$out_dir" \
        > "$log" 2>&1
    echo "  done ${tag}: $out_dir"
done

echo "==> all P1.1 cross-source done"
