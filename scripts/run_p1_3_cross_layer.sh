#!/usr/bin/env bash
# P1.3: cross-layer sweep with the best (source, alpha) pair.
# Usage: run_p1_3_cross_layer.sh <alpha>
# Always uses Math source (best from P1.1).

set -euo pipefail
cd /root/AI/geoprobe
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe

ALPHA="${1:?expected alpha as $1}"
SOURCE=/root/AI/runs/2026-05-17_extract-pilot-gsm8k-100
RUNS=/root/AI/runs

for layer in 5 10 20 24; do
    out_dir="$RUNS/2026-05-17_p1-source-math-L${layer}-a${ALPHA}"
    log="$RUNS/2026-05-17_p1-source-math-L${layer}-a${ALPHA}_console.log"
    echo "==> layer ${layer} alpha ${ALPHA}"
    python scripts/c4_steer_r1.py \
        --source-run "$SOURCE" \
        --target-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --target-source modelscope \
        --target-device cuda:1 \
        --layer "$layer" \
        --alphas 0.0 "$ALPHA" \
        --n-questions 100 \
        --out "$out_dir" \
        > "$log" 2>&1
    echo "  done layer ${layer}: $out_dir"
done

echo "==> P1.3 cross-layer done"
