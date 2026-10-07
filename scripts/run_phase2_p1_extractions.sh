#!/usr/bin/env bash
# Phase 2 P1: launch the 3 cross-model extractions.
# GPU 0 runs Llama-1B then Qwen-1.5B (sequential, same device).
# GPU 1 runs DeepSeek-R1-Distill-1.5B in parallel.
# Total wall time ~25 min (3 model downloads + ~10 min generation each).

set -euo pipefail

PROJ="/root/AI/geoprobe"
RUNS="/root/AI/runs"
cd "$PROJ"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoprobe

# GPU 1: DeepSeek (background)
nohup python scripts/extract_trajectories.py \
    --config configs/2026-05-17_extract-deepseek-r1-distill-1.5b-100.yaml \
    > "$RUNS/2026-05-17_extract-deepseek-r1-distill-1.5b-100_console.log" 2>&1 &
PID_DS=$!
echo "DeepSeek-R1-Distill-1.5B on cuda:1 -> PID=$PID_DS"

# GPU 0: Llama first, then Qwen (sequential)
(
    python scripts/extract_trajectories.py \
        --config configs/2026-05-17_extract-llama-1b-100.yaml \
        > "$RUNS/2026-05-17_extract-llama-1b-100_console.log" 2>&1
    python scripts/extract_trajectories.py \
        --config configs/2026-05-17_extract-qwen-1.5b-nomath-100.yaml \
        > "$RUNS/2026-05-17_extract-qwen-1.5b-nomath-100_console.log" 2>&1
) > "$RUNS/phase2_p1_gpu0_chain.log" 2>&1 &
PID_GPU0=$!
echo "Llama-1B then Qwen-1.5B-nomath on cuda:0 -> PID=$PID_GPU0"

echo
echo "Tail logs with:"
echo "  tail -f $RUNS/2026-05-17_extract-deepseek-r1-distill-1.5b-100_console.log"
echo "  tail -f $RUNS/2026-05-17_extract-llama-1b-100_console.log"
echo "  tail -f $RUNS/2026-05-17_extract-qwen-1.5b-nomath-100_console.log"
echo
echo "Sentinel DONE files appear at $RUNS/<exp-id>/DONE when each finishes."
