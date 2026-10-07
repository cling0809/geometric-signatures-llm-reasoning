#!/usr/bin/env bash
# Clean held-out CrossSteer baselines.
#
# Calibrate steering vectors on existing source runs over ids 0-99, then evaluate
# on ids 100-199. This gives a direct comparison between a cross-model source
# direction and a same-target direction without using target labels from the
# evaluation questions.

set -euo pipefail

cd /root/AI/geoprobe
if [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
  # Server runs keep dependencies in the geoprobe conda environment.
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate geoprobe
fi

RUNS_ROOT="${RUNS_ROOT:-/root/AI/runs}"
OUT_ROOT="${OUT_ROOT:-$RUNS_ROOT/2026-05-24_c4-holdout-baselines}"
CROSS_DEVICE="${CROSS_DEVICE:-cuda:0}"
SAME_DEVICE="${SAME_DEVICE:-cuda:1}"
PARALLEL="${PARALLEL:-1}"
TARGET_MODEL="${TARGET_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
TARGET_SOURCE="${TARGET_SOURCE:-modelscope}"
N_QUESTIONS="${N_QUESTIONS:-100}"
EVAL_START="${EVAL_START:-100}"
DATASET="${DATASET:-gsm8k}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LAYER="${LAYER:-14}"
ALPHAS="${ALPHAS:-0 1 2}"

mkdir -p "$OUT_ROOT"

run_one() {
  local name="$1"
  local source_run="$2"
  local target_device="$3"
  local out_dir="$OUT_ROOT/$name"
  echo "==> $name on $target_device"
  python scripts/c4_steer_r1.py \
    --source-run "$source_run" \
    --target-model "$TARGET_MODEL" \
    --target-source "$TARGET_SOURCE" \
    --target-device "$target_device" \
    --dataset "$DATASET" \
    --n-questions "$N_QUESTIONS" \
    --eval-start "$EVAL_START" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --layer "$LAYER" \
    --alphas $ALPHAS \
    --out "$out_dir"
}

if [[ "$PARALLEL" == "1" ]]; then
  run_one "cross_source_qwen_math_to_r1" \
    "$RUNS_ROOT/2026-05-17_extract-pilot-gsm8k-100" \
    "$CROSS_DEVICE" > "$OUT_ROOT/cross_source_qwen_math_to_r1_console.log" 2>&1 &
  pid_cross=$!

  run_one "same_target_r1_to_r1" \
    "$RUNS_ROOT/2026-05-17_extract-deepseek-r1-distill-1.5b-100" \
    "$SAME_DEVICE" > "$OUT_ROOT/same_target_r1_to_r1_console.log" 2>&1 &
  pid_same=$!

  wait "$pid_cross"
  wait "$pid_same"
else
  run_one "cross_source_qwen_math_to_r1" \
    "$RUNS_ROOT/2026-05-17_extract-pilot-gsm8k-100" \
    "$CROSS_DEVICE"

  run_one "same_target_r1_to_r1" \
    "$RUNS_ROOT/2026-05-17_extract-deepseek-r1-distill-1.5b-100" \
    "$SAME_DEVICE"
fi

python - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ.get("OUT_ROOT", os.environ.get("RUNS_ROOT", "/root/AI/runs") + "/2026-05-24_c4-holdout-baselines"))
rows = []
for path in sorted(root.glob("*/c4_alpha_accuracy.csv")):
    df = pd.read_csv(path)
    for _, rec in df.iterrows():
        rows.append({"run": path.parent.name, **rec.to_dict()})
out = pd.DataFrame(rows)
out.to_csv(root / "holdout_baseline_summary.csv", index=False)
print(out.to_string(index=False))
PY

echo "==> wrote $OUT_ROOT/holdout_baseline_summary.csv"
