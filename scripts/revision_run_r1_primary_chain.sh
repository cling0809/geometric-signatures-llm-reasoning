#!/usr/bin/env bash
# Execute the predeclared R1 primary-target chain after the Qwen baseline chain.
# This script never reads validation scores and never launches a locked test.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
R1_MODEL=${R1_MODEL:-/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B}
R1_SOURCE=${R1_SOURCE:-huggingface}
MATH_SOURCE_RUN=${MATH_SOURCE_RUN:-/root/AI/runs/tacl-revision/revision_gsm8k_qwen_math_source_calibration_100}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/r1-primary-chain-v1"
mkdir -p "$CHAIN_DIR"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
fail_if_failed() {
  local path
  for path in "$@"; do
    if [[ -f "$path/FAILURE_REASON" ]]; then
      log "refusing continuation: $(cat "$path/FAILURE_REASON")"
      exit 2
    fi
  done
}
wait_done() {
  local path=$1
  while [[ ! -f "$path/DONE" ]]; do
    fail_if_failed "$path"
    sleep 60
  done
}
audit_is_eligible() {
  local path=$1
  "$PY" - "$path" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
runs = payload.get("runs")
ok = (
    payload.get("protocol") == "tacl-11241-calibration-capability-audit-v1"
    and isinstance(runs, list)
    and len(runs) == 2
    and all(isinstance(run, dict) and bool(run.get("eligible_for_direction")) for run in runs)
)
raise SystemExit(0 if ok else 1)
PY
}
run_grid() {
  local method=$1 vector=$2 device=$3 out=$4
  mkdir -p "$out"
  log "starting $method on $device"
  "$PY" -u scripts/revision_steering_validation_grid.py \
    --vector "$method=$vector" \
    --target-model "$R1_MODEL" --target-source "$R1_SOURCE" --target-device "$device" \
    --eval-start 100 --eval-count 100 \
    --layers 8,14,20,24 --alphas 0.025,0.05,0.10,0.20 \
    --schedule constant --injection-mode absolute --max-new-tokens "$MAX_NEW_TOKENS" \
    --out "$out" >"$out/run.log" 2>&1 &
  echo $! >"$out/PID"
}

cd "$REPO"
# Prefer the checked-out revision source over a stale editable/wheel install.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export GEOPROBE_RUNS="$RUNS_ROOT"
# Run after the first fully predeclared target-control comparison, not after a
# selected score. The condition is completion only, never a positive result.
wait_done "$ROOT/baseline-chain-v1"

R1_CALIB="$RUNS_ROOT/revision_gsm8k_r1_target_calibration_100"
if [[ ! -f "$R1_CALIB/DONE" ]]; then
  log "extracting fresh R1 target calibration trajectories on IDs 0--99"
  "$PY" -u scripts/extract_trajectories.py \
    --config configs/revision_gsm8k_r1_target_calibration_100.yaml \
    >"$CHAIN_DIR/r1-calibration.log" 2>&1
fi
[[ -f "$R1_CALIB/DONE" ]] || { log "R1 calibration incomplete"; exit 3; }

CAPABILITY_AUDIT="$CHAIN_DIR/r1-source-target-capability-audit.json"
if [[ ! -f "$CAPABILITY_AUDIT" ]] || ! audit_is_eligible "$CAPABILITY_AUDIT"; then
  log "auditing R1/source calibration completeness and token-budget saturation"
  if ! "$PY" -u scripts/revision_audit_calibration.py \
    --run "qwen_math_source=$MATH_SOURCE_RUN" --run "r1_target=$R1_CALIB" \
    --expected-id-range 0:100 --max-new-tokens "$MAX_NEW_TOKENS" \
    --min-class-count 10 --max-budget-hit-rate 0.50 --out "$CAPABILITY_AUDIT" \
    >"$CHAIN_DIR/r1-capability-audit.log" 2>&1; then
    echo "R1 calibration capability audit failed; inspect r1-capability-audit.log before vector construction" >"$CHAIN_DIR/FAILURE_REASON"
    exit 4
  fi
fi
[[ -f "$CAPABILITY_AUDIT" ]] || { log "R1 capability audit artifact missing"; exit 4; }
audit_is_eligible "$CAPABILITY_AUDIT" || {
  echo "R1 calibration capability audit is ineligible; do not build steering vectors" >"$CHAIN_DIR/FAILURE_REASON"
  exit 4
}

REGISTRY="$ROOT/vector-registry-r1-gsm8k-ids0-99"
if [[ ! -f "$REGISTRY/manifest.json" ]]; then
  log "building source and target-calibrated R1 vector registry"
  "$PY" -u scripts/revision_build_vector_registry.py \
    --source-run "$MATH_SOURCE_RUN" --target-run "$R1_CALIB" --out "$REGISTRY" \
    >"$CHAIN_DIR/build-r1-vector-registry.log" 2>&1
fi
[[ -f "$REGISTRY/manifest.json" ]] || { log "R1 vector registry missing"; exit 4; }

SOURCE_OUT="$ROOT/validation-grid-v1-r1-crosssteer_source"
TARGET_OUT="$ROOT/validation-grid-v1-r1-target_calibrated"
if [[ ! -f "$SOURCE_OUT/DONE" && ! -f "$TARGET_OUT/DONE" ]]; then
  run_grid crosssteer_source "$REGISTRY/crosssteer_source.pt" cuda:0 "$SOURCE_OUT"
  SOURCE_PID=$(cat "$SOURCE_OUT/PID")
  run_grid target_calibrated "$REGISTRY/target_calibrated.pt" cuda:1 "$TARGET_OUT"
  TARGET_PID=$(cat "$TARGET_OUT/PID")
  wait "$SOURCE_PID"
  wait "$TARGET_PID"
elif [[ ! -f "$SOURCE_OUT/DONE" ]]; then
  run_grid crosssteer_source "$REGISTRY/crosssteer_source.pt" cuda:0 "$SOURCE_OUT"
  wait "$(cat "$SOURCE_OUT/PID")"
elif [[ ! -f "$TARGET_OUT/DONE" ]]; then
  run_grid target_calibrated "$REGISTRY/target_calibrated.pt" cuda:0 "$TARGET_OUT"
  wait "$(cat "$TARGET_OUT/PID")"
fi
fail_if_failed "$SOURCE_OUT" "$TARGET_OUT"
[[ -f "$SOURCE_OUT/DONE" && -f "$TARGET_OUT/DONE" ]] || { log "R1 source/target grids incomplete"; exit 5; }

POOL="$RUNS_ROOT/revision_gsm8k_r1_caa_source_k4"
if [[ ! -f "$POOL/DONE" ]]; then
  log "generating R1 same-question CAA/ActAdd completion pool"
  "$PY" -u scripts/revision_generate_contrast_pool.py \
    --config configs/revision_gsm8k_r1_caa_source_k4.yaml \
    >"$CHAIN_DIR/r1-contrast-pool.log" 2>&1
fi
[[ -f "$POOL/DONE" ]] || { log "R1 contrast pool incomplete"; exit 6; }

PROMPT_REGISTRY="$ROOT/prompt-contrast-registry-r1-gsm8k-ids0-99"
if [[ ! -f "$PROMPT_REGISTRY/manifest.json" ]]; then
  log "building R1 prompt-final CAA/ActAdd registry"
  "$PY" -u scripts/revision_build_prompt_contrast_registry.py \
    --target-run "$POOL" --model "$R1_MODEL" --model-source "$R1_SOURCE" \
    --device cuda:0 --out "$PROMPT_REGISTRY" \
    >"$CHAIN_DIR/build-r1-prompt-registry.log" 2>&1
fi
[[ -f "$PROMPT_REGISTRY/manifest.json" ]] || { log "R1 prompt registry missing"; exit 7; }

CAA_OUT="$ROOT/validation-grid-v1-r1-caa_target_prompt_final"
ACTADD_OUT="$ROOT/validation-grid-v1-r1-actadd_target_prompt_final"
RANDOM_REGISTRY="$ROOT/matched-random-registry-r1-gsm8k-ids0-99"
RANDOM_OUT="$ROOT/validation-grid-v1-r1-matched_norm_random"
NEGATIVE_REGISTRY="$ROOT/negative-direction-registry-r1-gsm8k-ids0-99"
NEGATIVE_OUT="$ROOT/validation-grid-v1-r1-negative_crosssteer_source"
SAE_REGISTRY="$ROOT/sae-sparse-registry-r1-gsm8k-ids0-99"
SAE_OUT="$ROOT/validation-grid-v1-r1-sae_sparse_activation"
SPARSE_OUT="$ROOT/validation-grid-v1-r1-sparse_caa_coordinate_10pct"
if [[ ! -f "$CAA_OUT/DONE" && ! -f "$ACTADD_OUT/DONE" ]]; then
  run_grid caa_target_prompt_final "$PROMPT_REGISTRY/caa_target_prompt_final.pt" cuda:0 "$CAA_OUT"
  CAA_PID=$(cat "$CAA_OUT/PID")
  run_grid actadd_target_prompt_final "$PROMPT_REGISTRY/actadd_target_prompt_final.pt" cuda:1 "$ACTADD_OUT"
  ACTADD_PID=$(cat "$ACTADD_OUT/PID")
  wait "$CAA_PID"
  wait "$ACTADD_PID"
elif [[ ! -f "$CAA_OUT/DONE" ]]; then
  run_grid caa_target_prompt_final "$PROMPT_REGISTRY/caa_target_prompt_final.pt" cuda:0 "$CAA_OUT"
  wait "$(cat "$CAA_OUT/PID")"
elif [[ ! -f "$ACTADD_OUT/DONE" ]]; then
  run_grid actadd_target_prompt_final "$PROMPT_REGISTRY/actadd_target_prompt_final.pt" cuda:0 "$ACTADD_OUT"
  wait "$(cat "$ACTADD_OUT/PID")"
fi
fail_if_failed "$CAA_OUT" "$ACTADD_OUT"
[[ -f "$CAA_OUT/DONE" && -f "$ACTADD_OUT/DONE" ]] || { log "R1 CAA/ActAdd grids incomplete"; exit 8; }

if [[ ! -f "$RANDOM_REGISTRY/manifest.json" ]]; then
  log "building R1 fixed-seed matched-norm random direction control"
  "$PY" -u scripts/revision_build_matched_random_vector.py \
    --reference "$REGISTRY/crosssteer_source.pt" --seed 11241 --out "$RANDOM_REGISTRY" \
    >"$CHAIN_DIR/build-r1-matched-random.log" 2>&1
fi
[[ -f "$RANDOM_REGISTRY/manifest.json" ]] || { log "R1 matched random registry missing"; exit 9; }

if [[ ! -f "$NEGATIVE_REGISTRY/manifest.json" ]]; then
  log "building R1 exact sign-reversed CrossSteer direction control"
  "$PY" -u scripts/revision_build_negative_direction.py \
    --reference "$REGISTRY/crosssteer_source.pt" --out "$NEGATIVE_REGISTRY" \
    >"$CHAIN_DIR/build-r1-negative-direction.log" 2>&1
fi
[[ -f "$NEGATIVE_REGISTRY/manifest.json" ]] || { log "R1 negative direction registry missing"; exit 10; }

if [[ ! -f "$SAE_REGISTRY/manifest.json" ]]; then
  log "building R1 target SAE-space sparse-activation baseline"
  "$PY" -u scripts/revision_build_sae_sparse_registry.py \
    --calibration-run "$R1_CALIB" --contrast-run "$POOL" \
    --model "$R1_MODEL" --model-source "$R1_SOURCE" --device cuda:0 \
    --out "$SAE_REGISTRY" >"$CHAIN_DIR/build-r1-sae-sparse.log" 2>&1
fi
[[ -f "$SAE_REGISTRY/manifest.json" ]] || { log "R1 SAE sparse registry missing"; exit 11; }

if [[ ! -f "$RANDOM_OUT/DONE" && ! -f "$NEGATIVE_OUT/DONE" ]]; then
  run_grid matched_norm_random "$RANDOM_REGISTRY/matched_norm_random.pt" cuda:0 "$RANDOM_OUT"
  RANDOM_PID=$(cat "$RANDOM_OUT/PID")
  run_grid negative_crosssteer_source "$NEGATIVE_REGISTRY/negative_crosssteer_source.pt" cuda:1 "$NEGATIVE_OUT"
  NEGATIVE_PID=$(cat "$NEGATIVE_OUT/PID")
  wait "$RANDOM_PID"
  wait "$NEGATIVE_PID"
elif [[ ! -f "$RANDOM_OUT/DONE" ]]; then
  run_grid matched_norm_random "$RANDOM_REGISTRY/matched_norm_random.pt" cuda:0 "$RANDOM_OUT"
  wait "$(cat "$RANDOM_OUT/PID")"
elif [[ ! -f "$NEGATIVE_OUT/DONE" ]]; then
  run_grid negative_crosssteer_source "$NEGATIVE_REGISTRY/negative_crosssteer_source.pt" cuda:0 "$NEGATIVE_OUT"
  wait "$(cat "$NEGATIVE_OUT/PID")"
fi
fail_if_failed "$RANDOM_OUT" "$NEGATIVE_OUT"
[[ -f "$RANDOM_OUT/DONE" && -f "$NEGATIVE_OUT/DONE" ]] || { log "R1 matched random/negative grids incomplete"; exit 12; }

if [[ ! -f "$SAE_OUT/DONE" && ! -f "$SPARSE_OUT/DONE" ]]; then
  run_grid sae_sparse_activation "$SAE_REGISTRY/sae_sparse_activation.pt" cuda:0 "$SAE_OUT"
  SAE_PID=$(cat "$SAE_OUT/PID")
  run_grid sparse_caa_coordinate_10pct "$PROMPT_REGISTRY/sparse_caa_coordinate_10pct.pt" cuda:1 "$SPARSE_OUT"
  SPARSE_PID=$(cat "$SPARSE_OUT/PID")
  wait "$SAE_PID"
  wait "$SPARSE_PID"
elif [[ ! -f "$SAE_OUT/DONE" ]]; then
  run_grid sae_sparse_activation "$SAE_REGISTRY/sae_sparse_activation.pt" cuda:0 "$SAE_OUT"
  wait "$(cat "$SAE_OUT/PID")"
elif [[ ! -f "$SPARSE_OUT/DONE" ]]; then
  run_grid sparse_caa_coordinate_10pct "$PROMPT_REGISTRY/sparse_caa_coordinate_10pct.pt" cuda:0 "$SPARSE_OUT"
  wait "$(cat "$SPARSE_OUT/PID")"
fi
fail_if_failed "$SAE_OUT" "$SPARSE_OUT"
[[ -f "$SAE_OUT/DONE" && -f "$SPARSE_OUT/DONE" ]] || { log "R1 SAE/coordinate-sparse grids incomplete"; exit 13; }

log "R1 primary target chain complete; selection/locked testing remains manual and audited"
touch "$CHAIN_DIR/DONE"
