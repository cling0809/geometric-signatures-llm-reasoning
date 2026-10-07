#!/usr/bin/env bash
# Execute the predeclared post-validation baseline chain for TACL-11241.
#
# This script never reads validation scores and never launches a locked test. It
# waits for the two V1 source/target grids to finish successfully, creates the
# source-training-only CAA/ActAdd contrast pool, builds its prompt-final registry,
# then evaluates CAA, ActAdd, an SAE-space sparse-activation baseline,
# matched-norm random and sign-reversed direction controls, and a predeclared
# 10% coordinate-sparse CAA ablation under the same V1 validation grid.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
MODEL=${MODEL:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct}
MODEL_SOURCE=${MODEL_SOURCE:-huggingface}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/baseline-chain-v1"
mkdir -p "$CHAIN_DIR"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
# A background grid can make `wait` return nonzero under `set -e`; leave a
# durable marker so downstream frozen queues fail closed rather than polling
# forever for a DONE file that will never appear.
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN_DIR/FAILURE_REASON" ]]; then printf "unexpected Qwen baseline-chain failure (exit=%s)\n" "$rc" >"$CHAIN_DIR/FAILURE_REASON"; fi' EXIT
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
run_grid() {
  local method=$1 vector=$2 device=$3 out=$4
  mkdir -p "$out"
  log "starting $method on $device"
  "$PY" -u scripts/revision_steering_validation_grid.py \
    --vector "$method=$vector" \
    --target-model "$MODEL" --target-source "$MODEL_SOURCE" --target-device "$device" \
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

SOURCE_GRID="$ROOT/validation-grid-v1-crosssteer_source"
TARGET_GRID="$ROOT/validation-grid-v1-target_calibrated"
log "waiting for predeclared source/target validation grids"
wait_done "$SOURCE_GRID"
wait_done "$TARGET_GRID"
fail_if_failed "$SOURCE_GRID" "$TARGET_GRID"

POOL="$RUNS_ROOT/revision_gsm8k_qwen_instruct_caa_source_k4"
if [[ ! -f "$POOL/DONE" ]]; then
  log "generating source-training-only same-question CAA/ActAdd completion pool"
  "$PY" -u scripts/revision_generate_contrast_pool.py \
    --config configs/revision_gsm8k_qwen_instruct_caa_source_k4.yaml \
    >"$CHAIN_DIR/contrast-pool.log" 2>&1
fi
[[ -f "$POOL/DONE" ]] || { log "contrast pool did not complete"; exit 3; }

REGISTRY="$ROOT/prompt-contrast-registry-gsm8k-ids0-99"
if [[ ! -f "$REGISTRY/manifest.json" ]]; then
  log "building audited prompt-final CAA/ActAdd registry"
  "$PY" -u scripts/revision_build_prompt_contrast_registry.py \
    --target-run "$POOL" --model "$MODEL" --model-source "$MODEL_SOURCE" \
    --device cuda:0 --out "$REGISTRY" \
    >"$CHAIN_DIR/build-prompt-registry.log" 2>&1
fi
[[ -f "$REGISTRY/manifest.json" ]] || { log "prompt contrast registry missing"; exit 4; }

CAA_OUT="$ROOT/validation-grid-v1-caa_target_prompt_final"
ACTADD_OUT="$ROOT/validation-grid-v1-actadd_target_prompt_final"
RANDOM_REGISTRY="$ROOT/matched-random-registry-qwen-gsm8k-ids0-99"
RANDOM_OUT="$ROOT/validation-grid-v1-matched_norm_random"
NEGATIVE_REGISTRY="$ROOT/negative-direction-registry-qwen-gsm8k-ids0-99"
NEGATIVE_OUT="$ROOT/validation-grid-v1-negative_crosssteer_source"
SAE_REGISTRY="$ROOT/sae-sparse-registry-qwen-gsm8k-ids0-99"
SAE_OUT="$ROOT/validation-grid-v1-sae_sparse_activation"
SPARSE_OUT="$ROOT/validation-grid-v1-sparse_caa_coordinate_10pct"

if [[ ! -f "$CAA_OUT/DONE" && ! -f "$ACTADD_OUT/DONE" ]]; then
  run_grid caa_target_prompt_final "$REGISTRY/caa_target_prompt_final.pt" cuda:0 "$CAA_OUT"
  CAA_PID=$(cat "$CAA_OUT/PID")
  run_grid actadd_target_prompt_final "$REGISTRY/actadd_target_prompt_final.pt" cuda:1 "$ACTADD_OUT"
  ACTADD_PID=$(cat "$ACTADD_OUT/PID")
  wait "$CAA_PID"
  wait "$ACTADD_PID"
elif [[ ! -f "$CAA_OUT/DONE" ]]; then
  run_grid caa_target_prompt_final "$REGISTRY/caa_target_prompt_final.pt" cuda:0 "$CAA_OUT"
  wait "$(cat "$CAA_OUT/PID")"
elif [[ ! -f "$ACTADD_OUT/DONE" ]]; then
  run_grid actadd_target_prompt_final "$REGISTRY/actadd_target_prompt_final.pt" cuda:0 "$ACTADD_OUT"
  wait "$(cat "$ACTADD_OUT/PID")"
fi
fail_if_failed "$CAA_OUT" "$ACTADD_OUT"
[[ -f "$CAA_OUT/DONE" && -f "$ACTADD_OUT/DONE" ]] || { log "CAA/ActAdd grid incomplete"; exit 5; }

if [[ ! -f "$RANDOM_REGISTRY/manifest.json" ]]; then
  log "building fixed-seed matched-norm random direction control"
  "$PY" -u scripts/revision_build_matched_random_vector.py \
    --reference "$ROOT/vector-registry-gsm8k-ids0-99/crosssteer_source.pt" \
    --seed 11241 --out "$RANDOM_REGISTRY" >"$CHAIN_DIR/build-matched-random.log" 2>&1
fi
[[ -f "$RANDOM_REGISTRY/manifest.json" ]] || { log "matched random registry missing"; exit 6; }

if [[ ! -f "$NEGATIVE_REGISTRY/manifest.json" ]]; then
  log "building exact sign-reversed CrossSteer direction control"
  "$PY" -u scripts/revision_build_negative_direction.py \
    --reference "$ROOT/vector-registry-gsm8k-ids0-99/crosssteer_source.pt" \
    --out "$NEGATIVE_REGISTRY" >"$CHAIN_DIR/build-negative-direction.log" 2>&1
fi
[[ -f "$NEGATIVE_REGISTRY/manifest.json" ]] || { log "negative direction registry missing"; exit 7; }

if [[ ! -f "$SAE_REGISTRY/manifest.json" ]]; then
  log "building target SAE-space sparse-activation baseline"
  "$PY" -u scripts/revision_build_sae_sparse_registry.py \
    --calibration-run "$ROOT/revision_gsm8k_qwen_instruct_calibration_100" \
    --contrast-run "$POOL" --model "$MODEL" --model-source "$MODEL_SOURCE" --device cuda:0 \
    --out "$SAE_REGISTRY" >"$CHAIN_DIR/build-sae-sparse.log" 2>&1
fi
[[ -f "$SAE_REGISTRY/manifest.json" ]] || { log "SAE sparse registry missing"; exit 8; }

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
[[ -f "$RANDOM_OUT/DONE" && -f "$NEGATIVE_OUT/DONE" ]] || { log "matched random/negative grids incomplete"; exit 9; }

if [[ ! -f "$SAE_OUT/DONE" && ! -f "$SPARSE_OUT/DONE" ]]; then
  run_grid sae_sparse_activation "$SAE_REGISTRY/sae_sparse_activation.pt" cuda:0 "$SAE_OUT"
  SAE_PID=$(cat "$SAE_OUT/PID")
  run_grid sparse_caa_coordinate_10pct "$REGISTRY/sparse_caa_coordinate_10pct.pt" cuda:1 "$SPARSE_OUT"
  SPARSE_PID=$(cat "$SPARSE_OUT/PID")
  wait "$SAE_PID"
  wait "$SPARSE_PID"
elif [[ ! -f "$SAE_OUT/DONE" ]]; then
  run_grid sae_sparse_activation "$SAE_REGISTRY/sae_sparse_activation.pt" cuda:0 "$SAE_OUT"
  wait "$(cat "$SAE_OUT/PID")"
elif [[ ! -f "$SPARSE_OUT/DONE" ]]; then
  run_grid sparse_caa_coordinate_10pct "$REGISTRY/sparse_caa_coordinate_10pct.pt" cuda:0 "$SPARSE_OUT"
  wait "$(cat "$SPARSE_OUT/PID")"
fi
fail_if_failed "$SAE_OUT" "$SPARSE_OUT"
[[ -f "$SAE_OUT/DONE" && -f "$SPARSE_OUT/DONE" ]] || { log "SAE/coordinate-sparse grids incomplete"; exit 10; }

log "baseline chain complete; selection/locked testing remains a separate manual audited step"
touch "$CHAIN_DIR/DONE"
