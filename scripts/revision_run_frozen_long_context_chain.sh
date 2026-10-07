#!/usr/bin/env bash
# Run predeclared 4k/32k stability schedules after the all-method OOD artifact.
# It never changes a selected direction/layer/alpha and never branches on scores.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
QWEN_MODEL=${QWEN_MODEL:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct}
R1_MODEL=${R1_MODEL:-/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B}
MODEL_SOURCE=${MODEL_SOURCE:-huggingface}

ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/frozen-long-context-chain-v1"
mkdir -p "$CHAIN_DIR"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

wait_done() {
  local path=$1
  while [[ ! -f "$path/DONE" ]]; do
    if [[ -f "$path/FAILURE_REASON" ]]; then
      log "refusing continuation because $path failed: $(cat "$path/FAILURE_REASON")"
      exit 2
    fi
    sleep 60
  done
}

eligible_long_methods() {
  local selection=$1
  "$PY" - "$selection" <<'PY'
import json
import sys
allowed={"crosssteer_source", "target_calibrated"}
for item in json.load(open(sys.argv[1]))["methods"]:
    if item["method"] in allowed and item["decision"].get("eligible"):
        print(item["method"])
PY
}

run_target() {
  local tag=$1 selection=$2 model=$3
  local locked_root="$ROOT/locked-gsm8k-${tag}-v1"
  if [[ -f "$locked_root/QUARANTINED_CAPABILITY" ]]; then
    log "$tag is quarantined by its predeclared capability gate; no long-context claim is permitted"
    mkdir -p "$ROOT/long-context-${tag}-v1"
    touch "$ROOT/long-context-${tag}-v1/QUARANTINED_CAPABILITY"
    return 0
  fi
  [[ -f "$selection" ]] || { log "$tag selection is missing without a capability quarantine"; exit 3; }
  local -a methods
  mapfile -t methods < <(eligible_long_methods "$selection")
  if [[ ${#methods[@]} -eq 0 ]]; then
    log "$tag has no eligible CrossSteer/target-calibrated method for long-context study"
    mkdir -p "$ROOT/long-context-${tag}-v1"
    touch "$ROOT/long-context-${tag}-v1/NO_ELIGIBLE_LONG_CONTEXT_METHOD"
    return 0
  fi
  local budget method root out device
  for budget in 4096 32768; do
    local -a pids=()
    local idx=0
    log "$tag launching frozen long-context budget $budget for: ${methods[*]}"
    for method in "${methods[@]}"; do
      root="$ROOT/long-context-${tag}-${method}-b${budget}"
      out="$ROOT/long-context-report-${tag}-${method}-b${budget}"
      device="cuda:$((idx % 2))"
      mkdir -p "$root" "$out"
      "$PY" -u scripts/revision_launch_long_context.py \
        --selection "$selection" --method "$method" \
        --target-model "$model" --target-source "$MODEL_SOURCE" --target-device "$device" \
        --budget "$budget" --out "$root" --execute >"$root/run.log" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      if [[ ${#pids[@]} -eq 2 ]]; then
        wait "${pids[0]}"
        wait "${pids[1]}"
        pids=()
      fi
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    for method in "${methods[@]}"; do
      root="$ROOT/long-context-${tag}-${method}-b${budget}"
      out="$ROOT/long-context-report-${tag}-${method}-b${budget}"
      for policy in constant prefix-256 exponential-1024 relative-hidden-rms; do
        [[ -f "$root/$policy/DONE" ]] || { log "$tag/$method/b$budget incomplete policy: $policy"; exit 3; }
      done
      log "$tag building long-context report for $method at $budget"
      "$PY" -u scripts/revision_build_long_context_report.py \
        --selection "$selection" --method "$method" \
        --plan "$root/long_context_launch_manifest.json" --out "$out" >"$out/build.log" 2>&1
      [[ -f "$out/long_context_report_manifest.json" ]] || { log "$tag/$method/b$budget report missing"; exit 4; }
    done
  done
}

cd "$REPO"
# Prefer the checked-out revision source over a stale editable/wheel install.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
wait_done "$ROOT/frozen-ood-chain-v1"
run_target qwen-instruct "$ROOT/selection-qwen-instruct-v1.json" "$QWEN_MODEL"
run_target r1 "$ROOT/selection-r1-v1.json" "$R1_MODEL"
log "frozen long-context chain complete"
touch "$CHAIN_DIR/DONE"
