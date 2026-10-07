#!/usr/bin/env bash
# Execute complete-family frozen MATH-500 and SVAMP OOD evaluations only after
# the locked family report. Completion, never the sign of a result, releases
# this stage. Both OOD datasets were frozen before validation scores were read.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
QWEN_MODEL=${QWEN_MODEL:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct}
R1_MODEL=${R1_MODEL:-/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B}
MODEL_SOURCE=${MODEL_SOURCE:-huggingface}

ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/frozen-ood-chain-v1"
MATH_GRADER_AUDIT="$ROOT/math500-symbolic-grader-audit-v1.json"
SVAMP_GRADER_AUDIT="$ROOT/svamp-numeric-grader-audit-v1.json"
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

eligible_methods() {
  local selection=$1
  "$PY" - "$selection" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1]))["methods"]:
    if item["decision"].get("eligible"):
        print(item["method"])
PY
}

run_dataset() {
  local tag=$1 selection=$2 model=$3 dataset=$4 grader_audit=$5
  local locked_root="$ROOT/locked-gsm8k-${tag}-v1"
  local ood_root="$ROOT/ood-${dataset}-${tag}-v1"
  local report_root="$ROOT/ood-report-${dataset}-${tag}-v1"
  if [[ -f "$locked_root/QUARANTINED_CAPABILITY" ]]; then
    log "$tag is quarantined by its predeclared capability gate; no $dataset OOD claim is permitted"
    mkdir -p "$ood_root" "$report_root"
    touch "$ood_root/QUARANTINED_CAPABILITY" "$report_root/QUARANTINED_CAPABILITY"
    return 0
  fi
  [[ -f "$selection" ]] || { log "$tag selection is missing without a capability quarantine"; exit 3; }
  if [[ -f "$locked_root/NO_ELIGIBLE_METHODS" ]]; then
    log "$tag has no eligible locked method; no $dataset OOD launch is permitted"
    mkdir -p "$ood_root"
    touch "$ood_root/NO_ELIGIBLE_METHODS"
    return 0
  fi
  local -a methods
  mapfile -t methods < <(eligible_methods "$selection")
  if [[ ${#methods[@]} -eq 0 ]]; then
    log "$tag selection has no eligible methods; no $dataset OOD launch is permitted"
    mkdir -p "$ood_root"
    touch "$ood_root/NO_ELIGIBLE_METHODS"
    return 0
  fi
  mkdir -p "$ood_root" "$report_root"
  log "$tag launching $dataset OOD for complete eligible family: ${methods[*]}"
  local -a pids=()
  local idx=0 method out device launcher
  case "$dataset" in
    math500) launcher=scripts/revision_launch_ood_selected.py ;;
    svamp) launcher=scripts/revision_launch_svamp_ood_selected.py ;;
    *) log "unsupported frozen OOD dataset: $dataset"; exit 3 ;;
  esac
  for method in "${methods[@]}"; do
    out="$ood_root/$method"
    device="cuda:$((idx % 2))"
    mkdir -p "$out"
    "$PY" -u "$launcher" \
      --selection "$selection" --method "$method" --grader-audit "$grader_audit" \
      --target-model "$model" --target-source "$MODEL_SOURCE" --target-device "$device" \
      --out "$out" --execute >"$out/run.log" 2>&1 &
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
    [[ -f "$ood_root/$method/DONE" ]] || { log "$tag $dataset OOD run incomplete: $method"; exit 4; }
  done
  local -a report_args=(--selection "$selection" --dataset "$dataset" --out "$report_root")
  for method in "${methods[@]}"; do
    report_args+=(--run "$method=$ood_root/$method")
  done
  log "$tag building complete-family $dataset OOD report"
  "$PY" -u scripts/revision_build_ood_report.py "${report_args[@]}" >"$report_root/build.log" 2>&1
  [[ -f "$report_root/ood_report_manifest.json" ]] || { log "$tag $dataset OOD report missing"; exit 5; }
}

cd "$REPO"
# Prefer the checked-out revision source over a stale editable/wheel install.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
wait_done "$ROOT/frozen-locked-chain-v1"
[[ -f "$MATH_GRADER_AUDIT" ]] || { log "formal MATH-500 grader audit missing"; exit 6; }
[[ -f "$SVAMP_GRADER_AUDIT" ]] || { log "formal SVAMP numeric grader audit missing"; exit 7; }
for tag_model in "qwen-instruct:$QWEN_MODEL" "r1:$R1_MODEL"; do
  tag=${tag_model%%:*}
  model=${tag_model#*:}
  selection="$ROOT/selection-${tag}-v1.json"
  run_dataset "$tag" "$selection" "$model" math500 "$MATH_GRADER_AUDIT"
  run_dataset "$tag" "$selection" "$model" svamp "$SVAMP_GRADER_AUDIT"
done
log "frozen MATH-500 plus SVAMP OOD chain complete"
touch "$CHAIN_DIR/DONE"
