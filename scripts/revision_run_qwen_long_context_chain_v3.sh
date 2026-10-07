#!/usr/bin/env bash
# Run Qwen long-context safety probes after the corrected v2 OOD suite.
# Supersedes revision_run_qwen_long_context_chain.sh, which gated on the v1
# OOD chain.  The selected direction/layer/alpha are unchanged; only the
# predeclared schedule/budget grid is varied.  GPU 0 remains isolated from the
# R1 official-context worker.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
QWEN_MODEL=${QWEN_MODEL:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct}
MODEL_SOURCE=${MODEL_SOURCE:-huggingface}
QWEN_DEVICE=${QWEN_DEVICE:-cuda:0}
ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/qwen-long-context-chain-v3"
SELECTION="$ROOT/selection-qwen-instruct-v3.json"
mkdir -p "$CHAIN_DIR"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$CHAIN_DIR/chain.log"; }
fail() { printf '%s\n' "$1" >"$CHAIN_DIR/FAILURE_REASON"; log "FAILED: $1"; exit 1; }
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN_DIR/FAILURE_REASON" ]]; then printf "unexpected Qwen long-context v3 failure (exit=%s)\n" "$rc" >"$CHAIN_DIR/FAILURE_REASON"; fi' EXIT
wait_done() {
  local path=$1
  while [[ ! -f "$path/DONE" ]]; do
    [[ -f "$path/FAILURE_REASON" ]] && fail "dependency failed: $path"
    sleep 60
  done
}
eligible_long_methods() {
  "$PY" - "$SELECTION" <<'PY'
import json
import sys
allowed={"crosssteer_source", "target_calibrated"}
for item in json.load(open(sys.argv[1]))["methods"]:
    if item["method"] in allowed and item["decision"].get("eligible"):
        print(item["method"])
PY
}
run_step() {
  local name=$1
  shift
  log "starting $name"
  if ! "$@" >"$CHAIN_DIR/$name.log" 2>&1; then fail "$name failed; see $CHAIN_DIR/$name.log"; fi
}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# Serialise GPU 0 descendants.  The label-efficiency curve is already frozen
# from the v2 selection; waiting for it is resource scheduling only.
wait_done "$ROOT/qwen-label-efficiency-chain-v3"
[[ -f "$SELECTION" ]] || fail "v3 selection artifact missing"
mapfile -t METHODS < <(eligible_long_methods)
if [[ ${#METHODS[@]} -eq 0 ]]; then
  log "no eligible CrossSteer/target-calibrated Qwen method for long-context probe"
  touch "$CHAIN_DIR/NO_ELIGIBLE_LONG_CONTEXT_METHOD" "$CHAIN_DIR/DONE"
  exit 0
fi
for budget in 4096 32768; do
  for method in "${METHODS[@]}"; do
    root="$ROOT/long-context-qwen-instruct-${method}-b${budget}-v2"
    report="$ROOT/long-context-report-qwen-instruct-${method}-b${budget}-v2"
    mkdir -p "$root" "$report"
    if [[ ! -f "$root/long_context_launch_manifest.json" ]]; then
      run_step "launch-${method}-b${budget}" "$PY" -u scripts/revision_launch_long_context.py \
        --selection "$SELECTION" --method "$method" --target-model "$QWEN_MODEL" \
        --target-source "$MODEL_SOURCE" --target-device "$QWEN_DEVICE" --budget "$budget" \
        --out "$root" --execute
    fi
    for policy in constant prefix-256 exponential-1024 relative-hidden-rms; do
      [[ -f "$root/$policy/DONE" ]] || fail "$method/b$budget incomplete policy: $policy"
    done
    run_step "report-${method}-b${budget}" "$PY" -u scripts/revision_build_long_context_report.py \
      --selection "$SELECTION" --method "$method" --plan "$root/long_context_launch_manifest.json" \
      --out "$report"
    [[ -f "$report/long_context_report_manifest.json" ]] || fail "long-context report missing: $method/b$budget"
  done
done
log "Qwen frozen long-context v2 chain complete"
touch "$CHAIN_DIR/DONE"
