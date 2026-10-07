#!/usr/bin/env bash
# Run the predeclared R1 official-context long-generation safety suite after
# its own frozen 32k formal comparison.  This is not a new R1 selection stage:
# it reuses the immutable layer, alpha, source direction and sampled decoder.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
R1_MODEL=${R1_MODEL:-/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B}
R1_SOURCE=${R1_SOURCE:-huggingface}
R1_VISIBLE_DEVICE=${R1_VISIBLE_DEVICE:-1}
ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/r1-long-context-chain-v2"
FORMAL_CHAIN="$ROOT/r1-formal-comparison-chain-v2"
SELECTION="$ROOT/frozen-selection-r1official32k-v2.json"

mkdir -p "$CHAIN_DIR"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$CHAIN_DIR/chain.log"; }
fail() { printf '%s\n' "$1" >"$CHAIN_DIR/FAILURE_REASON"; log "FAILED: $1"; exit 1; }
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN_DIR/FAILURE_REASON" ]]; then printf "unexpected R1 long-context failure (exit=%s)\n" "$rc" >"$CHAIN_DIR/FAILURE_REASON"; fi' EXIT
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

allowed = {"crosssteer_source", "target_calibrated"}
for item in json.load(open(sys.argv[1]))["methods"]:
    if item["method"] in allowed and item["decision"].get("eligible"):
        print(item["method"])
PY
}
run_step() {
  local name=$1
  shift
  log "starting $name"
  if ! "$@" >"$CHAIN_DIR/$name.log" 2>&1; then
    fail "$name failed; see $CHAIN_DIR/$name.log"
  fi
}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export GEOPROBE_RUNS="$RUNS_ROOT"
# Present physical GPU 1 as cuda:0, matching the completed R1 formal chain.
export CUDA_VISIBLE_DEVICES="$R1_VISIBLE_DEVICE"

wait_done "$FORMAL_CHAIN"
[[ -f "$SELECTION" ]] || fail "R1 formal comparison finished without selection artifact"
mapfile -t METHODS < <(eligible_long_methods)
if [[ ${#METHODS[@]} -eq 0 ]]; then
  log "no eligible R1 CrossSteer/target-calibrated method for long-context probe"
  touch "$CHAIN_DIR/NO_ELIGIBLE_LONG_CONTEXT_METHOD" "$CHAIN_DIR/DONE"
  exit 0
fi

for budget in 4096 32768; do
  for method in "${METHODS[@]}"; do
    root="$ROOT/long-context-r1official32k-v2-${method}-b${budget}"
    report="$ROOT/long-context-report-r1official32k-v2-${method}-b${budget}"
    mkdir -p "$root" "$report"
    if [[ ! -f "$root/long_context_launch_manifest.json" ]]; then
      run_step "launch-${method}-b${budget}" "$PY" -u scripts/revision_launch_long_context.py \
        --selection "$SELECTION" --method "$method" --target-model "$R1_MODEL" \
        --target-source "$R1_SOURCE" --target-device cuda:0 --budget "$budget" \
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

log "R1 official-context frozen long-context chain complete"
touch "$CHAIN_DIR/DONE"
