#!/usr/bin/env bash
# Run the frozen complete-family Qwen OOD suite.  It consumes GPU 0 only so the
# independent R1 official-context path can use GPU 1 without shared-state runs.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
QWEN_MODEL=${QWEN_MODEL:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct}
MODEL_SOURCE=${MODEL_SOURCE:-huggingface}
QWEN_DEVICE=${QWEN_DEVICE:-cuda:0}

ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/qwen-ood-chain-v1"
SELECTION="$ROOT/selection-qwen-instruct-v1.json"
LOCKED_ROOT="$ROOT/locked-gsm8k-qwen-instruct-v1"
MATH_GRADER_AUDIT="$ROOT/math500-symbolic-grader-audit-v1.json"
SVAMP_GRADER_AUDIT="$ROOT/svamp-numeric-grader-audit-v1.json"
mkdir -p "$CHAIN_DIR"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$CHAIN_DIR/chain.log"; }
fail() { printf '%s\n' "$1" >"$CHAIN_DIR/FAILURE_REASON"; log "FAILED: $1"; exit 1; }
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN_DIR/FAILURE_REASON" ]]; then printf "unexpected Qwen OOD-chain failure (exit=%s)\n" "$rc" >"$CHAIN_DIR/FAILURE_REASON"; fi' EXIT
wait_done() {
  local path=$1
  while [[ ! -f "$path/DONE" ]]; do
    [[ -f "$path/FAILURE_REASON" ]] && fail "dependency failed: $path"
    sleep 60
  done
}
run_step() {
  local name=$1
  shift
  log "starting $name"
  if ! "$@" >"$CHAIN_DIR/$name.log" 2>&1; then fail "$name failed; see $CHAIN_DIR/$name.log"; fi
}
eligible_methods() {
  "$PY" - "$SELECTION" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1]))["methods"]:
    if item["decision"].get("eligible"):
        print(item["method"])
PY
}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
wait_done "$ROOT/qwen-locked-chain-v1"
[[ -f "$MATH_GRADER_AUDIT" ]] || fail "formal MATH-500 grader audit missing"
[[ -f "$SVAMP_GRADER_AUDIT" ]] || fail "formal SVAMP grader audit missing"
if [[ -f "$LOCKED_ROOT/NO_ELIGIBLE_METHODS" ]]; then
  log "no behavior-eligible Qwen locked method; OOD launch is prohibited"
  touch "$CHAIN_DIR/NO_ELIGIBLE_METHODS" "$CHAIN_DIR/DONE"
  exit 0
fi
mapfile -t METHODS < <(eligible_methods)
if [[ ${#METHODS[@]} -eq 0 ]]; then
  fail "selection has no eligible method but lacks NO_ELIGIBLE_METHODS marker"
fi

for dataset in math500 svamp; do
  case "$dataset" in
    math500) launcher=scripts/revision_launch_ood_selected.py; audit="$MATH_GRADER_AUDIT" ;;
    svamp) launcher=scripts/revision_launch_svamp_ood_selected.py; audit="$SVAMP_GRADER_AUDIT" ;;
  esac
  OOD_ROOT="$ROOT/ood-${dataset}-qwen-instruct-v1"
  REPORT_ROOT="$ROOT/ood-report-${dataset}-qwen-instruct-v1"
  mkdir -p "$OOD_ROOT" "$REPORT_ROOT"
  log "launching frozen $dataset OOD serially on $QWEN_DEVICE for: ${METHODS[*]}"
  for method in "${METHODS[@]}"; do
    out="$OOD_ROOT/$method"
    [[ -f "$out/DONE" ]] && continue
    mkdir -p "$out"
    run_step "$dataset-$method" "$PY" -u "$launcher" \
      --selection "$SELECTION" --method "$method" --grader-audit "$audit" \
      --target-model "$QWEN_MODEL" --target-source "$MODEL_SOURCE" --target-device "$QWEN_DEVICE" \
      --out "$out" --execute
  done
  for method in "${METHODS[@]}"; do [[ -f "$OOD_ROOT/$method/DONE" ]] || fail "$dataset run incomplete: $method"; done
  REPORT_ARGS=(--selection "$SELECTION" --dataset "$dataset" --out "$REPORT_ROOT")
  for method in "${METHODS[@]}"; do REPORT_ARGS+=(--run "$method=$OOD_ROOT/$method"); done
  run_step "build-$dataset-report" "$PY" -u scripts/revision_build_ood_report.py "${REPORT_ARGS[@]}"
  [[ -f "$REPORT_ROOT/ood_report_manifest.json" ]] || fail "$dataset report missing"
done

log "Qwen complete-family frozen OOD chain complete"
touch "$CHAIN_DIR/DONE"
