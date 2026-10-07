#!/usr/bin/env bash
# Freeze the complete Qwen baseline-family selection and run exactly one locked
# confirmation for every eligible method.  R1 is intentionally excluded: its
# model-valid official-context path has a separate decoder and chain.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
QWEN_MODEL=${QWEN_MODEL:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct}
MODEL_SOURCE=${MODEL_SOURCE:-huggingface}

ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/qwen-locked-chain-v1"
SELECTION="$ROOT/selection-qwen-instruct-v1.json"
LOCKED_ROOT="$ROOT/locked-gsm8k-qwen-instruct-v1"
REPORT_ROOT="$ROOT/locked-report-qwen-instruct-v1"
VECTOR_ROOT="$ROOT/label-budget-vectors-qwen-instruct-v1"
CURVE_ROOT="$ROOT/label-efficiency-qwen-instruct-v1"
CURVE_REPORT="$ROOT/label-efficiency-report-qwen-instruct-v1"
mkdir -p "$CHAIN_DIR"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$CHAIN_DIR/chain.log"; }
fail() {
  local message=$1
  printf '%s\n' "$message" >"$CHAIN_DIR/FAILURE_REASON"
  log "FAILED: $message"
  exit 1
}
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN_DIR/FAILURE_REASON" ]]; then printf "unexpected Qwen locked-chain failure (exit=%s)\n" "$rc" >"$CHAIN_DIR/FAILURE_REASON"; fi' EXIT
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
  if ! "$@" >"$CHAIN_DIR/$name.log" 2>&1; then
    fail "$name failed; see $CHAIN_DIR/$name.log"
  fi
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
source_vector_path() {
  "$PY" - "$SELECTION" <<'PY'
import json
import sys
from pathlib import Path
payload=json.load(open(sys.argv[1]))
entries=[item for item in payload["methods"] if item["method"] == "crosssteer_source"]
if len(entries) != 1:
    raise SystemExit("selection lacks exactly one crosssteer_source entry")
manifest=json.loads((Path(entries[0]["run_dir"]) / "resolved_manifest.json").read_text())
vector=manifest["vectors"].get("crosssteer_source")
if not vector:
    raise SystemExit("crosssteer_source vector missing from selected manifest")
print(vector)
PY
}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export GEOPROBE_RUNS="$RUNS_ROOT"
wait_done "$ROOT/baseline-chain-v1"

if [[ ! -f "$SELECTION" ]]; then
  run_step freeze-selection "$PY" -u scripts/revision_select_validation.py \
    --run "crosssteer_source=$ROOT/validation-grid-v1-crosssteer_source" \
    --run "target_calibrated=$ROOT/validation-grid-v1-target_calibrated" \
    --run "caa_target_prompt_final=$ROOT/validation-grid-v1-caa_target_prompt_final" \
    --run "actadd_target_prompt_final=$ROOT/validation-grid-v1-actadd_target_prompt_final" \
    --run "sparse_caa_coordinate_10pct=$ROOT/validation-grid-v1-sparse_caa_coordinate_10pct" \
    --run "matched_norm_random=$ROOT/validation-grid-v1-matched_norm_random" \
    --run "sae_sparse_activation=$ROOT/validation-grid-v1-sae_sparse_activation" \
    --run "negative_crosssteer_source=$ROOT/validation-grid-v1-negative_crosssteer_source" \
    --out "$SELECTION"
fi
[[ -f "$SELECTION" ]] || fail "selection artifact is missing"

mapfile -t METHODS < <(eligible_methods)
if [[ ${#METHODS[@]} -eq 0 ]]; then
  log "selection has no behavior-eligible Qwen method; locked test is prohibited"
  mkdir -p "$LOCKED_ROOT" "$REPORT_ROOT"
  touch "$LOCKED_ROOT/NO_ELIGIBLE_METHODS"
  touch "$CHAIN_DIR/DONE"
  exit 0
fi

mkdir -p "$LOCKED_ROOT" "$REPORT_ROOT"
log "launching one locked Qwen run for every eligible method: ${METHODS[*]}"
PIDS=()
idx=0
for method in "${METHODS[@]}"; do
  out="$LOCKED_ROOT/$method"
  if [[ -f "$out/DONE" ]]; then
    continue
  fi
  device="cuda:$((idx % 2))"
  mkdir -p "$out"
  "$PY" -u scripts/revision_launch_locked_selected.py \
    --selection "$SELECTION" --method "$method" \
    --target-model "$QWEN_MODEL" --target-source "$MODEL_SOURCE" --target-device "$device" \
    --out "$out" --execute >"$out/run.log" 2>&1 &
  PIDS+=("$!")
  idx=$((idx + 1))
  if [[ ${#PIDS[@]} -eq 2 ]]; then
    wait "${PIDS[0]}" || fail "one Qwen locked method process failed"
    wait "${PIDS[1]}" || fail "one Qwen locked method process failed"
    PIDS=()
  fi
done
for pid in "${PIDS[@]}"; do
  wait "$pid" || fail "one Qwen locked method process failed"
done
for method in "${METHODS[@]}"; do
  [[ -f "$LOCKED_ROOT/$method/DONE" ]] || fail "Qwen locked run incomplete: $method"
done

REPORT_ARGS=(--selection "$SELECTION" --out "$REPORT_ROOT")
for method in "${METHODS[@]}"; do REPORT_ARGS+=(--run "$method=$LOCKED_ROOT/$method"); done
run_step build-locked-report "$PY" -u scripts/revision_build_locked_report.py "${REPORT_ARGS[@]}"
[[ -f "$REPORT_ROOT/locked_report_manifest.json" ]] || fail "complete Qwen locked report is missing"

if [[ " ${METHODS[*]} " != *" target_calibrated "* ]]; then
  log "target-calibrated Qwen method is ineligible; label-efficiency curve is prohibited"
  mkdir -p "$CURVE_ROOT"
  touch "$CURVE_ROOT/NO_TARGET_CALIBRATED_ELIGIBLE"
  touch "$CHAIN_DIR/DONE"
  exit 0
fi

mkdir -p "$VECTOR_ROOT" "$CURVE_ROOT" "$CURVE_REPORT"
if [[ ! -f "$VECTOR_ROOT/manifest.json" ]]; then
  run_step build-label-budget-vectors "$PY" -u scripts/revision_build_label_budget_vectors.py \
    --target-run "$ROOT/revision_gsm8k_qwen_instruct_calibration_100" --out "$VECTOR_ROOT"
fi
[[ -f "$VECTOR_ROOT/manifest.json" ]] || fail "label-budget vector registry missing"
SOURCE_VECTOR=$(source_vector_path)
run_step launch-label-efficiency "$PY" -u scripts/revision_launch_label_efficiency_locked.py \
  --selection "$SELECTION" --source-vector "$SOURCE_VECTOR" --label-vector-root "$VECTOR_ROOT" \
  --target-model "$QWEN_MODEL" --target-source "$MODEL_SOURCE" --target-device cuda:0 \
  --out "$CURVE_ROOT" --execute

CURVE_ARGS=(--plan "$CURVE_ROOT/label_efficiency_locked_launch_manifest.json" --out "$CURVE_REPORT")
while IFS=$'\t' read -r method_path; do CURVE_ARGS+=(--run "$method_path"); done < <(
  "$PY" - "$CURVE_ROOT/label_efficiency_locked_launch_manifest.json" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1]))["methods"]:
    print(f'{item["method"]}={item["out"]}')
PY
)
run_step build-label-efficiency-report "$PY" -u scripts/revision_build_label_efficiency_report.py "${CURVE_ARGS[@]}"
[[ -f "$CURVE_REPORT/label_efficiency_report_manifest.json" ]] || fail "label-efficiency report missing"

log "Qwen frozen locked and label-efficiency chain complete; OOD/long context remain separate"
touch "$CHAIN_DIR/DONE"
