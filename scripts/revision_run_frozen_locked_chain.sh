#!/usr/bin/env bash
# Execute validation selection, one-shot locked evaluations, and label-efficiency
# curves only after the predeclared Qwen and R1 validation chains complete.
#
# This chain does not inspect a score to decide whether to run a baseline.  It
# selects exactly once from completed validation artifacts, evaluates every
# eligible method on the locked set, and builds complete-family reports.  OOD and
# long-context runs remain separate frozen stages because they consume different
# datasets/budgets and must never be confused with in-domain confirmation.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
QWEN_MODEL=${QWEN_MODEL:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct}
R1_MODEL=${R1_MODEL:-/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B}
MODEL_SOURCE=${MODEL_SOURCE:-huggingface}

ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/frozen-locked-chain-v1"
mkdir -p "$CHAIN_DIR"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

fail_if_failed() {
  local path
  for path in "$@"; do
    if [[ -f "$path/FAILURE_REASON" ]]; then
      log "refusing continuation because $path failed: $(cat "$path/FAILURE_REASON")"
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

r1_capability_quarantined() {
  local path=$1
  [[ -f "$path/FAILURE_REASON" ]] && grep -Fq "R1 calibration capability audit failed" "$path/FAILURE_REASON"
}

selection_eligible_methods() {
  local selection=$1
  "$PY" - "$selection" <<'PY'
import json
import sys
payload=json.load(open(sys.argv[1]))
for item in payload["methods"]:
    if item["decision"].get("eligible"):
        print(item["method"])
PY
}

source_vector_path() {
  local selection=$1
  "$PY" - "$selection" <<'PY'
import json
import sys
from pathlib import Path
payload=json.load(open(sys.argv[1]))
entries=[item for item in payload["methods"] if item["method"] == "crosssteer_source"]
if len(entries) != 1:
    raise SystemExit("selection lacks exactly one crosssteer_source entry")
manifest=json.loads((Path(entries[0]["run_dir"]) / "resolved_manifest.json").read_text())
path=manifest["vectors"].get("crosssteer_source")
if not path:
    raise SystemExit("crosssteer_source validation manifest lacks vector path")
print(path)
PY
}

run_locked_target() {
  local tag=$1 selection=$2 model=$3 target_calibration=$4
  local locked_root="$ROOT/locked-gsm8k-${tag}-v1"
  local report_root="$ROOT/locked-report-${tag}-v1"
  mkdir -p "$locked_root" "$report_root"
  local -a methods
  mapfile -t methods < <(selection_eligible_methods "$selection")
  if [[ ${#methods[@]} -eq 0 ]]; then
    log "$tag selection has no eligible method; no locked test is permitted"
    touch "$locked_root/NO_ELIGIBLE_METHODS"
    return 0
  fi

  log "$tag launching one frozen locked run for every eligible method: ${methods[*]}"
  local -a pids=()
  local idx=0 method out device
  for method in "${methods[@]}"; do
    out="$locked_root/$method"
    device="cuda:$((idx % 2))"
    mkdir -p "$out"
    "$PY" -u scripts/revision_launch_locked_selected.py \
      --selection "$selection" --method "$method" \
      --target-model "$model" --target-source "$MODEL_SOURCE" --target-device "$device" \
      --out "$out" --execute >"$out/run.log" 2>&1 &
    pids+=("$!")
    idx=$((idx + 1))
    # Keep at most two model processes resident at once.
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
    [[ -f "$locked_root/$method/DONE" ]] || { log "$tag locked run incomplete: $method"; exit 3; }
  done

  local -a report_args=(--selection "$selection" --out "$report_root")
  for method in "${methods[@]}"; do
    report_args+=(--run "$method=$locked_root/$method")
  done
  log "$tag building complete-family locked report"
  "$PY" -u scripts/revision_build_locked_report.py "${report_args[@]}" >"$report_root/build.log" 2>&1
  [[ -f "$report_root/locked_report_manifest.json" ]] || { log "$tag locked report missing"; exit 4; }

  # The label-efficiency curve is meaningful only if the target-calibrated
  # vector itself passed the predeclared validation eligibility gate.
  if [[ " ${methods[*]} " != *" target_calibrated "* ]]; then
    log "$tag target-calibrated method is ineligible; label-efficiency curve is not permitted"
    mkdir -p "$ROOT/label-efficiency-${tag}-v1"
    touch "$ROOT/label-efficiency-${tag}-v1/NO_TARGET_CALIBRATED_ELIGIBLE"
    return 0
  fi

  local vector_root="$ROOT/label-budget-vectors-${tag}-v1"
  local curve_root="$ROOT/label-efficiency-${tag}-v1"
  local curve_report="$ROOT/label-efficiency-report-${tag}-v1"
  mkdir -p "$vector_root" "$curve_root" "$curve_report"
  if [[ ! -f "$vector_root/manifest.json" ]]; then
    log "$tag building deterministic target-label-budget vectors"
    "$PY" -u scripts/revision_build_label_budget_vectors.py \
      --target-run "$target_calibration" --out "$vector_root" >"$vector_root/build.log" 2>&1
  fi
  [[ -f "$vector_root/manifest.json" ]] || { log "$tag label-budget vector build failed"; exit 5; }
  local source_vector
  source_vector=$(source_vector_path "$selection")
  log "$tag launching frozen target-label-efficiency curve"
  "$PY" -u scripts/revision_launch_label_efficiency_locked.py \
    --selection "$selection" --source-vector "$source_vector" --label-vector-root "$vector_root" \
    --target-model "$model" --target-source "$MODEL_SOURCE" --target-device cuda:0 \
    --out "$curve_root" --execute >"$curve_root/run.log" 2>&1

  local -a curve_args=(--plan "$curve_root/label_efficiency_locked_launch_manifest.json" --out "$curve_report")
  local method_path
  while IFS=$'\t' read -r method_path; do
    curve_args+=(--run "$method_path")
  done < <("$PY" - "$curve_root/label_efficiency_locked_launch_manifest.json" <<'PY'
import json
import sys
payload=json.load(open(sys.argv[1]))
for item in payload["methods"]:
    print(f'{item["method"]}={item["out"]}')
PY
)
  log "$tag building frozen target-label-efficiency report"
  "$PY" -u scripts/revision_build_label_efficiency_report.py "${curve_args[@]}" >"$curve_report/build.log" 2>&1
  [[ -f "$curve_report/label_efficiency_report_manifest.json" ]] || { log "$tag label report missing"; exit 6; }
}

cd "$REPO"
# Prefer the checked-out revision source over a stale editable/wheel install.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export GEOPROBE_RUNS="$RUNS_ROOT"

# Completion, never score sign, releases the next predeclared stage.  A target
# that fails the predeclared calibration-capability gate is quarantined rather
# than allowed to block an otherwise independent eligible target.
wait_done "$ROOT/baseline-chain-v1"
fail_if_failed "$ROOT/baseline-chain-v1"
R1_CHAIN="$ROOT/r1-primary-chain-v1"
R1_AVAILABLE=0
if [[ -f "$R1_CHAIN/DONE" ]]; then
  R1_AVAILABLE=1
elif r1_capability_quarantined "$R1_CHAIN"; then
  log "R1 target is quarantined by its predeclared capability gate: $(cat "$R1_CHAIN/FAILURE_REASON")"
  printf '%s\n' "$(cat "$R1_CHAIN/FAILURE_REASON")" > "$CHAIN_DIR/R1_QUARANTINED_CAPABILITY"
elif [[ -f "$R1_CHAIN/FAILURE_REASON" ]]; then
  fail_if_failed "$R1_CHAIN"
else
  wait_done "$R1_CHAIN"
  R1_AVAILABLE=1
fi

QWEN_SELECTION="$ROOT/selection-qwen-instruct-v1.json"
if [[ ! -f "$QWEN_SELECTION" ]]; then
  log "freezing Qwen-Instruct validation selection"
  "$PY" -u scripts/revision_select_validation.py \
    --run "crosssteer_source=$ROOT/validation-grid-v1-crosssteer_source" \
    --run "target_calibrated=$ROOT/validation-grid-v1-target_calibrated" \
    --run "caa_target_prompt_final=$ROOT/validation-grid-v1-caa_target_prompt_final" \
    --run "actadd_target_prompt_final=$ROOT/validation-grid-v1-actadd_target_prompt_final" \
    --run "sparse_caa_coordinate_10pct=$ROOT/validation-grid-v1-sparse_caa_coordinate_10pct" \
    --run "matched_norm_random=$ROOT/validation-grid-v1-matched_norm_random" \
    --run "sae_sparse_activation=$ROOT/validation-grid-v1-sae_sparse_activation" \
    --run "negative_crosssteer_source=$ROOT/validation-grid-v1-negative_crosssteer_source" \
    --out "$QWEN_SELECTION" >"$CHAIN_DIR/qwen-selection.log" 2>&1
fi
run_locked_target qwen-instruct "$QWEN_SELECTION" "$QWEN_MODEL" "$RUNS_ROOT/revision_gsm8k_qwen_instruct_calibration_100"

if [[ "$R1_AVAILABLE" -eq 1 ]]; then
  R1_SELECTION="$ROOT/selection-r1-v1.json"
  if [[ ! -f "$R1_SELECTION" ]]; then
    log "freezing R1 validation selection"
    "$PY" -u scripts/revision_select_validation.py \
      --run "crosssteer_source=$ROOT/validation-grid-v1-r1-crosssteer_source" \
      --run "target_calibrated=$ROOT/validation-grid-v1-r1-target_calibrated" \
      --run "caa_target_prompt_final=$ROOT/validation-grid-v1-r1-caa_target_prompt_final" \
      --run "actadd_target_prompt_final=$ROOT/validation-grid-v1-r1-actadd_target_prompt_final" \
      --run "sparse_caa_coordinate_10pct=$ROOT/validation-grid-v1-r1-sparse_caa_coordinate_10pct" \
      --run "matched_norm_random=$ROOT/validation-grid-v1-r1-matched_norm_random" \
      --run "sae_sparse_activation=$ROOT/validation-grid-v1-r1-sae_sparse_activation" \
      --run "negative_crosssteer_source=$ROOT/validation-grid-v1-r1-negative_crosssteer_source" \
      --out "$R1_SELECTION" >"$CHAIN_DIR/r1-selection.log" 2>&1
  fi
  run_locked_target r1 "$R1_SELECTION" "$R1_MODEL" "$RUNS_ROOT/revision_gsm8k_r1_target_calibration_100"
else
  mkdir -p "$ROOT/locked-gsm8k-r1-v1" "$ROOT/locked-report-r1-v1"
  touch "$ROOT/locked-gsm8k-r1-v1/QUARANTINED_CAPABILITY"
  touch "$ROOT/locked-report-r1-v1/QUARANTINED_CAPABILITY"
fi

log "frozen selection, locked, and target-label-efficiency chain complete; OOD and long-context launchers remain separately frozen stages"
touch "$CHAIN_DIR/DONE"
