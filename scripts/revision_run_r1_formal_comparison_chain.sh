#!/usr/bin/env bash
# Execute the separately declared R1 official-context formal comparison.
#
# This runner is intentionally isolated from the short Qwen 512-token chain.
# It starts only after the correctness-blind official-context readiness and
# calibration chain has completed, preserves R1's 32k sampled decoder, gives
# every R1 method the same 16-cell validation budget, freezes selection before
# generating locked IDs, and does not use Qwen or R1 scores as a scheduler
# condition.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
R1_MODEL=${R1_MODEL:-/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B}
R1_SOURCE=${R1_SOURCE:-huggingface}
R1_VISIBLE_DEVICE=${R1_VISIBLE_DEVICE:-1}
MATH_SOURCE_RUN=${MATH_SOURCE_RUN:-/root/AI/runs/tacl-revision/revision_gsm8k_qwen_math_source_calibration_100}
ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/r1-formal-comparison-chain-v2"
PARENT_CHAIN="$ROOT/r1-official-context-chain-v2"
REGISTRY="$ROOT/vector-registry-r1-official32k-gsm8k-ids0-99-v2"
POOL="$RUNS_ROOT/revision_gsm8k_r1_official32k_caa_source_k4"
PROMPT_REGISTRY="$ROOT/prompt-contrast-registry-r1-official32k-gsm8k-ids0-99-v2"
RANDOM_REGISTRY="$ROOT/matched-random-registry-r1-official32k-gsm8k-ids0-99-v2"
NEGATIVE_REGISTRY="$ROOT/negative-direction-registry-r1-official32k-gsm8k-ids0-99-v2"
SELECTION="$ROOT/frozen-selection-r1official32k-v2.json"
LOCKED_REPORT="$ROOT/locked-report-r1official32k-v2"

# The SAE-space baseline is deliberately excluded from this R1 matrix: the
# approved 32k compact replay stores exact pooled means, while SAE requires the
# time-indexed state representation.  Qwen remains the primary eight-method SAE
# comparison.  This chain includes every R1-admissible baseline declared in
# revision/R1_FORMAL_COMPARISON_PROTOCOL.md.
METHODS=(
  crosssteer_source
  target_calibrated
  caa_target_prompt_final
  actadd_target_prompt_final
  matched_norm_random
  negative_crosssteer_source
  sparse_caa_coordinate_10pct
)

mkdir -p "$CHAIN_DIR"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$CHAIN_DIR/chain.log"; }
fail() {
  local message=$1
  printf '%s\n' "$message" >"$CHAIN_DIR/FAILURE_REASON"
  log "FAILED: $message"
  exit 1
}
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN_DIR/FAILURE_REASON" ]]; then printf "unexpected R1 formal-comparison failure (exit=%s)\n" "$rc" >"$CHAIN_DIR/FAILURE_REASON"; fi' EXIT
wait_done() {
  local path=$1
  while [[ ! -f "$path/DONE" ]]; do
    [[ -f "$path/FAILURE_REASON" ]] && fail "required predecessor failed: $(cat "$path/FAILURE_REASON")"
    sleep 60
  done
}
require_file() { [[ -f "$1" ]] || fail "missing required artifact: $1"; }
run_step() {
  local name=$1
  shift
  log "starting $name"
  if ! "$@" >"$CHAIN_DIR/$name.log" 2>&1; then
    fail "$name failed; see $CHAIN_DIR/$name.log"
  fi
}
run_grid() {
  local method=$1 vector=$2 out=$3
  mkdir -p "$out"
  if [[ -f "$out/DONE" ]]; then
    log "reusing completed validation grid: $method"
    return
  fi
  [[ ! -f "$out/FAILURE_REASON" ]] || fail "$method already has FAILURE_REASON: $(cat "$out/FAILURE_REASON")"
  run_step "validation-${method}" "$PY" -u scripts/revision_steering_validation_grid.py \
    --vector "$method=$vector" \
    --target-model "$R1_MODEL" --target-source "$R1_SOURCE" --target-device cuda:0 \
    --eval-start 100 --eval-count 100 \
    --layers 8,14,20,24 --alphas 0.025,0.05,0.10,0.20 \
    --schedule constant --injection-mode absolute --max-new-tokens 32768 \
    --do-sample --temperature 0.6 --top-p 0.95 --seed 11241 \
    --out "$out"
  require_file "$out/DONE"
}
selected_methods() {
  "$PY" - "$SELECTION" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
for item in payload["methods"]:
    decision = item["decision"]
    if decision.get("eligible"):
        print(item["method"])
PY
}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export GEOPROBE_RUNS="$RUNS_ROOT"
# Expose physical GPU 1 as cuda:0 so every nested runner records the same
# logical target device without competing with the Qwen OOD/long-context chain.
export CUDA_VISIBLE_DEVICES="$R1_VISIBLE_DEVICE"

wait_done "$PARENT_CHAIN"
require_file "$REGISTRY/manifest.json"
require_file "$REGISTRY/crosssteer_source.pt"
require_file "$REGISTRY/target_calibrated.pt"

if [[ ! -f "$POOL/DONE" ]]; then
  run_step contrast-pool "$PY" -u scripts/revision_generate_contrast_pool.py \
    --config configs/revision_gsm8k_r1_official32k_caa_source_k4.yaml
fi
require_file "$POOL/DONE"

if [[ ! -f "$PROMPT_REGISTRY/manifest.json" ]]; then
  run_step prompt-contrast-registry "$PY" -u scripts/revision_build_prompt_contrast_registry.py \
    --target-run "$POOL" --model "$R1_MODEL" --model-source "$R1_SOURCE" \
    --device cuda:0 --out "$PROMPT_REGISTRY"
fi
require_file "$PROMPT_REGISTRY/manifest.json"
require_file "$PROMPT_REGISTRY/caa_target_prompt_final.pt"
require_file "$PROMPT_REGISTRY/actadd_target_prompt_final.pt"
require_file "$PROMPT_REGISTRY/sparse_caa_coordinate_10pct.pt"

if [[ ! -f "$RANDOM_REGISTRY/manifest.json" ]]; then
  run_step matched-random-registry "$PY" -u scripts/revision_build_matched_random_vector.py \
    --reference "$REGISTRY/crosssteer_source.pt" --seed 11241 --out "$RANDOM_REGISTRY"
fi
if [[ ! -f "$NEGATIVE_REGISTRY/manifest.json" ]]; then
  run_step negative-direction-registry "$PY" -u scripts/revision_build_negative_direction.py \
    --reference "$REGISTRY/crosssteer_source.pt" --out "$NEGATIVE_REGISTRY"
fi
require_file "$RANDOM_REGISTRY/matched_norm_random.pt"
require_file "$NEGATIVE_REGISTRY/negative_crosssteer_source.pt"

run_grid crosssteer_source "$REGISTRY/crosssteer_source.pt" "$ROOT/validation-grid-v2-r1official32k-crosssteer_source"
run_grid target_calibrated "$REGISTRY/target_calibrated.pt" "$ROOT/validation-grid-v2-r1official32k-target_calibrated"
run_grid caa_target_prompt_final "$PROMPT_REGISTRY/caa_target_prompt_final.pt" "$ROOT/validation-grid-v2-r1official32k-caa_target_prompt_final"
run_grid actadd_target_prompt_final "$PROMPT_REGISTRY/actadd_target_prompt_final.pt" "$ROOT/validation-grid-v2-r1official32k-actadd_target_prompt_final"
run_grid matched_norm_random "$RANDOM_REGISTRY/matched_norm_random.pt" "$ROOT/validation-grid-v2-r1official32k-matched_norm_random"
run_grid negative_crosssteer_source "$NEGATIVE_REGISTRY/negative_crosssteer_source.pt" "$ROOT/validation-grid-v2-r1official32k-negative_crosssteer_source"
run_grid sparse_caa_coordinate_10pct "$PROMPT_REGISTRY/sparse_caa_coordinate_10pct.pt" "$ROOT/validation-grid-v2-r1official32k-sparse_caa_coordinate_10pct"

if [[ ! -f "$SELECTION" ]]; then
  run_step select-validation "$PY" -u scripts/revision_select_validation.py \
    --run "crosssteer_source=$ROOT/validation-grid-v2-r1official32k-crosssteer_source" \
    --run "target_calibrated=$ROOT/validation-grid-v2-r1official32k-target_calibrated" \
    --run "caa_target_prompt_final=$ROOT/validation-grid-v2-r1official32k-caa_target_prompt_final" \
    --run "actadd_target_prompt_final=$ROOT/validation-grid-v2-r1official32k-actadd_target_prompt_final" \
    --run "matched_norm_random=$ROOT/validation-grid-v2-r1official32k-matched_norm_random" \
    --run "negative_crosssteer_source=$ROOT/validation-grid-v2-r1official32k-negative_crosssteer_source" \
    --run "sparse_caa_coordinate_10pct=$ROOT/validation-grid-v2-r1official32k-sparse_caa_coordinate_10pct" \
    --expected-cells 16 --out "$SELECTION"
fi
require_file "$SELECTION"

mapfile -t ELIGIBLE < <(selected_methods)
if [[ ${#ELIGIBLE[@]} -eq 0 ]]; then
  printf 'no behavior-eligible R1 official-context method after the frozen validation rule\n' >"$CHAIN_DIR/NO_ELIGIBLE_METHODS"
  log "no behavior-eligible method; locked generation is correctly not launched"
  touch "$CHAIN_DIR/DONE"
  exit 0
fi

for method in "${ELIGIBLE[@]}"; do
  out="$ROOT/locked-r1official32k-v2-${method}"
  if [[ ! -f "$out/DONE" ]]; then
    run_step "locked-${method}" "$PY" -u scripts/revision_launch_locked_selected.py \
      --selection "$SELECTION" --method "$method" \
      --target-model "$R1_MODEL" --target-source "$R1_SOURCE" --target-device cuda:0 \
      --out "$out" --execute
  fi
  require_file "$out/DONE"
done

if [[ ! -f "$LOCKED_REPORT/locked_report_manifest.json" ]]; then
  args=(--selection "$SELECTION" --out "$LOCKED_REPORT")
  for method in "${ELIGIBLE[@]}"; do
    args+=(--run "$method=$ROOT/locked-r1official32k-v2-${method}")
  done
  run_step locked-report "$PY" -u scripts/revision_build_locked_report.py "${args[@]}"
fi
require_file "$LOCKED_REPORT/locked_report_manifest.json"
log "R1 official-context formal comparison complete"
touch "$CHAIN_DIR/DONE"
