#!/usr/bin/env bash
# One-shot correction after the Qwen 512-token Math source calibration failed
# the formal 25% source-budget ceiling.  This script does not inspect a
# validation score before rebuilding the source/target directions.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
MODEL=${MODEL:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct}
MODEL_SOURCE=${MODEL_SOURCE:-huggingface}

ROOT="$RUNS_ROOT/tacl-revision"
V2_CHAIN="$ROOT/qwen-validation-v2-chain"
CHAIN="$ROOT/qwen-calibration-correction-v3-chain"
SOURCE_RUN="$ROOT/revision_gsm8k_qwen_math_source_calibration_100_v2"
TARGET_RUN="$ROOT/revision_gsm8k_qwen_instruct_calibration_100_v2"
SOURCE_AUDIT="$ROOT/qwen-math-source-calibration-v2-audit.json"
TARGET_AUDIT="$ROOT/qwen-instruct-target-calibration-v2-audit.json"
REGISTRY="$ROOT/vector-registry-gsm8k-ids0-99-v2"
RANDOM_REGISTRY="$ROOT/matched-random-registry-qwen-gsm8k-ids0-99-v3"
NEGATIVE_REGISTRY="$ROOT/negative-direction-registry-qwen-gsm8k-ids0-99-v3"
SAE_REGISTRY="$ROOT/sae-sparse-registry-qwen-gsm8k-ids0-99-v3"
CONTRAST_POOL="$RUNS_ROOT/revision_gsm8k_qwen_instruct_caa_source_k4_v2"
SELECTION="$ROOT/selection-qwen-instruct-v3.json"
LOCKED_ROOT="$ROOT/locked-gsm8k-qwen-instruct-v3"
REPORT_ROOT="$ROOT/locked-report-qwen-instruct-v3"
mkdir -p "$CHAIN"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$CHAIN/chain.log"; }
fail() {
  local message=$1
  printf '%s\n' "$message" >"$CHAIN/FAILURE_REASON"
  log "FAILED: $message"
  exit 1
}
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN/FAILURE_REASON" ]]; then printf "unexpected correction-chain failure (exit=%s)\n" "$rc" >"$CHAIN/FAILURE_REASON"; fi' EXIT
wait_gpu_free() {
  local waited=0
  while :; do
    local n
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [[ "$n" -eq 0 ]]; then
      return 0
    fi
    waited=$((waited + 60))
    if [[ $waited -gt 7200 ]]; then
      fail "waited 120m for both GPUs to become free"
    fi
    sleep 60
  done
}
wait_done() {
  local path=$1
  while [[ ! -f "$path/DONE" ]]; do
    [[ -f "$path/FAILURE_REASON" ]] && fail "dependency failed: $path"
    sleep 60
  done
}
run_grid() {
  local method=$1 vector=$2 device=$3 out=$4
  mkdir -p "$out"
  log "starting corrected v3 $method on $device"
  "$PY" -u scripts/revision_steering_validation_grid.py \
    --vector "$method=$vector" \
    --target-model "$MODEL" --target-source "$MODEL_SOURCE" --target-device "$device" \
    --eval-start 100 --eval-count 100 \
    --layers 8,14,20,24 --alphas 0.025,0.05,0.10,0.20 \
    --schedule constant --injection-mode absolute --max-new-tokens 512 \
    --out "$out" >"$out/run.log" 2>&1 &
  echo $! >"$out/PID"
}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export GEOPROBE_RUNS="$ROOT"

# The previous v2 chain must first terminate at the expected score-blind source
# audit gate.  A grid crash or a different selection failure must be diagnosed,
# not silently converted into this corrective protocol.
while [[ ! -f "$V2_CHAIN/FAILURE_REASON" ]]; do
  [[ -f "$V2_CHAIN/DONE" ]] && fail "v2 chain unexpectedly reached DONE; refuse corrective relaunch"
  sleep 60
done
# The parent launcher records a generic failure marker; the score-blind
# selection preflight records the specific quality-gate exception in its log.
# Accept only the exact provenance failure from either artifact, never a generic
# grid crash or a score-derived selection failure.
EXPECTED_GATE='lacks a bound source_calibration_audit|formal selection requires audited CrossSteer'
V2_SELECTION_LOG="$V2_CHAIN/freeze-selection.log"
if ! grep -Eq "$EXPECTED_GATE" "$V2_CHAIN/FAILURE_REASON" \
  && { [[ ! -f "$V2_SELECTION_LOG" ]] || ! grep -Eq "$EXPECTED_GATE" "$V2_SELECTION_LOG"; }; then
  fail "v2 chain did not stop at the expected source-calibration quality gate"
fi
{
  printf 'validated expected v2 source-quality stop\n'
  printf 'failure_reason_path=%s\n' "$V2_CHAIN/FAILURE_REASON"
  printf 'selection_preflight_log=%s\n' "$V2_SELECTION_LOG"
} >"$CHAIN/V2_GATE_CONFIRMED"
log "validated expected v2 source-quality stop"

wait_gpu_free

# Both directions are rebuilt from telemetry-complete calibration runs.  No
# validation or locked answer is opened in this phase.
if [[ ! -f "$SOURCE_RUN/DONE" ]]; then
  log "starting one-shot Math source calibration at 2048 tokens on cuda:0"
  "$PY" -u scripts/extract_trajectories.py \
    --config configs/revision_gsm8k_qwen_math_source_calibration_100_v2.yaml \
    >"$CHAIN/source-extract.log" 2>&1 &
  source_pid=$!
else
  source_pid=''
fi
if [[ ! -f "$TARGET_RUN/DONE" ]]; then
  log "starting provenance-complete target calibration at 512 tokens on cuda:1"
  "$PY" -u scripts/extract_trajectories.py \
    --config configs/revision_gsm8k_qwen_instruct_calibration_100_v2.yaml \
    >"$CHAIN/target-extract.log" 2>&1 &
  target_pid=$!
else
  target_pid=''
fi
[[ -z "$source_pid" ]] || wait "$source_pid" || fail "source calibration extraction failed"
[[ -z "$target_pid" ]] || wait "$target_pid" || fail "target calibration extraction failed"
[[ -f "$SOURCE_RUN/DONE" && -f "$TARGET_RUN/DONE" ]] || fail "corrective calibration run incomplete"

if [[ ! -f "$SOURCE_AUDIT" ]]; then
  "$PY" -u scripts/revision_audit_calibration.py \
    --run "qwen_math_source=$SOURCE_RUN" --expected-id-range 0:100 \
    --max-new-tokens 2048 --min-class-count 10 --max-budget-hit-rate 0.25 \
    --require-generation-provenance --out "$SOURCE_AUDIT" \
    >"$CHAIN/source-audit.log" 2>&1 || fail "source calibration audit failed"
fi
if [[ ! -f "$TARGET_AUDIT" ]]; then
  "$PY" -u scripts/revision_audit_calibration.py \
    --run "qwen_instruct_target=$TARGET_RUN" --expected-id-range 0:100 \
    --max-new-tokens 512 --min-class-count 10 --max-budget-hit-rate 0.25 \
    --require-generation-provenance --out "$TARGET_AUDIT" \
    >"$CHAIN/target-audit.log" 2>&1 || fail "target calibration audit failed"
fi

if [[ ! -f "$REGISTRY/manifest.json" ]]; then
  "$PY" -u scripts/revision_build_vector_registry.py \
    --source-run "$SOURCE_RUN" --target-run "$TARGET_RUN" \
    --source-calibration-audit "$SOURCE_AUDIT" \
    --target-calibration-audit "$TARGET_AUDIT" \
    --out "$REGISTRY" >"$CHAIN/build-registry.log" 2>&1 || fail "audited vector registry build failed"
fi
[[ -f "$REGISTRY/manifest.json" ]] || fail "audited vector registry missing"

# CrossSteer, target calibration, the matched-random control and the exact
# negative control all depend on the corrected source direction. SAE also uses
# target-calibration trajectory states, so it is rebuilt from the provenance-
# complete target run. Only the independent CAA/ActAdd/sparse-CAA v2 controls
# are reused verbatim. Every v3 grid regenerates the deterministic no-steering
# baseline, which the complete selection audit must verify byte-for-byte.
if [[ ! -f "$RANDOM_REGISTRY/manifest.json" ]]; then
  "$PY" -u scripts/revision_build_matched_random_vector.py \
    --reference "$REGISTRY/crosssteer_source.pt" --out "$RANDOM_REGISTRY" \
    >"$CHAIN/build-random-control.log" 2>&1 || fail "matched-random control build failed"
fi
if [[ ! -f "$NEGATIVE_REGISTRY/manifest.json" ]]; then
  "$PY" -u scripts/revision_build_negative_direction.py \
    --reference "$REGISTRY/crosssteer_source.pt" --out "$NEGATIVE_REGISTRY" \
    >"$CHAIN/build-negative-control.log" 2>&1 || fail "negative-direction control build failed"
fi
if [[ ! -f "$SAE_REGISTRY/manifest.json" ]]; then
  "$PY" -u scripts/revision_build_sae_sparse_registry.py \
    --calibration-run "$TARGET_RUN" --calibration-audit "$TARGET_AUDIT" \
    --contrast-run "$CONTRAST_POOL" \
    --model "$MODEL" --model-source "$MODEL_SOURCE" --device cuda:0 --out "$SAE_REGISTRY" \
    >"$CHAIN/build-sae-control.log" 2>&1 || fail "SAE sparse control build failed"
fi
CROSS_OUT="$ROOT/validation-grid-v3-crosssteer_source"
TARGET_OUT="$ROOT/validation-grid-v3-target_calibrated"
RANDOM_OUT="$ROOT/validation-grid-v3-matched_norm_random"
NEGATIVE_OUT="$ROOT/validation-grid-v3-negative_crosssteer_source"
SAE_OUT="$ROOT/validation-grid-v3-sae_sparse_activation"
if [[ ! -f "$CROSS_OUT/DONE" || ! -f "$TARGET_OUT/DONE" ]]; then
  wait_gpu_free
  run_grid crosssteer_source "$REGISTRY/crosssteer_source.pt" cuda:0 "$CROSS_OUT"
  run_grid target_calibrated "$REGISTRY/target_calibrated.pt" cuda:1 "$TARGET_OUT"
  wait "$(cat "$CROSS_OUT/PID")" || fail "v3 CrossSteer grid failed"
  wait "$(cat "$TARGET_OUT/PID")" || fail "v3 target-calibrated grid failed"
fi
if [[ ! -f "$RANDOM_OUT/DONE" || ! -f "$NEGATIVE_OUT/DONE" ]]; then
  wait_gpu_free
  run_grid matched_norm_random "$RANDOM_REGISTRY/matched_norm_random.pt" cuda:0 "$RANDOM_OUT"
  run_grid negative_crosssteer_source "$NEGATIVE_REGISTRY/negative_crosssteer_source.pt" cuda:1 "$NEGATIVE_OUT"
  wait "$(cat "$RANDOM_OUT/PID")" || fail "v3 matched-random grid failed"
  wait "$(cat "$NEGATIVE_OUT/PID")" || fail "v3 negative-direction grid failed"
fi
if [[ ! -f "$SAE_OUT/DONE" ]]; then
  wait_gpu_free
  run_grid sae_sparse_activation "$SAE_REGISTRY/sae_sparse_activation.pt" cuda:0 "$SAE_OUT"
  wait "$(cat "$SAE_OUT/PID")" || fail "v3 SAE grid failed"
fi
[[ -f "$CROSS_OUT/DONE" && -f "$TARGET_OUT/DONE" && -f "$RANDOM_OUT/DONE" && -f "$NEGATIVE_OUT/DONE" && -f "$SAE_OUT/DONE" ]] \
  || fail "corrected direction/control grids incomplete"

if [[ ! -f "$SELECTION" ]]; then
  "$PY" -u scripts/revision_select_validation.py \
    --run "crosssteer_source=$CROSS_OUT" \
    --run "target_calibrated=$TARGET_OUT" \
    --run "caa_target_prompt_final=$ROOT/validation-grid-v2-caa_target_prompt_final" \
    --run "actadd_target_prompt_final=$ROOT/validation-grid-v2-actadd_target_prompt_final" \
    --run "sparse_caa_coordinate_10pct=$ROOT/validation-grid-v2-sparse_caa_coordinate_10pct" \
    --run "matched_norm_random=$RANDOM_OUT" \
    --run "sae_sparse_activation=$SAE_OUT" \
    --run "negative_crosssteer_source=$NEGATIVE_OUT" \
    --out "$SELECTION" >"$CHAIN/freeze-selection.log" 2>&1 || fail "v3 selection preflight failed"
fi
[[ -f "$SELECTION" ]] || fail "corrected selection artifact missing"

mapfile -t METHODS < <("$PY" - "$SELECTION" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1]))["methods"]:
    if item["decision"].get("eligible"):
        print(item["method"])
PY
)
if [[ ${#METHODS[@]} -eq 0 ]]; then
  log "no behavior-eligible method after audited v3 selection; locked test prohibited"
  mkdir -p "$LOCKED_ROOT" "$REPORT_ROOT"
  touch "$LOCKED_ROOT/NO_ELIGIBLE_METHODS"
  touch "$CHAIN/DONE"
  exit 0
fi

mkdir -p "$LOCKED_ROOT" "$REPORT_ROOT"
PIDS=()
idx=0
for method in "${METHODS[@]}"; do
  out="$LOCKED_ROOT/$method"
  [[ -f "$out/DONE" ]] && continue
  mkdir -p "$out"
  device="cuda:$((idx % 2))"
  "$PY" -u scripts/revision_launch_locked_selected.py \
    --selection "$SELECTION" --method "$method" \
    --target-model "$MODEL" --target-source "$MODEL_SOURCE" --target-device "$device" \
    --out "$out" --execute >"$out/run.log" 2>&1 &
  PIDS+=("$!")
  idx=$((idx + 1))
  if [[ ${#PIDS[@]} -eq 2 ]]; then
    wait "${PIDS[0]}" || fail "one v3 locked method failed"
    wait "${PIDS[1]}" || fail "one v3 locked method failed"
    PIDS=()
  fi
done
for pid in "${PIDS[@]}"; do wait "$pid" || fail "one v3 locked method failed"; done
for method in "${METHODS[@]}"; do [[ -f "$LOCKED_ROOT/$method/DONE" ]] || fail "locked method incomplete: $method"; done

REPORT_ARGS=(--selection "$SELECTION" --out "$REPORT_ROOT")
for method in "${METHODS[@]}"; do REPORT_ARGS+=(--run "$method=$LOCKED_ROOT/$method"); done
"$PY" -u scripts/revision_build_locked_report.py "${REPORT_ARGS[@]}" >"$CHAIN/build-locked-report.log" 2>&1 \
  || fail "v3 locked report failed"
[[ -f "$REPORT_ROOT/locked_report_manifest.json" ]] || fail "v3 locked report missing"
touch "$CHAIN/DONE"
log "audited Qwen v3 correction + selection + locked chain complete"
