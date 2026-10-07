#!/usr/bin/env bash
# Fresh, score-blind confirmation of Claim 1 (trajectory signatures).
#
# This chain waits for the higher-priority frozen steering family to finish and
# then regenerates the matrix inputs under complete extraction provenance.  It
# never selects metrics, physical layers, prompt variants, budgets or model
# subsets from a diagnostic result.  A cell which fails the input/support gate
# is recorded as unqualified; it is not regenerated with a more favorable setup.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
ROOT="$RUNS_ROOT/tacl-revision"
# A restart must not append to or reinterpret a paused chain
# from an earlier operational attempt.  Defaults preserve the registered v1
# artifact name; a fresh launcher supplies a distinct chain name.
SIGNATURE_CHAIN_NAME=${SIGNATURE_CHAIN_NAME:-qwen-signature-confirmation-v1}
CHAIN="$ROOT/$SIGNATURE_CHAIN_NAME"
PRIORITY_CHAIN="$ROOT/qwen-calibration-correction-v3-chain"
mkdir -p "$CHAIN"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$CHAIN/chain.log"; }
fail() {
  printf '%s\n' "$1" > "$CHAIN/FAILURE_REASON"
  log "FAILED: $1"
  exit 1
}
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN/FAILURE_REASON" ]]; then printf "unexpected signature-confirmation failure (exit=%s)\n" "$rc" > "$CHAIN/FAILURE_REASON"; fi' EXIT

wait_for_priority_chain() {
  while [[ ! -f "$PRIORITY_CHAIN/DONE" && ! -f "$PRIORITY_CHAIN/FAILURE_REASON" ]]; do
    sleep 120
  done
  if [[ -f "$PRIORITY_CHAIN/FAILURE_REASON" ]]; then
    # A steering-chain implementation failure must be diagnosed and restarted
    # before this lower-priority diagnostic workload is allowed to take both
    # GPUs.  Preserve partial trajectory files for deterministic resumption;
    # do not turn the parent failure into an implicit scheduling decision.
    touch "$CHAIN/PAUSED_FOR_CORRECTION_FAILURE"
    log "steering priority chain failed; signature confirmation paused pending correction restart"
    exit 0
  fi
  log "steering priority chain completed; starting diagnostic confirmation after GPUs drain"
}

wait_for_model_cache() {
  local path=$1 label=$2 waited=0
  while [[ ! -f "$path/config.json" ]]; do
    waited=$((waited + 60))
    if [[ $waited -gt 86400 ]]; then fail "timed out waiting for required model cache: $label"; fi
    log "waiting for required model cache: $label"
    sleep 60
  done
}

wait_gpu_free() {
  local waited=0
  while :; do
    local n
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [[ "$n" -eq 0 ]]; then return 0; fi
    waited=$((waited + 60))
    if [[ $waited -gt 172800 ]]; then fail "waited 48h for GPUs to drain"; fi
    sleep 60
  done
}

run_pair() {
  local config_a=$1 config_b=$2 label=$3
  local exp_a exp_b
  exp_a=$(awk '/^exp_id:/ {print $2; exit}' "$config_a")
  exp_b=$(awk '/^exp_id:/ {print $2; exit}' "$config_b")
  if [[ -f "$ROOT/$exp_a/DONE" && -f "$ROOT/$exp_b/DONE" ]]; then
    log "reusing completed pair ${label}"
    return 0
  fi
  log "starting ${label}: $(basename "$config_a") + $(basename "$config_b")"
  "$PY" -u scripts/extract_trajectories.py --config "$config_a" >"$CHAIN/$(basename "${config_a%.yaml}").log" 2>&1 &
  local pid_a=$!
  "$PY" -u scripts/extract_trajectories.py --config "$config_b" >"$CHAIN/$(basename "${config_b%.yaml}").log" 2>&1 &
  local pid_b=$!
  wait "$pid_a" || fail "generation failed: $config_a"
  wait "$pid_b" || fail "generation failed: $config_b"
}

build_metrics() {
  local run=$1
  "$PY" - "$run" <<'PY'
import sys
from pathlib import Path
from geoprobe.analysis import write_run_metrics
write_run_metrics(Path(sys.argv[1]))
PY
}

run_audit() {
  # With `set -u`, Bash expands all assignments in a single `local` command
  # before binding the earlier names.  Keep dependent values on separate lines
  # so the first completed generation family can reach its audit instead of
  # failing with an unbound `family` variable.
  local family=$1
  local budget=$2
  local out="$ROOT/signature-confirmation-${family}-gsm8k-b${budget}-v1"
  shift 2
  local -a runs=("$@")
  if [[ -f "$out/DONE" ]]; then
    log "reusing completed diagnostic audit: $out"
    return 0
  fi
  local -a args=()
  local pair
  for pair in "${runs[@]}"; do args+=(--run "$pair"); done
  if "$PY" -u scripts/revision_diagnostic_generalization.py \
    "${args[@]}" \
    --out "$out" \
    --partition-salt "tacl-11241-signature-confirmation-v1-${family}-gsm8k-b${budget}" \
    --max-new-tokens "$budget" --max-truncation-rate 0.25 \
    --n-bootstrap 2000 --n-permutations 10000 --seed 11241 \
    --relative-depth-bins 29 --min-per-class 10 \
    >"$CHAIN/${family}-b${budget}-audit.log" 2>&1
  then
    touch "$out/DONE"
    log "completed diagnostic audit: ${family} b${budget}"
  else
    # Underpowered or provenance-ineligible is an empirical boundary, not an
    # excuse to discard the remaining preregistered conditions.
    touch "$out/UNQUALIFIED"
    printf 'The frozen input/support gate did not qualify this condition. See %s.\n'       "$CHAIN/${family}-b${budget}-audit.log" > "$out/STATUS.md"
    log "diagnostic condition unqualified: ${family} b${budget}; continuing registered cells"
  fi
}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export GEOPROBE_RUNS="$ROOT"

wait_for_priority_chain
wait_for_model_cache /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen2.5-Coder-1.5B-Instruct
wait_for_model_cache /root/.cache/modelscope/hub/models/LLM-Research/gemma-2-2b Gemma-2-2B
wait_for_model_cache /root/.cache/modelscope/hub/models/LLM-Research/gemma-2-2b-it Gemma-2-2B-IT
wait_gpu_free

# Qwen primary within-lineage family: four fixed post-training roles.
for budget in 512 2048; do
  run_pair "configs/revision_signature_qwen_base_gsm8k_300_b${budget}_v1.yaml" \
           "configs/revision_signature_qwen_instruct_gsm8k_300_b${budget}_v1.yaml" \
           "qwen-${budget}-base-instruct"
  run_pair "configs/revision_signature_qwen_math_gsm8k_300_b${budget}_v1.yaml" \
           "configs/revision_signature_qwen_coder_gsm8k_300_b${budget}_v1.yaml" \
           "qwen-${budget}-math-coder"
  for role in qwen_base qwen_instruct qwen_math qwen_coder; do
    build_metrics "$ROOT/revision_signature_${role}_gsm8k_300_b${budget}_v1" \
      >"$CHAIN/${role}-b${budget}-metrics.log" 2>&1 || fail "metric build failed: ${role} b${budget}"
  done
  run_audit qwen "$budget" \
    "Base=$ROOT/revision_signature_qwen_base_gsm8k_300_b${budget}_v1" \
    "Instruct=$ROOT/revision_signature_qwen_instruct_gsm8k_300_b${budget}_v1" \
    "Math-Instruct=$ROOT/revision_signature_qwen_math_gsm8k_300_b${budget}_v1" \
    "Coder-Instruct=$ROOT/revision_signature_qwen_coder_gsm8k_300_b${budget}_v1"
done

# A predeclared independent matched base/instruct replication. It is analysed
# separately and can only strengthen a Qwen result; it cannot calibrate Qwen.
for budget in 512 2048; do
  run_pair "configs/revision_signature_gemma_base_gsm8k_300_b${budget}_v1.yaml" \
           "configs/revision_signature_gemma_instruct_gsm8k_300_b${budget}_v1.yaml" \
           "gemma-${budget}-base-instruct"
  for role in gemma_base gemma_instruct; do
    build_metrics "$ROOT/revision_signature_${role}_gsm8k_300_b${budget}_v1" \
      >"$CHAIN/${role}-b${budget}-metrics.log" 2>&1 || fail "metric build failed: ${role} b${budget}"
  done
  run_audit gemma "$budget" \
    "Base=$ROOT/revision_signature_gemma_base_gsm8k_300_b${budget}_v1" \
    "Instruct=$ROOT/revision_signature_gemma_instruct_gsm8k_300_b${budget}_v1"
done

touch "$CHAIN/DONE"
log "signature confirmation chain complete"
