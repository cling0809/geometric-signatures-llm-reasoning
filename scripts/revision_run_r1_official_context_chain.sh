#!/usr/bin/env bash
# Run the R1 model-valid extraction path.  This chain deliberately stops before
# any validation score is inspected or steering hyperparameter is selected.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
# Expose only the physical R1 GPU; configs continue to address it as cuda:0.
R1_VISIBLE_DEVICE=${R1_VISIBLE_DEVICE:-1}
MATH_SOURCE_RUN=${MATH_SOURCE_RUN:-/root/AI/runs/tacl-revision/revision_gsm8k_qwen_math_source_calibration_100}

ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/r1-official-context-chain-v2"
CAPABILITY_RUN="$RUNS_ROOT/revision_gsm8k_r1_official32k_capability20"
CALIBRATION_RUN="$RUNS_ROOT/revision_gsm8k_r1_official32k_calibration100"
CAPABILITY_AUDIT="$CHAIN_DIR/capability20-readiness.json"
CALIBRATION_AUDIT="$CHAIN_DIR/calibration100-capability-audit.json"
REGISTRY="$ROOT/vector-registry-r1-official32k-gsm8k-ids0-99-v2"
mkdir -p "$CHAIN_DIR"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$CHAIN_DIR/chain.log"; }
fail() {
  local message=$1
  printf '%s\n' "$message" >"$CHAIN_DIR/FAILURE_REASON"
  log "FAILED: $message"
  exit 1
}
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN_DIR/FAILURE_REASON" ]]; then printf "unexpected R1 official-context chain failure (exit=%s)\n" "$rc" >"$CHAIN_DIR/FAILURE_REASON"; fi' EXIT
run_step() {
  local name=$1
  shift
  log "starting $name"
  if ! "$@" >"$CHAIN_DIR/$name.log" 2>&1; then
    fail "$name failed; see $CHAIN_DIR/$name.log"
  fi
}
wait_r1_gpu_free() {
  # This is an operational resource gate only.  R1 has its own frozen decoder,
  # calibration, selection, and evidence path, so a Qwen score or failure must
  # never decide whether the R1 experiment exists.
  local waited=0
  while nvidia-smi -i "$R1_VISIBLE_DEVICE" --query-compute-apps=pid --format=csv,noheader \
      2>/dev/null | grep -q '[0-9]'; do
    waited=$((waited + 60))
    if [[ $waited -gt 3600 ]]; then
      fail "waited 60m for physical R1 GPU $R1_VISIBLE_DEVICE to become free"
    fi
    sleep 60
  done
}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export GEOPROBE_RUNS="$RUNS_ROOT"

# The physical-GPU gate prevents concurrent allocation but carries no scientific
# dependency: unlike the superseded Qwen v1 chain, it cannot propagate a Qwen
# decode/configuration failure into this separately preregistered R1 protocol.
wait_r1_gpu_free
export CUDA_VISIBLE_DEVICES="$R1_VISIBLE_DEVICE"

if [[ ! -f "$CAPABILITY_RUN/DONE" ]]; then
  run_step capability20-extract "$PY" -u scripts/revision_extract_pooled_calibration.py \
    --config configs/revision_gsm8k_r1_official32k_capability20.yaml
fi
[[ -f "$CAPABILITY_RUN/DONE" ]] || fail "20-problem R1 official-context extraction has no DONE marker"

run_step capability20-readiness "$PY" -u scripts/revision_audit_pooled_readiness.py \
  --run "$CAPABILITY_RUN" --expected-id-range 0:20 --max-new-tokens 32768 \
  --max-budget-hit-rate 0.25 --max-severe-repetition-rate 0.05 \
  --severe-repetition-fraction 0.95 --out "$CAPABILITY_AUDIT"

if [[ ! -f "$CALIBRATION_RUN/DONE" ]]; then
  run_step calibration100-extract "$PY" -u scripts/revision_extract_pooled_calibration.py \
    --config configs/revision_gsm8k_r1_official32k_calibration100.yaml
fi
[[ -f "$CALIBRATION_RUN/DONE" ]] || fail "100-problem R1 official-context calibration has no DONE marker"

run_step calibration100-audit "$PY" -u scripts/revision_audit_calibration.py \
  --run "qwen_math_source=$MATH_SOURCE_RUN" --run "r1_target=$CALIBRATION_RUN" \
  --expected-id-range 0:100 --max-new-tokens 32768 --min-class-count 10 \
  --max-budget-hit-rate 0.25 --out "$CALIBRATION_AUDIT"

if [[ ! -f "$REGISTRY/manifest.json" ]]; then
  run_step build-vector-registry "$PY" -u scripts/revision_build_vector_registry.py \
    --source-run "$MATH_SOURCE_RUN" --target-run "$CALIBRATION_RUN" --out "$REGISTRY"
fi
[[ -f "$REGISTRY/manifest.json" ]] || fail "R1 official-context vector registry is missing"
log "R1 official-context calibration chain complete; formal paired validation remains a separate frozen protocol"
touch "$CHAIN_DIR/DONE"
