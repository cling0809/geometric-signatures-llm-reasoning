#!/usr/bin/env bash
# Read-only figure postprocessing for frozen signature-confirmation audits.
# This never opens raw trajectory scores or changes the running generation chain.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
ROOT="$RUNS_ROOT/tacl-revision"
# Keep a postprocessor bound to the exact confirmation-chain attempt it
# observes.  This prevents stale paused markers from a previous launch from
# being mistaken for terminal state of a fresh execution.
SIGNATURE_CHAIN_NAME=${SIGNATURE_CHAIN_NAME:-qwen-signature-confirmation-v1}
SIGNATURE_FIGURE_CHAIN_NAME=${SIGNATURE_FIGURE_CHAIN_NAME:-qwen-signature-figure-postprocess-v1}
CHAIN="$ROOT/$SIGNATURE_FIGURE_CHAIN_NAME"
PARENT="$ROOT/$SIGNATURE_CHAIN_NAME"
mkdir -p "$CHAIN"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$CHAIN/chain.log"; }
fail() { printf '%s\n' "$1" > "$CHAIN/FAILURE_REASON"; log "FAILED: $1"; exit 1; }
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN/FAILURE_REASON" ]]; then printf "unexpected figure postprocess failure (exit=%s)\n" "$rc" > "$CHAIN/FAILURE_REASON"; fi' EXIT

render_if_complete() {
  local family=$1 budget=$2 label=$3
  local audit="$ROOT/signature-confirmation-${family}-gsm8k-b${budget}-v1"
  local output="$ROOT/signature-figures-${family}-gsm8k-b${budget}-v1"
  if [[ -f "$output/DONE" ]]; then return 0; fi
  if [[ -f "$audit/UNQUALIFIED" ]]; then
    mkdir -p "$output"
    printf 'Input/support gate did not qualify this condition; no confirmation figure was rendered.\n' > "$output/UNQUALIFIED"
    log "not rendering unqualified condition: ${family} b${budget}"
    return 0
  fi
  if [[ -f "$audit/DONE" ]]; then
    log "rendering all-cell figure: ${family} b${budget}"
    "$PY" -u scripts/revision_plot_signature_confirmation.py \
      --audit-dir "$audit" --out "$output" --label "$label, ${budget} tokens" \
      > "$output/render.log" 2>&1 || fail "figure render failed: ${family} b${budget}"
    touch "$output/DONE"
  fi
}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

while :; do
  render_if_complete qwen 512 "Qwen matched lineage"
  render_if_complete qwen 2048 "Qwen matched lineage"
  render_if_complete gemma 512 "Gemma matched lineage"
  render_if_complete gemma 2048 "Gemma matched lineage"
  terminal=0
  for family in qwen gemma; do
    for budget in 512 2048; do
      audit="$ROOT/signature-confirmation-${family}-gsm8k-b${budget}-v1"
      [[ -f "$audit/DONE" || -f "$audit/UNQUALIFIED" ]] && terminal=$((terminal + 1))
    done
  done
  if [[ $terminal -eq 4 ]]; then
    touch "$CHAIN/DONE"
    log "all registered signature conditions reached terminal state"
    exit 0
  fi
  if [[ -f "$PARENT/FAILURE_REASON" ]]; then
    fail "signature-confirmation parent failed before all conditions reached terminal state"
  fi
  sleep 120
done
