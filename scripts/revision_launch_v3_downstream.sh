#!/usr/bin/env bash
# Launch the v3 downstream evidence chains after the v3 correction chain DONE.
#
# The v3 correction chain writes selection-qwen-instruct-v3.json and
# locked-gsm8k-qwen-instruct-v3, but it does not itself start the OOD /
# target-label-efficiency / long-context stages.  This launcher waits for the
# correction chain to finish, waits for a free GPU, then starts the three v3
# downstream chains serially (OOD → label-efficiency → long-context), matching
# the dependencies already encoded in the v3 scripts.
#
# Fail-closed: it never runs a stage whose predecessor failed, and never starts
# until the v3 correction chain has a DONE (or FAILURE) marker.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-qwen-v2}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/qwen-v3-downstream-launcher"
mkdir -p "$CHAIN_DIR"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$CHAIN_DIR/chain.log"; }
fail() { printf '%s\n' "$1" >"$CHAIN_DIR/FAILURE_REASON"; log "FAILED: $1"; exit 1; }
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN_DIR/FAILURE_REASON" ]]; then printf "unexpected v3 downstream launcher failure (exit=%s)\n" "$rc" >"$CHAIN_DIR/FAILURE_REASON"; fi' EXIT

CORRECTION_CHAIN="$ROOT/qwen-calibration-correction-v3-chain"
# The diagnostic signature chain and the OOD chain share GPU 0.  Waiting only
# for the correction chain creates a race: the signature launcher may start an
# independent Gemma pair after Qwen finishes while OOD is already loading the
# target model.  Make signature completion an explicit scientific/resource
# prerequisite, not an implicit timing assumption.
SIGNATURE_CHAIN="${SIGNATURE_CHAIN:-$ROOT/qwen-signature-confirmation-v3}"
SELECTION="$ROOT/selection-qwen-instruct-v3.json"
OOD_CHAIN="$ROOT/qwen-ood-chain-v3"
LABEL_CHAIN="$ROOT/qwen-label-efficiency-chain-v3"
LONG_CHAIN="$ROOT/qwen-long-context-chain-v3"

cd "$REPO"

# Wait for the v3 correction chain to reach a terminal state.
log "waiting for v3 correction chain"
while true; do
  if [[ -f "$CORRECTION_CHAIN/FAILURE_REASON" ]]; then
    fail "v3 correction chain failed: $(cat "$CORRECTION_CHAIN/FAILURE_REASON")"
  fi
  if [[ -f "$CORRECTION_CHAIN/DONE" ]]; then
    break
  fi
  sleep 60
done
log "v3 correction chain complete"

# Do not let downstream GPU work overlap the independent diagnostic signature
# confirmation.  A signature failure is terminal for the diagnostic chain but
# still does not authorize OOD to race with an unresolved/partially audited
# signature workload.
log "waiting for signature confirmation chain: $SIGNATURE_CHAIN"
while true; do
  if [[ -f "$SIGNATURE_CHAIN/FAILURE_REASON" ]]; then
    fail "signature confirmation chain failed: $(cat "$SIGNATURE_CHAIN/FAILURE_REASON")"
  fi
  if [[ -f "$SIGNATURE_CHAIN/DONE" ]]; then
    break
  fi
  sleep 60
done
log "signature confirmation chain complete"

[[ -f "$SELECTION" ]] || fail "v3 selection artifact missing"

# Wait for a free GPU before starting OOD (the v3 grids and the correction
# chain's own workers may still hold the GPUs when the chain writes DONE).
while :; do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [[ "$n" -eq 0 ]]; then
    log "GPUs free"
    break
  fi
  sleep 60
done

# Serialise the three downstream stages; each v3 script blocks on its own
# predecessor, so launching them sequentially and letting them wait is safe.
log "starting v3 OOD chain"
bash "$REPO/scripts/revision_run_qwen_ood_chain_v3.sh" >>"$CHAIN_DIR/ood.log" 2>&1 &
OOD_PID=$!
[[ -f "$OOD_CHAIN/FAILURE_REASON" ]] && fail "v3 OOD chain failed"
wait "$OOD_PID" || fail "v3 OOD chain exited non-zero"

log "starting v3 label-efficiency chain"
bash "$REPO/scripts/revision_run_qwen_label_efficiency_chain_v3.sh" >>"$CHAIN_DIR/label.log" 2>&1 &
LABEL_PID=$!
wait "$LABEL_PID" || fail "v3 label-efficiency chain exited non-zero"

log "starting v3 long-context chain"
bash "$REPO/scripts/revision_run_qwen_long_context_chain_v3.sh" >>"$CHAIN_DIR/long.log" 2>&1 &
LONG_PID=$!
wait "$LONG_PID" || fail "v3 long-context chain exited non-zero"

touch "$CHAIN_DIR/DONE"
log "v3 downstream evidence chains complete"
