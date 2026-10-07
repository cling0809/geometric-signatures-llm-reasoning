#!/usr/bin/env bash
# Run the frozen target-label-efficiency curve for the corrected v2 Qwen
# pipeline.  Supersedes the inline label-efficiency block of the superseded
# v1 locked chain (which gated on the invalidated v1 selection).  Reviewer B.3
# / editor E3 require the 0/5/10/20/50/100 target-label curve; this step keeps
# that requirement on the v2 evidence path.
set -euo pipefail

REPO=${REPO:-/root/AI/geoprobe-tacl-revision}
RUNS_ROOT=${RUNS_ROOT:-/root/AI/runs}
PY=${PY:-/root/miniconda3/envs/geoprobe/bin/python}
QWEN_MODEL=${QWEN_MODEL:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct}
MODEL_SOURCE=${MODEL_SOURCE:-huggingface}

ROOT="$RUNS_ROOT/tacl-revision"
CHAIN_DIR="$ROOT/qwen-label-efficiency-chain-v2"
SELECTION="$ROOT/selection-qwen-instruct-v2.json"
LOCKED_ROOT="$ROOT/locked-gsm8k-qwen-instruct-v2"
LOCKED_REPORT="$ROOT/locked-report-qwen-instruct-v2"
VECTOR_ROOT="$ROOT/label-budget-vectors-qwen-instruct-v2"
CURVE_ROOT="$ROOT/label-efficiency-qwen-instruct-v2"
CURVE_REPORT="$ROOT/label-efficiency-report-qwen-instruct-v2"
TARGET_CALIBRATION="$ROOT/revision_gsm8k_qwen_instruct_calibration_100"
mkdir -p "$CHAIN_DIR"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$CHAIN_DIR/chain.log"; }
fail() { printf '%s\n' "$1" >"$CHAIN_DIR/FAILURE_REASON"; log "FAILED: $1"; exit 1; }
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$CHAIN_DIR/FAILURE_REASON" ]]; then printf "unexpected v2 label-efficiency chain failure (exit=%s)\n" "$rc" >"$CHAIN_DIR/FAILURE_REASON"; fi' EXIT
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
source_vector_path() {
  "$PY" - "$SELECTION" <<'PY'
import json
import sys
from pathlib import Path
payload = json.load(open(sys.argv[1]))
entries = [item for item in payload["methods"] if item["method"] == "crosssteer_source"]
if len(entries) != 1:
    raise SystemExit("selection lacks exactly one crosssteer_source entry")
manifest = json.loads((Path(entries[0]["run_dir"]) / "resolved_manifest.json").read_text())
vector = manifest["vectors"].get("crosssteer_source")
if not vector:
    raise SystemExit("crosssteer_source vector missing from selected manifest")
print(vector)
PY
}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# Serialise GPU 0 descendants.  OOD has no access to target-label curve
# outcomes; this is a resource gate only, not a scientific dependency.
wait_done "$ROOT/qwen-validation-v2-chain"
wait_done "$ROOT/qwen-ood-chain-v2"
[[ -f "$SELECTION" ]] || fail "v2 selection artifact missing"
[[ -f "$LOCKED_REPORT/locked_report_manifest.json" ]] || fail "v2 locked report missing"

# Only the full target-calibrated method can anchor the label-efficiency curve.
if ! "$PY" - "$SELECTION" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if not any(item["method"] == "target_calibrated" and item["decision"].get("eligible")
           for item in payload["methods"]):
    raise SystemExit(1)
PY
then
  log "target-calibrated Qwen method ineligible; label-efficiency curve is prohibited"
  mkdir -p "$CURVE_ROOT"
  touch "$CURVE_ROOT/NO_TARGET_CALIBRATED_ELIGIBLE"
  touch "$CHAIN_DIR/DONE"
  exit 0
fi

mkdir -p "$VECTOR_ROOT" "$CURVE_ROOT" "$CURVE_REPORT"
if [[ ! -f "$VECTOR_ROOT/manifest.json" ]]; then
  run_step build-label-budget-vectors "$PY" -u scripts/revision_build_label_budget_vectors.py \
    --target-run "$TARGET_CALIBRATION" --out "$VECTOR_ROOT"
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

log "Qwen v2 frozen locked and label-efficiency chain complete; OOD/long-context remain separate"
touch "$CHAIN_DIR/DONE"
