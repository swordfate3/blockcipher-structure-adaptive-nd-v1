#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_COMMIT="${1:-}"
U3_RUN_ID="i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_20260725"
U4_RUN_ID="i1_uknit64_runtime_e4_window_swap_u4_20260725"
U3_ROOT="${PROJECT_ROOT}/outputs/local_diagnostic/${U3_RUN_ID}"
U3_MONITOR_ROOT="${PROJECT_ROOT}/outputs/local_diagnostic/${U3_RUN_ID}_successor_monitor"
U4_ROOT="${PROJECT_ROOT}/outputs/local_audits/${U4_RUN_ID}"
MONITOR_ROOT="${PROJECT_ROOT}/outputs/local_audits/${U4_RUN_ID}_successor_monitor"
PLAN="${PROJECT_ROOT}/configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1.csv"

PROTECTED_PATHS=(
  "configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1.csv"
  "configs/runtime/spn/uknit64.json"
  "docs/experiments/innovation1-uknit-runtime-e4-window-swap-u4-conditional-plan.md"
  "scripts/audit-runtime-spn-recurrent-window-counterfactual"
  "src/blockcipher_nd/cli/audit_runtime_spn_recurrent_window_counterfactual.py"
  "src/blockcipher_nd/data/differential"
  "src/blockcipher_nd/evaluation/result_index.py"
  "src/blockcipher_nd/models/structure/spn/runtime_parameterized.py"
  "src/blockcipher_nd/models/structure/spn/runtime_structure.py"
  "src/blockcipher_nd/registry/model_families/spn.py"
  "src/blockcipher_nd/tasks/innovation1/runtime_spn_recurrent_window.py"
  "src/blockcipher_nd/tasks/innovation1/runtime_spn_recurrent_window_counterfactual.py"
  "src/blockcipher_nd/training/metrics.py"
)

mkdir -p "${MONITOR_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

record_stop() {
  local reason="$1"
  printf '%s\n' "${reason}" > "${MONITOR_ROOT}/u4_not_run.marker"
  echo "$(timestamp) u4_not_run reason=${reason}" >> "${MONITOR_ROOT}/monitor.log"
  exit 0
}

source_is_frozen() {
  [[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || return 1
  git cat-file -e "${SOURCE_COMMIT}^{commit}" 2>/dev/null || return 1
  git merge-base --is-ancestor "${SOURCE_COMMIT}" origin/main 2>/dev/null || return 1
  git diff --quiet "${SOURCE_COMMIT}..HEAD" -- "${PROTECTED_PATHS[@]}" || return 1
  [[ -z "$(git status --porcelain -- "${PROTECTED_PATHS[@]}")" ]] || return 1
}

u3_evidence_ready() {
  [[ -f "${U3_MONITOR_ROOT}/u3_complete.marker" ]] || return 1
  [[ -f "${U3_ROOT}/results.jsonl" ]] || return 1
  [[ -f "${U3_ROOT}/gate.json" ]] || return 1
  [[ -f "${U3_ROOT}/validation.json" ]] || return 1
  [[ -f "${U3_ROOT}/plan_validation.json" ]] || return 1
  [[ -f "${U3_ROOT}/visual_qa_passed.marker" ]] || return 1
}

if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <pushed-source-commit>" >&2
  exit 6
fi

cd "${PROJECT_ROOT}" || exit 2
if ! source_is_frozen; then
  touch "${MONITOR_ROOT}/source_gate_failed.marker"
  echo "$(timestamp) source_gate_failed" >> "${MONITOR_ROOT}/monitor.log"
  exit 7
fi

if [[ -f "${MONITOR_ROOT}/u4_complete.marker" && -f "${U4_ROOT}/gate.json" ]]; then
  echo "$(timestamp) u4_already_complete" >> "${MONITOR_ROOT}/monitor.log"
  exit 0
fi

if [[ -e "${U4_ROOT}" && ! -f "${U4_ROOT}/gate.json" ]]; then
  touch "${MONITOR_ROOT}/partial_audit_requires_review.marker"
  echo "$(timestamp) partial_audit_requires_review" >> "${MONITOR_ROOT}/monitor.log"
  exit 8
fi

echo "$(timestamp) waiting_for_verified_u3_evidence" >> "${MONITOR_ROOT}/monitor.log"
while ! u3_evidence_ready; do
  if [[ -f "${U3_MONITOR_ROOT}/u3_not_launched.marker" ]]; then
    reason="$(tr -d '\r\n' < "${U3_MONITOR_ROOT}/u3_not_launched.marker")"
    record_stop "u3_not_launched_${reason:-unknown}"
  fi
  for failure in \
    authorization_failed \
    training_failed \
    validation_failed \
    plot_failed \
    result_gate_failed \
    index_failed \
    visual_qa_failed; do
    if [[ -f "${U3_MONITOR_ROOT}/${failure}.marker" ]]; then
      record_stop "u3_${failure}"
    fi
  done
  sleep 300
done

if ! source_is_frozen; then
  touch "${MONITOR_ROOT}/source_drift_before_audit.marker"
  echo "$(timestamp) source_drift_before_audit" >> "${MONITOR_ROOT}/monitor.log"
  exit 7
fi

echo "$(timestamp) u4_audit_started" >> "${MONITOR_ROOT}/monitor.log"
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/audit-runtime-spn-recurrent-window-counterfactual \
  --run-id "${U4_RUN_ID}" \
  --u3-root "${U3_ROOT}" \
  --plan "${PLAN}" \
  --output-root "${U4_ROOT}" \
  --device cpu \
  --batch-size 256 \
  >> "${MONITOR_ROOT}/audit.log" \
  2>> "${MONITOR_ROOT}/audit_stderr.log"
audit_exit=$?
if [[ "${audit_exit}" -ne 0 ]]; then
  touch "${MONITOR_ROOT}/audit_failed.marker"
  echo "$(timestamp) u4_audit_failed exit=${audit_exit}" >> "${MONITOR_ROOT}/monitor.log"
  exit 9
fi

gate_state="$(python -c 'import json,sys; g=json.load(open(sys.argv[1], encoding="utf-8")); print("|".join((str(g.get("status")), str(g.get("decision")))))' "${U4_ROOT}/gate.json" 2>> "${MONITOR_ROOT}/audit_stderr.log" || true)"
if [[ "${gate_state}" != pass\|* && "${gate_state}" != hold\|* ]]; then
  touch "${MONITOR_ROOT}/gate_state_invalid.marker"
  echo "$(timestamp) u4_gate_state_invalid state=${gate_state}" >> "${MONITOR_ROOT}/monitor.log"
  exit 10
fi

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
  >> "${MONITOR_ROOT}/index.log" \
  2>> "${MONITOR_ROOT}/index_stderr.log" || {
    touch "${MONITOR_ROOT}/index_failed.marker"
    exit 11
  }

touch "${U4_ROOT}/visual_qa_pending.marker"
touch "${MONITOR_ROOT}/u4_result_ready_visual_qa_pending.marker"
echo "$(timestamp) u4_result_ready state=${gate_state}" >> "${MONITOR_ROOT}/monitor.log"

while [[ ! -f "${U4_ROOT}/visual_qa_passed.marker" ]]; do
  if [[ -f "${U4_ROOT}/visual_qa_failed.marker" ]]; then
    touch "${MONITOR_ROOT}/visual_qa_failed.marker"
    echo "$(timestamp) u4_visual_qa_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 12
  fi
  sleep 300
done

touch "${MONITOR_ROOT}/u4_complete.marker"
echo "$(timestamp) u4_visual_qa_passed_complete" >> "${MONITOR_ROOT}/monitor.log"
