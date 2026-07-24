#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_COMMIT="${1:-}"
RTG3_SEED0_ID="i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725"
RTG3_SEED1_ID="i1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_20260725"
RTG3_JOINT_ID="i1_rtg3a_skinny64_general_gf2_formal_1000000_joint_seed0_seed1_20260725"
U3_RUN_ID="i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_20260725"
SEED0_ROOT="${PROJECT_ROOT}/outputs/remote_results/${RTG3_SEED0_ID}"
SEED1_ROOT="${PROJECT_ROOT}/outputs/remote_results/${RTG3_SEED1_ID}"
JOINT_ROOT="${PROJECT_ROOT}/outputs/remote_results_incomplete/${RTG3_JOINT_ID}"
SEED0_SUCCESSOR_ROOT="${PROJECT_ROOT}/outputs/remote_results_incomplete/i1_rtg3a_seed1_after_seed0_20260725_monitor"
SEED1_MONITOR_ROOT="${PROJECT_ROOT}/outputs/remote_results_incomplete/${RTG3_SEED1_ID}_monitor"
READINESS_ROOT="${PROJECT_ROOT}/outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725"
AUTH_ROOT="${PROJECT_ROOT}/outputs/local_readiness/i1_uknit64_runtime_e4_recurrent_window_r5_u3_authorization_20260725_${SOURCE_COMMIT:0:12}"
PLAN="${PROJECT_ROOT}/configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1.csv"
RUN_ROOT="${PROJECT_ROOT}/outputs/local_diagnostic/${U3_RUN_ID}"
MONITOR_ROOT="${PROJECT_ROOT}/outputs/local_diagnostic/${U3_RUN_ID}_successor_monitor"

PROTECTED_PATHS=(
  "configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1.csv"
  "configs/runtime/spn/uknit64.json"
  "scripts/train"
  "scripts/check-runtime-spn-uknit-u3-launch"
  "scripts/check-runtime-spn-recurrent-window-readiness"
  "scripts/gate-runtime-spn-recurrent-window"
  "src/blockcipher_nd/cli/check_runtime_spn_uknit_u3_launch.py"
  "src/blockcipher_nd/cli/check_runtime_spn_recurrent_window_readiness.py"
  "src/blockcipher_nd/cli/gate_runtime_spn_recurrent_window.py"
  "src/blockcipher_nd/data"
  "src/blockcipher_nd/engine"
  "src/blockcipher_nd/models/structure/spn/runtime_parameterized.py"
  "src/blockcipher_nd/models/structure/spn/runtime_structure.py"
  "src/blockcipher_nd/planning/matrix.py"
  "src/blockcipher_nd/registry/model_families/spn.py"
  "src/blockcipher_nd/tasks/innovation1/runtime_spn_recurrent_window.py"
  "src/blockcipher_nd/tasks/innovation1/runtime_spn_recurrent_window_readiness.py"
  "src/blockcipher_nd/tasks/innovation1/runtime_spn_uknit_u3_launch.py"
  "src/blockcipher_nd/training"
)

mkdir -p "${MONITOR_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

record_stop() {
  local reason="$1"
  printf '%s\n' "${reason}" > "${MONITOR_ROOT}/u3_not_launched.marker"
  echo "$(timestamp) u3_not_launched reason=${reason}" >> "${MONITOR_ROOT}/monitor.log"
  exit 0
}

source_is_frozen() {
  [[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || return 1
  git cat-file -e "${SOURCE_COMMIT}^{commit}" 2>/dev/null || return 1
  git merge-base --is-ancestor "${SOURCE_COMMIT}" origin/main 2>/dev/null || return 1
  git diff --quiet "${SOURCE_COMMIT}..HEAD" -- "${PROTECTED_PATHS[@]}" || return 1
  [[ -z "$(git status --porcelain -- "${PROTECTED_PATHS[@]}")" ]] || return 1
}

required_joint_evidence_ready() {
  [[ -f "${SEED0_ROOT}/gate.local.json" ]] || return 1
  [[ -f "${SEED0_ROOT}/visual_qa_passed.marker" ]] || return 1
  [[ -f "${SEED1_ROOT}/gate.local.json" ]] || return 1
  [[ -f "${SEED1_ROOT}/visual_qa_passed.marker" ]] || return 1
  [[ -f "${JOINT_ROOT}/gate.json" ]] || return 1
  [[ -f "${JOINT_ROOT}/validation.json" ]] || return 1
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

if [[ -f "${MONITOR_ROOT}/u3_complete.marker" && -f "${RUN_ROOT}/gate.json" ]]; then
  echo "$(timestamp) u3_already_complete" >> "${MONITOR_ROOT}/monitor.log"
  exit 0
fi

if [[ -e "${RUN_ROOT}" && ! -f "${RUN_ROOT}/gate.json" ]]; then
  touch "${MONITOR_ROOT}/partial_run_requires_audit.marker"
  echo "$(timestamp) partial_run_requires_audit" >> "${MONITOR_ROOT}/monitor.log"
  exit 8
fi

echo "$(timestamp) waiting_for_verified_rtg3_joint_evidence" >> "${MONITOR_ROOT}/monitor.log"
while ! required_joint_evidence_ready; do
  if [[ -f "${SEED0_SUCCESSOR_ROOT}/seed1_not_launched.marker" ]]; then
    reason="$(tr -d '\r\n' < "${SEED0_SUCCESSOR_ROOT}/seed1_not_launched.marker")"
    record_stop "rtg3_seed1_not_launched_${reason:-unknown}"
  fi
  if [[ -f "${SEED1_MONITOR_ROOT}/remote_failed.marker" ]]; then
    record_stop "rtg3_seed1_remote_failed"
  fi
  sleep 300
done

if ! source_is_frozen; then
  touch "${MONITOR_ROOT}/source_drift_before_authorization.marker"
  echo "$(timestamp) source_drift_before_authorization" >> "${MONITOR_ROOT}/monitor.log"
  exit 7
fi

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-runtime-spn-uknit-u3-launch \
  --seed0-root "${SEED0_ROOT}" \
  --rtg3-joint-root "${JOINT_ROOT}" \
  --readiness-root "${READINESS_ROOT}" \
  --plan "${PLAN}" \
  --repository "${PROJECT_ROOT}" \
  --output-root "${AUTH_ROOT}" \
  >> "${MONITOR_ROOT}/authorization.log" \
  2>> "${MONITOR_ROOT}/authorization_stderr.log"
authorization_exit=$?

authorization_state="$(python -c 'import json,sys; g=json.load(open(sys.argv[1], encoding="utf-8")); print("|".join((str(g.get("status")), str(g.get("decision")), str(g.get("execution_authorized")))))' "${AUTH_ROOT}/gate.json" 2>> "${MONITOR_ROOT}/authorization_stderr.log" || true)"
echo "$(timestamp) authorization_exit=${authorization_exit} state=${authorization_state}" >> "${MONITOR_ROOT}/monitor.log"
if [[ "${authorization_exit}" -ne 0 || "${authorization_state}" != "pass|innovation1_runtime_spn_uknit_u3_execution_authorized|True" ]]; then
  if [[ "${authorization_state}" == hold\|* ]]; then
    record_stop "authorization_${authorization_state}"
  fi
  touch "${MONITOR_ROOT}/authorization_failed.marker"
  exit 9
fi

if ! source_is_frozen; then
  touch "${MONITOR_ROOT}/source_drift_before_training.marker"
  echo "$(timestamp) source_drift_before_training" >> "${MONITOR_ROOT}/monitor.log"
  exit 7
fi

echo "$(timestamp) u3_training_started" >> "${MONITOR_ROOT}/monitor.log"
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan "${PLAN}" \
  --device cpu \
  --dataset-cache-root "${RUN_ROOT}/cache" \
  --dataset-cache-chunk-size 1024 \
  --dataset-cache-workers 1 \
  --checkpoint-output-dir "${RUN_ROOT}/checkpoints" \
  --progress-output "${RUN_ROOT}/progress.jsonl" \
  --output "${RUN_ROOT}/results.jsonl" \
  >> "${MONITOR_ROOT}/training.log" \
  2>> "${MONITOR_ROOT}/training_stderr.log" || {
    touch "${MONITOR_ROOT}/training_failed.marker"
    echo "$(timestamp) u3_training_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 10
  }

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan "${PLAN}" \
  --results "${RUN_ROOT}/results.jsonl" \
  --expected-rows 10 \
  --output "${RUN_ROOT}/plan_validation.json" \
  >> "${MONITOR_ROOT}/validation.log" \
  2>> "${MONITOR_ROOT}/validation_stderr.log" || {
    touch "${MONITOR_ROOT}/validation_failed.marker"
    exit 11
  }

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results "${RUN_ROOT}/results.jsonl" \
  --output "${RUN_ROOT}/curves.svg" \
  --history-csv "${RUN_ROOT}/history.csv" \
  --title "创新1 U3：uKNIT 五轮异构双窗口运行时 SPN 复验" \
  --validation-only \
  >> "${MONITOR_ROOT}/plot.log" \
  2>> "${MONITOR_ROOT}/plot_stderr.log" || {
    touch "${MONITOR_ROOT}/plot_failed.marker"
    exit 12
  }

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-runtime-spn-recurrent-window \
  --run-id "${U3_RUN_ID}" \
  --run-root "${RUN_ROOT}" \
  >> "${MONITOR_ROOT}/gate.log" \
  2>> "${MONITOR_ROOT}/gate_stderr.log" || {
    touch "${MONITOR_ROOT}/result_gate_failed.marker"
    exit 13
  }

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
  >> "${MONITOR_ROOT}/index.log" \
  2>> "${MONITOR_ROOT}/index_stderr.log" || {
    touch "${MONITOR_ROOT}/index_failed.marker"
    exit 14
  }

touch "${RUN_ROOT}/visual_qa_pending.marker"
touch "${MONITOR_ROOT}/u3_result_ready_visual_qa_pending.marker"
echo "$(timestamp) u3_result_ready_indexed_visual_qa_pending" >> "${MONITOR_ROOT}/monitor.log"

while [[ ! -f "${RUN_ROOT}/visual_qa_passed.marker" ]]; do
  if [[ -f "${RUN_ROOT}/visual_qa_failed.marker" ]]; then
    touch "${MONITOR_ROOT}/visual_qa_failed.marker"
    echo "$(timestamp) visual_qa_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 15
  fi
  sleep 300
done

touch "${MONITOR_ROOT}/u3_complete.marker"
echo "$(timestamp) u3_visual_qa_passed_complete" >> "${MONITOR_ROOT}/monitor.log"
