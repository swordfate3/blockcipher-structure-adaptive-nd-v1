#!/usr/bin/env bash
set -u

RUN_ID="i1_bit_transition_spectrum_r7_262k_seed1_gpu1_20260702"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"
PLAN="configs/experiment/innovation1/innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1.json"
PLAN_DOC="docs/experiments/innovation1-bit-transition-spectrum-plan.md"
EXPECTED_ROWS="4"

mkdir -p "${LOCAL_ROOT}" "${MONITOR_DIR}" "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/results"
touch "${MONITOR_DIR}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_artifacts() {
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/logs" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/results" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_DIR}/monitor.log"
  sync_artifacts

  if compgen -G "${LOCAL_ROOT}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) failed" >> "${MONITOR_DIR}/monitor.log"
    exit 1
  fi

  result_file="${LOCAL_ROOT}/results/${RUN_ID}.jsonl"
  result_rows=0
  if [[ -f "${result_file}" ]]; then
    result_rows=$(grep -cve '^[[:space:]]*$' "${result_file}" || true)
  fi

  if [[ "${result_rows}" -ge "${EXPECTED_ROWS}" ]]; then
    echo "$(timestamp) result_ready" >> "${MONITOR_DIR}/monitor.log"
    env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-transition-spectrum \
      --plan "${PLAN}" \
      --results "${result_file}" \
      --output-dir "${LOCAL_ROOT}" \
      --run-id "${RUN_ID}" \
      --expected-rows "${EXPECTED_ROWS}" \
      --update-plan-doc "${PLAN_DOC}" \
      > "${MONITOR_DIR}/postprocess.log" 2> "${MONITOR_DIR}/postprocess_stderr.log"
    postprocess_status=$?
    if [[ "${postprocess_status}" -eq 0 ]]; then
      echo "$(timestamp) postprocess_done" >> "${MONITOR_DIR}/monitor.log"
    else
      echo "$(timestamp) postprocess_failed" >> "${MONITOR_DIR}/monitor.log"
    fi
    exit "${postprocess_status}"
  fi

  if compgen -G "${LOCAL_ROOT}/logs/*done.marker" > /dev/null; then
    echo "$(timestamp) completed_missing_or_incomplete_results rows=${result_rows}" >> "${MONITOR_DIR}/monitor.log"
    exit 2
  fi

  echo "$(timestamp) running" >> "${MONITOR_DIR}/monitor.log"
  sleep 840
done
