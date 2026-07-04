#!/usr/bin/env bash
set -u

RUN_ID="i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"
PLAN="configs/experiment/innovation1/innovation1_spn_present_active_auxiliary_r7_262k_seed0.json"
PLAN_DOC="docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md"
REMOTE_CONFIG="configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed0_gpu1_20260703.json"
EXPECTED_ROWS="3"

mkdir -p "${LOCAL_ROOT}" "${MONITOR_DIR}" "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/results"
touch "${MONITOR_DIR}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_artifacts() {
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/logs" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/results" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
}

progress_stamp() {
  if [[ -f "${LOCAL_ROOT}/logs/active_auxiliary_progress.jsonl" ]]; then
    stat -c %Y "${LOCAL_ROOT}/logs/active_auxiliary_progress.jsonl" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

failed_marker_is_stale() {
  local progress_mtime
  progress_mtime="$(progress_stamp)"
  [[ "${progress_mtime}" -gt 0 ]] || return 1
  for marker in "${LOCAL_ROOT}"/logs/*failed.marker; do
    [[ -e "${marker}" ]] || return 1
    marker_mtime="$(stat -c %Y "${marker}" 2>/dev/null || echo 0)"
    if [[ "${marker_mtime}" -ge "${progress_mtime}" ]]; then
      return 1
    fi
  done
  return 0
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_DIR}/monitor.log"
  progress_before="$(progress_stamp)"
  sync_artifacts
  progress_after="$(progress_stamp)"

  if compgen -G "${LOCAL_ROOT}/logs/*failed.marker" > /dev/null; then
    if [[ "${progress_after}" -gt "${progress_before}" ]] || failed_marker_is_stale; then
      echo "$(timestamp) stale_failed_marker_ignored progress_before=${progress_before} progress_after=${progress_after}" >> "${MONITOR_DIR}/monitor.log"
    else
    echo "$(timestamp) failed" >> "${MONITOR_DIR}/monitor.log"
    exit 1
    fi
  fi

  result_file="${LOCAL_ROOT}/results/${RUN_ID}.jsonl"
  result_rows=0
  if [[ -f "${result_file}" ]]; then
    result_rows=$(grep -cve '^[[:space:]]*$' "${result_file}" || true)
  fi

  if [[ "${result_rows}" -ge "${EXPECTED_ROWS}" ]]; then
    echo "$(timestamp) result_ready" >> "${MONITOR_DIR}/monitor.log"
    env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-active-auxiliary \
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
      echo "$(timestamp) postprocess_failed plan=${PLAN}" >> "${MONITOR_DIR}/monitor.log"
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
