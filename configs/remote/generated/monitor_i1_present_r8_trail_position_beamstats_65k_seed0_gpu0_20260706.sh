#!/usr/bin/env bash
set -u

RUN_ID="i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"
EXPECTED_ROWS="2"

mkdir -p "${LOCAL_ROOT}" "${MONITOR_DIR}" "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/results" "${LOCAL_ROOT}/checkpoints" "${LOCAL_ROOT}/score_artifacts"
touch "${MONITOR_DIR}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_artifacts() {
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/logs" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/results" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/checkpoints" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/score_artifacts" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_DIR}/monitor.log"
  sync_artifacts

  if compgen -G "${LOCAL_ROOT}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) failed" >> "${MONITOR_DIR}/monitor.log"
    exit 1
  fi

  train_file="${LOCAL_ROOT}/results/train_matrix.jsonl"
  global_stats_scores="${LOCAL_ROOT}/score_artifacts/global_stats_control/models.json"
  trail_position_scores="${LOCAL_ROOT}/score_artifacts/trail_position/models.json"
  result_rows=0
  if [[ -f "${train_file}" ]]; then
    result_rows=$(grep -cve '^[[:space:]]*$' "${train_file}" || true)
  fi

  if [[ "${result_rows}" -ge "${EXPECTED_ROWS}" && -f "${global_stats_scores}" && -f "${trail_position_scores}" ]]; then
    echo "$(timestamp) result_ready" >> "${MONITOR_DIR}/monitor.log"
    echo "$(timestamp) score_artifacts_ready global_stats_control/models.json trail_position/models.json" >> "${MONITOR_DIR}/monitor.log"
    exit 0
  fi

  if compgen -G "${LOCAL_ROOT}/logs/*done.marker" > /dev/null; then
    echo "$(timestamp) completed_missing_or_incomplete_results rows=${result_rows}" >> "${MONITOR_DIR}/monitor.log"
    exit 2
  fi

  echo "$(timestamp) running" >> "${MONITOR_DIR}/monitor.log"
  sleep 840
done
