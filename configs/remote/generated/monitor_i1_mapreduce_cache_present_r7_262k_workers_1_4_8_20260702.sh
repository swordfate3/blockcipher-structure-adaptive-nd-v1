#!/usr/bin/env bash
set -u

RUN_ID="i1_mapreduce_cache_present_r7_262k_workers_1_4_8_20260702"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"
SUMMARY="${LOCAL_ROOT}/results/dataset_cache_bench/summary.json"

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

  if [[ -f "${SUMMARY}" ]]; then
    echo "$(timestamp) summary_ready" >> "${MONITOR_DIR}/monitor.log"
    exit 0
  fi

  if compgen -G "${LOCAL_ROOT}/logs/*done.marker" > /dev/null; then
    echo "$(timestamp) done_without_summary" >> "${MONITOR_DIR}/monitor.log"
    exit 2
  fi

  echo "$(timestamp) running" >> "${MONITOR_DIR}/monitor.log"
  sleep 300
done
