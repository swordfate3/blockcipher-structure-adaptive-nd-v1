#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
RUN_ID="i2_speck32_hwang_phase_b_singlekey_gpu0_20260717"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
REMOTE_ROOT="${RUNS_ROOT}/${RUN_ID}"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
DESTINATION="outputs/remote_results/${RUN_ID}"

mkdir -p "${MONITOR_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_logs() {
  mkdir -p "${MONITOR_ROOT}/${RUN_ID}"
  scp -r "${REMOTE}:${REMOTE_ROOT}/logs" "${MONITOR_ROOT}/${RUN_ID}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

retrieve_archive() {
  local staging
  staging=$(mktemp -d /tmp/i2-speck-phase-b-retrieval.XXXXXX) || return 1
  scp -r "${REMOTE}:${REMOTE_ROOT}/source/results_archive/${RUN_ID}" \
    "${staging}/" >> "${MONITOR_ROOT}/scp.log" \
    2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  mkdir -p "${DESTINATION}"
  cp -a "${staging}/${RUN_ID}/." "${DESTINATION}/" || return 1
  rm -rf "${staging}"
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_ROOT}/monitor.log"
  sync_logs
  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) remote_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi
  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/${RUN_ID}_done.marker" > /dev/null; then
    retrieve_archive || exit 2
    (
      cd "${DESTINATION}" || exit 1
      sha256sum -c <(sed 's/\r$//' SHA256SUMS)
    ) >> "${MONITOR_ROOT}/hash.log" 2>> "${MONITOR_ROOT}/hash_stderr.log" || exit 2
    touch "${DESTINATION}/retrieved_from_verified_result_branch.marker"
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
      >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || exit 3
    echo "$(timestamp) verified_result_retrieved_and_indexed" >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi
  echo "$(timestamp) running" >> "${MONITOR_ROOT}/monitor.log"
  sleep 60
done
