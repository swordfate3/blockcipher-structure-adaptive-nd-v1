#!/usr/bin/env bash
set -uo pipefail

REMOTE="lxy-a6000"
RUN_ID="i2_speck32_hwang_topology_pairs_gpu0_20260717"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
REMOTE_ROOT="${RUNS_ROOT}/${RUN_ID}"
RESULT_REF="refs/remotes/origin/results/${RUN_ID}"
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

retrieve_verified_branch() {
  local staging
  staging=$(mktemp -d /tmp/i2-speck-topology-retrieval.XXXXXX) || return 1
  git fetch origin "refs/heads/results/${RUN_ID}:${RESULT_REF}" \
    >> "${MONITOR_ROOT}/git_fetch.log" 2>> "${MONITOR_ROOT}/git_fetch_stderr.log" \
    || return 1
  git archive --format=tar "${RESULT_REF}" "results_archive/${RUN_ID}" \
    | tar -xf - -C "${staging}" || return 1
  mkdir -p "${DESTINATION}"
  cp -a "${staging}/results_archive/${RUN_ID}/." "${DESTINATION}/" || return 1
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
    retrieve_verified_branch || exit 2
    (
      cd "${DESTINATION}" || exit 1
      sha256sum -c <(sed 's/\r$//' SHA256SUMS)
    ) >> "${MONITOR_ROOT}/hash.log" 2>> "${MONITOR_ROOT}/hash_stderr.log" || exit 2
    touch "${DESTINATION}/retrieved_from_verified_result_branch.marker"
    echo "$(timestamp) verified_result_branch_retrieved" >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi
  echo "$(timestamp) running" >> "${MONITOR_ROOT}/monitor.log"
  sleep 60
done
