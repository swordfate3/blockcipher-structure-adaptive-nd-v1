#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
SEED2_ID="i1_gift64_cross_spn_target_adaptation_r4_65536_seed2"
SEED3_ID="i1_gift64_cross_spn_target_adaptation_r4_65536_seed3"
JOINT_ID="i1_gift64_cross_spn_target_adaptation_r4_65536_joint_seed2_seed3"
MONITOR_ROOT="outputs/remote_results_incomplete/i1_gift64_cross_spn_target_adaptation_r4_65536_monitor"

mkdir -p "${MONITOR_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_logs() {
  local run_id="$1"
  local destination="${MONITOR_ROOT}/${run_id}"
  mkdir -p "${destination}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${run_id}/logs" "${destination}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

retrieve_verified_seed() {
  local run_id="$1"
  local destination="outputs/remote_results"
  mkdir -p "${destination}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${run_id}/source/results_archive/${run_id}" \
    "${destination}/" >> "${MONITOR_ROOT}/scp.log" \
    2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  scp -r "${REMOTE}:${RUNS_ROOT}/${run_id}/checkpoints" \
    "${destination}/${run_id}/" >> "${MONITOR_ROOT}/scp.log" \
    2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  touch "${destination}/${run_id}/retrieved_from_verified_result_branch.marker"
  if [[ ! -f "${destination}/${run_id}/curves.svg" ]]; then
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
      --results "${destination}/${run_id}/results.jsonl" \
      --output "${destination}/${run_id}/curves.svg" \
      --history-csv "${destination}/${run_id}/history.csv" \
      --title "${run_id}" >> "${MONITOR_ROOT}/plot.log" \
      2>> "${MONITOR_ROOT}/plot_stderr.log" || return 1
  fi
}

retrieve_verified_joint() {
  local destination="outputs/remote_results"
  mkdir -p "${destination}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${JOINT_ID}/results_archive/${JOINT_ID}" \
    "${destination}/" >> "${MONITOR_ROOT}/scp.log" \
    2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  touch "${destination}/${JOINT_ID}/retrieved_from_verified_result_branch.marker"
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_ROOT}/monitor.log"
  sync_logs "${SEED2_ID}"
  sync_logs "${SEED3_ID}"

  if compgen -G "${MONITOR_ROOT}/${SEED2_ID}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) seed2_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi
  if compgen -G "${MONITOR_ROOT}/${SEED3_ID}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) seed3_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi

  seed2_ready=false
  seed3_ready=false
  if compgen -G "${MONITOR_ROOT}/${SEED2_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    seed2_ready=true
  fi
  if compgen -G "${MONITOR_ROOT}/${SEED3_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    seed3_ready=true
  fi

  if [[ "${seed2_ready}" == true && "${seed3_ready}" == true ]]; then
    mkdir -p "${MONITOR_ROOT}/${JOINT_ID}"
    scp -r "${REMOTE}:${RUNS_ROOT}/${JOINT_ID}" "${MONITOR_ROOT}/${JOINT_ID}/" \
      >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
    if find "${MONITOR_ROOT}/${JOINT_ID}" -name result_branch_pushed.marker -print -quit | grep -q .; then
      retrieve_verified_seed "${SEED2_ID}" || exit 2
      retrieve_verified_seed "${SEED3_ID}" || exit 2
      retrieve_verified_joint || exit 2
      UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
        >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || exit 3
      echo "$(timestamp) verified_results_retrieved_and_indexed" >> "${MONITOR_ROOT}/monitor.log"
      exit 0
    fi
  fi

  echo "$(timestamp) running seed2_branch=${seed2_ready} seed3_branch=${seed3_ready}" >> "${MONITOR_ROOT}/monitor.log"
  sleep 300
done
