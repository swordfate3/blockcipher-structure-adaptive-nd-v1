#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
SEED0_ID="i1_gift64_cross_spn_typed_transfer_r3_65536_seed0"
SEED1_ID="i1_gift64_cross_spn_typed_transfer_r3_65536_seed1"
JOINT_ID="i1_gift64_cross_spn_typed_transfer_r3_65536_joint_seed0_seed1"
MONITOR_ROOT="outputs/remote_results_incomplete/i1_gift64_cross_spn_typed_transfer_r3_65536_monitor"

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
  sync_logs "${SEED0_ID}"
  sync_logs "${SEED1_ID}"

  if compgen -G "${MONITOR_ROOT}/${SEED0_ID}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) seed0_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi
  if compgen -G "${MONITOR_ROOT}/${SEED1_ID}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) seed1_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi

  seed0_ready=false
  seed1_ready=false
  if compgen -G "${MONITOR_ROOT}/${SEED0_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    seed0_ready=true
  fi
  if compgen -G "${MONITOR_ROOT}/${SEED1_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    seed1_ready=true
  fi

  if [[ "${seed0_ready}" == true && "${seed1_ready}" == true ]]; then
    mkdir -p "${MONITOR_ROOT}/${JOINT_ID}"
    scp -r "${REMOTE}:${RUNS_ROOT}/${JOINT_ID}" "${MONITOR_ROOT}/${JOINT_ID}/" \
      >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
    if find "${MONITOR_ROOT}/${JOINT_ID}" -name result_branch_pushed.marker -print -quit | grep -q .; then
      retrieve_verified_seed "${SEED0_ID}" || exit 2
      retrieve_verified_seed "${SEED1_ID}" || exit 2
      retrieve_verified_joint || exit 2
      UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
        >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || exit 3
      echo "$(timestamp) verified_results_retrieved_and_indexed" >> "${MONITOR_ROOT}/monitor.log"
      exit 0
    fi
  fi

  echo "$(timestamp) running seed0_branch=${seed0_ready} seed1_branch=${seed1_ready}" >> "${MONITOR_ROOT}/monitor.log"
  sleep 840
done
