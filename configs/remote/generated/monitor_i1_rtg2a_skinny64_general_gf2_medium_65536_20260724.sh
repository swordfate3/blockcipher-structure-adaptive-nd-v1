#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
SEED0_ID="i1_rtg2a_skinny64_general_gf2_medium_65536_seed0_20260724"
SEED1_ID="i1_rtg2a_skinny64_general_gf2_medium_65536_seed1_20260724"
LAUNCH_SCRIPT="launch_i1_rtg2a_skinny64_general_gf2_medium_65536_20260724.cmd"
MONITOR_ROOT="outputs/remote_results_incomplete/i1_rtg2a_skinny64_general_gf2_medium_65536_monitor"
RESULT_ROOT="outputs/remote_results"

mkdir -p "${MONITOR_ROOT}" "${RESULT_ROOT}"
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

branch_exists() {
  local run_id="$1"
  git ls-remote --exit-code origin "refs/heads/results/${run_id}" \
    >> "${MONITOR_ROOT}/branch.log" 2>> "${MONITOR_ROOT}/branch_stderr.log"
}

retrieve_and_readjudicate() {
  local run_id="$1"
  local seed="$2"
  local destination="${RESULT_ROOT}/${run_id}"
  local staging="${MONITOR_ROOT}/staging_${run_id}_$(date +%s)"
  if [[ -e "${destination}" ]]; then
    echo "$(timestamp) destination_exists run_id=${run_id}" >> "${MONITOR_ROOT}/monitor.log"
    return 1
  fi
  mkdir -p "${staging}"
  branch_exists "${run_id}" || return 1
  scp -r "${REMOTE}:${RUNS_ROOT}/${run_id}/source/results_archive/${run_id}" \
    "${staging}/" >> "${MONITOR_ROOT}/scp.log" \
    2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  (
    cd "${staging}/${run_id}" || exit 1
    sed 's/\r$//' SHA256SUMS | sha256sum -c -
  ) >> "${MONITOR_ROOT}/sha256.log" 2>> "${MONITOR_ROOT}/sha256_stderr.log" || return 1
  cp -a "${staging}/${run_id}" "${destination}" || return 1
  touch "${destination}/retrieved_from_verified_result_branch.marker"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
    --plan "${destination}/plan.csv" \
    --results "${destination}/results.jsonl" \
    --expected-rows 3 \
    --output "${destination}/validation.local.json" \
    >> "${MONITOR_ROOT}/readjudication.log" \
    2>> "${MONITOR_ROOT}/readjudication_stderr.log" || return 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-runtime-spn-skinny-medium \
    --run-id "${run_id}" \
    --run-root "${destination}" \
    --seed "${seed}" \
    --progress "${destination}/progress.jsonl" \
    >> "${MONITOR_ROOT}/readjudication.log" \
    2>> "${MONITOR_ROOT}/readjudication_stderr.log" || return 1
  touch "${destination}/visual_qa_pending.marker"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
    >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || return 1
}

gate_status() {
  local run_id="$1"
  python -c "import json,pathlib; print(json.loads((pathlib.Path(r'${RESULT_ROOT}/${run_id}')/'gate.json').read_text(encoding='utf-8'))['status'])"
}

launch_seed1() {
  local revision
  revision="$(tr -d '\r\n' < "${RESULT_ROOT}/${SEED0_ID}/git_revision.txt")"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE}" \
    "cmd.exe /c ${RUNS_ROOT}\\${SEED0_ID}\\source\\configs\\remote\\generated\\${LAUNCH_SCRIPT} ${revision} 1 0" \
    >> "${MONITOR_ROOT}/seed1_launch.log" \
    2>> "${MONITOR_ROOT}/seed1_launch_stderr.log"
}

seed0_handled=false
seed1_launched=false

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_ROOT}/monitor.log"
  sync_logs "${SEED0_ID}"
  if [[ "${seed1_launched}" == true ]]; then
    sync_logs "${SEED1_ID}"
  fi

  if [[ "${seed0_handled}" != true ]] && compgen -G "${MONITOR_ROOT}/${SEED0_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    retrieve_and_readjudicate "${SEED0_ID}" 0 || exit 2
    seed0_handled=true
    if [[ "$(gate_status "${SEED0_ID}")" == "pass" ]]; then
      launch_seed1 || exit 4
      seed1_launched=true
      echo "$(timestamp) conditional_seed1_launched" >> "${MONITOR_ROOT}/monitor.log"
    else
      echo "$(timestamp) seed0_gate_did_not_pass_stop" >> "${MONITOR_ROOT}/monitor.log"
      touch "${MONITOR_ROOT}/seed0_stop.marker"
      exit 0
    fi
  fi

  if [[ "${seed0_handled}" != true ]] && compgen -G "${MONITOR_ROOT}/${SEED0_ID}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) seed0_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi

  if [[ "${seed1_launched}" == true ]] && compgen -G "${MONITOR_ROOT}/${SEED1_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    retrieve_and_readjudicate "${SEED1_ID}" 1 || exit 2
    touch "${MONITOR_ROOT}/medium_pair_complete.marker"
    echo "$(timestamp) medium_pair_retrieved_and_indexed" >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi

  if [[ "${seed1_launched}" == true ]] && compgen -G "${MONITOR_ROOT}/${SEED1_ID}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) seed1_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi

  echo "$(timestamp) running seed0_handled=${seed0_handled} seed1_launched=${seed1_launched}" >> "${MONITOR_ROOT}/monitor.log"
  sleep 300
done
