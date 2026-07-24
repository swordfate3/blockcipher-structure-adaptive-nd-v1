#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
SEED0_ID="i1_rtg2a_skinny64_general_gf2_medium_65536_seed0_20260724"
SEED1_ID="i1_rtg2a_skinny64_general_gf2_medium_65536_seed1_20260724"
JOINT_ID="i1_rtg2a_skinny64_general_gf2_medium_65536_joint_seed0_seed1_20260724"
LAUNCH_SCRIPT="launch_i1_rtg2a_skinny64_general_gf2_medium_65536_20260724.cmd"
MONITOR_ROOT="outputs/remote_results_incomplete/i1_rtg2a_skinny64_general_gf2_medium_65536_monitor"
RESULT_ROOT="outputs/remote_results"
SEED1_LAUNCHED_MARKER="${MONITOR_ROOT}/conditional_seed1_launched.marker"

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
  local gate_exit=0
  local destination="${RESULT_ROOT}/${run_id}"
  local staging="${MONITOR_ROOT}/staging_${run_id}_$(date +%s)"
  if [[ -e "${destination}" ]]; then
    if [[ -f "${destination}/retrieved_from_verified_result_branch.marker" \
      && -f "${destination}/validation.local.json" \
      && -f "${destination}/gate.json" ]]; then
      echo "$(timestamp) validated_destination_reused run_id=${run_id}" >> "${MONITOR_ROOT}/monitor.log"
      return 0
    fi
    echo "$(timestamp) incomplete_destination_exists run_id=${run_id}" >> "${MONITOR_ROOT}/monitor.log"
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
    2>> "${MONITOR_ROOT}/readjudication_stderr.log" || gate_exit=$?
  if [[ ! -f "${destination}/gate.json" ]]; then
    return 1
  fi
  touch "${destination}/visual_qa_pending.marker"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
    >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || return 1
  return "${gate_exit}"
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

adjudicate_joint() {
  local destination="${RESULT_ROOT}/${JOINT_ID}"
  local gate_exit=0
  if [[ -e "${destination}" ]]; then
    echo "$(timestamp) joint_destination_exists" >> "${MONITOR_ROOT}/monitor.log"
    return 1
  fi
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-runtime-spn-skinny-medium-joint \
    --run-id "${JOINT_ID}" \
    --seed0-gate "${RESULT_ROOT}/${SEED0_ID}/gate.json" \
    --seed1-gate "${RESULT_ROOT}/${SEED1_ID}/gate.json" \
    --output-root "${destination}" \
    >> "${MONITOR_ROOT}/joint.log" \
    2>> "${MONITOR_ROOT}/joint_stderr.log" || gate_exit=$?
  if [[ ! -f "${destination}/gate.json" ]]; then
    return 1
  fi
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
    >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || return 1
  return "${gate_exit}"
}

seed0_handled=false
seed1_launched=false

if grep -q "conditional_seed1_launched" "${MONITOR_ROOT}/monitor.log"; then
  touch "${SEED1_LAUNCHED_MARKER}"
fi
if [[ -f "${SEED1_LAUNCHED_MARKER}" ]]; then
  seed1_launched=true
fi
if [[ -f "${RESULT_ROOT}/${SEED0_ID}/gate.json" ]]; then
  seed0_handled=true
  if [[ "$(gate_status "${SEED0_ID}")" == "pass" ]]; then
    if [[ "${seed1_launched}" != true ]]; then
      launch_seed1 || exit 4
      seed1_launched=true
      touch "${SEED1_LAUNCHED_MARKER}"
      echo "$(timestamp) conditional_seed1_launched_after_resume" >> "${MONITOR_ROOT}/monitor.log"
    fi
  else
    touch "${MONITOR_ROOT}/seed0_stop.marker"
    echo "$(timestamp) resumed_seed0_gate_did_not_pass_stop" >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi
fi
if [[ -f "${RESULT_ROOT}/${SEED1_ID}/gate.json" ]]; then
  if [[ ! -f "${RESULT_ROOT}/${JOINT_ID}/gate.json" ]]; then
    adjudicate_joint || exit 5
  fi
  touch "${MONITOR_ROOT}/medium_pair_complete.marker"
  echo "$(timestamp) resumed_medium_pair_joint_adjudicated_and_indexed" >> "${MONITOR_ROOT}/monitor.log"
  exit 0
fi

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
      touch "${SEED1_LAUNCHED_MARKER}"
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
    adjudicate_joint || exit 5
    touch "${MONITOR_ROOT}/medium_pair_complete.marker"
    echo "$(timestamp) medium_pair_joint_adjudicated_and_indexed" >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi

  if [[ "${seed1_launched}" == true ]] && compgen -G "${MONITOR_ROOT}/${SEED1_ID}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) seed1_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi

  echo "$(timestamp) running seed0_handled=${seed0_handled} seed1_launched=${seed1_launched}" >> "${MONITOR_ROOT}/monitor.log"
  sleep 300
done
