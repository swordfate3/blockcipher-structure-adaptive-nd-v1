#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
RUN_ID="i1_rtg2b_skinny64_general_gf2_scale_262144_seed1_20260724"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
REMOTE_LAUNCH_ROOT="G:\\lxy\\launcher-clones\\${RUN_ID}"
REPO_URL="git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git"
LAUNCH_SCRIPT="configs\\remote\\generated\\launch_i1_rtg2b_skinny64_general_gf2_scale_262144_seed1_20260724.cmd"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
RESULT_ROOT="outputs/remote_results"
DESTINATION="${RESULT_ROOT}/${RUN_ID}"
SOURCE_COMMIT="${1:-}"
LAUNCH_GATE_PATH="${2:-outputs/local_readiness/i1_rtg2b_skinny64_general_gf2_scale_262144_seed1_launch_gate_20260724/gate.json}"
LAUNCHED_MARKER="${MONITOR_ROOT}/remote_launch_completed.marker"

if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <pushed-source-commit> [launch-gate.json]" >&2
  exit 6
fi
if [[ ! -f "${LAUNCH_GATE_PATH}" ]]; then
  echo "missing launch gate: ${LAUNCH_GATE_PATH}" >&2
  exit 7
fi
python -c "import json,pathlib,sys; g=json.loads(pathlib.Path(r'${LAUNCH_GATE_PATH}').read_text(encoding='utf-8')); ok=g.get('status') == 'pass' and g.get('decision') == 'innovation1_rtg2b_seed1_remote_launch_authorized' and g.get('should_ssh') is True and g.get('ssh_allowed') is True and g.get('launch_authorized') is True and g.get('source_commit') == '${SOURCE_COMMIT}'; sys.exit(0 if ok else 1)" || {
  echo "launch gate does not authorize remote contact" >&2
  exit 7
}

mkdir -p "${MONITOR_ROOT}" "${RESULT_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

launch_remote() {
  local remote_command
  remote_command="cmd.exe /d /s /c \"if exist ${REMOTE_LAUNCH_ROOT} (exit /b 9) else (set GIT_SSH_COMMAND=ssh -i C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new&& git clone --no-checkout ${REPO_URL} ${REMOTE_LAUNCH_ROOT} && cd /d ${REMOTE_LAUNCH_ROOT} && git checkout --detach ${SOURCE_COMMIT} && call ${LAUNCH_SCRIPT} ${SOURCE_COMMIT} 0)\""
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE}" \
    "${remote_command}" \
    >> "${MONITOR_ROOT}/launch.log" 2>> "${MONITOR_ROOT}/launch_stderr.log"
}

confirm_started_bounded() {
  local attempt
  for attempt in $(seq 1 30); do
    if ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE}" \
      "cmd.exe /c if exist G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\${RUN_ID}\\logs\\${RUN_ID}_started.marker (type G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\${RUN_ID}\\logs\\${RUN_ID}_started.marker) else exit /b 8" \
      >> "${MONITOR_ROOT}/start_confirmation.log" \
      2>> "${MONITOR_ROOT}/start_confirmation_stderr.log"; then
      return 0
    fi
    echo "$(timestamp) start_confirmation_pending attempt=${attempt}/30" \
      >> "${MONITOR_ROOT}/monitor.log"
    sleep 2
  done
  return 1
}

sync_logs() {
  mkdir -p "${MONITOR_ROOT}/${RUN_ID}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/logs" "${MONITOR_ROOT}/${RUN_ID}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

branch_exists() {
  git ls-remote --exit-code origin "refs/heads/results/${RUN_ID}" \
    >> "${MONITOR_ROOT}/branch.log" 2>> "${MONITOR_ROOT}/branch_stderr.log"
}

retrieve_and_readjudicate() {
  local staging="${MONITOR_ROOT}/staging_${RUN_ID}_$(date +%s)"
  if [[ -e "${DESTINATION}" ]]; then
    if [[ -f "${DESTINATION}/retrieved_from_verified_result_branch.marker" \
      && -f "${DESTINATION}/validation.local.json" \
      && -f "${DESTINATION}/gate.json" ]]; then
      echo "$(timestamp) validated_destination_reused" >> "${MONITOR_ROOT}/monitor.log"
      return 0
    fi
    echo "$(timestamp) incomplete_destination_exists" >> "${MONITOR_ROOT}/monitor.log"
    return 1
  fi
  branch_exists || return 1
  mkdir -p "${staging}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/source/results_archive/${RUN_ID}" \
    "${staging}/" >> "${MONITOR_ROOT}/scp.log" \
    2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  (
    cd "${staging}/${RUN_ID}" || exit 1
    sed 's/\r$//' SHA256SUMS | sha256sum -c -
  ) >> "${MONITOR_ROOT}/sha256.log" 2>> "${MONITOR_ROOT}/sha256_stderr.log" || return 1
  cp -a "${staging}/${RUN_ID}" "${DESTINATION}" || return 1
  touch "${DESTINATION}/retrieved_from_verified_result_branch.marker"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
    --plan "${DESTINATION}/plan.csv" \
    --results "${DESTINATION}/results.jsonl" \
    --expected-rows 3 \
    --output "${DESTINATION}/validation.local.json" \
    >> "${MONITOR_ROOT}/readjudication.log" \
    2>> "${MONITOR_ROOT}/readjudication_stderr.log" || return 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-runtime-spn-skinny-medium \
    --run-id "${RUN_ID}" \
    --run-root "${DESTINATION}" \
    --seed 1 \
    --phase rtg2b \
    --progress "${DESTINATION}/progress.jsonl" \
    >> "${MONITOR_ROOT}/readjudication.log" \
    2>> "${MONITOR_ROOT}/readjudication_stderr.log" || true
  [[ -f "${DESTINATION}/gate.json" ]] || return 1
  touch "${DESTINATION}/visual_qa_pending.marker"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
    >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || return 1
  return 0
}

if [[ -f "${DESTINATION}/gate.json" ]]; then
  echo "$(timestamp) result_already_retrieved" >> "${MONITOR_ROOT}/monitor.log"
  exit 0
fi

if [[ ! -f "${LAUNCHED_MARKER}" ]]; then
  launch_remote || exit 4
  echo "$(timestamp) remote_launcher_returned" >> "${MONITOR_ROOT}/monitor.log"
  confirm_started_bounded || exit 8
  touch "${LAUNCHED_MARKER}"
  echo "$(timestamp) bounded_start_confirmation_passed" >> "${MONITOR_ROOT}/monitor.log"
fi

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_ROOT}/monitor.log"
  sync_logs
  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    retrieve_and_readjudicate || exit 2
    touch "${MONITOR_ROOT}/result_retrieved.marker"
    echo "$(timestamp) verified_result_retrieved_readjudicated_and_indexed" >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi
  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*failed.marker" > /dev/null; then
    touch "${MONITOR_ROOT}/remote_failed.marker"
    echo "$(timestamp) remote_run_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi
  echo "$(timestamp) running" >> "${MONITOR_ROOT}/monitor.log"
  sleep 300
done
