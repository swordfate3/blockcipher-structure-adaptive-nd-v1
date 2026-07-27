#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_COMMIT="${1:-}"
REMOTE="lxy-a6000"
RUN_ID="i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727"
RCT2_ID="i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_20260725"
RCT2_ROOT="${PROJECT_ROOT}/outputs/remote_results/${RCT2_ID}"
RCT2_MONITOR_ROOT="${PROJECT_ROOT}/outputs/remote_results_incomplete/${RCT2_ID}_monitor"
GATE_ROOT="${PROJECT_ROOT}/outputs/local_readiness/${RUN_ID}_launch_gate_${SOURCE_COMMIT:0:12}"
MONITOR_ROOT="${PROJECT_ROOT}/outputs/remote_results_incomplete/i1_rct3_after_rct2_20260727_monitor"
RESULT_MONITOR_SESSION="i1_rct3_rectangle80_scale_monitor"
RESULT_MONITOR="${PROJECT_ROOT}/configs/remote/generated/monitor_i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727.sh"
REPO_URL="git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git"
GITHUB_SSH_KEY="C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519"
REMOTE_RUN_ROOT="G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\${RUN_ID}"
REMOTE_SOURCE_ROOT="${REMOTE_RUN_ROOT}\\source"
REMOTE_LAUNCHER="${REMOTE_SOURCE_ROOT}\\configs\\remote\\generated\\launch_i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727.cmd"
REMOTE_STARTED_MARKER="${REMOTE_RUN_ROOT}\\logs\\${RUN_ID}_started.marker"
REMOTE_SCHEDULE_DISABLED_MARKER="G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\launcher_logs\\${RUN_ID}_schedule_disabled.marker"
REMOTE_RUN_LOCK="${REMOTE_RUN_ROOT}\\run.lock\\NUL"

PROTECTED_PATHS=(
  "configs/experiment/innovation1/innovation1_spn_rectangle80_runtime_e4_medium_rct2_65536_seed0.csv"
  "configs/experiment/innovation1/innovation1_spn_rectangle80_runtime_e4_scale_rct3_262144_seed0.csv"
  "configs/runtime/spn/rectangle64.json"
  "configs/remote/innovation1_rct3_rectangle80_runtime_e4_scale_262144_seed0_gpu1_20260727.json"
  "configs/remote/generated/run_i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727.cmd"
  "configs/remote/generated/launch_i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727.cmd"
  "configs/remote/generated/monitor_i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727.sh"
  "configs/remote/generated/monitor_i1_rct3_after_rct2_20260727.sh"
  "scripts/train"
  "scripts/check-runtime-spn-rectangle-rct3-launch"
  "scripts/gate-runtime-spn-rectangle-medium"
  "src/blockcipher_nd/cli/check_runtime_spn_rectangle_rct3_launch.py"
  "src/blockcipher_nd/cli/gate_runtime_spn_rectangle_medium.py"
  "src/blockcipher_nd/data"
  "src/blockcipher_nd/engine"
  "src/blockcipher_nd/models/structure/spn/runtime_parameterized.py"
  "src/blockcipher_nd/models/structure/spn/runtime_structure.py"
  "src/blockcipher_nd/planning/matrix.py"
  "src/blockcipher_nd/registry/model_families/spn.py"
  "src/blockcipher_nd/tasks/innovation1/runtime_spn_rectangle_attribution.py"
  "src/blockcipher_nd/tasks/innovation1/runtime_spn_rectangle_rct3_launch.py"
  "src/blockcipher_nd/training"
)

mkdir -p "${MONITOR_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"
cd "${PROJECT_ROOT}" || exit 2

timestamp() {
  date --iso-8601=seconds
}

record_stop() {
  local reason="$1"
  printf '%s\n' "${reason}" > "${MONITOR_ROOT}/rct3_not_launched.marker"
  echo "$(timestamp) rct3_not_launched reason=${reason}" >> "${MONITOR_ROOT}/monitor.log"
  exit 0
}

record_failure() {
  local reason="$1"
  printf '%s\n' "${reason}" > "${MONITOR_ROOT}/rct3_launch_failed.marker"
  echo "$(timestamp) rct3_launch_failed reason=${reason}" >> "${MONITOR_ROOT}/monitor.log"
  exit 1
}

source_is_frozen() {
  [[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || return 1
  git cat-file -e "${SOURCE_COMMIT}^{commit}" 2>/dev/null || return 1
  git merge-base --is-ancestor "${SOURCE_COMMIT}" origin/main 2>/dev/null || return 1
  git diff --quiet "${SOURCE_COMMIT}..HEAD" -- "${PROTECTED_PATHS[@]}" || return 1
  [[ -z "$(git status --porcelain -- "${PROTECTED_PATHS[@]}")" ]] || return 1
}

rct2_status() {
  python -c "import json,pathlib; p=pathlib.Path(r'${RCT2_ROOT}/gate.local.json'); print(json.loads(p.read_text(encoding='utf-8')).get('status', 'missing') if p.is_file() else 'missing')" 2>/dev/null
}

rct2_local_authority_ready() {
  [[ -f "${RCT2_ROOT}/results.jsonl" ]] || return 1
  [[ -f "${RCT2_ROOT}/gate.local.json" ]] || return 1
  [[ -f "${RCT2_ROOT}/validation.local.json" ]] || return 1
  [[ -f "${RCT2_ROOT}/retrieved_from_verified_result_branch.marker" ]] || return 1
  [[ -f "${RCT2_ROOT}/visual_qa_passed.marker" ]] || return 1
}

if ! source_is_frozen; then
  record_stop "source_not_frozen_or_published"
fi

if tmux has-session -t "${RESULT_MONITOR_SESSION}" 2>/dev/null; then
  record_stop "result_monitor_already_running"
fi

echo "$(timestamp) waiting_for_local_rct2_authority" >> "${MONITOR_ROOT}/monitor.log"
while true; do
  status="$(rct2_status)"
  if [[ "${status}" == "hold" || "${status}" == "fail" ]]; then
    record_stop "rct2_gate_${status}"
  fi
  if [[ "${status}" == "pass" ]] && rct2_local_authority_ready; then
    break
  fi
  if [[ -f "${RCT2_MONITOR_ROOT}/remote_failed.marker" ]]; then
    record_stop "rct2_remote_failed"
  fi
  if ! source_is_frozen; then
    record_stop "protected_source_drift_while_waiting"
  fi
  echo "$(timestamp) waiting rct2_status=${status} visual_qa=$([[ -f "${RCT2_ROOT}/visual_qa_passed.marker" ]] && echo pass || echo pending)" \
    >> "${MONITOR_ROOT}/monitor.log"
  sleep 300
done

if ! source_is_frozen; then
  record_stop "protected_source_drift_before_authorization"
fi

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-runtime-spn-rectangle-rct3-launch \
  --rct2-root "${RCT2_ROOT}" \
  --source-commit "${SOURCE_COMMIT}" \
  --repository "${PROJECT_ROOT}" \
  --output-root "${GATE_ROOT}" \
  >> "${MONITOR_ROOT}/authorization_stdout.log" \
  2>> "${MONITOR_ROOT}/authorization_stderr.log"
authorization_exit=$?
if [[ "${authorization_exit}" -ne 0 ]]; then
  record_failure "authorization_exit_${authorization_exit}"
fi

python -c "import json,pathlib,sys; g=json.loads(pathlib.Path(r'${GATE_ROOT}/gate.json').read_text(encoding='utf-8')); ok=g.get('status') == 'pass' and g.get('decision') == 'innovation1_rct3_rectangle_remote_launch_authorized' and g.get('should_ssh') is True and g.get('ssh_allowed') is True and g.get('launch_authorized') is True and g.get('source_commit') == '${SOURCE_COMMIT}'; sys.exit(0 if ok else 1)" \
  || record_failure "authorization_payload_invalid"

if ! source_is_frozen; then
  record_stop "protected_source_drift_before_remote_bootstrap"
fi

ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE}" \
  "cmd.exe /c if exist \"${REMOTE_RUN_ROOT}\" (exit /b 3) else (mkdir \"${REMOTE_RUN_ROOT}\")" \
  > "${MONITOR_ROOT}/bootstrap_precondition_stdout.log" \
  2> "${MONITOR_ROOT}/bootstrap_precondition_stderr.log" \
  || record_failure "remote_source_path_already_exists_or_unreachable"

ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE}" \
  "cmd.exe /c git -c core.sshCommand=\"ssh -i ${GITHUB_SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new\" clone --no-checkout \"${REPO_URL}\" \"${REMOTE_SOURCE_ROOT}\"" \
  > "${MONITOR_ROOT}/bootstrap_clone_stdout.log" \
  2> "${MONITOR_ROOT}/bootstrap_clone_stderr.log" \
  || record_failure "run_owned_clone_failed"

ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE}" \
  "cmd.exe /c git -C \"${REMOTE_SOURCE_ROOT}\" -c core.sshCommand=\"ssh -i ${GITHUB_SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new\" fetch origin" \
  > "${MONITOR_ROOT}/bootstrap_fetch_stdout.log" \
  2> "${MONITOR_ROOT}/bootstrap_fetch_stderr.log" \
  || record_failure "run_owned_fetch_failed"

ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE}" \
  "cmd.exe /c git -C \"${REMOTE_SOURCE_ROOT}\" checkout --detach ${SOURCE_COMMIT}" \
  > "${MONITOR_ROOT}/bootstrap_checkout_stdout.log" \
  2> "${MONITOR_ROOT}/bootstrap_checkout_stderr.log" \
  || record_failure "run_owned_checkout_failed"

remote_head="$(
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE}" \
    "cmd.exe /c git -C \"${REMOTE_SOURCE_ROOT}\" rev-parse HEAD" \
    2>> "${MONITOR_ROOT}/bootstrap_revision_stderr.log" \
    | tr -d '\r\n'
)"
[[ "${remote_head}" == "${SOURCE_COMMIT}" ]] \
  || record_failure "run_owned_revision_mismatch"

remote_status="$(
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE}" \
    "cmd.exe /c git -C \"${REMOTE_SOURCE_ROOT}\" status --porcelain" \
    2>> "${MONITOR_ROOT}/bootstrap_status_stderr.log" \
    | tr -d '\r\n'
)"
[[ -z "${remote_status}" ]] || record_failure "run_owned_clone_dirty"

ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE}" \
  "cmd.exe /c call \"${REMOTE_LAUNCHER}\" ${SOURCE_COMMIT} 1" \
  > "${MONITOR_ROOT}/remote_launch_stdout.log" \
  2> "${MONITOR_ROOT}/remote_launch_stderr.log" \
  || record_failure "remote_launcher_failed"
echo "$(timestamp) remote_launcher_returned" >> "${MONITOR_ROOT}/monitor.log"

started=false
for attempt in $(seq 1 30); do
  if ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE}" \
    "cmd.exe /c if exist \"${REMOTE_STARTED_MARKER}\" (if exist \"${REMOTE_SCHEDULE_DISABLED_MARKER}\" (if exist \"${REMOTE_RUN_LOCK}\" (type \"${REMOTE_STARTED_MARKER}\") else exit /b 10) else exit /b 9) else exit /b 8" \
    >> "${MONITOR_ROOT}/start_confirmation_stdout.log" \
    2>> "${MONITOR_ROOT}/start_confirmation_stderr.log"; then
    started=true
    echo "$(timestamp) bounded_start_confirmation_passed attempt=${attempt}" \
      >> "${MONITOR_ROOT}/monitor.log"
    break
  fi
  sleep 2
done
[[ "${started}" == "true" ]] || record_failure "bounded_start_confirmation_failed"

tmux new-session -d -s "${RESULT_MONITOR_SESSION}" \
  "cd '${PROJECT_ROOT}' && bash '${RESULT_MONITOR}' '${SOURCE_COMMIT}'"
touch "${MONITOR_ROOT}/rct3_result_monitor_started.marker"
echo "$(timestamp) result_monitor_started session=${RESULT_MONITOR_SESSION}" \
  >> "${MONITOR_ROOT}/monitor.log"
