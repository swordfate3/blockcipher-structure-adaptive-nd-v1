#!/usr/bin/env bash
set -euo pipefail

RUN_ID="i1_present_r8_residual_focus_262k_retry1"
PACKAGE_REPORT="outputs/local_audits/i1_present_r8_residual_focus_262k_retry1_remote_package.json"
LOCAL_MONITOR_DIR="outputs/remote_results/i1_present_r8_residual_focus_262k_retry1/monitor"
REMOTE_RUN_ROOT="G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_residual_focus_262k_retry1"
REMOTE_SOURCE_ROOT="G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_residual_focus_262k_retry1\source"
REMOTE_RUN_CMD="G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_residual_focus_262k_retry1\source\configs\remote\generated\run_i1_present_r8_residual_focus_262k_retry1_20260707.cmd"
REPO_URL="git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git"
MONITOR_SCRIPT="configs/remote/generated/monitor_i1_present_r8_residual_focus_262k_retry1_20260707.sh"
MONITOR_SESSION="monitor_i1_present_r8_residual_focus_262k_retry1_20260707"

mkdir -p "${LOCAL_MONITOR_DIR}"
echo "$(date -Is) launch_start" >> "${LOCAL_MONITOR_DIR}/launch.log"

launch_allowed="$(UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import json; print(json.load(open('${PACKAGE_REPORT}')).get('launch_allowed'))")"
if [[ "${launch_allowed}" != "True" ]]; then
  echo "$(date -Is) launch_blocked source_gate_not_pass launch_allowed=${launch_allowed}" >> "${LOCAL_MONITOR_DIR}/launch.log"
  echo "blocked" > "${LOCAL_MONITOR_DIR}/launch_blocked.marker"
  exit 3
fi

if tmux has-session -t "${MONITOR_SESSION}" >/dev/null 2>&1; then
  echo "$(date -Is) monitor_already_running monitor=${MONITOR_SESSION}" >> "${LOCAL_MONITOR_DIR}/launch.log"
else
  tmux new-session -d -s monitor_i1_present_r8_residual_focus_262k_retry1_20260707 "${MONITOR_SCRIPT}"
  echo "$(date -Is) monitor_started monitor=${MONITOR_SESSION}" >> "${LOCAL_MONITOR_DIR}/launch.log"
fi

set +e
ssh lxy-a6000 "cmd.exe /c if not exist \"${REMOTE_RUN_ROOT}\" mkdir \"${REMOTE_RUN_ROOT}\" && if exist \"${REMOTE_SOURCE_ROOT}\.git\" (cd /d \"${REMOTE_SOURCE_ROOT}\" && git fetch origin main && git checkout main && git pull --ff-only origin main) else (git clone --branch main \"${REPO_URL}\" \"${REMOTE_SOURCE_ROOT}\") && call \"${REMOTE_RUN_CMD}\"" \
  >> "${LOCAL_MONITOR_DIR}/launch.log" \
  2>> "${LOCAL_MONITOR_DIR}/launch_stderr.log"
status=$?
set -e

if [[ ${status} -ne 0 ]]; then
  echo "$(date -Is) launch_failed status=${status}" >> "${LOCAL_MONITOR_DIR}/launch.log"
  echo "failed" > "${LOCAL_MONITOR_DIR}/launch_failed.marker"
  exit "${status}"
fi

echo "$(date -Is) launch_done monitor=${MONITOR_SESSION}" >> "${LOCAL_MONITOR_DIR}/launch.log"
echo "done" > "${LOCAL_MONITOR_DIR}/launch_done.marker"
