#!/usr/bin/env bash
set -euo pipefail

RUN_ID="i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706"
LOCAL_MONITOR_DIR="outputs/remote_results/i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706/monitor"
REMOTE_RUN_CMD="G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\${RUN_ID}\\source\\configs\\remote\\generated\\run_i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706.cmd"
MONITOR_SCRIPT="configs/remote/generated/monitor_i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706.sh"
MONITOR_SESSION="monitor_i1_present_r8_trailpos_262k_seed0_20260706"

mkdir -p "${LOCAL_MONITOR_DIR}"
echo "$(date -Is) launch_start" >> "${LOCAL_MONITOR_DIR}/launch.log"

set +e
ssh lxy-a6000 "cmd.exe /c call \"${REMOTE_RUN_CMD}\"" \
  >> "${LOCAL_MONITOR_DIR}/launch.log" \
  2>> "${LOCAL_MONITOR_DIR}/launch_stderr.log"
status=$?
set -e

if [[ ${status} -ne 0 ]]; then
  echo "$(date -Is) launch_failed status=${status}" >> "${LOCAL_MONITOR_DIR}/launch.log"
  echo "failed" > "${LOCAL_MONITOR_DIR}/launch_failed.marker"
  exit "${status}"
fi

tmux new-session -d -s "${MONITOR_SESSION}" "${MONITOR_SCRIPT}"
echo "$(date -Is) launch_done monitor=${MONITOR_SESSION}" >> "${LOCAL_MONITOR_DIR}/launch.log"
echo "done" > "${LOCAL_MONITOR_DIR}/launch_done.marker"
