#!/usr/bin/env bash
set -euo pipefail

RUN_ID="i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706"
LOCAL_MONITOR_DIR="outputs/remote_results/i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706/monitor"
REMOTE_RUN_ROOT="G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\${RUN_ID}"
REMOTE_SOURCE_ROOT="${REMOTE_RUN_ROOT}\\source"
REPO_URL="git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git"
REMOTE_REPAIR_CMD="G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\${RUN_ID}\\source\\configs\\remote\\generated\\repair_i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706_score_export.cmd"

mkdir -p "${LOCAL_MONITOR_DIR}"
echo "$(date -Is) score_export_repair_launch_start" >> "${LOCAL_MONITOR_DIR}/score_export_repair_launch.log"

set +e
ssh lxy-a6000 "cmd.exe /c if not exist \"${REMOTE_RUN_ROOT}\" mkdir \"${REMOTE_RUN_ROOT}\" && if exist \"${REMOTE_SOURCE_ROOT}\\.git\" (cd /d \"${REMOTE_SOURCE_ROOT}\" && git fetch origin main && git checkout main && git pull --ff-only origin main) else (git clone --branch main \"${REPO_URL}\" \"${REMOTE_SOURCE_ROOT}\") && call \"${REMOTE_REPAIR_CMD}\"" \
  >> "${LOCAL_MONITOR_DIR}/score_export_repair_launch.log" \
  2>> "${LOCAL_MONITOR_DIR}/score_export_repair_launch_stderr.log"
status=$?
set -e

if [[ ${status} -ne 0 ]]; then
  echo "$(date -Is) score_export_repair_launch_failed status=${status}" >> "${LOCAL_MONITOR_DIR}/score_export_repair_launch.log"
  echo "failed" > "${LOCAL_MONITOR_DIR}/score_export_repair_launch_failed.marker"
  exit "${status}"
fi

echo "$(date -Is) score_export_repair_launch_done" >> "${LOCAL_MONITOR_DIR}/score_export_repair_launch.log"
echo "done" > "${LOCAL_MONITOR_DIR}/score_export_repair_launch_done.marker"
