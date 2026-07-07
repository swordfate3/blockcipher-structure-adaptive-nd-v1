#!/usr/bin/env bash
set -u

RUN_ID="i1_present_r8_residual_focus_262k"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
REMOTE_ARTIFACT_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/i1_present_r8_residual_focus_262k/artifacts"
LOCAL_ARTIFACT_ROOT="outputs/local_audits/i1_present_r8_residual_focus_262k"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"

mkdir -p "${LOCAL_ARTIFACT_ROOT}" "${LOCAL_ROOT}" "${MONITOR_DIR}" "${LOCAL_ROOT}/logs"
touch "${MONITOR_DIR}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_artifacts() {
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/logs" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_ARTIFACT_ROOT}/"* "${LOCAL_ARTIFACT_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_DIR}/monitor.log"
  sync_artifacts

  if compgen -G "${LOCAL_ROOT}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) failed" >> "${MONITOR_DIR}/monitor.log"
    exit 1
  fi

  missing=0
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/raw117_report.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus05_report.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus05_slice_eval.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_labelshuffle_report.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_labelshuffle_slice_eval.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_report.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_slice_eval.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_uniform_report.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_uniform_slice_eval.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/raw117_report.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus05_report.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus05_slice_eval.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_labelshuffle_report.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_labelshuffle_slice_eval.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_report.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_slice_eval.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_uniform_report.json" ]] || missing=$((missing + 1))
  [[ -f "outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_uniform_slice_eval.json" ]] || missing=$((missing + 1))
  if [[ "${missing}" -eq 0 ]]; then
    echo "$(timestamp) result_ready" >> "${MONITOR_DIR}/monitor.log"
    exit 0
  fi

  if compgen -G "${LOCAL_ROOT}/logs/*done.marker" > /dev/null; then
    echo "$(timestamp) completed_missing_outputs missing=${missing}" >> "${MONITOR_DIR}/monitor.log"
    exit 2
  fi

  echo "$(timestamp) running missing=${missing}" >> "${MONITOR_DIR}/monitor.log"
  sleep 840
done
