#!/usr/bin/env bash
set -u

RUN_ID="i1_present_r8_residual_focus_262k_retry1"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
REMOTE_ARTIFACT_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/i1_present_r8_residual_focus_262k_retry1/artifacts"
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
  sync_path_list <<SYNC_IMMEDIATE_ARTIFACTS
${REMOTE_ARTIFACT_ROOT}/seed0/dataset_cache/seed0_train_feature_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/dataset_cache/seed0_train_feature_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed0/dataset_cache/seed0_train_score_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/dataset_cache/seed0_train_score_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed0/dataset_cache/seed0_validation_feature_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/dataset_cache/seed0_validation_feature_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed0/raw117_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/raw117_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus05_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus05_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus05_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus05_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_labelshuffle_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_labelshuffle_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_labelshuffle_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_labelshuffle_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_uniform_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_uniform_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_uniform_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_uniform_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed1/dataset_cache/seed1_train_feature_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/dataset_cache/seed1_train_feature_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed1/dataset_cache/seed1_train_score_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/dataset_cache/seed1_train_score_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed1/dataset_cache/seed1_validation_feature_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/dataset_cache/seed1_validation_feature_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed1/raw117_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/raw117_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus05_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus05_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus05_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus05_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_labelshuffle_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_labelshuffle_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_labelshuffle_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_labelshuffle_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_uniform_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_uniform_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_uniform_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_uniform_slice_eval.json
SYNC_IMMEDIATE_ARTIFACTS
}

sync_final_artifacts() {
  sync_path_list <<SYNC_FINAL_ARTIFACTS
${REMOTE_ARTIFACT_ROOT}/residual_axis_spectrum_summary.json|outputs/local_audits/i1_present_r8_residual_focus_262k/residual_axis_spectrum_summary.json
${REMOTE_ARTIFACT_ROOT}/seed0/dataset_cache/seed0_train_feature_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/dataset_cache/seed0_train_feature_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed0/dataset_cache/seed0_train_score_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/dataset_cache/seed0_train_score_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed0/dataset_cache/seed0_validation_feature_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/dataset_cache/seed0_validation_feature_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed0/raw117_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/raw117_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus05_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus05_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus05_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus05_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus05_source_selected_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus05_source_selected_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus05_source_selected_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus05_source_selected_train_scores
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus05_source_selected_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus05_source_selected_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus05_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus05_train_scores
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus05_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus05_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_labelshuffle_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_labelshuffle_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_labelshuffle_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_labelshuffle_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_labelshuffle_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_labelshuffle_train_scores
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_labelshuffle_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_labelshuffle_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_source_selected_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_source_selected_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_source_selected_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_source_selected_train_scores
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_source_selected_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_source_selected_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_train_scores
${REMOTE_ARTIFACT_ROOT}/seed0/residual_focus10_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_focus10_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed0/residual_uniform_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_uniform_report.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_uniform_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_uniform_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed0/residual_uniform_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_uniform_train_scores
${REMOTE_ARTIFACT_ROOT}/seed0/residual_uniform_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/residual_uniform_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed0/train_hard_error_axis_spectrum.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/train_hard_error_axis_spectrum.json
${REMOTE_ARTIFACT_ROOT}/seed0/train_raw117_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/train_raw117_scores
${REMOTE_ARTIFACT_ROOT}/seed0/train_residual_loss_axis_spectrum.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/train_residual_loss_axis_spectrum.json
${REMOTE_ARTIFACT_ROOT}/seed0/train_span_summary_features|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/train_span_summary_features
${REMOTE_ARTIFACT_ROOT}/seed0/train_trail_position_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/train_trail_position_scores
${REMOTE_ARTIFACT_ROOT}/seed0/validation_raw117_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/validation_raw117_scores
${REMOTE_ARTIFACT_ROOT}/seed0/validation_span_summary_features|outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/validation_span_summary_features
${REMOTE_ARTIFACT_ROOT}/seed1/dataset_cache/seed1_train_feature_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/dataset_cache/seed1_train_feature_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed1/dataset_cache/seed1_train_score_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/dataset_cache/seed1_train_score_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed1/dataset_cache/seed1_validation_feature_export_progress.jsonl|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/dataset_cache/seed1_validation_feature_export_progress.jsonl
${REMOTE_ARTIFACT_ROOT}/seed1/raw117_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/raw117_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus05_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus05_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus05_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus05_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus05_source_selected_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus05_source_selected_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus05_source_selected_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus05_source_selected_train_scores
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus05_source_selected_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus05_source_selected_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus05_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus05_train_scores
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus05_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus05_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_labelshuffle_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_labelshuffle_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_labelshuffle_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_labelshuffle_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_labelshuffle_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_labelshuffle_train_scores
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_labelshuffle_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_labelshuffle_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_source_selected_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_source_selected_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_source_selected_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_source_selected_train_scores
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_source_selected_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_source_selected_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_train_scores
${REMOTE_ARTIFACT_ROOT}/seed1/residual_focus10_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_focus10_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed1/residual_uniform_report.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_uniform_report.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_uniform_slice_eval.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_uniform_slice_eval.json
${REMOTE_ARTIFACT_ROOT}/seed1/residual_uniform_train_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_uniform_train_scores
${REMOTE_ARTIFACT_ROOT}/seed1/residual_uniform_validation_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/residual_uniform_validation_scores
${REMOTE_ARTIFACT_ROOT}/seed1/train_hard_error_axis_spectrum.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/train_hard_error_axis_spectrum.json
${REMOTE_ARTIFACT_ROOT}/seed1/train_raw117_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/train_raw117_scores
${REMOTE_ARTIFACT_ROOT}/seed1/train_residual_loss_axis_spectrum.json|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/train_residual_loss_axis_spectrum.json
${REMOTE_ARTIFACT_ROOT}/seed1/train_span_summary_features|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/train_span_summary_features
${REMOTE_ARTIFACT_ROOT}/seed1/train_trail_position_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/train_trail_position_scores
${REMOTE_ARTIFACT_ROOT}/seed1/validation_raw117_scores|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/validation_raw117_scores
${REMOTE_ARTIFACT_ROOT}/seed1/validation_span_summary_features|outputs/local_audits/i1_present_r8_residual_focus_262k/seed1/validation_span_summary_features
SYNC_FINAL_ARTIFACTS
}

sync_path_list() {
  while IFS='|' read -r remote_path local_path; do
    if [[ -z "${remote_path}" || -z "${local_path}" ]]; then
      continue
    fi
    mkdir -p "$(dirname "${local_path}")"
    scp -r "${REMOTE}:${remote_path}" "${local_path}" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  done
}

count_missing_outputs() {
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
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_DIR}/monitor.log"
  sync_artifacts

  if compgen -G "${LOCAL_ROOT}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) failed" >> "${MONITOR_DIR}/monitor.log"
    exit 1
  fi

  count_missing_outputs

  if compgen -G "${LOCAL_ROOT}/logs/*done.marker" > /dev/null; then
    sync_final_artifacts
    count_missing_outputs
    if [[ "${missing}" -eq 0 ]]; then
      echo "$(timestamp) result_ready" >> "${MONITOR_DIR}/monitor.log"
      exit 0
    fi
    echo "$(timestamp) completed_missing_outputs missing=${missing}" >> "${MONITOR_DIR}/monitor.log"
    exit 2
  fi

  if [[ "${missing}" -eq 0 ]]; then
    echo "$(timestamp) outputs_ready_waiting_done" >> "${MONITOR_DIR}/monitor.log"
  else
    echo "$(timestamp) running missing=${missing}" >> "${MONITOR_DIR}/monitor.log"
  fi
  sleep 840
done
