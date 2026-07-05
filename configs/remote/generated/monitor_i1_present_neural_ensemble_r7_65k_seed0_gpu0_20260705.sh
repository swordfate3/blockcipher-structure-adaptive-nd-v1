#!/usr/bin/env bash
set -u

RUN_ID="i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"
PLAN_DOC="docs/experiments/innovation1-present-neural-ensemble-aggregation-plan.md"
EXPECTED_ROWS="3"

mkdir -p "${LOCAL_ROOT}" "${MONITOR_DIR}" "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/results" "${LOCAL_ROOT}/checkpoints" "${LOCAL_ROOT}/score_artifacts"
touch "${MONITOR_DIR}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_artifacts() {
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/logs" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/results" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/checkpoints" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/score_artifacts" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
}

commit_plan_doc_if_changed() {
  if git diff --quiet -- "${PLAN_DOC}"; then
    echo "$(timestamp) plan_doc_unchanged" >> "${MONITOR_DIR}/monitor.log"
    return 0
  fi

  git add "${PLAN_DOC}" >> "${MONITOR_DIR}/git_commit.log" 2>> "${MONITOR_DIR}/git_commit_stderr.log"
  git commit -m "docs: record ${RUN_ID} result" >> "${MONITOR_DIR}/git_commit.log" 2>> "${MONITOR_DIR}/git_commit_stderr.log"
  commit_status=$?
  if [[ "${commit_status}" -ne 0 ]]; then
    echo "$(timestamp) plan_doc_commit_failed" >> "${MONITOR_DIR}/monitor.log"
    return "${commit_status}"
  fi

  git push >> "${MONITOR_DIR}/git_push.log" 2>> "${MONITOR_DIR}/git_push_stderr.log"
  push_status=$?
  if [[ "${push_status}" -ne 0 ]]; then
    echo "$(timestamp) plan_doc_push_failed" >> "${MONITOR_DIR}/monitor.log"
    return "${push_status}"
  fi

  echo "$(timestamp) plan_doc_committed_and_pushed" >> "${MONITOR_DIR}/monitor.log"
  return 0
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_DIR}/monitor.log"
  sync_artifacts

  if compgen -G "${LOCAL_ROOT}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) failed" >> "${MONITOR_DIR}/monitor.log"
    exit 1
  fi

  train_file="${LOCAL_ROOT}/results/train_matrix.jsonl"
  ensemble_file="${LOCAL_ROOT}/results/neural_ensemble_summary.json"
  zhang_wang_scores="${LOCAL_ROOT}/score_artifacts/zhang_wang/models.json"
  invp_scores="${LOCAL_ROOT}/score_artifacts/invp_only/models.json"
  ddt_graph_scores="${LOCAL_ROOT}/score_artifacts/ddt_graph/models.json"
  result_rows=0
  if [[ -f "${train_file}" ]]; then
    result_rows=$(grep -cve '^[[:space:]]*$' "${train_file}" || true)
  fi

  if [[ "${result_rows}" -ge "${EXPECTED_ROWS}" && -f "${ensemble_file}" && -f "${zhang_wang_scores}" && -f "${invp_scores}" && -f "${ddt_graph_scores}" ]]; then
    echo "$(timestamp) result_ready" >> "${MONITOR_DIR}/monitor.log"
    env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-neural-ensemble \
      --train-results "${train_file}" \
      --ensemble-summary "${ensemble_file}" \
      --output-dir "${LOCAL_ROOT}" \
      --run-id "${RUN_ID}" \
      --expected-rows "${EXPECTED_ROWS}" \
      --update-plan-doc "${PLAN_DOC}" \
      > "${MONITOR_DIR}/postprocess.log" 2> "${MONITOR_DIR}/postprocess_stderr.log"
    postprocess_status=$?
    if [[ "${postprocess_status}" -eq 0 ]]; then
      commit_plan_doc_if_changed
      commit_status=$?
      if [[ "${commit_status}" -eq 0 ]]; then
        echo "$(timestamp) postprocess_done" >> "${MONITOR_DIR}/monitor.log"
      else
        echo "$(timestamp) postprocess_done_commit_failed" >> "${MONITOR_DIR}/monitor.log"
      fi
      exit "${commit_status}"
    else
      echo "$(timestamp) postprocess_failed" >> "${MONITOR_DIR}/monitor.log"
    fi
    exit "${postprocess_status}"
  fi

  if compgen -G "${LOCAL_ROOT}/logs/*done.marker" > /dev/null; then
    echo "$(timestamp) completed_missing_or_incomplete_results rows=${result_rows}" >> "${MONITOR_DIR}/monitor.log"
    exit 2
  fi

  echo "$(timestamp) running" >> "${MONITOR_DIR}/monitor.log"
  sleep 840
done
