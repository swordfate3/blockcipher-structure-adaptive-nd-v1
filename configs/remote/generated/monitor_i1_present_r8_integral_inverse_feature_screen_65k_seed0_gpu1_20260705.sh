#!/usr/bin/env bash
set -u

RUN_ID="i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"
PLAN="configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0.csv"
PLAN_DOC="docs/experiments/innovation1-present-r8-integral-inverse-feature-screen-plan.md"
EXPECTED_ROWS="3"

mkdir -p "${LOCAL_ROOT}" "${MONITOR_DIR}" "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/results"
touch "${MONITOR_DIR}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_artifacts() {
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/logs" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/results" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
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

  result_file="${LOCAL_ROOT}/results/${RUN_ID}.jsonl"
  result_rows=0
  if [[ -f "${result_file}" ]]; then
    result_rows=$(grep -cve '^[[:space:]]*$' "${result_file}" || true)
  fi

  if [[ "${result_rows}" -ge "${EXPECTED_ROWS}" ]]; then
    echo "$(timestamp) result_ready" >> "${MONITOR_DIR}/monitor.log"
    env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/advance-integral-inverse-feature-result \
      --results "${result_file}" \
      --output-dir "${LOCAL_ROOT}" \
      --run-id "${RUN_ID}" \
      --plan "${PLAN}" \
      --expected-rows "${EXPECTED_ROWS}" \
      --update-plan-doc "${PLAN_DOC}" \
      > "${MONITOR_DIR}/advance.log" 2> "${MONITOR_DIR}/advance_stderr.log"
    advance_status=$?
    if [[ "${advance_status}" -eq 0 ]]; then
      commit_plan_doc_if_changed
      commit_status=$?
      if [[ "${commit_status}" -eq 0 ]]; then
        echo "$(timestamp) advance_done" >> "${MONITOR_DIR}/monitor.log"
      else
        echo "$(timestamp) advance_done_commit_failed" >> "${MONITOR_DIR}/monitor.log"
      fi
      exit "${commit_status}"
    else
      echo "$(timestamp) advance_failed plan=${PLAN}" >> "${MONITOR_DIR}/monitor.log"
    fi
    exit "${advance_status}"
  fi

  if compgen -G "${LOCAL_ROOT}/logs/*done.marker" > /dev/null; then
    echo "$(timestamp) completed_missing_or_incomplete_results rows=${result_rows}" >> "${MONITOR_DIR}/monitor.log"
    exit 2
  fi

  echo "$(timestamp) running" >> "${MONITOR_DIR}/monitor.log"
  sleep 840
done
