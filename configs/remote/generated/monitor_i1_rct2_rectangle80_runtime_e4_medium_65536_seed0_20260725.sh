#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
RUN_ID="i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_20260725"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
RESULT_ROOT="outputs/remote_results"
DESTINATION="${RESULT_ROOT}/${RUN_ID}"
SOURCE_COMMIT="${1:-}"

if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <pushed-source-commit>" >&2
  exit 6
fi

mkdir -p "${MONITOR_ROOT}" "${RESULT_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_logs() {
  mkdir -p "${MONITOR_ROOT}/${RUN_ID}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/logs" \
    "${MONITOR_ROOT}/${RUN_ID}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

retrieve_verified_archive() {
  local staging="${MONITOR_ROOT}/staging_${RUN_ID}_$(date +%s)"
  local local_gate_root="${DESTINATION}/local_adjudication"
  if [[ -e "${DESTINATION}" ]]; then
    if [[ -f "${DESTINATION}/retrieved_from_verified_result_branch.marker" \
      && -f "${DESTINATION}/validation.local.json" \
      && -f "${DESTINATION}/gate.local.json" ]]; then
      return 0
    fi
    echo "$(timestamp) incomplete_destination_exists" >> "${MONITOR_ROOT}/monitor.log"
    return 1
  fi
  git ls-remote --exit-code origin "refs/heads/results/${RUN_ID}" \
    >> "${MONITOR_ROOT}/branch.log" 2>> "${MONITOR_ROOT}/branch_stderr.log" \
    || return 1
  mkdir -p "${staging}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/source/results_archive/${RUN_ID}" \
    "${staging}/" >> "${MONITOR_ROOT}/scp.log" \
    2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  (
    cd "${staging}/${RUN_ID}" || exit 1
    sed 's/\r$//' SHA256SUMS | sha256sum -c -
  ) >> "${MONITOR_ROOT}/sha256.log" \
    2>> "${MONITOR_ROOT}/sha256_stderr.log" || return 1
  [[ "$(tr -d '\r\n' < "${staging}/${RUN_ID}/git_revision.txt")" == "${SOURCE_COMMIT}" ]] \
    || return 1
  cp -a "${staging}/${RUN_ID}" "${DESTINATION}" || return 1
  touch "${DESTINATION}/retrieved_from_verified_result_branch.marker"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
    --plan "${DESTINATION}/plan.csv" \
    --results "${DESTINATION}/results.jsonl" \
    --expected-rows 3 \
    --output "${DESTINATION}/validation.local.json" \
    >> "${MONITOR_ROOT}/readjudication.log" \
    2>> "${MONITOR_ROOT}/readjudication_stderr.log" || return 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-runtime-spn-rectangle-medium \
    --run-id "${RUN_ID}" \
    --run-root "${DESTINATION}" \
    --output-root "${local_gate_root}" \
    --seed 0 \
    --progress "${local_gate_root}/progress.jsonl" \
    >> "${MONITOR_ROOT}/readjudication.log" \
    2>> "${MONITOR_ROOT}/readjudication_stderr.log" || true
  [[ -f "${local_gate_root}/gate.json" ]] || return 1
  cp "${local_gate_root}/gate.json" "${DESTINATION}/gate.local.json" || return 1
  cp "${local_gate_root}/curves.svg" "${DESTINATION}/curves.svg" || return 1
  touch "${DESTINATION}/visual_qa_pending.marker"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
    >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" \
    || return 1
  return 0
}

if [[ -f "${DESTINATION}/gate.local.json" \
  && -f "${DESTINATION}/validation.local.json" ]]; then
  touch "${MONITOR_ROOT}/result_retrieved.marker"
  exit 0
fi

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_ROOT}/monitor.log"
  sync_logs
  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    retrieve_verified_archive || exit 2
    touch "${MONITOR_ROOT}/result_retrieved.marker"
    echo "$(timestamp) retrieved_readjudicated_indexed_visual_qa_pending" \
      >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi
  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*failed.marker" > /dev/null; then
    touch "${MONITOR_ROOT}/remote_failed.marker"
    echo "$(timestamp) remote_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi
  echo "$(timestamp) waiting" >> "${MONITOR_ROOT}/monitor.log"
  sleep 300
done
