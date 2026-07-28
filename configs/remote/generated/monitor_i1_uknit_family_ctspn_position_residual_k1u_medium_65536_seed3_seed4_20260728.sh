#!/usr/bin/env bash
set -u
set -o pipefail

REMOTE="lxy-a6000"
RUN_ID="i1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4_20260728"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
VERIFIED_ROOT="outputs/remote_results"
FALLBACK_ROOT="outputs/remote_results_incomplete"
SOURCE_COMMIT="${1:-}"
LAUNCH_GATE="${2:-}"

if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <pushed-source-commit> <launch-gate.json>" >&2
  exit 6
fi
if [[ ! -f "${LAUNCH_GATE}" ]]; then
  echo "missing launch gate: ${LAUNCH_GATE}" >&2
  exit 7
fi
python -c "import json,pathlib,sys; g=json.loads(pathlib.Path(r'${LAUNCH_GATE}').read_text(encoding='utf-8')); ok=g.get('status') == 'pass' and g.get('decision') == 'innovation1_uknit_family_ctspn_k1u_remote_launch_authorized' and g.get('should_ssh') is True and g.get('ssh_allowed') is True and g.get('launch_authorized') is True and g.get('source_commit') == '${SOURCE_COMMIT}'; sys.exit(0 if ok else 1)" || exit 7

mkdir -p "${MONITOR_ROOT}" "${VERIFIED_ROOT}" "${FALLBACK_ROOT}"
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

branch_exists() {
  git ls-remote --exit-code origin "refs/heads/results/${RUN_ID}" \
    >> "${MONITOR_ROOT}/branch.log" 2>> "${MONITOR_ROOT}/branch_stderr.log"
}

retrieve_archive() {
  local mode="$1"
  local destination_root="${VERIFIED_ROOT}"
  local marker="retrieved_from_verified_result_branch.marker"
  local staging="${MONITOR_ROOT}/staging_${RUN_ID}_$(date +%s)"
  local result_ref="refs/remotes/origin/results/${RUN_ID}"
  local destination
  local adjudication
  if [[ "${mode}" == "raw" ]]; then
    destination_root="${FALLBACK_ROOT}"
    marker="fallback_retrieved.marker"
  fi
  destination="${destination_root}/${RUN_ID}"
  adjudication="${destination}/local_adjudication"
  if [[ -e "${destination}" ]]; then
    if [[ -f "${destination}/${marker}" \
      && -f "${destination}/validation.local.json" \
      && -f "${destination}/gate.local.json" ]]; then
      return 0
    fi
    echo "$(timestamp) incomplete_destination_exists mode=${mode}" >> "${MONITOR_ROOT}/monitor.log"
    return 1
  fi
  mkdir -p "${staging}"
  if [[ "${mode}" == "verified" ]]; then
    mkdir -p "${staging}/${RUN_ID}"
    git fetch --force origin \
      "refs/heads/results/${RUN_ID}:${result_ref}" \
      >> "${MONITOR_ROOT}/branch.log" \
      2>> "${MONITOR_ROOT}/branch_stderr.log" || return 1
    git archive "${result_ref}" "results_archive/${RUN_ID}" \
      | tar -x -C "${staging}/${RUN_ID}" --strip-components=2 \
      >> "${MONITOR_ROOT}/branch.log" \
      2>> "${MONITOR_ROOT}/branch_stderr.log" || return 1
  else
    scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/source/results_archive/${RUN_ID}" \
      "${staging}/" >> "${MONITOR_ROOT}/scp.log" \
      2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  fi
  (
    cd "${staging}/${RUN_ID}" || exit 1
    sed 's/\r$//' SHA256SUMS | sha256sum -c -
  ) >> "${MONITOR_ROOT}/sha256.log" \
    2>> "${MONITOR_ROOT}/sha256_stderr.log" || return 1
  [[ "$(tr -d '\r\n' < "${staging}/${RUN_ID}/git_revision.txt")" == "${SOURCE_COMMIT}" ]] \
    || return 1
  cp -a "${staging}/${RUN_ID}" "${destination}" || return 1
  touch "${destination}/${marker}"
  if [[ "${mode}" == "raw" ]]; then
    printf '%s\n' \
      "RAW FALLBACK RETRIEVAL: remote training produced a valid archive but the result branch was not verified." \
      "This remains fallback-retrieved evidence until all local gates and publication evidence are complete." \
      > "${destination}/RAW_RETRIEVAL_NOTICE.txt"
  fi
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
    --plan "${destination}/innovation1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4.csv" \
    --results "${destination}/results.jsonl" \
    --expected-rows 6 \
    --output "${destination}/validation.local.json" \
    >> "${MONITOR_ROOT}/readjudication.log" \
    2>> "${MONITOR_ROOT}/readjudication_stderr.log" || return 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-uknit-family-ctspn-k1u \
    --plan "${destination}/innovation1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4.csv" \
    --results "${destination}/results.jsonl" \
    --progress "${destination}/progress.jsonl" \
    --source-commit-file "${destination}/git_revision.txt" \
    --expected-source-commit-file "${destination}/source_expected_commit.txt" \
    --output-root "${adjudication}" \
    >> "${MONITOR_ROOT}/readjudication.log" \
    2>> "${MONITOR_ROOT}/readjudication_stderr.log" || true
  [[ -f "${adjudication}/gate.json" ]] || return 1
  cp "${adjudication}/gate.json" "${destination}/gate.local.json" || return 1
  cp "${adjudication}/validation.json" "${destination}/validation.k1u.local.json" || return 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-uknit-family-ctspn-k1u \
    --gate "${destination}/gate.local.json" \
    --output "${destination}/curves.svg" \
    --report "${destination}/plot_report.json" \
    >> "${MONITOR_ROOT}/plot.log" 2>> "${MONITOR_ROOT}/plot_stderr.log" \
    || return 1
  touch "${destination}/visual_qa_pending.marker"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
    >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" \
    || return 1
  return 0
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_ROOT}/monitor.log"
  sync_logs
  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    branch_exists || {
      echo "$(timestamp) result_branch_marker_without_remote_ref" >> "${MONITOR_ROOT}/monitor.log"
      sleep 300
      continue
    }
    retrieve_archive verified || exit 2
    touch "${MONITOR_ROOT}/result_retrieved.marker"
    echo "$(timestamp) verified_result_retrieved_readjudicated_indexed_visual_qa_pending" \
      >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi
  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*raw_ready.marker" > /dev/null; then
    retrieve_archive raw || exit 3
    touch "${MONITOR_ROOT}/fallback_result_retrieved.marker"
    echo "$(timestamp) fallback_result_retrieved_readjudicated_indexed_visual_qa_pending" \
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
