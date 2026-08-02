#!/usr/bin/env bash
set -u
set -o pipefail

REMOTE="lxy-a6000"
RUN_ID="i1_dialga128_runtime_e4_dmc2_r4_262144_seed0_seed1_20260801"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
VERIFIED_ROOT="outputs/remote_results"
FALLBACK_ROOT="outputs/remote_results_incomplete"
SOURCE_COMMIT="${1:-}"
LAUNCH_GATE="${2:-}"

if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || [[ ! -f "${LAUNCH_GATE}" ]]; then
  echo "usage: $0 <pushed-source-commit> <launch-gate.json>" >&2
  exit 6
fi
python -c "import json,pathlib,sys; g=json.loads(pathlib.Path(r'${LAUNCH_GATE}').read_text(encoding='utf-8')); ok=g.get('status') == 'pass' and g.get('decision') == 'innovation1_dialga_dmc2_remote_launch_authorized' and g.get('should_ssh') is True and g.get('ssh_allowed') is True and g.get('launch_authorized') is True and g.get('source_commit') == '${SOURCE_COMMIT}'; sys.exit(0 if ok else 1)" || exit 7
mkdir -p "${MONITOR_ROOT}" "${VERIFIED_ROOT}" "${FALLBACK_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() { date --iso-8601=seconds; }

sync_logs() {
  mkdir -p "${MONITOR_ROOT}/${RUN_ID}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/logs" "${MONITOR_ROOT}/${RUN_ID}/" >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

retrieve_archive() {
  local mode="$1"
  local destination_root="${VERIFIED_ROOT}"
  local marker="retrieved_from_verified_result_branch.marker"
  local staging="${MONITOR_ROOT}/staging_${RUN_ID}_$(date +%s)"
  local result_ref="refs/remotes/origin/results/${RUN_ID}"
  if [[ "${mode}" == "raw" ]]; then
    destination_root="${FALLBACK_ROOT}"
    marker="fallback_retrieved.marker"
  fi
  local destination="${destination_root}/${RUN_ID}"
  [[ ! -e "${destination}" ]] || return 1
  mkdir -p "${staging}"
  if [[ "${mode}" == "verified" ]]; then
    mkdir -p "${staging}/${RUN_ID}"
    git fetch --force origin "refs/heads/results/${RUN_ID}:${result_ref}" >> "${MONITOR_ROOT}/branch.log" 2>> "${MONITOR_ROOT}/branch_stderr.log" || return 1
    git archive "${result_ref}" "results_archive/${RUN_ID}" | tar -x -C "${staging}/${RUN_ID}" --strip-components=2 >> "${MONITOR_ROOT}/branch.log" 2>> "${MONITOR_ROOT}/branch_stderr.log" || return 1
  else
    scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/source/results_archive/${RUN_ID}" "${staging}/" >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  fi
  (cd "${staging}/${RUN_ID}" && sed 's/\r$//' SHA256SUMS | sha256sum -c -) >> "${MONITOR_ROOT}/sha256.log" 2>> "${MONITOR_ROOT}/sha256_stderr.log" || return 1
  [[ "$(tr -d '\r\n' < "${staging}/${RUN_ID}/git_revision.txt")" == "${SOURCE_COMMIT}" ]] || return 1
  cp -a "${staging}/${RUN_ID}" "${destination}" || return 1
  touch "${destination}/${marker}"
  if [[ "${mode}" == "raw" ]]; then
    printf '%s\n' "RAW FALLBACK RETRIEVAL: the remote archive was retrieved directly because its result branch was unavailable or incomplete." "Treat this as fallback-retrieved evidence until local gates pass." > "${destination}/RAW_RETRIEVAL_NOTICE.txt"
  fi
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results --plan "${destination}/plan.csv" --results "${destination}/results.jsonl" --expected-rows 6 --output "${destination}/validation.local.json" >> "${MONITOR_ROOT}/readjudication.log" 2>> "${MONITOR_ROOT}/readjudication_stderr.log" || return 1
  mkdir -p "${destination}/local_adjudication"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-dialga-r4-dmc2 --plan "${destination}/plan.csv" --results "${destination}/results.jsonl" --progress "${destination}/progress.jsonl" --checkpoint-root "${destination}/checkpoints" --source-commit-file "${destination}/git_revision.txt" --expected-source-commit-file "${destination}/source_expected_commit.txt" --output-root "${destination}/local_adjudication" >> "${MONITOR_ROOT}/readjudication.log" 2>> "${MONITOR_ROOT}/readjudication_stderr.log" || true
  [[ -f "${destination}/local_adjudication/gate.json" ]] || return 1
  cp "${destination}/local_adjudication/gate.json" "${destination}/gate.local.json"
  touch "${destination}/visual_qa_pending.marker"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || return 1
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_ROOT}/monitor.log"
  sync_logs
  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    if retrieve_archive verified; then
      touch "${MONITOR_ROOT}/result_retrieved.marker"
      echo "$(timestamp) verified_result_retrieved_readjudicated_indexed_visual_qa_pending" >> "${MONITOR_ROOT}/monitor.log"
      exit 0
    fi
    echo "$(timestamp) verified_result_incomplete_trying_raw_fallback" >> "${MONITOR_ROOT}/monitor.log"
    if retrieve_archive raw; then
      touch "${MONITOR_ROOT}/fallback_result_retrieved.marker"
      echo "$(timestamp) fallback_result_retrieved_readjudicated_indexed_visual_qa_pending" >> "${MONITOR_ROOT}/monitor.log"
      exit 0
    fi
    exit 2
  fi
  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*raw_ready.marker" > /dev/null; then
    retrieve_archive raw || exit 3
    touch "${MONITOR_ROOT}/fallback_result_retrieved.marker"
    echo "$(timestamp) fallback_result_retrieved_readjudicated_indexed_visual_qa_pending" >> "${MONITOR_ROOT}/monitor.log"
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
