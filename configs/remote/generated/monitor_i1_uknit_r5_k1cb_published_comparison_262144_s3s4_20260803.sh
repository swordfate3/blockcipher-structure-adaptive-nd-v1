#!/usr/bin/env bash
set -u
set -o pipefail

REMOTE="lxy-a6000"
RUN_ID="i1_uknit_r5_k1cb_published_comparison_262144_s3s4_20260803"
SOURCE_RUN_ID="i1_uknit_r5_k1ca_invariant_autond_262144_s3s4_20260803"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
REMOTE_LAUNCH_ROOT="G:/lxy/scheduled-runs"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
SOURCE_MONITOR_ROOT="outputs/remote_results_incomplete/${SOURCE_RUN_ID}_monitor"
VERIFIED_ROOT="outputs/remote_results"
FALLBACK_ROOT="outputs/remote_results_incomplete"
SOURCE_COMMIT="${1:-}"
LAUNCHER="configs/remote/generated/launch_${RUN_ID}.cmd"
REMOTE_LAUNCHER="${REMOTE_LAUNCH_ROOT}/launch_${RUN_ID}.cmd"
LAUNCH_GATE_ROOT="${MONITOR_ROOT}/launch_gate"
LAUNCH_GATE="${LAUNCH_GATE_ROOT}/gate.json"

if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <pushed-source-commit>" >&2
  exit 6
fi
mkdir -p "${MONITOR_ROOT}" "${VERIFIED_ROOT}" "${FALLBACK_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() { date --iso-8601=seconds; }

source_result_root() {
  if [[ -d "${VERIFIED_ROOT}/${SOURCE_RUN_ID}" ]]; then
    printf '%s\n' "${VERIFIED_ROOT}/${SOURCE_RUN_ID}"
    return 0
  fi
  if [[ -d "${FALLBACK_ROOT}/${SOURCE_RUN_ID}" ]]; then
    printf '%s\n' "${FALLBACK_ROOT}/${SOURCE_RUN_ID}"
    return 0
  fi
  return 1
}

source_gate_valid() {
  local source_root="$1"
  local gate="${source_root}/gate.local.json"
  [[ -f "${gate}" ]] || gate="${source_root}/local_adjudication/gate.json"
  [[ -f "${gate}" ]] || gate="${source_root}/gate.json"
  [[ -f "${gate}" ]] || return 1
  python3 -c "import json,pathlib,sys; g=json.loads(pathlib.Path(r'${gate}').read_text(encoding='utf-8')); p=g.get('protocol_checks',{}); ok=g.get('status') in {'pass','hold'} and bool(p) and all(p.values()); sys.exit(0 if ok else 1)"
}

wait_for_source() {
  while true; do
    local source_root=""
    source_root="$(source_result_root 2>/dev/null || true)"
    if [[ -n "${source_root}" ]] && source_gate_valid "${source_root}"; then
      printf '%s\n' "${source_root}" > "${MONITOR_ROOT}/source_result_root.txt"
      echo "$(timestamp) source_k1ca_protocol_valid" >> "${MONITOR_ROOT}/monitor.log"
      return 0
    fi
    if [[ -f "${SOURCE_MONITOR_ROOT}/remote_failed.marker" ]]; then
      echo "$(timestamp) source_k1ca_failed" >> "${MONITOR_ROOT}/monitor.log"
      return 1
    fi
    echo "$(timestamp) waiting_for_source_k1ca" >> "${MONITOR_ROOT}/monitor.log"
    sleep 60
  done
}

prepare_launch_gate() {
  local source_root=""
  source_root="$(tr -d '\r\n' < "${MONITOR_ROOT}/source_result_root.txt")"
  local remote_main=""
  remote_main="$(git ls-remote origin refs/heads/main | awk '{print $1}')" || return 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-uknit-r5-k1cb-paper-comparison-launch \
    --source-k1ca-root "${source_root}" \
    --source-commit "${SOURCE_COMMIT}" \
    --remote-main-sha "${remote_main}" \
    --repository . \
    --output-root "${LAUNCH_GATE_ROOT}" \
    >> "${MONITOR_ROOT}/launch_gate.log" 2>> "${MONITOR_ROOT}/launch_gate_stderr.log" || return 1
  python3 -c "import json,pathlib,sys; g=json.loads(pathlib.Path(r'${LAUNCH_GATE}').read_text(encoding='utf-8')); ok=g.get('status') == 'pass' and g.get('decision') == 'innovation1_uknit_k1cb_remote_launch_authorized' and g.get('should_ssh') is True and g.get('ssh_allowed') is True and g.get('launch_authorized') is True and g.get('source_commit') == '${SOURCE_COMMIT}' and g.get('remote_main_sha') == '${SOURCE_COMMIT}'; sys.exit(0 if ok else 1)" || return 1
  touch "${MONITOR_ROOT}/launch_gate_passed.marker"
  echo "$(timestamp) launch_gate_passed" >> "${MONITOR_ROOT}/monitor.log"
}

launch_remote() {
  scp "${LAUNCHER}" "${REMOTE}:${REMOTE_LAUNCHER}" \
    >> "${MONITOR_ROOT}/launch_scp.log" 2>> "${MONITOR_ROOT}/launch_scp_stderr.log" || return 1
  ssh "${REMOTE}" "cmd.exe /c G:\\lxy\\scheduled-runs\\launch_${RUN_ID}.cmd ${SOURCE_COMMIT} 0" \
    >> "${MONITOR_ROOT}/launch.log" 2>> "${MONITOR_ROOT}/launch_stderr.log" || return 1
  printf '%s\n' "${SOURCE_COMMIT}" > "${MONITOR_ROOT}/launched_source_commit.txt"
  touch "${MONITOR_ROOT}/launch_submitted.marker"
  echo "$(timestamp) launch_submitted" >> "${MONITOR_ROOT}/monitor.log"
  sleep 20
  sync_logs
  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*started.marker" > /dev/null; then
    touch "${MONITOR_ROOT}/bounded_start_confirmed.marker"
    echo "$(timestamp) bounded_start_confirmed" >> "${MONITOR_ROOT}/monitor.log"
  elif compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*failed.marker" > /dev/null; then
    touch "${MONITOR_ROOT}/bounded_start_failed.marker"
    return 1
  else
    touch "${MONITOR_ROOT}/bounded_start_pending.marker"
    echo "$(timestamp) bounded_start_pending" >> "${MONITOR_ROOT}/monitor.log"
  fi
}

sync_logs() {
  mkdir -p "${MONITOR_ROOT}/${RUN_ID}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/logs" "${MONITOR_ROOT}/${RUN_ID}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
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
    git fetch --force origin "refs/heads/results/${RUN_ID}:${result_ref}" \
      >> "${MONITOR_ROOT}/branch.log" 2>> "${MONITOR_ROOT}/branch_stderr.log" || return 1
    git archive "${result_ref}" "results_archive/${RUN_ID}" | tar -x -C "${staging}/${RUN_ID}" --strip-components=2 \
      >> "${MONITOR_ROOT}/branch.log" 2>> "${MONITOR_ROOT}/branch_stderr.log" || return 1
  else
    scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/source/results_archive/${RUN_ID}" "${staging}/" \
      >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  fi
  (cd "${staging}/${RUN_ID}" && sed 's/\r$//' SHA256SUMS | sha256sum -c -) \
    >> "${MONITOR_ROOT}/sha256.log" 2>> "${MONITOR_ROOT}/sha256_stderr.log" || return 1
  [[ "$(tr -d '\r\n' < "${staging}/${RUN_ID}/git_revision.txt")" == "${SOURCE_COMMIT}" ]] || return 1
  cp -a "${staging}/${RUN_ID}" "${destination}" || return 1
  touch "${destination}/${marker}"
  if [[ "${mode}" == "raw" ]]; then
    printf '%s\n' \
      "RAW FALLBACK RETRIEVAL: the remote archive was retrieved directly because its result branch was unavailable or incomplete." \
      "Treat this as fallback-retrieved evidence until local gates pass." \
      > "${destination}/RAW_RETRIEVAL_NOTICE.txt"
  fi
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
    --plan "${destination}/plan.csv" \
    --results "${destination}/results.jsonl" \
    --expected-rows 6 \
    --output "${destination}/validation.local.json" \
    >> "${MONITOR_ROOT}/readjudication.log" 2>> "${MONITOR_ROOT}/readjudication_stderr.log" || return 1
  mkdir -p "${destination}/local_adjudication"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-uknit-r5-k1cb-paper-comparison \
    --plan "${destination}/plan.csv" \
    --results "${destination}/results.jsonl" \
    --progress "${destination}/progress.jsonl" \
    --checkpoint-root "${destination}/checkpoints" \
    --source-cache-audit "${destination}/source_cache_audit.json" \
    --source-k1ca-results "${destination}/source_k1ca/results.jsonl" \
    --source-k1ca-gate "${destination}/source_k1ca/gate.json" \
    --source-commit-file "${destination}/git_revision.txt" \
    --expected-source-commit-file "${destination}/source_expected_commit.txt" \
    --output-root "${destination}/local_adjudication" \
    >> "${MONITOR_ROOT}/readjudication.log" 2>> "${MONITOR_ROOT}/readjudication_stderr.log" || return 1
  cp "${destination}/local_adjudication/gate.json" "${destination}/gate.local.json"
  touch "${destination}/visual_qa_pending.marker"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
    >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || return 1
}

wait_for_source || {
  touch "${MONITOR_ROOT}/source_failed.marker"
  exit 8
}
prepare_launch_gate || {
  touch "${MONITOR_ROOT}/launch_gate_failed.marker"
  echo "$(timestamp) launch_gate_failed" >> "${MONITOR_ROOT}/monitor.log"
  exit 9
}
launch_remote || {
  touch "${MONITOR_ROOT}/launch_failed.marker"
  echo "$(timestamp) launch_failed" >> "${MONITOR_ROOT}/monitor.log"
  exit 10
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
