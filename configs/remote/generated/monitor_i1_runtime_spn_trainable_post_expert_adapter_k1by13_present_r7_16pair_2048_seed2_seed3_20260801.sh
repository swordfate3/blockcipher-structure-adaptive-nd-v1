#!/usr/bin/env bash
set -u
set -o pipefail

REMOTE="lxy-a6000"
RUN_ID="i1_runtime_spn_trainable_post_expert_adapter_k1by13_present_r7_16pair_2048_seed2_seed3_20260801"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
DESTINATION="outputs/remote_results_incomplete/${RUN_ID}"
SOURCE_COMMIT="${1:-}"
LAUNCH_GATE="${2:-}"

if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || [[ ! -f "${LAUNCH_GATE}" ]]; then
  echo "usage: $0 <pushed-source-commit> <launch-gate.json>" >&2
  exit 6
fi
python -c "import json,pathlib,sys; g=json.loads(pathlib.Path(r'${LAUNCH_GATE}').read_text(encoding='utf-8')); ok=g.get('status') == 'pass' and g.get('decision') == 'innovation1_runtime_spn_k1by13_remote_launch_authorized' and g.get('should_ssh') is True and g.get('ssh_allowed') is True and g.get('launch_authorized') is True and g.get('source_commit') == '${SOURCE_COMMIT}'; sys.exit(0 if ok else 1)" || exit 7
mkdir -p "${MONITOR_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() { date --iso-8601=seconds; }

sync_logs() {
  mkdir -p "${MONITOR_ROOT}/${RUN_ID}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/logs" "${MONITOR_ROOT}/${RUN_ID}/" >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

retrieve_raw() {
  local staging="${MONITOR_ROOT}/staging_${RUN_ID}_$(date +%s)"
  [[ ! -e "${DESTINATION}" ]] || return 1
  mkdir -p "${staging}/${RUN_ID}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/output" "${staging}/${RUN_ID}/" >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  scp -r "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/logs" "${staging}/${RUN_ID}/" >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  scp "${REMOTE}:${RUNS_ROOT}/${RUN_ID}/source_expected_commit.txt" "${staging}/${RUN_ID}/" >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  local remote_revision
  remote_revision="$(tr -d '\r\n' < "${staging}/${RUN_ID}/logs/git_revision.txt")"
  [[ "${remote_revision}" == "${SOURCE_COMMIT}" ]] || return 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
    --plan configs/experiment/innovation1/innovation1_runtime_spn_trainable_post_expert_adapter_k1by13_present_r7_16pair_2048_seed2_seed3.csv \
    --results "${staging}/${RUN_ID}/output/results.jsonl" \
    --expected-rows 8 \
    --output "${staging}/${RUN_ID}/output/validation.local.json" \
    >> "${MONITOR_ROOT}/validation.log" 2>> "${MONITOR_ROOT}/validation_stderr.log" || return 1
  python -c "import json,pathlib,sys; root=pathlib.Path(r'${staging}/${RUN_ID}/output'); gate=json.loads((root/'gate.json').read_text(encoding='utf-8')); validation=json.loads((root/'validation.local.json').read_text(encoding='utf-8')); ok=gate.get('status') in {'pass','hold'} and validation.get('status') == 'pass' and validation.get('result_rows') == 8; sys.exit(0 if ok else 1)" || return 1
  printf '%s\n' \
    "RAW FALLBACK RETRIEVAL: artifacts were retrieved directly from the approved G:/lxy run root." \
    "The remote gate is present and plan validation passed locally; rendered-pixel visual QA and final documentation remain pending." \
    > "${staging}/${RUN_ID}/RAW_RETRIEVAL_NOTICE.txt"
  touch "${staging}/${RUN_ID}/fallback_retrieved.marker"
  cp -a "${staging}/${RUN_ID}" "${DESTINATION}" || return 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || return 1
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_ROOT}/monitor.log"
  sync_logs
  if [[ -f "${MONITOR_ROOT}/${RUN_ID}/logs/raw_ready.marker" ]]; then
    retrieve_raw || exit 3
    touch "${MONITOR_ROOT}/fallback_result_retrieved.marker"
    echo "$(timestamp) fallback_result_retrieved_validated_indexed_visual_qa_pending" >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi
  if [[ -f "${MONITOR_ROOT}/${RUN_ID}/logs/failed.marker" ]]; then
    touch "${MONITOR_ROOT}/remote_failed.marker"
    echo "$(timestamp) remote_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi
  echo "$(timestamp) waiting" >> "${MONITOR_ROOT}/monitor.log"
  sleep 120
done
