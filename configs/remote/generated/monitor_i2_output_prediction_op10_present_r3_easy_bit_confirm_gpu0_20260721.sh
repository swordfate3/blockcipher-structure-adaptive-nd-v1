#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
SOURCE_COMMIT="${1:?pushed source commit is required}"
LAUNCHER_ROOT="G:\\lxy\\blockcipher-structure-adaptive-nd-op9-launcher-20260721"
RUN_ID="i2_output_prediction_op10_present_r3_easy_bit_confirm_gpu0_20260721"
SOURCE_RUN_ID="i2_output_prediction_op9_present_r3_kimura_lstm_2p17_seed0_gpu0_20260721"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
REMOTE_ROOT="${RUNS_ROOT}/${RUN_ID}"
SOURCE_LOCAL_MARKER="outputs/remote_results/${SOURCE_RUN_ID}/retrieved_from_verified_result_branch.marker"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
DESTINATION="outputs/remote_results/${RUN_ID}"
LAUNCH_SCRIPT="${LAUNCHER_ROOT}\\configs\\remote\\generated\\launch_i2_output_prediction_op10_present_r3_easy_bit_confirm_gpu0_20260721.cmd"

mkdir -p "${MONITOR_ROOT}/${RUN_ID}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

while [[ ! -f "${SOURCE_LOCAL_MARKER}" ]]; do
  echo "$(timestamp) waiting_for_verified_op9_retrieval" >> "${MONITOR_ROOT}/monitor.log"
  sleep 120
done

echo "$(timestamp) launching_after_verified_op9" >> "${MONITOR_ROOT}/monitor.log"
ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE}" \
  "cmd.exe /c ${LAUNCH_SCRIPT} ${SOURCE_COMMIT}" \
  > "${MONITOR_ROOT}/launch_stdout.log" \
  2> "${MONITOR_ROOT}/launch_stderr.log" || exit 5

sync_live_artifacts() {
  rm -rf "${MONITOR_ROOT}/${RUN_ID}/logs"
  scp -r "${REMOTE}:${REMOTE_ROOT}/logs" "${MONITOR_ROOT}/${RUN_ID}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
  mkdir -p "${MONITOR_ROOT}/${RUN_ID}/results"
  scp "${REMOTE}:${REMOTE_ROOT}/results/progress.jsonl" \
    "${MONITOR_ROOT}/${RUN_ID}/results/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

retrieve_archive() {
  local staging
  staging=$(mktemp -d /tmp/i2-op10-bit-retrieval.XXXXXX) || return 1
  if ! scp -r "${REMOTE}:${REMOTE_ROOT}/source/results_archive/${RUN_ID}" \
    "${staging}/" >> "${MONITOR_ROOT}/scp.log" \
    2>> "${MONITOR_ROOT}/scp_stderr.log"; then
    rm -rf "${staging}"
    return 1
  fi
  mkdir -p "${DESTINATION}"
  if ! cp -a "${staging}/${RUN_ID}/." "${DESTINATION}/"; then
    rm -rf "${staging}"
    return 1
  fi
  rm -rf "${staging}"
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_ROOT}/monitor.log"
  sync_live_artifacts

  if [[ -f "${MONITOR_ROOT}/${RUN_ID}/logs/${RUN_ID}_failed.marker" ]]; then
    echo "$(timestamp) remote_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi

  if [[ -f "${MONITOR_ROOT}/${RUN_ID}/logs/${RUN_ID}_result_branch_pushed.marker" ]]; then
    retrieve_archive || exit 2
    (
      cd "${DESTINATION}" || exit 1
      sha256sum -c <(sed 's/\r$//' SHA256SUMS)
    ) >> "${MONITOR_ROOT}/hash.log" 2>> "${MONITOR_ROOT}/hash_stderr.log" || exit 2
    UV_CACHE_DIR=/tmp/uv-cache uv run python -c \
      "import hashlib,json,pathlib; root=pathlib.Path(r'${DESTINATION}'); gate=json.loads((root/'gate.json').read_text()); cache=json.loads((root/'fresh_cache_metadata.json').read_text()); expected=(root/'candidates.sha256').read_text().split()[0]; actual=hashlib.sha256((root/'candidates.json').read_bytes()).hexdigest(); assert gate['status'] in {'pass','hold'} and all(gate['protocol_checks'].values()) and cache['status']=='complete' and cache['completed_rows']==65536 and actual==expected" \
      >> "${MONITOR_ROOT}/validation.log" 2>> "${MONITOR_ROOT}/validation_stderr.log" || exit 3
    MPLCONFIGDIR=/tmp/mplconfig UV_CACHE_DIR=/tmp/uv-cache uv run python \
      scripts/plot-innovation2-output-bit-discovery \
      --summary "${DESTINATION}/summary.json" \
      --output "${DESTINATION}/curves.svg" \
      >> "${MONITOR_ROOT}/plot.log" 2>> "${MONITOR_ROOT}/plot_stderr.log" || exit 3
    touch "${DESTINATION}/retrieved_from_verified_result_branch.marker"
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
      >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || exit 4
    echo "$(timestamp) verified_results_retrieved_and_indexed" >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi

  echo "$(timestamp) running" >> "${MONITOR_ROOT}/monitor.log"
  sleep 120
done
