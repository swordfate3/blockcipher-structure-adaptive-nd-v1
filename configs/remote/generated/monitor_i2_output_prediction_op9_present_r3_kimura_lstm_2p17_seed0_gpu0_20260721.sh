#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
RUN_ID="i2_output_prediction_op9_present_r3_kimura_lstm_2p17_seed0_gpu0_20260721"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
REMOTE_ROOT="${RUNS_ROOT}/${RUN_ID}"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
DESTINATION="outputs/remote_results/${RUN_ID}"

mkdir -p "${MONITOR_ROOT}/${RUN_ID}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_live_artifacts() {
  scp -r "${REMOTE}:${REMOTE_ROOT}/logs" "${MONITOR_ROOT}/${RUN_ID}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
  mkdir -p "${MONITOR_ROOT}/${RUN_ID}/results"
  scp "${REMOTE}:${REMOTE_ROOT}/results/progress.jsonl" \
    "${MONITOR_ROOT}/${RUN_ID}/results/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

retrieve_archive() {
  local staging
  staging=$(mktemp -d /tmp/i2-op9-kimura-retrieval.XXXXXX) || return 1
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
      "import json,pathlib; root=pathlib.Path(r'${DESTINATION}'); gate=json.loads((root/'gate.json').read_text()); meta=json.loads((root/'metadata.json').read_text()); cache=json.loads((root/'cache_metadata.json').read_text()); assert gate['status'] in {'pass','hold'} and all(gate['protocol_checks'].values()) and meta['sample_classification'] is False and cache['status']=='complete' and cache['completed_rows']==196608" \
      >> "${MONITOR_ROOT}/validation.log" 2>> "${MONITOR_ROOT}/validation_stderr.log" || exit 3
    MPLCONFIGDIR=/tmp/mplconfig UV_CACHE_DIR=/tmp/uv-cache uv run python \
      scripts/plot-innovation2-output-prediction-kimura-lstm \
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
