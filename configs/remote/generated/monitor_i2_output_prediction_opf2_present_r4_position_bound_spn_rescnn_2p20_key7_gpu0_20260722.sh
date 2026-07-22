#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
SOURCE_COMMIT="${1:?pushed source commit is required}"
RUN_ID="i2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_key7_gpu0_20260722"
REMOTE_DIR="i2_opf2_r4_poshead_2p20_k7_20260722"
ARCHIVE_NAME="i2_opf2_r4_poshead_2p20_k7_20260722"
RESULT_REF="refs/remotes/origin/results/${RUN_ID}"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
REMOTE_ROOT="${RUNS_ROOT}/${REMOTE_DIR}"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
DESTINATION="outputs/remote_results/${RUN_ID}"
OPC1_GATE_SHA256="ebb86a9feab6d2d9993937f5c0a7f4afe1bfe3597c8c1dff083956381e0310b4"
OPN1_GATE_SHA256="887a7db3643e73bdda67958bcaae470881a09db25ab0ba5ff6c3d6bb0a2503d7"
OPD1_GATE_SHA256="3d63163ab94e95b6c8c859be0867cc0a6b1f91382bd842e32dd3adbe04863579"
OPF1_GATE_SHA256="dad638c7180682074f134233e807ed0b58bb40d7d671f7a5d01540721e399354"

mkdir -p "${MONITOR_ROOT}/${RUN_ID}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_live_artifacts() {
  rm -rf "${MONITOR_ROOT}/${RUN_ID}/logs"
  scp -r "${REMOTE}:${REMOTE_ROOT}/logs" "${MONITOR_ROOT}/${RUN_ID}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
  mkdir -p "${MONITOR_ROOT}/${RUN_ID}/results"
  scp "${REMOTE}:${REMOTE_ROOT}/results/progress.jsonl" \
    "${MONITOR_ROOT}/${RUN_ID}/results/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

retrieve_verified_branch() {
  local staging
  staging=$(mktemp -d /tmp/i2-opf2-r4-2p20-retrieval.XXXXXX) || return 1
  if ! git fetch origin "refs/heads/results/${RUN_ID}:${RESULT_REF}" \
    >> "${MONITOR_ROOT}/git_fetch.log" \
    2>> "${MONITOR_ROOT}/git_fetch_stderr.log"; then
    rm -rf "${staging}"
    return 1
  fi
  if ! git archive --format=tar "${RESULT_REF}" \
    "results_archive/${ARCHIVE_NAME}" | tar -xf - -C "${staging}"; then
    rm -rf "${staging}"
    return 1
  fi
  mkdir -p "${DESTINATION}"
  if ! cp -a "${staging}/results_archive/${ARCHIVE_NAME}/." \
    "${DESTINATION}/"; then
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
    retrieve_verified_branch || exit 2
    (
      cd "${DESTINATION}" || exit 1
      sha256sum -c <(sed 's/\r$//' SHA256SUMS)
    ) >> "${MONITOR_ROOT}/hash.log" 2>> "${MONITOR_ROOT}/hash_stderr.log" || exit 2
    UV_CACHE_DIR=/tmp/uv-cache uv run python -c \
      "import csv,hashlib,json,pathlib; root=pathlib.Path(r'${DESTINATION}'); gate=json.loads((root/'gate.json').read_text()); meta=json.loads((root/'metadata.json').read_text()); cache=json.loads((root/'cache_metadata.json').read_text()); checkpoints=json.loads((root/'checkpoint_manifest.json').read_text()); results=[json.loads(line) for line in (root/'results.jsonl').read_text().splitlines() if line.strip()]; source=[root/'opc1_gate.json',root/'opn1_gate.json',root/'opd1_gate.json',root/'opf1_gate.json']; history=list(csv.DictReader((root/'history.csv').open())); revision=(root/'git_revision.txt').read_text().strip(); layout=cache['split_layout']; assert revision=='${SOURCE_COMMIT}' and [hashlib.sha256(p.read_bytes()).hexdigest() for p in source]==['${OPC1_GATE_SHA256}','${OPN1_GATE_SHA256}','${OPD1_GATE_SHA256}','${OPF1_GATE_SHA256}'] and gate['status'] in {'pass','hold'} and all(gate['protocol_checks'].values()) and all(gate['execution_checks'].values()) and meta['sample_classification'] is False and meta['config']['seed']==7 and meta['config']['rounds']==4 and meta['config']['mode']=='scale_extension' and cache['status']=='complete' and cache['completed_rows']==1114112 and layout['train_index_segments']==[[0,131072],[196608,1114112]] and layout['test_index_segment']==[131072,196608] and len(results)==40 and len(history)==500 and len(checkpoints)==5" \
      >> "${MONITOR_ROOT}/validation.log" 2>> "${MONITOR_ROOT}/validation_stderr.log" || exit 3
    MPLCONFIGDIR=/tmp/mplconfig UV_CACHE_DIR=/tmp/uv-cache uv run python \
      -m blockcipher_nd.cli.plot_innovation2_selected_output_position_bound_spn_rescnn \
      --summary "${DESTINATION}/summary.json" \
      --output "${DESTINATION}/curves.svg" \
      >> "${MONITOR_ROOT}/plot.log" 2>> "${MONITOR_ROOT}/plot_stderr.log" || exit 3
    touch "${DESTINATION}/visual_qa_pending.marker"
    touch "${DESTINATION}/retrieved_from_verified_result_branch.marker"
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
      >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || exit 4
    echo "$(timestamp) verified_results_retrieved_plotted_and_indexed_visual_qa_pending" >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi

  echo "$(timestamp) running" >> "${MONITOR_ROOT}/monitor.log"
  sleep 120
done
