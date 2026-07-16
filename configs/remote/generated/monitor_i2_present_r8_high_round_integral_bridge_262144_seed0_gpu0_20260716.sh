#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
RUN_ID="i2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
REMOTE_ROOT="${RUNS_ROOT}/${RUN_ID}"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
DESTINATION="outputs/remote_results/${RUN_ID}"

mkdir -p "${MONITOR_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_live_artifacts() {
  mkdir -p "${MONITOR_ROOT}/${RUN_ID}"
  scp -r "${REMOTE}:${REMOTE_ROOT}/logs" "${MONITOR_ROOT}/${RUN_ID}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_ROOT}/results" "${MONITOR_ROOT}/${RUN_ID}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

retrieve_archive() {
  local staging
  staging=$(mktemp -d /tmp/i2-r8-integral-bridge-retrieval.XXXXXX) || return 1
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

  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) remote_failed" >> "${MONITOR_ROOT}/monitor.log"
    exit 1
  fi

  if compgen -G "${MONITOR_ROOT}/${RUN_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    retrieve_archive || exit 2
    (
      cd "${DESTINATION}" || exit 1
      sha256sum -c <(sed 's/\r$//' SHA256SUMS)
    ) >> "${MONITOR_ROOT}/hash.log" 2>> "${MONITOR_ROOT}/hash_stderr.log" || exit 2
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
      --results "${DESTINATION}/results.jsonl" \
      --output "${DESTINATION}/curves.svg" \
      --history-csv "${DESTINATION}/history.csv" \
      --title "创新2 H0：PRESENT 8轮高轮积分神经桥接（262144总训练行，seed 0）" \
      >> "${MONITOR_ROOT}/plot.log" 2>> "${MONITOR_ROOT}/plot_stderr.log" || exit 3
    UV_CACHE_DIR=/tmp/uv-cache uv run python -c \
      "import json,pathlib; from blockcipher_nd.cli.run_innovation2_high_round_integral import validate_artifacts; root=pathlib.Path(r'${DESTINATION}'); report=validate_artifacts(root,expected_rows=4); (root/'validation.local.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\\n',encoding='utf-8'); raise SystemExit(0 if report['status']=='pass' else 1)" \
      >> "${MONITOR_ROOT}/validation.log" 2>> "${MONITOR_ROOT}/validation_stderr.log" || exit 3
    UV_CACHE_DIR=/tmp/uv-cache uv run python -c \
      "import json,pathlib; root=pathlib.Path(r'${DESTINATION}'); gate=json.loads((root/'gate.json').read_text()); data=json.loads((root/'dataset_summary.json').read_text()); cache=json.loads((root/'cache_metadata.json').read_text()); assert all(gate['bridge_plan_checks'].values()); assert gate['artifact_validation']['status']=='pass'; assert data['status']=='pass'; assert len(cache)==3 and all(v['status']=='complete' for v in cache.values())" \
      >> "${MONITOR_ROOT}/readjudication.log" 2>> "${MONITOR_ROOT}/readjudication_stderr.log" || exit 3
    touch "${DESTINATION}/retrieved_from_verified_result_branch.marker"
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
      >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || exit 4
    echo "$(timestamp) verified_results_retrieved_and_indexed" >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi

  echo "$(timestamp) running" >> "${MONITOR_ROOT}/monitor.log"
  sleep 120
done
