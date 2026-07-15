#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
RUN_ID="i1_feistel_sm4_position_resnet_calibration_2048_seed0"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
REMOTE_ROOT="${RUNS_ROOT}/${RUN_ID}"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
DESTINATION="outputs/remote_results/${RUN_ID}"

mkdir -p "${MONITOR_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_logs() {
  mkdir -p "${MONITOR_ROOT}/${RUN_ID}"
  scp -r "${REMOTE}:${REMOTE_ROOT}/logs" "${MONITOR_ROOT}/${RUN_ID}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

retrieve_archive() {
  local staging
  staging=$(mktemp -d /tmp/sm4-position-retrieval.XXXXXX) || return 1
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
  sync_logs

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
    touch "${DESTINATION}/retrieved_from_verified_result_branch.marker"
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
      --plan "${DESTINATION}/plan.csv" \
      --results "${DESTINATION}/results.jsonl" \
      --expected-rows 4 \
      --output "${DESTINATION}/validation.local.json" \
      >> "${MONITOR_ROOT}/readjudication.log" \
      2>> "${MONITOR_ROOT}/readjudication_stderr.log" || exit 3
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-feistel-sm4 \
      --plan "${DESTINATION}/plan.csv" \
      --results "${DESTINATION}/results.jsonl" \
      --samples-per-class 2048 \
      --seeds 0 \
      --epochs 10 \
      --final-repeats 3 \
      --position-calibration \
      --output "${DESTINATION}/gate.local.json" \
      >> "${MONITOR_ROOT}/readjudication.log" \
      2>> "${MONITOR_ROOT}/readjudication_stderr.log" || exit 3
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
      --results "${DESTINATION}/results.jsonl" \
      --output "${DESTINATION}/curves.svg" \
      --history-csv "${DESTINATION}/history.csv" \
      --title "创新1 Feistel / SM4 位置保留论文家族骨干校准" \
      >> "${MONITOR_ROOT}/plot.log" 2>> "${MONITOR_ROOT}/plot_stderr.log" || exit 3
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
      >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || exit 4
    echo "$(timestamp) verified_results_retrieved_and_indexed" >> "${MONITOR_ROOT}/monitor.log"
    exit 0
  fi

  echo "$(timestamp) running" >> "${MONITOR_ROOT}/monitor.log"
  sleep 60
done
