#!/usr/bin/env bash
set -euo pipefail

RUN_ID="i2_speck32_hwang_contexts_32plus32_gpu0_20260717"
SOURCE_COMMIT="9e8f3ea35d2a0b691f702791064e7867247270a2"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
DESTINATION="outputs/remote_results/${RUN_ID}"
POSTPROCESS_LOG="${MONITOR_ROOT}/postprocess.log"

mkdir -p "${MONITOR_ROOT}"
touch "${POSTPROCESS_LOG}"

timestamp() {
  date --iso-8601=seconds
}

while [[ ! -f "${DESTINATION}/retrieved_from_verified_result_branch.marker" ]]; do
  if [[ -f "${MONITOR_ROOT}/monitor.log" ]] \
    && grep -q "remote_failed" "${MONITOR_ROOT}/monitor.log"; then
    echo "$(timestamp) retrieval_monitor_failed" >> "${POSTPROCESS_LOG}"
    exit 1
  fi
  sleep 30
done

echo "$(timestamp) verified_archive_detected" >> "${POSTPROCESS_LOG}"
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/validate-innovation2-speck-hwang-contexts \
  --artifact-root "${DESTINATION}" \
  --expected-source-commit "${SOURCE_COMMIT}" \
  >> "${POSTPROCESS_LOG}" 2>&1

UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/plot-innovation2-speck-hwang-contexts \
  --results "${DESTINATION}/results.jsonl" \
  --gate "${DESTINATION}/gate.local.json" \
  --output "${DESTINATION}/curves.svg" \
  >> "${POSTPROCESS_LOG}" 2>&1

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
  >> "${POSTPROCESS_LOG}" 2>&1
touch "${DESTINATION}/visual_qa_pending.marker"
echo "$(timestamp) local_validation_plot_index_complete_visual_qa_pending" \
  >> "${POSTPROCESS_LOG}"
