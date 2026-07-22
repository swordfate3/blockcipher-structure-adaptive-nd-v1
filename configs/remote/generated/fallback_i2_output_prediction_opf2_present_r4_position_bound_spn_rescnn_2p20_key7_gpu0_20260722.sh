#!/usr/bin/env bash
set -u

RUN_ID="i2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_key7_gpu0_20260722"
REMOTE="lxy-a6000"
REMOTE_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/i2_opf2_r4_poshead_2p20_k7_20260722"
MONITOR_ROOT="outputs/remote_results_incomplete/${RUN_ID}_monitor"
LIVE_ROOT="${MONITOR_ROOT}/${RUN_ID}"
VERIFIED_ROOT="outputs/remote_results/${RUN_ID}"
FALLBACK_ROOT="outputs/remote_results_incomplete/${RUN_ID}_raw_fallback"
GUARD_LOG="${MONITOR_ROOT}/fallback_guardian.log"
MAX_CHECKS=720
PUSHED_GRACE_CHECKS=5
pushed_checks=0

mkdir -p "${MONITOR_ROOT}"

timestamp() {
  date --iso-8601=seconds
}

record() {
  printf '%s %s\n' "$(timestamp)" "$1" >> "${GUARD_LOG}"
}

retrieve_raw() {
  local reason="$1"
  local incoming
  local history_root
  incoming=$(mktemp -d /tmp/i2-opf2-r4-2p20-raw-fallback.XXXXXX) || return 1
  if ! scp -r "${REMOTE}:${REMOTE_ROOT}/logs" "${incoming}/" \
    >> "${GUARD_LOG}" 2>&1; then
    rm -rf "${incoming}"
    return 1
  fi
  if ! scp -r "${REMOTE}:${REMOTE_ROOT}/results" "${incoming}/" \
    >> "${GUARD_LOG}" 2>&1; then
    rm -rf "${incoming}"
    return 1
  fi
  if [[ -e "${FALLBACK_ROOT}" ]]; then
    history_root="${FALLBACK_ROOT}_history_$(date +%Y%m%d-%H%M%S)"
    mv "${FALLBACK_ROOT}" "${history_root}" || {
      rm -rf "${incoming}"
      return 1
    }
    record "archived_prior_fallback root=${history_root}"
  fi
  mv "${incoming}" "${FALLBACK_ROOT}" || return 1
  {
    printf '%s\n' "# Raw Fallback Retrieval"
    printf '\n'
    printf '%s\n' "Retrieved from \`${REMOTE}:${REMOTE_ROOT}\` after \`${reason}\`."
    printf '%s\n' "This directory is incomplete remote evidence, not a verified result-branch archive."
    printf '%s\n' "Validate source revision, hashes, cache, split, and all result artifacts before interpretation."
  } > "${FALLBACK_ROOT}/RAW_RETRIEVAL_NOTICE.md"
  printf 'reason=%s\nretrieved_at=%s\n' "${reason}" "$(timestamp)" \
    > "${FALLBACK_ROOT}/raw_fallback_retrieved.marker"
  if [[ -f "${FALLBACK_ROOT}/results/gate.json" \
    || -f "${FALLBACK_ROOT}/results/results.jsonl" ]]; then
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
      >> "${GUARD_LOG}" 2>&1 || return 1
  fi
  record "raw_fallback_retrieved reason=${reason} root=${FALLBACK_ROOT}"
}

record "fallback_guardian_started"
for ((check=1; check<=MAX_CHECKS; check++)); do
  if [[ -f "${VERIFIED_ROOT}/retrieved_from_verified_result_branch.marker" ]]; then
    record "verified_retrieval_present"
    exit 0
  fi
  if [[ -f "${LIVE_ROOT}/logs/${RUN_ID}_failed.marker" ]]; then
    if retrieve_raw "remote_failed_marker"; then
      exit 0
    fi
    record "raw_fallback_retry_after_remote_failed check=${check}"
  elif [[ -f "${LIVE_ROOT}/logs/${RUN_ID}_result_branch_pushed.marker" ]]; then
    pushed_checks=$((pushed_checks + 1))
    if [[ ${pushed_checks} -ge ${PUSHED_GRACE_CHECKS} ]]; then
      if retrieve_raw "verified_branch_retrieval_timeout"; then
        exit 0
      fi
      record "raw_fallback_retry_after_branch_timeout check=${check}"
    fi
  else
    pushed_checks=0
  fi
  sleep 120
done

record "fallback_guardian_timeout"
exit 3
