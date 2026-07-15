#!/usr/bin/env bash
set -u

REMOTE="lxy-a6000"
RUNS_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs"
SEED6_ID="i1_gift64_mainstream_performance_1m_seed6"
SEED7_ID="i1_gift64_mainstream_performance_1m_seed7"
JOINT_ID="i1_gift64_mainstream_performance_1m_joint_seed6_seed7"
MONITOR_ROOT="outputs/remote_results_incomplete/i1_gift64_mainstream_performance_1m_monitor"

mkdir -p "${MONITOR_ROOT}"
touch "${MONITOR_ROOT}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_logs() {
  local run_id="$1"
  local destination="${MONITOR_ROOT}/${run_id}"
  mkdir -p "${destination}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${run_id}/logs" "${destination}/" \
    >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
}

retrieve_seed() {
  local run_id="$1"
  local destination="outputs/remote_results/${run_id}"
  mkdir -p "${destination}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${run_id}/source/results_archive/${run_id}/." \
    "${destination}/" >> "${MONITOR_ROOT}/scp.log" \
    2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  scp "${REMOTE}:${RUNS_ROOT}/${run_id}/results/primary_scores.npz" \
    "${destination}/" >> "${MONITOR_ROOT}/scp.log" \
    2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python -c \
    "import hashlib,pathlib; root=pathlib.Path('${destination}'); expected=(root/'primary_scores.sha256').read_text().split()[0]; actual=hashlib.sha256((root/'primary_scores.npz').read_bytes()).hexdigest(); assert actual==expected, (actual, expected)" \
    >> "${MONITOR_ROOT}/score_hash.log" \
    2>> "${MONITOR_ROOT}/score_hash_stderr.log" || return 1
  touch "${destination}/retrieved_from_verified_result_branch.marker"
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
    --results "${destination}/results.jsonl" \
    --output "${destination}/curves.svg" \
    --history-csv "${destination}/history.csv" \
    --title "创新1 GIFT-64 大规模主流网络性能对比" \
    >> "${MONITOR_ROOT}/plot.log" 2>> "${MONITOR_ROOT}/plot_stderr.log" || return 1
}

retrieve_joint() {
  local destination="outputs/remote_results/${JOINT_ID}"
  mkdir -p "${destination}"
  scp -r "${REMOTE}:${RUNS_ROOT}/${JOINT_ID}/results_archive/${JOINT_ID}/." \
    "${destination}/" >> "${MONITOR_ROOT}/scp.log" \
    2>> "${MONITOR_ROOT}/scp_stderr.log" || return 1
  touch "${destination}/retrieved_from_verified_result_branch.marker"
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_ROOT}/monitor.log"
  sync_logs "${SEED6_ID}"
  sync_logs "${SEED7_ID}"

  seed6_ready=false
  seed7_ready=false
  if compgen -G "${MONITOR_ROOT}/${SEED6_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    seed6_ready=true
  fi
  if compgen -G "${MONITOR_ROOT}/${SEED7_ID}/logs/*result_branch_pushed.marker" > /dev/null; then
    seed7_ready=true
  fi

  for run_id in "${SEED6_ID}" "${SEED7_ID}"; do
    recovery_started=false
    if compgen -G "${MONITOR_ROOT}/${run_id}/logs/*recovery_started.marker" > /dev/null; then
      recovery_started=true
    fi
    if [[ "${recovery_started}" == true ]] && compgen -G "${MONITOR_ROOT}/${run_id}/logs/*recovery_failed.marker" > /dev/null; then
      echo "$(timestamp) ${run_id}_recovery_failed" >> "${MONITOR_ROOT}/monitor.log"
      exit 1
    fi
    if [[ "${recovery_started}" != true ]] && compgen -G "${MONITOR_ROOT}/${run_id}/logs/*failed.marker" > /dev/null; then
      echo "$(timestamp) ${run_id}_failed" >> "${MONITOR_ROOT}/monitor.log"
      exit 1
    fi
  done

  if [[ "${seed6_ready}" == true && "${seed7_ready}" == true ]]; then
    mkdir -p "${MONITOR_ROOT}/${JOINT_ID}"
    scp -r "${REMOTE}:${RUNS_ROOT}/${JOINT_ID}" "${MONITOR_ROOT}/${JOINT_ID}/" \
      >> "${MONITOR_ROOT}/scp.log" 2>> "${MONITOR_ROOT}/scp_stderr.log" || true
    if find "${MONITOR_ROOT}/${JOINT_ID}" -name '*recovery_failed.marker' -print -quit | grep -q .; then
      echo "$(timestamp) joint_recovery_failed" >> "${MONITOR_ROOT}/monitor.log"
      exit 1
    fi
    if find "${MONITOR_ROOT}/${JOINT_ID}" -name result_branch_pushed.marker -print -quit | grep -q .; then
      retrieve_seed "${SEED6_ID}" || exit 2
      retrieve_seed "${SEED7_ID}" || exit 2
      retrieve_joint || exit 2
      UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
        --plan "outputs/remote_results/${SEED6_ID}/plan.csv" \
        --results "outputs/remote_results/${SEED6_ID}/results.jsonl" \
        --expected-rows 5 \
        --output "outputs/remote_results/${SEED6_ID}/validation.local.json" \
        >> "${MONITOR_ROOT}/readjudication.log" \
        2>> "${MONITOR_ROOT}/readjudication_stderr.log" || exit 3
      UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
        --plan "outputs/remote_results/${SEED7_ID}/plan.csv" \
        --results "outputs/remote_results/${SEED7_ID}/results.jsonl" \
        --expected-rows 5 \
        --output "outputs/remote_results/${SEED7_ID}/validation.local.json" \
        >> "${MONITOR_ROOT}/readjudication.log" \
        2>> "${MONITOR_ROOT}/readjudication_stderr.log" || exit 3
      UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-cross-spn-mainstream-performance-joint \
        --seed6-gate "outputs/remote_results/${SEED6_ID}/gate.json" \
        --seed7-gate "outputs/remote_results/${SEED7_ID}/gate.json" \
        --output "outputs/remote_results/${JOINT_ID}/gate.local.json" \
        >> "${MONITOR_ROOT}/readjudication.log" \
        2>> "${MONITOR_ROOT}/readjudication_stderr.log" || exit 3
      UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/index-results \
        >> "${MONITOR_ROOT}/index.log" 2>> "${MONITOR_ROOT}/index_stderr.log" || exit 4
      echo "$(timestamp) verified_results_retrieved_and_indexed" >> "${MONITOR_ROOT}/monitor.log"
      exit 0
    fi
  fi

  echo "$(timestamp) running seed6_branch=${seed6_ready} seed7_branch=${seed7_ready}" >> "${MONITOR_ROOT}/monitor.log"
  sleep 300
done
