#!/usr/bin/env bash
set -u

RUN_ID="i1_present_r8_pairset_1m_seed0_gpu1_20260705"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"
PLAN="configs/experiment/innovation1/innovation1_spn_present_pairset_r8_1m_seed0.csv"
EXPECTED_ROWS="2"

mkdir -p "${LOCAL_ROOT}" "${MONITOR_DIR}" "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/results"
touch "${MONITOR_DIR}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_artifacts() {
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/logs" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/results" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
}

write_gate_note() {
  result_file="${LOCAL_ROOT}/results/${RUN_ID}.jsonl"
  env UV_CACHE_DIR=/tmp/uv-cache uv run python - "$result_file" "${LOCAL_ROOT}/${RUN_ID}_gate_note.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]

def metric(row, name):
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get(name)
    return row.get(name)

models = {
    str(row.get("model") or row.get("selected_model") or row.get("model_key")): {
        "auc": metric(row, "auc"),
        "accuracy": metric(row, "accuracy"),
        "calibrated_accuracy": metric(row, "calibrated_accuracy"),
        "loss": metric(row, "loss"),
    }
    for row in rows
}
baseline_auc = models.get("present_zhang_wang_keras_mcnd", {}).get("auc")
candidate_auc = models.get("present_nibble_invp_pair_consistency_spn_only", {}).get("auc")
delta_vs_baseline = None
if isinstance(candidate_auc, (int, float)) and isinstance(baseline_auc, (int, float)):
    delta_vs_baseline = candidate_auc - baseline_auc
if not isinstance(candidate_auc, (int, float)):
    decision = "manual_review_missing_pairset_auc"
elif isinstance(delta_vs_baseline, (int, float)) and delta_vs_baseline >= 0.005:
    decision = "support_r8_pairset_1m_confirmation"
elif isinstance(delta_vs_baseline, (int, float)) and delta_vs_baseline > 0:
    decision = "weak_r8_pairset_1m_positive_needs_seed1_or_controls"
else:
    decision = "stop_or_rethink_r8_pairset_scale"
report = {
    "run_id": result_path.stem,
    "status": "paper_scale_single_seed_only",
    "claim_scope": "PRESENT r8 1000000/class single-seed pair-set confirmation, not formal evidence",
    "candidate_model": "present_nibble_invp_pair_consistency_spn_only",
    "candidate_auc": candidate_auc,
    "baseline_auc": baseline_auc,
    "delta_vs_baseline": delta_vs_baseline,
    "decision": decision,
    "models": models,
}
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, sort_keys=True))
PY
}

while true; do
  echo "$(timestamp) sync" >> "${MONITOR_DIR}/monitor.log"
  sync_artifacts

  if compgen -G "${LOCAL_ROOT}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) failed" >> "${MONITOR_DIR}/monitor.log"
    exit 1
  fi

  result_file="${LOCAL_ROOT}/results/${RUN_ID}.jsonl"
  result_rows=0
  if [[ -f "${result_file}" ]]; then
    result_rows=$(grep -cve '^[[:space:]]*$' "${result_file}" || true)
  fi

  if [[ "${result_rows}" -ge "${EXPECTED_ROWS}" ]]; then
    echo "$(timestamp) result_ready" >> "${MONITOR_DIR}/monitor.log"
    env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
      --plan "${PLAN}" \
      --results "${result_file}" \
      --expected-rows "${EXPECTED_ROWS}" \
      --output "${LOCAL_ROOT}/${RUN_ID}_validation.json" \
      > "${MONITOR_DIR}/validate.log" 2> "${MONITOR_DIR}/validate_stderr.log"
    validate_status=$?
    env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
      --results "${result_file}" \
      --output "${LOCAL_ROOT}/${RUN_ID}_curves.svg" \
      --history-csv "${LOCAL_ROOT}/${RUN_ID}_history.csv" \
      --title "${RUN_ID}" \
      > "${MONITOR_DIR}/plot.log" 2> "${MONITOR_DIR}/plot_stderr.log"
    plot_status=$?
    write_gate_note > "${MONITOR_DIR}/gate_note.log" 2> "${MONITOR_DIR}/gate_note_stderr.log"
    gate_status=$?
    if [[ "${validate_status}" -eq 0 && "${plot_status}" -eq 0 && "${gate_status}" -eq 0 ]]; then
      echo "$(timestamp) postprocess_done" >> "${MONITOR_DIR}/monitor.log"
      exit 0
    fi
    echo "$(timestamp) postprocess_failed validate=${validate_status} plot=${plot_status} gate=${gate_status}" >> "${MONITOR_DIR}/monitor.log"
    exit 3
  fi

  if compgen -G "${LOCAL_ROOT}/logs/*done.marker" > /dev/null; then
    echo "$(timestamp) completed_missing_or_incomplete_results rows=${result_rows}" >> "${MONITOR_DIR}/monitor.log"
    exit 2
  fi

  echo "$(timestamp) running" >> "${MONITOR_DIR}/monitor.log"
  sleep 840
done
