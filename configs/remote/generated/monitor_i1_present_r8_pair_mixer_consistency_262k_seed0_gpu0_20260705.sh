#!/usr/bin/env bash
set -u

RUN_ID="i1_present_r8_pair_mixer_consistency_262k_seed0_gpu0_20260705"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"
PLAN="configs/experiment/innovation1/innovation1_spn_present_pair_mixer_consistency_r8_262k_seed0.csv"
PLAN_DOC="docs/experiments/innovation1-present-pair-mixer-consistency-plan.md"
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
anchor_auc = models.get("present_nibble_invp_pair_consistency_spn_only", {}).get("auc")
candidate_auc = models.get("present_nibble_invp_pair_mixer_consistency_spn_only", {}).get("auc")
delta_vs_anchor = None
if isinstance(candidate_auc, (int, float)) and isinstance(anchor_auc, (int, float)):
    delta_vs_anchor = candidate_auc - anchor_auc
if not isinstance(candidate_auc, (int, float)):
    decision = "manual_review_missing_pair_mixer_auc"
elif isinstance(delta_vs_anchor, (int, float)) and delta_vs_anchor >= 0.003:
    decision = "support_pair_mixer_consistency_route"
elif isinstance(delta_vs_anchor, (int, float)) and delta_vs_anchor > 0:
    decision = "weak_pair_mixer_positive_needs_seed_or_scale_check"
else:
    decision = "stop_pair_mixer_route_for_now"
report = {
    "run_id": result_path.stem,
    "status": "medium_diagnostic_only",
    "claim_scope": "PRESENT r8 262144/class single-seed pair-mixer diagnostic, not formal evidence",
    "anchor_model": "present_nibble_invp_pair_consistency_spn_only",
    "candidate_model": "present_nibble_invp_pair_mixer_consistency_spn_only",
    "anchor_auc": anchor_auc,
    "candidate_auc": candidate_auc,
    "delta_vs_anchor": delta_vs_anchor,
    "decision": decision,
    "models": models,
}
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, sort_keys=True))
PY
}

commit_plan_doc_if_changed() {
  if git diff --quiet -- "${PLAN_DOC}"; then
    echo "$(timestamp) plan_doc_unchanged" >> "${MONITOR_DIR}/monitor.log"
    return 0
  fi

  git add "${PLAN_DOC}" >> "${MONITOR_DIR}/git_commit.log" 2>> "${MONITOR_DIR}/git_commit_stderr.log"
  git commit -m "docs: record ${RUN_ID} result" >> "${MONITOR_DIR}/git_commit.log" 2>> "${MONITOR_DIR}/git_commit_stderr.log"
  commit_status=$?
  if [[ "${commit_status}" -ne 0 ]]; then
    echo "$(timestamp) plan_doc_commit_failed" >> "${MONITOR_DIR}/monitor.log"
    return "${commit_status}"
  fi

  git push >> "${MONITOR_DIR}/git_push.log" 2>> "${MONITOR_DIR}/git_push_stderr.log"
  push_status=$?
  if [[ "${push_status}" -ne 0 ]]; then
    echo "$(timestamp) plan_doc_push_failed" >> "${MONITOR_DIR}/monitor.log"
    return "${push_status}"
  fi

  echo "$(timestamp) plan_doc_committed_and_pushed" >> "${MONITOR_DIR}/monitor.log"
  return 0
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
    env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-pair-mixer-consistency \
      --plan "${PLAN}" \
      --results "${result_file}" \
      --output-dir "${LOCAL_ROOT}" \
      --run-id "${RUN_ID}" \
      --expected-rows "${EXPECTED_ROWS}" \
      --update-plan-doc "${PLAN_DOC}" \
      > "${MONITOR_DIR}/postprocess.log" 2> "${MONITOR_DIR}/postprocess_stderr.log"
    postprocess_status=$?
    if [[ "${validate_status}" -eq 0 && "${plot_status}" -eq 0 && "${postprocess_status}" -eq 0 ]]; then
      commit_plan_doc_if_changed
      commit_status=$?
      if [[ "${commit_status}" -eq 0 ]]; then
        echo "$(timestamp) postprocess_done" >> "${MONITOR_DIR}/monitor.log"
      else
        echo "$(timestamp) postprocess_done_commit_failed" >> "${MONITOR_DIR}/monitor.log"
      fi
      exit "${commit_status}"
    fi
    echo "$(timestamp) postprocess_failed validate=${validate_status} plot=${plot_status} postprocess=${postprocess_status}" >> "${MONITOR_DIR}/monitor.log"
    exit 3
  fi

  if compgen -G "${LOCAL_ROOT}/logs/*done.marker" > /dev/null; then
    echo "$(timestamp) completed_missing_or_incomplete_results rows=${result_rows}" >> "${MONITOR_DIR}/monitor.log"
    exit 2
  fi

  echo "$(timestamp) running" >> "${MONITOR_DIR}/monitor.log"
  sleep 840
done
