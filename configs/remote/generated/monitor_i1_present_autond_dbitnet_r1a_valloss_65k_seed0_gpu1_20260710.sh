#!/usr/bin/env bash
set -u

RUN_ID="i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"
PLAN="configs/experiment/innovation1/innovation1_spn_present_autond_dbitnet_r1a_valloss_65k_seed0.csv"
ANCHOR_RESULT="outputs/remote_results/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710/results/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710.jsonl"
EXPECTED_ROWS="1"

mkdir -p "${LOCAL_ROOT}" "${MONITOR_DIR}" "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/results" "${LOCAL_ROOT}/results_archive"
touch "${MONITOR_DIR}/monitor.log"

timestamp() {
  date --iso-8601=seconds
}

sync_artifacts() {
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/logs" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/results" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
  scp -r "${REMOTE}:${REMOTE_RUN_ROOT}/results_archive" "${LOCAL_ROOT}/" >> "${MONITOR_DIR}/scp.log" 2>> "${MONITOR_DIR}/scp_stderr.log" || true
}

write_gate() {
  local result_file="$1"
  local output_file="$2"
  local anchor_file="$3"
  env UV_CACHE_DIR=/tmp/uv-cache uv run python - "${result_file}" "${output_file}" "${anchor_file}" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
anchor_path = Path(sys.argv[3])
row = json.loads(result_path.read_text(encoding="utf-8"))
anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
pretraining = row.get("training", {}).get("pretraining", {})
stages = {stage["rounds"]: stage for stage in pretraining.get("curriculum_stages", [])}
integrity = {
    "model_key": row.get("selected_model") == "autond_dbitnet2023",
    "input_bits": row.get("training", {}).get("input_bits") == 128,
    "dilations": row.get("dilations") == [63, 31, 15, 7, 3],
    "amsgrad": row.get("training", {}).get("amsgrad") is True,
    "negative_mode": row.get("negative_mode") == "encrypted_random_plaintexts",
    "pairs_per_sample": row.get("pairs_per_sample") == 1,
    "round_sequence": pretraining.get("round_sequence") == [5, 6, 7, 8],
    "checkpoint_metric": row.get("training", {}).get("checkpoint_metric") == "val_loss",
    "curriculum_checkpoint_metric": all(
        stage.get("checkpoint_metric") == "val_loss" for stage in stages.values()
    ),
    "optimizer_state_transition": pretraining.get("optimizer_state_transition") == "reset_each_stage",
}
accuracies = {
    str(rounds): stage.get("metrics", {}).get("accuracy")
    for rounds, stage in sorted(stages.items())
}
accuracies["9"] = row.get("metrics", {}).get("accuracy")
aucs = {
    str(rounds): stage.get("metrics", {}).get("auc")
    for rounds, stage in sorted(stages.items())
}
aucs["9"] = row.get("metrics", {}).get("auc")
anchor_pretraining = anchor.get("training", {}).get("pretraining", {})
anchor_stages = {
    stage["rounds"]: stage
    for stage in anchor_pretraining.get("curriculum_stages", [])
}
anchor_accuracies = {
    str(rounds): stage.get("metrics", {}).get("accuracy")
    for rounds, stage in sorted(anchor_stages.items())
}
anchor_accuracies["9"] = anchor.get("metrics", {}).get("accuracy")
anchor_aucs = {
    str(rounds): stage.get("metrics", {}).get("auc")
    for rounds, stage in sorted(anchor_stages.items())
}
anchor_aucs["9"] = anchor.get("metrics", {}).get("auc")
if not all(integrity.values()):
    decision = "protocol_invalid"
elif accuracies.get("5", 0.0) < 0.75 or accuracies.get("6", 0.0) < 0.60:
    decision = "stop_and_audit_lower_round_pipeline"
elif accuracies.get("7", 0.0) < 0.52:
    decision = "stop_and_audit_r7_transfer"
elif accuracies.get("8", 0.0) <= 0.505:
    decision = "stop_scale_up_and_audit_protocol_mismatch"
else:
    decision = "allow_r2_design_review"
report = {
    "run_id": row.get("selected_model") and result_path.stem,
    "status": "medium_diagnostic_only",
    "claim_scope": "65536/class single-seed R1A-C1 checkpoint attribution; not formal evidence or a ceiling claim",
    "integrity": integrity,
    "stage_accuracy": accuracies,
    "stage_auc": aucs,
    "anchor_stage_accuracy": anchor_accuracies,
    "anchor_stage_auc": anchor_aucs,
    "accuracy_delta_vs_r1": {
        rounds: accuracies[rounds] - anchor_accuracies[rounds]
        for rounds in sorted(accuracies)
    },
    "auc_delta_vs_r1": {
        rounds: aucs[rounds] - anchor_aucs[rounds]
        for rounds in sorted(aucs)
    },
    "decision": decision,
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, sort_keys=True))
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

  if [[ -f "${LOCAL_ROOT}/logs/${RUN_ID}_done.marker" && "${result_rows}" -ge "${EXPECTED_ROWS}" ]]; then
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
    write_gate "${result_file}" "${LOCAL_ROOT}/${RUN_ID}_gate.json" "${ANCHOR_RESULT}" \
      > "${MONITOR_DIR}/gate.log" 2> "${MONITOR_DIR}/gate_stderr.log"
    gate_status=$?
    if [[ "${validate_status}" -eq 0 && "${plot_status}" -eq 0 && "${gate_status}" -eq 0 ]]; then
      echo "$(timestamp) postprocess_done" >> "${MONITOR_DIR}/monitor.log"
      exit 0
    fi
    echo "$(timestamp) postprocess_failed validate=${validate_status} plot=${plot_status} gate=${gate_status}" >> "${MONITOR_DIR}/monitor.log"
    exit 3
  fi

  if [[ -f "${LOCAL_ROOT}/logs/${RUN_ID}_done.marker" ]]; then
    echo "$(timestamp) completed_missing_or_incomplete_results rows=${result_rows}" >> "${MONITOR_DIR}/monitor.log"
    exit 2
  fi

  echo "$(timestamp) running" >> "${MONITOR_DIR}/monitor.log"
  sleep 840
done
