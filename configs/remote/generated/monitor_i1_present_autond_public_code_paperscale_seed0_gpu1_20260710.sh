#!/usr/bin/env bash
set -u

RUN_ID="i1_present_autond_public_code_paperscale_seed0_gpu1_20260710"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"
PLAN="configs/experiment/innovation1/innovation1_spn_present_autond_public_code_paperscale_seed0.csv"
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
  env UV_CACHE_DIR=/tmp/uv-cache uv run python - "${result_file}" "${output_file}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
row = json.loads(result_path.read_text(encoding="utf-8"))
training = row.get("training", {})
pretraining = training.get("pretraining", {})
stages = {
    stage["rounds"]: stage
    for stage in pretraining.get("curriculum_stages", [])
}
stages[row.get("rounds")] = training | {"rounds": row.get("rounds")}
ordered_rounds = [5, 6, 7, 8, 9]
ordered_stages = [stages.get(rounds, {}) for rounds in ordered_rounds]
final = row.get("final_evaluation", {})
repeat_metrics = final.get("metrics_by_repeat", [])

optimizer_step_continuity = (
    len(stages) == 5
    and ordered_stages[0].get("optimizer_state_reused") is False
    and ordered_stages[0].get("optimizer_state_step_before") == 0
    and all(stage.get("optimizer_state_reused") is True for stage in ordered_stages[1:])
    and all(
        current.get("optimizer_state_step_before")
        == previous.get("optimizer_state_step_after")
        for previous, current in zip(ordered_stages, ordered_stages[1:])
    )
    and all(
        isinstance(stage.get("optimizer_state_step_before"), int)
        and isinstance(stage.get("optimizer_state_step_after"), int)
        and stage["optimizer_state_step_after"] > stage["optimizer_state_step_before"]
        for stage in ordered_stages
    )
)
exact_split_rows = all(
    stage.get("train_rows") == 10_000_000
    and stage.get("validation_rows") == 1_000_000
    and stage.get("train_positive_rows", 0) > 0
    and stage.get("train_negative_rows", 0) > 0
    and stage.get("train_positive_rows", 0) + stage.get("train_negative_rows", 0)
    == 10_000_000
    and stage.get("validation_positive_rows", 0)
    + stage.get("validation_negative_rows", 0)
    == 1_000_000
    for stage in ordered_stages
)
accuracies = [float(item["accuracy"]) for item in repeat_metrics]
aucs = [float(item["auc"]) for item in repeat_metrics]
final_test_aggregation = (
    final.get("repeats") == 5
    and final.get("samples_total_per_repeat") == 1_000_000
    and final.get("seeds") == [50_000, 50_001, 50_002, 50_003, 50_004]
    and len(repeat_metrics) == 5
    and all(item.get("samples_total") == 1_000_000 for item in repeat_metrics)
    and final.get("accuracy_mean") == statistics.mean(accuracies)
    and final.get("accuracy_std") == statistics.pstdev(accuracies)
    and final.get("auc_mean") == statistics.mean(aucs)
    and final.get("auc_std") == statistics.pstdev(aucs)
)
integrity = {
    "model_key": row.get("selected_model") == "autond_dbitnet2023",
    "input_bits": training.get("input_bits") == 128,
    "dilations": row.get("dilations") == [63, 31, 15, 7, 3],
    "dataset_label_mode": row.get("dataset_label_mode") == "random_labels_total",
    "negative_mode": row.get("negative_mode") == "random_ciphertext",
    "key_rotation_interval": row.get("key_rotation_interval") == 1,
    "round_sequence": pretraining.get("round_sequence") == [5, 6, 7, 8],
    "checkpoint_metric": training.get("checkpoint_metric") == "val_loss",
    "curriculum_checkpoint_metric": all(
        stages[rounds].get("checkpoint_metric") == "val_loss"
        for rounds in [5, 6, 7, 8]
    ),
    "optimizer_state_transition": (
        training.get("optimizer_state_transition") == "carry_across_stages"
        and pretraining.get("optimizer_state_transition") == "carry_across_stages"
    ),
    "exact_split_rows": exact_split_rows,
    "optimizer_step_continuity": optimizer_step_continuity,
    "final_test_aggregation": final_test_aggregation,
}
paper_r9_accuracy = 0.5092
observed_accuracy = final.get("accuracy_mean")
decision = (
    "protocol_invalid"
    if not all(integrity.values())
    else "paper_scale_public_code_comparison_complete"
)
report = {
    "run_id": result_path.stem,
    "status": "paper_scale_public_code_aligned",
    "claim_scope": (
        "AutoND public-code-aligned paper-scale PRESENT r9 reproduction; "
        "not exact paper-and-code agreement because the negative definitions differ"
    ),
    "integrity": integrity,
    "optimizer_step_audit": {
        str(rounds): {
            "reused": stages[rounds].get("optimizer_state_reused"),
            "step_before": stages[rounds].get("optimizer_state_step_before"),
            "step_after": stages[rounds].get("optimizer_state_step_after"),
        }
        for rounds in ordered_rounds
    },
    "final_evaluation": final,
    "paper_r9_accuracy": paper_r9_accuracy,
    "observed_r9_accuracy_mean": observed_accuracy,
    "accuracy_delta_vs_paper": (
        observed_accuracy - paper_r9_accuracy
        if isinstance(observed_accuracy, (int, float))
        else None
    ),
    "decision": decision,
    "next_action": (
        "audit paper/public-code agreement and seed variance before any publication claim"
    ),
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
    write_gate "${result_file}" "${LOCAL_ROOT}/${RUN_ID}_gate.json" \
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
  sleep 1800
done
