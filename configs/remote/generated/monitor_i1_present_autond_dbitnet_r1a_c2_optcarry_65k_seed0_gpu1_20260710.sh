#!/usr/bin/env bash
set -u

RUN_ID="i1_present_autond_dbitnet_r1a_c2_optcarry_65k_seed0_gpu1_20260710"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="G:/lxy/blockcipher-structure-adaptive-nd-runs/${RUN_ID}"
LOCAL_ROOT="outputs/remote_results/${RUN_ID}"
MONITOR_DIR="${LOCAL_ROOT}/monitor"
PLAN="configs/experiment/innovation1/innovation1_spn_present_autond_dbitnet_r1a_c2_optcarry_65k_seed0.csv"
C1_RESULT="outputs/remote_results/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710/results/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710.jsonl"
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
  local c1_file="$3"
  env UV_CACHE_DIR=/tmp/uv-cache uv run python - "${result_file}" "${output_file}" "${c1_file}" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
c1_path = Path(sys.argv[3])
row = json.loads(result_path.read_text(encoding="utf-8"))
c1 = json.loads(c1_path.read_text(encoding="utf-8"))


def stages_by_round(result):
    pretraining = result.get("training", {}).get("pretraining", {})
    stages = {
        stage["rounds"]: stage
        for stage in pretraining.get("curriculum_stages", [])
    }
    stages[result.get("rounds")] = result.get("training", {}) | {
        "rounds": result.get("rounds"),
        "metrics": result.get("metrics", {}),
    }
    return stages


def metric_by_round(stages, metric):
    return {
        str(rounds): stage.get("metrics", {}).get(metric)
        for rounds, stage in sorted(stages.items())
    }


training = row.get("training", {})
pretraining = training.get("pretraining", {})
stages = stages_by_round(row)
c1_stages = stages_by_round(c1)
ordered_rounds = [5, 6, 7, 8, 9]
ordered_stages = [stages.get(rounds, {}) for rounds in ordered_rounds]

optimizer_step_continuity = (
    len(stages) == len(ordered_rounds)
    and ordered_stages[0].get("optimizer_state_reused") is False
    and ordered_stages[0].get("optimizer_state_step_before") == 0
    and all(
        stage.get("optimizer_state_reused") is True
        for stage in ordered_stages[1:]
    )
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
integrity = {
    "model_key": row.get("selected_model") == "autond_dbitnet2023",
    "input_bits": training.get("input_bits") == 128,
    "dilations": row.get("dilations") == [63, 31, 15, 7, 3],
    "amsgrad": training.get("amsgrad") is True,
    "negative_mode": row.get("negative_mode") == "encrypted_random_plaintexts",
    "pairs_per_sample": row.get("pairs_per_sample") == 1,
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
    "optimizer_step_continuity": optimizer_step_continuity,
}
accuracies = metric_by_round(stages, "accuracy")
aucs = metric_by_round(stages, "auc")
c1_accuracies = metric_by_round(c1_stages, "accuracy")
c1_aucs = metric_by_round(c1_stages, "auc")

if not all(integrity.values()):
    decision = "protocol_invalid"
elif accuracies.get("5", 0.0) < 0.75 or accuracies.get("6", 0.0) < 0.60:
    decision = "close_strict_medium_audit_lower_round_gate_failed"
elif accuracies.get("7", 0.0) < 0.52:
    decision = "close_strict_medium_audit_r7_gate_failed"
elif accuracies.get("8", 0.0) <= 0.505:
    decision = "close_strict_medium_audit_r8_gate_failed"
else:
    decision = "strict_medium_gates_passed_design_separate_paper_scale_phase"

report = {
    "run_id": result_path.stem,
    "status": "medium_diagnostic_only",
    "claim_scope": "65536/class single-seed R1A-C2 optimizer-carry attribution; not paper-scale evidence or a ceiling claim",
    "integrity": integrity,
    "optimizer_step_audit": {
        str(rounds): {
            "optimizer_state_reused": stages[rounds].get("optimizer_state_reused"),
            "step_before": stages[rounds].get("optimizer_state_step_before"),
            "step_after": stages[rounds].get("optimizer_state_step_after"),
        }
        for rounds in ordered_rounds
    },
    "stage_accuracy": accuracies,
    "stage_auc": aucs,
    "c1_stage_accuracy": c1_accuracies,
    "c1_stage_auc": c1_aucs,
    "accuracy_delta_vs_c1": {
        rounds: accuracies[rounds] - c1_accuracies[rounds]
        for rounds in sorted(accuracies)
    },
    "auc_delta_vs_c1": {
        rounds: aucs[rounds] - c1_aucs[rounds]
        for rounds in sorted(aucs)
    },
    "decision": decision,
    "next_action": "write a separate public-code-aligned paper-scale plan; do not mechanically scale this strict-medium run",
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
    write_gate "${result_file}" "${LOCAL_ROOT}/${RUN_ID}_gate.json" "${C1_RESULT}" \
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
