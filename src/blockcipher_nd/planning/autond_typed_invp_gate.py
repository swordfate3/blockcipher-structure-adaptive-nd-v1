from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


MODEL_ROLES = {
    "autond": "autond_dbitnet2023",
    "candidate": "present_nibble_invp_only_spn_only",
    "shuffled_p": "present_nibble_shuffled_paligned_spn_only",
    "delta_only": "present_nibble_delta_only_spn_only",
}


def gate_autond_typed_invp(
    results_path: Path,
    *,
    expected_rows: int = 4,
    required_margin: float = 0.01,
    train_rows: int = 16_384,
    validation_rows: int = 4_096,
    final_repeats: int = 3,
    final_rows: int = 4_096,
) -> dict[str, Any]:
    rows = _load_jsonl(results_path)
    errors = _row_set_errors(rows, expected_rows=expected_rows)
    by_model = {
        str(row.get("selected_model")): row
        for row in rows
        if row.get("selected_model") not in {None, ""}
    }
    for role, model in MODEL_ROLES.items():
        if model not in by_model:
            errors.append(f"missing_model={model}")
            continue
        errors.extend(
            f"{role}:{error}"
            for error in _protocol_errors(
                by_model[model],
                train_rows=train_rows,
                validation_rows=validation_rows,
                final_repeats=final_repeats,
                final_rows=final_rows,
            )
        )
    if errors:
        return {
            "status": "fail",
            "decision": "invalid_protocol",
            "errors": errors,
            "next_action": "repair_protocol_and_rerun_same_matrix",
            "claim_scope": "invalid local protocol evidence",
        }
    return _decision_report(by_model, required_margin=required_margin)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_set_errors(rows: list[dict[str, Any]], *, expected_rows: int) -> list[str]:
    errors: list[str] = []
    if len(rows) != expected_rows:
        errors.append(f"result_rows={len(rows)} expected={expected_rows}")
    models = [str(row.get("selected_model")) for row in rows]
    if len(models) != len(set(models)):
        errors.append("duplicate_selected_model")
    unexpected = sorted(set(models) - set(MODEL_ROLES.values()))
    if unexpected:
        errors.append(f"unexpected_models={unexpected}")
    return errors


def _protocol_errors(
    row: dict[str, Any],
    *,
    train_rows: int,
    validation_rows: int,
    final_repeats: int,
    final_rows: int,
) -> list[str]:
    errors: list[str] = []
    expected_fields = {
        "rounds": 9,
        "seed": 0,
        "train_samples_total": train_rows,
        "validation_samples_total": validation_rows,
        "final_test_samples_total": final_rows,
        "final_test_repeats": final_repeats,
        "dataset_label_mode": "random_labels_total",
        "negative_mode": "random_ciphertext",
        "key_rotation_interval": 1,
        "sample_structure": "independent_pairs",
        "pairs_per_sample": 1,
        "feature_encoding": "ciphertext_pair_bits",
    }
    for field, expected in expected_fields.items():
        if row.get(field) != expected:
            errors.append(f"{field}={row.get(field)} expected={expected}")

    training = row.get("training")
    if not isinstance(training, dict):
        return [*errors, "missing_training"]
    if training.get("optimizer_state_transition") != "carry_across_stages":
        errors.append("optimizer_state_transition")
    pretraining = training.get("pretraining")
    if not isinstance(pretraining, dict):
        return [*errors, "missing_pretraining"]
    if pretraining.get("round_sequence") != [5, 6, 7, 8]:
        errors.append("pretrain_round_sequence")
    if pretraining.get("optimizer_state_transition") != "carry_across_stages":
        errors.append("pretrain_optimizer_state_transition")
    curriculum = pretraining.get("curriculum_stages")
    if not isinstance(curriculum, list) or len(curriculum) != 4:
        return [*errors, "curriculum_stage_count"]
    stages = [*curriculum, {**training, "rounds": 9}]
    errors.extend(
        _stage_errors(
            stages,
            train_rows=train_rows,
            validation_rows=validation_rows,
        )
    )
    errors.extend(
        _final_errors(
            row.get("final_evaluation"),
            seed=int(row.get("seed", 0)),
            repeats=final_repeats,
            rows=final_rows,
        )
    )
    return errors


def _stage_errors(
    stages: list[dict[str, Any]],
    *,
    train_rows: int,
    validation_rows: int,
) -> list[str]:
    errors: list[str] = []
    if [stage.get("rounds") for stage in stages] != [5, 6, 7, 8, 9]:
        errors.append("round_stage_order")
    for index, stage in enumerate(stages):
        rounds = stage.get("rounds")
        if stage.get("checkpoint_metric") != "val_loss":
            errors.append(f"r{rounds}:checkpoint_metric")
        if stage.get("dataset_label_mode") != "random_labels_total":
            errors.append(f"r{rounds}:dataset_label_mode")
        if stage.get("train_rows") != train_rows:
            errors.append(f"r{rounds}:train_rows")
        if stage.get("validation_rows") != validation_rows:
            errors.append(f"r{rounds}:validation_rows")
        errors.extend(_label_count_errors(stage, rounds=rounds, split="train", rows=train_rows))
        errors.extend(
            _label_count_errors(
                stage,
                rounds=rounds,
                split="validation",
                rows=validation_rows,
            )
        )
        before = stage.get("optimizer_state_step_before")
        after = stage.get("optimizer_state_step_after")
        if not isinstance(before, int) or not isinstance(after, int) or after <= before:
            errors.append(f"r{rounds}:optimizer_steps_increasing")
        expected_reuse = index != 0
        if stage.get("optimizer_state_reused") is not expected_reuse:
            errors.append(f"r{rounds}:optimizer_state_reused")
        if index == 0 and before != 0:
            errors.append("r5:optimizer_step_before")
        if index and before != stages[index - 1].get("optimizer_state_step_after"):
            errors.append(f"r{rounds}:optimizer_step_continuity")
    return errors


def _label_count_errors(
    stage: dict[str, Any],
    *,
    rounds: Any,
    split: str,
    rows: int,
) -> list[str]:
    positive = stage.get(f"{split}_positive_rows")
    negative = stage.get(f"{split}_negative_rows")
    if not isinstance(positive, int) or not isinstance(negative, int):
        return [f"r{rounds}:{split}_label_counts"]
    if positive <= 0 or negative <= 0 or positive + negative != rows:
        return [f"r{rounds}:{split}_label_counts"]
    return []


def _final_errors(final: Any, *, seed: int, repeats: int, rows: int) -> list[str]:
    if not isinstance(final, dict):
        return ["missing_final_evaluation"]
    errors: list[str] = []
    expected_seeds = [seed + 50_000 + index for index in range(repeats)]
    if final.get("repeats") != repeats:
        errors.append("final_repeats")
    if final.get("samples_total_per_repeat") != rows:
        errors.append("final_rows")
    if final.get("seeds") != expected_seeds:
        errors.append("final_seeds")
    metrics = final.get("metrics_by_repeat")
    if not isinstance(metrics, list) or len(metrics) != repeats:
        return [*errors, "final_metric_count"]
    for index, metric in enumerate(metrics):
        if metric.get("seed") != expected_seeds[index] or metric.get("samples_total") != rows:
            errors.append(f"final_repeat_{index + 1}:identity")
        positive = metric.get("positive_rows")
        negative = metric.get("negative_rows")
        if (
            not isinstance(positive, int)
            or not isinstance(negative, int)
            or positive <= 0
            or negative <= 0
            or positive + negative != rows
        ):
            errors.append(f"final_repeat_{index + 1}:label_counts")
        for field in ("accuracy", "auc"):
            if not isinstance(metric.get(field), (int, float)):
                errors.append(f"final_repeat_{index + 1}:{field}")
    if errors:
        return errors
    accuracies = [float(metric["accuracy"]) for metric in metrics]
    aucs = [float(metric["auc"]) for metric in metrics]
    expected_aggregates = {
        "accuracy_mean": mean(accuracies),
        "accuracy_std": pstdev(accuracies),
        "auc_mean": mean(aucs),
        "auc_std": pstdev(aucs),
    }
    for field, expected in expected_aggregates.items():
        observed = final.get(field)
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(f"final_aggregation:{field}")
    return errors


def _decision_report(
    by_model: dict[str, dict[str, Any]],
    *,
    required_margin: float,
) -> dict[str, Any]:
    role_metrics = {
        role: {
            "model": model,
            "accuracy_mean": float(by_model[model]["final_evaluation"]["accuracy_mean"]),
            "accuracy_std": float(by_model[model]["final_evaluation"]["accuracy_std"]),
            "auc_mean": float(by_model[model]["final_evaluation"]["auc_mean"]),
            "auc_std": float(by_model[model]["final_evaluation"]["auc_std"]),
            "metrics_by_repeat": by_model[model]["final_evaluation"]["metrics_by_repeat"],
        }
        for role, model in MODEL_ROLES.items()
    }
    candidate_auc = role_metrics["candidate"]["auc_mean"]
    control_aucs = {
        role: metrics["auc_mean"]
        for role, metrics in role_metrics.items()
        if role != "candidate"
    }
    margins = {role: candidate_auc - auc for role, auc in control_aucs.items()}
    margin_vs_best = min(margins.values())
    repeatwise = all(
        float(candidate["auc"])
        > max(
            float(role_metrics[role]["metrics_by_repeat"][index]["auc"])
            for role in control_aucs
        )
        for index, candidate in enumerate(role_metrics["candidate"]["metrics_by_repeat"])
    )
    candidate_best = all(margin > 0.0 for margin in margins.values())
    if candidate_best and margin_vs_best >= required_margin and repeatwise:
        decision = "strong_local_support"
        next_action = "run_identical_seed1_local_gate"
    elif candidate_best:
        decision = "weak_or_fragile"
        next_action = "run_seed1_bounded_variance_adjudication"
    else:
        decision = "stop_public_typed_adapter"
        next_action = "do_not_scale_or_redesign_public_typed_adapter"
    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "required_margin": required_margin,
        "models": role_metrics,
        "candidate_margins_auc": margins,
        "candidate_margin_vs_best_control_auc": margin_vs_best,
        "candidate_above_all_controls_by_repeat": repeatwise,
        "next_action": next_action,
        "claim_scope": (
            "local AutoND public-code protocol representation diagnostic; "
            "not strict-negative, paper-scale, or novelty evidence"
        ),
    }


__all__ = ["MODEL_ROLES", "gate_autond_typed_invp"]
