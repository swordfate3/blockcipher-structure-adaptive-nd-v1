from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.evaluation.neural_ensemble import load_score_artifact
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment
from blockcipher_nd.training.metrics import binary_auc


MODEL_ROLES = {
    "typed_scratch": "gift_cross_spn_typed_cell_true",
    "typed_source0": "gift_cross_spn_typed_cell_true_from_present_true_s0",
    "typed_source1": "gift_cross_spn_typed_cell_true_from_present_true_s1",
    "lstm": "gift64_sun_style_lstm_pairset",
    "resnet": "gift64_gohr_style_resnet_pairset",
}


def performance_decision(
    *,
    primary_aucs: dict[str, float],
    final_mean_aucs: dict[str, float],
    epoch1_aucs: dict[str, float],
) -> dict[str, Any]:
    strongest_mainstream_role = max(
        ("lstm", "resnet"),
        key=lambda role: final_mean_aucs[role],
    )
    strongest_mainstream_auc = final_mean_aucs[strongest_mainstream_role]
    source_roles = ("typed_source0", "typed_source1")
    mainstream_margins = {
        role: final_mean_aucs[role] - strongest_mainstream_auc
        for role in source_roles
    }
    scratch_margins = {
        role: final_mean_aucs[role] - final_mean_aucs["typed_scratch"]
        for role in source_roles
    }
    epoch1_scratch_margins = {
        role: epoch1_aucs[role] - epoch1_aucs["typed_scratch"]
        for role in source_roles
    }
    primary_mainstream_margins = {
        role: primary_aucs[role] - primary_aucs[strongest_mainstream_role]
        for role in source_roles
    }

    superior = all(value >= 0.002 for value in mainstream_margins.values())
    competitive = all(value >= -0.001 for value in mainstream_margins.values())
    persistent_transfer = all(value >= 0.002 for value in scratch_margins.values())
    epoch1_transfer = all(value >= 0.004 for value in epoch1_scratch_margins.values())
    primary_ordering = all(value > 0.0 for value in primary_mainstream_margins.values())

    if superior and primary_ordering:
        decision = "large_scale_mainstream_superiority_candidate"
        next_action = "run_paired_interval_gate_then_freeze_exact_sun_protocol_plan"
    elif competitive:
        decision = "large_scale_mainstream_competitive_no_superiority"
        next_action = "retain_typed_method_without_accuracy_lead"
    else:
        decision = "large_scale_mainstream_performance_not_supported"
        next_action = "stop_performance_lead_claim_retain_topology_attribution"
    return {
        "decision": decision,
        "next_action": next_action,
        "strongest_mainstream_role": strongest_mainstream_role,
        "strongest_mainstream_auc": strongest_mainstream_auc,
        "mainstream_margins": mainstream_margins,
        "scratch_margins": scratch_margins,
        "epoch1_scratch_margins": epoch1_scratch_margins,
        "primary_mainstream_margins": primary_mainstream_margins,
        "gates": {
            "mainstream_superiority": superior,
            "mainstream_competitiveness": competitive,
            "persistent_transfer": persistent_transfer,
            "epoch1_source_robust_adaptation": epoch1_transfer,
            "primary_repeat_ordering": primary_ordering,
        },
    }


def gate_cross_spn_mainstream_performance(
    *,
    plan_path: Path,
    results_path: Path,
    score_artifact_paths: dict[str, Path],
    expected_seed: int,
    samples_per_class: int = 1_000_000,
    epochs: int = 10,
) -> dict[str, Any]:
    errors: list[str] = []
    if expected_seed not in {6, 7}:
        errors.append(f"expected_seed must be 6 or 7: {expected_seed}")
    rows = _read_jsonl(results_path, errors)
    alignment = validate_result_plan_alignment(plan_path, results_path, expected_rows=5)
    errors.extend(alignment["errors"])
    rows_by_model = {str(row.get("model")): row for row in rows}
    if len(rows_by_model) != len(rows):
        errors.append("result model keys must be unique")
    if set(rows_by_model) != set(MODEL_ROLES.values()):
        errors.append(
            f"result models mismatch expected={sorted(MODEL_ROLES.values())} "
            f"actual={sorted(rows_by_model)}"
        )

    final_mean_aucs: dict[str, float] = {}
    epoch1_aucs: dict[str, float] = {}
    parameter_counts: dict[str, int] = {}
    for role, model_key in MODEL_ROLES.items():
        row = rows_by_model.get(model_key)
        if row is None:
            continue
        expected_fields = {
            "cipher_key": "gift64",
            "rounds": 6,
            "seed": expected_seed,
            "samples_per_class": samples_per_class,
            "pairs_per_sample": 4,
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "independent_pairs",
            "final_test_repeats": 5,
            "final_test_samples_total": 1_000_000,
            "final_test_key": int("22" * 16, 16),
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                errors.append(
                    f"{model_key} {field}={row.get(field)!r} expected={expected!r}"
                )
        training = row.get("training")
        if not isinstance(training, dict) or training.get("epochs") != epochs:
            errors.append(f"{model_key} training epochs must equal {epochs}")
        history = row.get("history")
        if not isinstance(history, list) or len(history) != epochs:
            errors.append(f"{model_key} history must contain {epochs} epochs")
        else:
            epoch1_aucs[role] = float(history[0]["val_auc"])
        final = row.get("final_evaluation")
        if not isinstance(final, dict):
            errors.append(f"{model_key} missing final_evaluation")
        else:
            metrics_by_repeat = final.get("metrics_by_repeat")
            if not isinstance(metrics_by_repeat, list) or len(metrics_by_repeat) != 5:
                errors.append(f"{model_key} final evaluation must contain five repeats")
            if final.get("samples_total_per_repeat") != 1_000_000:
                errors.append(f"{model_key} final evaluation rows must equal 1000000")
            if final.get("final_test_key") != int("22" * 16, 16):
                errors.append(f"{model_key} final evaluation key mismatch")
            if isinstance(final.get("auc_mean"), (int, float)):
                final_mean_aucs[role] = float(final["auc_mean"])
        if isinstance(row.get("parameter_count"), int):
            parameter_counts[role] = int(row["parameter_count"])

    artifacts = {}
    for role, path in score_artifact_paths.items():
        if role not in MODEL_ROLES:
            errors.append(f"unknown score role: {role}")
            continue
        try:
            artifacts[role] = load_score_artifact(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot load score artifact {role}: {exc}")
    if set(artifacts) != set(MODEL_ROLES):
        errors.append(
            f"score roles mismatch expected={sorted(MODEL_ROLES)} actual={sorted(artifacts)}"
        )

    primary_aucs: dict[str, float] = {}
    if artifacts:
        first = next(iter(artifacts.values()))
        for role, artifact in artifacts.items():
            if not np.array_equal(first.labels, artifact.labels):
                errors.append(f"score labels differ: {role}")
            if not np.array_equal(first.sample_ids, artifact.sample_ids):
                errors.append(f"score sample_ids differ: {role}")
            metadata = artifact.metadata
            if metadata.get("model_key") != MODEL_ROLES[role]:
                errors.append(f"score model mismatch: {role}")
            if metadata.get("score_split") != "final_test_1":
                errors.append(f"score split mismatch: {role}")
            if metadata.get("score_samples_total") != 1_000_000:
                errors.append(f"score rows mismatch: {role}")
            if metadata.get("score_key") != int("22" * 16, 16):
                errors.append(f"score key mismatch: {role}")
            primary_aucs[role] = binary_auc(
                artifact.labels,
                artifact.probabilities,
            )

    if errors:
        return {
            "status": "fail",
            "decision": "invalid_large_scale_mainstream_performance_protocol",
            "errors": errors,
            "expected_seed": expected_seed,
            "alignment": alignment,
            "research_decision_applied": False,
        }

    decision = performance_decision(
        primary_aucs=primary_aucs,
        final_mean_aucs=final_mean_aucs,
        epoch1_aucs=epoch1_aucs,
    )
    return {
        "status": "pass",
        "errors": [],
        "expected_seed": expected_seed,
        "samples_per_class": samples_per_class,
        "epochs": epochs,
        "models": MODEL_ROLES,
        "primary_fresh_test_aucs": primary_aucs,
        "five_repeat_mean_aucs": final_mean_aucs,
        "epoch1_validation_aucs": epoch1_aucs,
        "parameter_counts": parameter_counts,
        "score_rows": len(next(iter(artifacts.values())).labels),
        "score_pairing": "identical final_test_1 labels and sample_ids",
        "alignment": alignment,
        "research_decision_applied": True,
        "claim_scope": (
            "GIFT-64 r6 1000000/class same-protocol multi-architecture large-scale "
            "benchmark cell; not exact Sun protocol, paired-CI evidence, SOTA, or breakthrough"
        ),
        "stopped_actions": [
            "e5_e6_scale_rescue",
            "posthoc_model_sweep",
            "mechanical_5000000_per_class_scale",
        ],
        **decision,
    }


def joint_mainstream_performance_gate(seed_gates: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if len(seed_gates) != 2:
        errors.append("joint gate requires exactly two target-seed gates")
    seeds = {gate.get("expected_seed") for gate in seed_gates}
    if seeds != {6, 7}:
        errors.append(f"joint gate seeds must equal [6, 7]: {sorted(seeds)}")
    for gate in seed_gates:
        if gate.get("status") != "pass" or gate.get("errors") != []:
            errors.append(f"seed{gate.get('expected_seed')} gate is not valid")
    if errors:
        return {
            "status": "fail",
            "decision": "invalid_joint_large_scale_mainstream_performance_protocol",
            "errors": errors,
            "research_decision_applied": False,
        }
    decisions = {gate["expected_seed"]: gate["decision"] for gate in seed_gates}
    if all(
        decision == "large_scale_mainstream_superiority_candidate"
        for decision in decisions.values()
    ):
        decision = "two_seed_large_scale_mainstream_superiority_candidate"
        next_action = "run_paired_interval_gate_then_freeze_exact_sun_protocol_plan"
    elif all(
        decision
        in {
            "large_scale_mainstream_superiority_candidate",
            "large_scale_mainstream_competitive_no_superiority",
        }
        for decision in decisions.values()
    ):
        decision = "two_seed_large_scale_mainstream_competitive_no_superiority"
        next_action = "retain_typed_method_without_accuracy_lead"
    else:
        decision = "two_seed_large_scale_mainstream_performance_not_supported"
        next_action = "stop_performance_lead_claim_retain_topology_attribution"
    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "seed_decisions": decisions,
        "next_action": next_action,
        "research_decision_applied": True,
        "claim_scope": (
            "two-target-seed GIFT-64 r6 1000000/class large-scale same-protocol "
            "architecture comparison; not exact paper-protocol or SOTA evidence"
        ),
    }


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                errors.append(f"result line {line_number} must be an object")
            else:
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read results: {exc}")
    return rows
