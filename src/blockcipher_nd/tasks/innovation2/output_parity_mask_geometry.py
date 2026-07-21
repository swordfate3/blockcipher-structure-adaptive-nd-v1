from __future__ import annotations

import math
from typing import Any

import numpy as np

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.tasks.innovation2.output_parity_prediction import (
    MASKS as CONTIGUOUS_MASKS,
    OutputParityPredictionConfig,
    generate_output_prediction_data,
    multilabel_metrics,
    parity_probabilities_from_bit_probabilities,
    train_output_prediction_row,
    validate_output_prediction_contract,
)


RUN_ID = "i2_output_parity_prediction_op2_mask_geometry_present_r1_seed0_20260721"
ALIGNED_MASKS = tuple(
    Present80.permutation_layer(0xF << (4 * sbox_index)) for sbox_index in range(16)
)


def build_mask_geometry_data(
    config: OutputParityPredictionConfig,
) -> dict[str, dict[str, Any]]:
    return {
        "contiguous": generate_output_prediction_data(config, CONTIGUOUS_MASKS),
        "aligned": generate_output_prediction_data(config, ALIGNED_MASKS),
    }


def validate_mask_geometry_contract(
    config: OutputParityPredictionConfig,
    datasets: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    contiguous = datasets["contiguous"]
    aligned = datasets["aligned"]
    contiguous_checks = validate_output_prediction_contract(config, contiguous)
    aligned_checks = validate_output_prediction_contract(config, aligned)
    same_arrays = all(
        np.array_equal(
            contiguous[split_name].plaintexts, aligned[split_name].plaintexts
        )
        and np.array_equal(
            contiguous[split_name].features, aligned[split_name].features
        )
        and np.array_equal(
            contiguous[split_name].full_targets,
            aligned[split_name].full_targets,
        )
        for split_name in ("train", "validation", "test")
    )
    return {
        "contiguous_output_prediction_contract_passed": all(contiguous_checks.values()),
        "aligned_output_prediction_contract_passed": all(aligned_checks.values()),
        "same_fixed_secret_key": contiguous["secret_key"] == aligned["secret_key"],
        "same_plaintexts_features_and_full_outputs": same_arrays,
        "same_ciphertexts": np.array_equal(
            contiguous["ciphertexts"], aligned["ciphertexts"]
        ),
        "sixteen_distinct_contiguous_masks": len(set(CONTIGUOUS_MASKS)) == 16,
        "sixteen_distinct_aligned_masks": len(set(ALIGNED_MASKS)) == 16,
        "all_masks_have_weight_four": all(
            mask.bit_count() == 4 for mask in (*CONTIGUOUS_MASKS, *ALIGNED_MASKS)
        ),
        "each_geometry_covers_output_once": sum(CONTIGUOUS_MASKS) == (1 << 64) - 1
        and sum(ALIGNED_MASKS) == (1 << 64) - 1,
        "aligned_masks_are_present_p_layer_images": all(
            aligned_mask == Present80.permutation_layer(contiguous_mask)
            for contiguous_mask, aligned_mask in zip(
                CONTIGUOUS_MASKS, ALIGNED_MASKS, strict=True
            )
        ),
        "labels_are_ciphertext_outputs_not_sample_classes": True,
    }


def train_mask_geometry_matrix(
    config: OutputParityPredictionConfig,
    datasets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    contiguous = datasets["contiguous"]
    aligned = datasets["aligned"]
    specifications = (
        ("full_output_mlp", "full", False, contiguous, config.seed),
        (
            "contiguous_parity_mlp",
            "parity",
            False,
            contiguous,
            config.seed + 1000,
        ),
        (
            "aligned_parity_mlp",
            "parity",
            False,
            aligned,
            config.seed + 1000,
        ),
        (
            "aligned_parity_label_shuffle",
            "parity",
            True,
            aligned,
            config.seed + 2000,
        ),
    )
    trained: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    for row_name, target_kind, shuffle_labels, data, seed in specifications:
        result = train_output_prediction_row(
            config,
            data,
            row_name=row_name,
            target_kind=target_kind,
            shuffle_train_labels=shuffle_labels,
            seed=seed,
        )
        trained[row_name] = result
        history.extend(result["history"])

    full_probabilities = trained["full_output_mlp"]["test_probabilities"]
    derived_metrics = {
        "contiguous": multilabel_metrics(
            parity_probabilities_from_bit_probabilities(
                full_probabilities, CONTIGUOUS_MASKS
            ),
            contiguous["test"].parity_targets,
        ),
        "aligned": multilabel_metrics(
            parity_probabilities_from_bit_probabilities(
                full_probabilities, ALIGNED_MASKS
            ),
            aligned["test"].parity_targets,
        ),
    }
    rows = []
    for row_name, target_kind, shuffle_labels, data, seed in specifications:
        result = trained[row_name]
        metrics = result["test_metrics"]
        geometry = (
            "full"
            if target_kind == "full"
            else "aligned"
            if row_name.startswith("aligned")
            else "contiguous"
        )
        row = {
            "run_id": config.run_id,
            "task": "innovation2_output_parity_prediction",
            "experiment": "op2_mask_geometry",
            "model": row_name,
            "target_kind": target_kind,
            "mask_geometry": geometry,
            "train_labels_shuffled": shuffle_labels,
            "model_seed": seed,
            "seed": config.seed,
            "rounds": config.rounds,
            "secret_key_scope": "single_fixed_unknown_key",
            "parameters": result["parameters"],
            "epochs": config.epochs,
            "train_rows": config.train_rows,
            "validation_rows": config.validation_rows,
            "test_rows": config.test_rows,
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
            "test_macro_auc": metrics["macro_auc"],
            "test_exact_match": metrics["exact_match"],
            "test_majority_accuracy": metrics["majority_accuracy"],
            "training_performed": True,
        }
        if target_kind == "full":
            row.update(
                {
                    "derived_contiguous_parity_accuracy": derived_metrics["contiguous"][
                        "accuracy"
                    ],
                    "derived_contiguous_parity_macro_auc": derived_metrics[
                        "contiguous"
                    ]["macro_auc"],
                    "derived_aligned_parity_accuracy": derived_metrics["aligned"][
                        "accuracy"
                    ],
                    "derived_aligned_parity_macro_auc": derived_metrics["aligned"][
                        "macro_auc"
                    ],
                }
            )
        rows.append(row)
    return {
        "rows": rows,
        "history": history,
        "trained": trained,
        "derived_parity_metrics": derived_metrics,
    }


def adjudicate_mask_geometry(
    config: OutputParityPredictionConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
) -> dict[str, Any]:
    rows = {row["model"]: row for row in training["rows"]}
    numeric_values = [
        float(row[key])
        for row in rows.values()
        for key in ("test_loss", "test_accuracy", "test_macro_auc", "test_exact_match")
    ]
    checks = {
        **protocol_checks,
        "four_training_rows_complete": len(rows) == 4,
        "all_training_metrics_finite": all(
            math.isfinite(value) for value in numeric_values
        ),
        "history_rows_complete": len(training["history"]) == config.epochs * 4,
        "paired_geometry_models_share_initialization_seed": rows[
            "contiguous_parity_mlp"
        ]["model_seed"]
        == rows["aligned_parity_mlp"]["model_seed"],
        "shuffle_control_uses_true_test_targets": training["trained"][
            "aligned_parity_label_shuffle"
        ]["test_target_identity"]
        == "true_parity_targets",
    }
    contiguous = rows["contiguous_parity_mlp"]
    aligned = rows["aligned_parity_mlp"]
    shuffled = rows["aligned_parity_label_shuffle"]
    aligned_minus_contiguous = aligned["test_macro_auc"] - contiguous["test_macro_auc"]
    aligned_minus_shuffled = aligned["test_macro_auc"] - shuffled["test_macro_auc"]
    performance_gate = (
        aligned["test_macro_auc"] >= 0.55
        and aligned_minus_shuffled >= 0.03
        and aligned_minus_contiguous >= 0.03
    )
    if not all(checks.values()):
        status = "fail"
        decision = "innovation2_output_parity_mask_geometry_protocol_invalid"
        next_adjudication = "repair_op2_protocol"
        action = "repair only the fixed-key output-prediction or paired-mask protocol"
    elif performance_gate:
        status = "pass"
        decision = "innovation2_output_parity_mask_geometry_supported"
        next_adjudication = (
            "op3_independent_fixed_key_confirmation"
            if config.rounds == 1
            else f"present_r{config.rounds}_independent_fixed_key_confirmation"
        )
        action = (
            "preregister an independent fixed-key confirmation at PRESENT "
            f"r{config.rounds} with the same aligned, contiguous, and shuffled matrix"
        )
    else:
        status = "hold"
        decision = (
            "innovation2_output_parity_mask_geometry_not_calibrated"
            if config.rounds == 1
            else f"innovation2_output_parity_present_r{config.rounds}_single_key_not_supported"
        )
        next_adjudication = (
            "output_prediction_literature_protocol_audit"
            if config.rounds == 1
            else f"present_r{config.rounds}_local_representation_redesign"
        )
        action = (
            "stop mechanical scaling and audit the exact output-prediction protocol"
            if config.rounds == 1
            else f"stop mechanical scaling and compare an SPN-local representation at "
            f"PRESENT r{config.rounds} with unchanged output targets"
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "checks": checks,
        "thresholds": {
            "aligned_macro_auc_min": 0.55,
            "aligned_minus_shuffled_macro_auc_min": 0.03,
            "aligned_minus_contiguous_macro_auc_min": 0.03,
        },
        "metrics": {
            "full_bit_accuracy": rows["full_output_mlp"]["test_accuracy"],
            "full_bit_macro_auc": rows["full_output_mlp"]["test_macro_auc"],
            "contiguous_parity_accuracy": contiguous["test_accuracy"],
            "contiguous_parity_macro_auc": contiguous["test_macro_auc"],
            "aligned_parity_accuracy": aligned["test_accuracy"],
            "aligned_parity_macro_auc": aligned["test_macro_auc"],
            "shuffled_aligned_parity_accuracy": shuffled["test_accuracy"],
            "shuffled_aligned_parity_macro_auc": shuffled["test_macro_auc"],
            "aligned_minus_contiguous_macro_auc": aligned_minus_contiguous,
            "aligned_minus_shuffled_macro_auc": aligned_minus_shuffled,
        },
        "claim_scope": (
            f"local PRESENT r{config.rounds} fixed-key known-plaintext ciphertext-output "
            "parity mask-geometry "
            "calibration; labels are real ciphertext parities rather than sample classes, "
            "and this is not a high-round attack, paper reproduction, or SOTA result"
        ),
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "remote_scale": False,
            "high_round_training": False,
            "sample_classification": False,
        },
    }


def adjudicate_two_key_confirmation(
    run_id: str,
    anchor_gate: dict[str, Any],
    current_gate: dict[str, Any],
    independence_checks: dict[str, bool],
    *,
    rounds: int = 1,
) -> dict[str, Any]:
    anchor_supported = (
        anchor_gate.get("status") == "pass"
        and anchor_gate.get("decision")
        == "innovation2_output_parity_mask_geometry_supported"
    )
    current_supported = (
        current_gate.get("status") == "pass"
        and current_gate.get("decision")
        == "innovation2_output_parity_mask_geometry_supported"
    )
    protocol_valid = (
        all(independence_checks.values())
        and anchor_gate.get("status") != "fail"
        and current_gate.get("status") != "fail"
    )
    if not protocol_valid:
        status = "fail"
        decision = "innovation2_output_parity_two_key_protocol_invalid"
        next_adjudication = "repair_op3_protocol"
        action = (
            "repair only the anchor, key, plaintext-independence, or output contract"
        )
    elif anchor_supported and current_supported:
        status = "pass"
        if rounds == 1:
            decision = "innovation2_output_parity_mask_geometry_two_key_confirmed"
            next_adjudication = "op4_present_r2_two_key_round_step"
            action = (
                "preregister OP4 changing only PRESENT rounds from r1 to r2 for both "
                "fixed-key seeds under the same mask, model, data, and epoch budgets"
            )
        else:
            decision = f"innovation2_output_parity_present_r{rounds}_two_key_supported"
            next_adjudication = f"present_r{rounds + 1}_two_key_round_step"
            action = (
                f"preregister the next round step changing only PRESENT rounds from r{rounds} "
                f"to r{rounds + 1} for both fixed-key seeds under the same budgets"
            )
    else:
        status = "hold"
        if rounds == 1:
            decision = "innovation2_output_parity_mask_geometry_two_key_not_confirmed"
            next_adjudication = "output_prediction_literature_protocol_audit"
            action = (
                "stop round, sample, epoch, capacity, and key scaling; audit the closest "
                "fixed-key ciphertext-output prediction protocol"
            )
        else:
            decision = (
                f"innovation2_output_parity_present_r{rounds}_two_key_not_supported"
            )
            next_adjudication = f"present_r{rounds}_local_representation_redesign"
            action = (
                f"stop mechanical scaling and compare an SPN-local representation against "
                f"the current MLP at PRESENT r{rounds} with unchanged output targets"
            )
    anchor_metrics = anchor_gate.get("metrics", {})
    current_metrics = current_gate.get("metrics", {})
    aligned_auc_values = [
        float(anchor_metrics.get("aligned_parity_macro_auc", float("nan"))),
        float(current_metrics.get("aligned_parity_macro_auc", float("nan"))),
    ]
    return {
        "run_id": run_id,
        "status": status,
        "decision": decision,
        "checks": {
            **independence_checks,
            "anchor_seed0_mask_geometry_supported": anchor_supported,
            "current_seed1_mask_geometry_supported": current_supported,
        },
        "thresholds": current_gate.get("thresholds", {}),
        "metrics": {
            "seed0": anchor_metrics,
            "seed1": current_metrics,
            "minimum_aligned_parity_macro_auc": min(aligned_auc_values),
            "mean_aligned_parity_macro_auc": float(np.mean(aligned_auc_values)),
            "aligned_parity_macro_auc_range": max(aligned_auc_values)
            - min(aligned_auc_values),
        },
        "claim_scope": (
            f"local PRESENT-80 r{rounds} confirmation across two independently generated "
            "fixed secret keys and disjoint plaintext sets; targets are true ciphertext "
            "output parities rather than sample classes, and this is not a high-round "
            "attack, paper reproduction, or SOTA result"
        ),
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "remote_scale": False,
            "high_round_training": False,
            "sample_classification": False,
        },
    }


def mask_positions(mask: int) -> tuple[int, ...]:
    return tuple(bit for bit in range(64) if (mask >> bit) & 1)
