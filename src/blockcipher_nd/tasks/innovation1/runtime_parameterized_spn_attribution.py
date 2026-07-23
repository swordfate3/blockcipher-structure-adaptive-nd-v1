from __future__ import annotations

import math
from typing import Any


EXPECTED_RUNTIME_MODELS = {
    "GIFT-64": {
        "true": "gift64_runtime_spn_true",
        "corrupted": "gift64_runtime_spn_corrupted",
        "independent": "gift64_runtime_spn_independent",
    },
    "PRESENT-80": {
        "true": "present_runtime_spn_true",
        "corrupted": "present_runtime_spn_corrupted",
        "independent": "present_runtime_spn_independent",
    },
}

EXPECTED_ANCHOR_MODELS = {
    "GIFT-64": "gift_cross_spn_typed_cell_true",
    "PRESENT-80": "present_cross_spn_typed_cell_true",
}

ANCHOR_TOLERANCE = 0.005
CONTROL_MARGIN = 0.005


def adjudicate_runtime_spn_r1(
    *,
    run_id: str,
    cipher: str,
    runtime_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    r0_gate: dict[str, Any],
) -> dict[str, Any]:
    if cipher not in EXPECTED_RUNTIME_MODELS:
        raise ValueError(f"unsupported RTG1 R1 cipher: {cipher}")
    runtime_by_role = _runtime_rows_by_role(cipher, runtime_rows)
    anchor = _anchor_row(cipher, anchor_rows)
    protocol_checks = _protocol_checks(
        cipher,
        runtime_by_role,
        anchor,
        r0_gate,
    )
    aucs = {
        role: float(row["metrics"]["auc"])
        for role, row in runtime_by_role.items()
    }
    aucs["anchor"] = float(anchor["metrics"]["auc"])
    margins = {
        "true_minus_anchor": aucs["true"] - aucs["anchor"],
        "true_minus_corrupted": aucs["true"] - aucs["corrupted"],
        "true_minus_independent": aucs["true"] - aucs["independent"],
    }
    research_checks = {
        "true_within_anchor_tolerance": margins["true_minus_anchor"]
        >= -ANCHOR_TOLERANCE,
        "true_exceeds_corrupted": margins["true_minus_corrupted"]
        >= CONTROL_MARGIN,
        "true_exceeds_independent": margins["true_minus_independent"]
        >= CONTROL_MARGIN,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_r1_protocol_invalid"
        next_action = "repair the invalid R1 protocol or artifacts before interpretation"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_runtime_spn_r1_seed0_supported"
        next_action = (
            "run the same cipher and the second SPN at the frozen seed0 budget; "
            "only a two-cipher pass may authorize seed1"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_r1_seed0_not_supported"
        next_action = (
            "replace only the bit-to-pair global pooling with a runtime cell-token "
            "interaction stage that preserves E4 per-pair aligned cell relations; "
            "rerun a small local GIFT calibration before any 8192/class repeat"
        )
    return {
        "run_id": run_id,
        "cipher": cipher,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": margins,
        "thresholds": {
            "anchor_tolerance": ANCHOR_TOLERANCE,
            "control_margin": CONTROL_MARGIN,
        },
        "claim_scope": (
            f"{cipher} seed0 8192/class local diagnostic only; not multi-seed, "
            "multi-cipher, formal, or paper-scale evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "run seed1 before a seed0 pass",
            "run the heavier second-cipher matrix after a decisive first-cipher miss",
            "remote scale-up",
            "claim stable topology superiority",
        ],
    }


def _runtime_rows_by_role(
    cipher: str,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected = EXPECTED_RUNTIME_MODELS[cipher]
    by_model = {str(row.get("model")): row for row in rows}
    missing = [model for model in expected.values() if model not in by_model]
    if missing:
        raise ValueError(f"missing RTG1 runtime result rows: {missing}")
    return {role: by_model[model] for role, model in expected.items()}


def _anchor_row(cipher: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = EXPECTED_ANCHOR_MODELS[cipher]
    matches = [row for row in rows if row.get("model") == expected]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {expected} anchor row, found {len(matches)}"
        )
    return matches[0]


def _protocol_checks(
    cipher: str,
    runtime: dict[str, dict[str, Any]],
    anchor: dict[str, Any],
    r0_gate: dict[str, Any],
) -> dict[str, bool]:
    rows = list(runtime.values())
    static_fields = (
        "cipher",
        "rounds",
        "seed",
        "samples_per_class",
        "dataset_label_mode",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "train_key",
        "validation_key",
    )
    training_fields = (
        "epochs",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
    )
    all_rows = rows + [anchor]
    return {
        "r0_readiness_passed": r0_gate.get("status") == "pass"
        and r0_gate.get("decision") == "innovation1_runtime_spn_r0_readiness_passed",
        "runtime_model_set_complete": {
            str(row.get("model")) for row in rows
        }
        == set(EXPECTED_RUNTIME_MODELS[cipher].values()),
        "cipher_and_seed_match": all(
            row.get("cipher") == cipher and row.get("seed") == 0 for row in all_rows
        ),
        "same_data_protocol_as_anchor": all(
            all(row.get(field) == anchor.get(field) for field in static_fields)
            for row in rows
        ),
        "same_training_protocol_as_anchor": all(
            all(
                row.get("training", {}).get(field)
                == anchor.get("training", {}).get(field)
                for field in training_fields
            )
            for row in rows
        ),
        "frozen_r1_scale": all(
            row.get("samples_per_class") == 8192
            and row.get("training", {}).get("train_rows") == 16384
            and row.get("training", {}).get("validation_rows") == 8192
            and row.get("training", {}).get("epochs") == 10
            for row in rows
        ),
        "encrypted_random_plaintext_negatives": all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in all_rows
        ),
        "equal_runtime_parameter_count": len(
            {int(row.get("parameter_count", -1)) for row in rows}
        )
        == 1,
        "finite_auc_metrics": all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in all_rows
        ),
        "best_validation_checkpoints_restored": all(
            row.get("training", {}).get("checkpoint_metric") == "val_auc"
            and row.get("training", {}).get("restore_best_checkpoint") is True
            and row.get("training", {}).get("selected_checkpoint") == "best"
            for row in all_rows
        ),
        "disk_backed_datasets": all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in rows
        ),
    }


__all__ = [
    "ANCHOR_TOLERANCE",
    "CONTROL_MARGIN",
    "adjudicate_runtime_spn_r1",
    "adjudicate_runtime_spn_r1a_cell_token",
    "adjudicate_runtime_spn_r1b_position",
    "adjudicate_runtime_spn_r1c_view_encoder",
    "adjudicate_runtime_spn_r1d_cell_mixer",
    "adjudicate_runtime_spn_r2a_e4_attribution",
    "adjudicate_runtime_spn_r2d_sbox_scale",
]


def adjudicate_runtime_spn_r1a_cell_token(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    r1_gate: dict[str, Any],
) -> dict[str, Any]:
    expected_models = {
        "current": "gift64_runtime_spn_true",
        "cell_true": "gift64_runtime_cell_token_true",
        "cell_corrupted": "gift64_runtime_cell_token_corrupted",
    }
    by_model = {str(row.get("model")): row for row in rows}
    missing = [model for model in expected_models.values() if model not in by_model]
    if missing:
        raise ValueError(f"missing RTG1-R1a rows: {missing}")
    by_role = {role: by_model[model] for role, model in expected_models.items()}
    reference = by_role["current"]
    static_fields = (
        "cipher",
        "rounds",
        "seed",
        "samples_per_class",
        "dataset_label_mode",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "train_key",
        "validation_key",
    )
    training_fields = (
        "epochs",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
    )
    protocol_checks = {
        "source_r1_was_valid_hold": r1_gate.get("status") == "hold"
        and r1_gate.get("decision")
        == "innovation1_runtime_spn_r1_seed0_not_supported",
        "three_calibration_rows_complete": set(by_model) == set(expected_models.values()),
        "same_data_protocol": all(
            all(row.get(field) == reference.get(field) for field in static_fields)
            for row in by_role.values()
        ),
        "same_training_protocol": all(
            all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in by_role.values()
        ),
        "frozen_calibration_scale": all(
            row.get("cipher") == "GIFT-64"
            and row.get("rounds") == 6
            and row.get("seed") == 0
            and row.get("samples_per_class") == 2048
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
            and row.get("training", {}).get("epochs") == 5
            for row in by_role.values()
        ),
        "encrypted_random_plaintext_negatives": all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in by_role.values()
        ),
        "cell_token_controls_equal_parameters": (
            by_role["cell_true"].get("parameter_count")
            == by_role["cell_corrupted"].get("parameter_count")
        ),
        "finite_auc_metrics": all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in by_role.values()
        ),
        "disk_backed_datasets": all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in by_role.values()
        ),
    }
    aucs = {
        role: float(row["metrics"]["auc"])
        for role, row in by_role.items()
    }
    margins = {
        "cell_true_minus_current": aucs["cell_true"] - aucs["current"],
        "cell_true_minus_corrupted": aucs["cell_true"] - aucs["cell_corrupted"],
    }
    research_checks = {
        "cell_true_improves_current_by_0p010": margins["cell_true_minus_current"]
        >= 0.010,
        "cell_true_exceeds_corrupted_by_0p005": margins[
            "cell_true_minus_corrupted"
        ]
        >= 0.005,
        "cell_true_auc_at_least_0p520": aucs["cell_true"] >= 0.520,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_cell_token_calibration_protocol_invalid"
        next_action = "repair the R1a protocol or artifacts before interpretation"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_runtime_spn_cell_token_calibration_supported"
        next_action = "return the cell-token candidate to the full 8192/class GIFT gate"
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_cell_token_calibration_not_supported"
        next_action = (
            "run a matched E4 position-identifiability audit: typed-cell true versus "
            "the same model with its 16-cell learned position embedding disabled; "
            "use the result to decide whether a runtime functional coordinate descriptor "
            "is required"
        )
    return {
        "run_id": run_id,
        "cipher": "GIFT-64",
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": margins,
        "thresholds": {
            "cell_true_minus_current": 0.010,
            "cell_true_minus_corrupted": 0.005,
            "cell_true_auc": 0.520,
        },
        "claim_scope": (
            "GIFT-64 seed0 2048/class architecture calibration only; not R1, "
            "multi-seed, formal, or paper-scale evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "repeat cell-token at 8192/class",
            "run PRESENT",
            "run seed1",
            "remote scale-up",
            "add another neural architecture before the E4 position audit",
        ],
    }


def adjudicate_runtime_spn_r1b_position(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    r1a_gate: dict[str, Any],
) -> dict[str, Any]:
    expected_models = {
        "learned": "gift_cross_spn_typed_cell_true",
        "zero": "gift_cross_spn_typed_cell_no_position",
    }
    by_model = {str(row.get("model")): row for row in rows}
    missing = [model for model in expected_models.values() if model not in by_model]
    if missing:
        raise ValueError(f"missing RTG1-R1b rows: {missing}")
    by_role = {role: by_model[model] for role, model in expected_models.items()}
    reference = by_role["learned"]
    static_fields = (
        "cipher",
        "rounds",
        "seed",
        "samples_per_class",
        "dataset_label_mode",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "train_key",
        "validation_key",
    )
    training_fields = (
        "epochs",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
    )
    protocol_checks = {
        "source_r1a_was_valid_hold": r1a_gate.get("status") == "hold"
        and r1a_gate.get("decision")
        == "innovation1_runtime_spn_cell_token_calibration_not_supported",
        "two_position_rows_complete": set(by_model) == set(expected_models.values()),
        "same_data_protocol": all(
            all(row.get(field) == reference.get(field) for field in static_fields)
            for row in by_role.values()
        ),
        "same_training_protocol": all(
            all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in by_role.values()
        ),
        "frozen_calibration_scale": all(
            row.get("cipher") == "GIFT-64"
            and row.get("rounds") == 6
            and row.get("seed") == 0
            and row.get("samples_per_class") == 2048
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
            and row.get("training", {}).get("epochs") == 5
            for row in by_role.values()
        ),
        "encrypted_random_plaintext_negatives": all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in by_role.values()
        ),
        "equal_parameter_geometry": len(
            {
                (
                    int(row.get("parameter_count", -1)),
                    int(row.get("trainable_parameter_count", -1)),
                )
                for row in by_role.values()
            }
        )
        == 1,
        "finite_auc_metrics": all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in by_role.values()
        ),
        "disk_backed_datasets": all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in by_role.values()
        ),
    }
    aucs = {
        role: float(row["metrics"]["auc"]) for role, row in by_role.items()
    }
    margin = aucs["learned"] - aucs["zero"]
    research_checks = {
        "learned_position_auc_at_least_0p520": aucs["learned"] >= 0.520,
        "learned_position_exceeds_zero_by_0p010": margin >= 0.010,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_position_audit_protocol_invalid"
        next_action = "repair the R1b protocol or artifacts before interpretation"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_runtime_spn_position_identity_supported"
        next_action = (
            "add fixed runtime cell-coordinate descriptors followed by a shared encoder; "
            "keep trainable parameter shapes independent of cell count and cipher"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_position_identity_not_supported"
        next_action = (
            "do not add runtime coordinates; audit E4 current/previous typed fusion "
            "against a shared-encoder ablation at the same budget"
        )
    return {
        "run_id": run_id,
        "cipher": "GIFT-64",
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": {"learned_minus_zero": margin},
        "thresholds": {"learned_auc": 0.520, "learned_minus_zero": 0.010},
        "claim_scope": (
            "GIFT-64 seed0 2048/class E4 position-identifiability calibration only; "
            "not runtime-topology superiority, multi-seed, formal, or paper-scale evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "run PRESENT",
            "run seed1",
            "increase samples or epochs",
            "remote scale-up",
            "claim stable topology superiority",
        ],
    }


def adjudicate_runtime_spn_r1c_view_encoder(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    r1b_gate: dict[str, Any],
) -> dict[str, Any]:
    expected_models = {
        "separate": "gift_cross_spn_typed_cell_no_position",
        "shared": "gift_cross_spn_typed_cell_shared_view_encoder",
    }
    by_model = {str(row.get("model")): row for row in rows}
    missing = [model for model in expected_models.values() if model not in by_model]
    if missing:
        raise ValueError(f"missing RTG1-R1c rows: {missing}")
    by_role = {role: by_model[model] for role, model in expected_models.items()}
    reference = by_role["separate"]
    static_fields = (
        "cipher",
        "rounds",
        "seed",
        "samples_per_class",
        "dataset_label_mode",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "train_key",
        "validation_key",
    )
    training_fields = (
        "epochs",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
    )
    protocol_checks = {
        "source_r1b_was_valid_hold": r1b_gate.get("status") == "hold"
        and r1b_gate.get("decision")
        == "innovation1_runtime_spn_position_identity_not_supported",
        "two_view_rows_complete": set(by_model) == set(expected_models.values()),
        "same_data_protocol": all(
            all(row.get(field) == reference.get(field) for field in static_fields)
            for row in by_role.values()
        ),
        "same_training_protocol": all(
            all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in by_role.values()
        ),
        "frozen_calibration_scale": all(
            row.get("cipher") == "GIFT-64"
            and row.get("rounds") == 6
            and row.get("seed") == 0
            and row.get("samples_per_class") == 2048
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
            and row.get("training", {}).get("epochs") == 5
            for row in by_role.values()
        ),
        "encrypted_random_plaintext_negatives": all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in by_role.values()
        ),
        "equal_parameter_geometry": len(
            {
                (
                    int(row.get("parameter_count", -1)),
                    int(row.get("trainable_parameter_count", -1)),
                )
                for row in by_role.values()
            }
        )
        == 1,
        "finite_auc_metrics": all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in by_role.values()
        ),
        "disk_backed_datasets": all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in by_role.values()
        ),
    }
    aucs = {
        role: float(row["metrics"]["auc"]) for role, row in by_role.items()
    }
    margin = aucs["separate"] - aucs["shared"]
    research_checks = {
        "separate_view_auc_at_least_0p520": aucs["separate"] >= 0.520,
        "separate_view_exceeds_shared_by_0p010": margin >= 0.010,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_view_encoder_audit_protocol_invalid"
        next_action = "repair the R1c protocol or artifacts before interpretation"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_runtime_spn_typed_view_identity_supported"
        next_action = (
            "add a fixed two-value current/previous view-role descriptor to the "
            "runtime shared encoder, then rerun a small topology-attribution calibration"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_typed_view_identity_not_supported"
        next_action = (
            "audit E4 fixed-cell Token-Mixer against a parameter-matched "
            "permutation-equivariant cell mixer before another runtime redesign"
        )
    return {
        "run_id": run_id,
        "cipher": "GIFT-64",
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": {"separate_minus_shared": margin},
        "thresholds": {"separate_auc": 0.520, "separate_minus_shared": 0.010},
        "claim_scope": (
            "GIFT-64 seed0 2048/class E4 view-encoder calibration only; not "
            "runtime-topology superiority, multi-seed, formal, or paper-scale evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "run PRESENT",
            "run seed1",
            "increase samples or epochs",
            "remote scale-up",
            "claim stable topology superiority",
        ],
    }


def adjudicate_runtime_spn_r1d_cell_mixer(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    r1c_gate: dict[str, Any],
) -> dict[str, Any]:
    expected_models = {
        "fixed": "gift_cross_spn_typed_cell_shared_view_encoder",
        "equivariant": "gift_cross_spn_typed_cell_equivariant_mixer",
    }
    by_model = {str(row.get("model")): row for row in rows}
    missing = [model for model in expected_models.values() if model not in by_model]
    if missing:
        raise ValueError(f"missing RTG1-R1d rows: {missing}")
    by_role = {role: by_model[model] for role, model in expected_models.items()}
    reference = by_role["fixed"]
    static_fields = (
        "cipher",
        "rounds",
        "seed",
        "samples_per_class",
        "dataset_label_mode",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "train_key",
        "validation_key",
    )
    training_fields = (
        "epochs",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
    )
    protocol_checks = {
        "source_r1c_was_valid_hold": r1c_gate.get("status") == "hold"
        and r1c_gate.get("decision")
        == "innovation1_runtime_spn_typed_view_identity_not_supported",
        "two_cell_mixer_rows_complete": set(by_model)
        == set(expected_models.values()),
        "same_data_protocol": all(
            all(row.get(field) == reference.get(field) for field in static_fields)
            for row in by_role.values()
        ),
        "same_training_protocol": all(
            all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in by_role.values()
        ),
        "frozen_calibration_scale": all(
            row.get("cipher") == "GIFT-64"
            and row.get("rounds") == 6
            and row.get("seed") == 0
            and row.get("samples_per_class") == 2048
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
            and row.get("training", {}).get("epochs") == 5
            for row in by_role.values()
        ),
        "encrypted_random_plaintext_negatives": all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in by_role.values()
        ),
        "equal_parameter_budget": len(
            {
                (
                    int(row.get("parameter_count", -1)),
                    int(row.get("trainable_parameter_count", -1)),
                )
                for row in by_role.values()
            }
        )
        == 1,
        "finite_auc_metrics": all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in by_role.values()
        ),
        "disk_backed_datasets": all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in by_role.values()
        ),
    }
    aucs = {
        role: float(row["metrics"]["auc"]) for role, row in by_role.items()
    }
    margin = aucs["fixed"] - aucs["equivariant"]
    research_checks = {
        "fixed_mixer_auc_at_least_0p520": aucs["fixed"] >= 0.520,
        "fixed_mixer_exceeds_equivariant_by_0p010": margin >= 0.010,
        "equivariant_mixer_auc_at_least_0p520": aucs["equivariant"] >= 0.520,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_cell_mixer_audit_protocol_invalid"
        next_action = "repair the R1d protocol or artifacts before interpretation"
    elif (
        research_checks["fixed_mixer_auc_at_least_0p520"]
        and research_checks["fixed_mixer_exceeds_equivariant_by_0p010"]
    ):
        status = "pass"
        decision = "innovation1_runtime_spn_fixed_cell_mixer_dependency_supported"
        next_action = (
            "design external topology-derived functional coordinates for the runtime "
            "cell mixer, without cipher IDs or fixed trainable position tables"
        )
    elif research_checks["equivariant_mixer_auc_at_least_0p520"]:
        status = "pass"
        decision = "innovation1_runtime_spn_equivariant_e4_backbone_supported"
        next_action = (
            "port the E4 per-pair current/exact-inverse cell-token frontend and "
            "equivariant mixer into the runtime model, then test true/corrupted/independent"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_cell_mixer_calibration_not_supported"
        next_action = (
            "stop E4 component peeling and redesign the runtime topology message-passing "
            "operator from first principles before additional training"
        )
    return {
        "run_id": run_id,
        "cipher": "GIFT-64",
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": {"fixed_minus_equivariant": margin},
        "thresholds": {
            "fixed_auc": 0.520,
            "fixed_minus_equivariant": 0.010,
            "equivariant_auc": 0.520,
        },
        "claim_scope": (
            "GIFT-64 seed0 2048/class E4 cell-mixer calibration only; not "
            "runtime-topology superiority, multi-seed, formal, or paper-scale evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "run PRESENT",
            "run seed1",
            "increase samples or epochs",
            "remote scale-up",
            "claim stable topology superiority",
        ],
    }


def adjudicate_runtime_spn_r2a_e4_attribution(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    r1d_gate: dict[str, Any],
) -> dict[str, Any]:
    expected_models = {
        "true": "gift64_runtime_e4_equivariant_true",
        "corrupted": "gift64_runtime_e4_equivariant_corrupted",
        "independent": "gift64_runtime_e4_equivariant_independent",
    }
    by_model = {str(row.get("model")): row for row in rows}
    missing = [model for model in expected_models.values() if model not in by_model]
    if missing:
        raise ValueError(f"missing RTG1-R2a rows: {missing}")
    by_role = {role: by_model[model] for role, model in expected_models.items()}
    reference = by_role["true"]
    static_fields = (
        "cipher",
        "rounds",
        "seed",
        "samples_per_class",
        "dataset_label_mode",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "train_key",
        "validation_key",
    )
    training_fields = (
        "epochs",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
    )
    protocol_checks = {
        "source_r1d_supported_equivariant_backbone": r1d_gate.get("status")
        == "pass"
        and r1d_gate.get("decision")
        == "innovation1_runtime_spn_equivariant_e4_backbone_supported",
        "three_runtime_controls_complete": set(by_model)
        == set(expected_models.values()),
        "same_data_protocol": all(
            all(row.get(field) == reference.get(field) for field in static_fields)
            for row in by_role.values()
        ),
        "same_training_protocol": all(
            all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in by_role.values()
        ),
        "frozen_calibration_scale": all(
            row.get("cipher") == "GIFT-64"
            and row.get("rounds") == 6
            and row.get("seed") == 0
            and row.get("samples_per_class") == 2048
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
            and row.get("training", {}).get("epochs") == 5
            for row in by_role.values()
        ),
        "encrypted_random_plaintext_negatives": all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in by_role.values()
        ),
        "runtime_bit_order_adapter_recorded": all(
            row.get("input_bit_order") == "project_msb_to_runtime_lsb"
            for row in by_role.values()
        ),
        "equal_parameter_geometry": len(
            {
                (
                    int(row.get("parameter_count", -1)),
                    int(row.get("trainable_parameter_count", -1)),
                )
                for row in by_role.values()
            }
        )
        == 1,
        "finite_auc_metrics": all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in by_role.values()
        ),
        "disk_backed_datasets": all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in by_role.values()
        ),
    }
    aucs = {
        role: float(row["metrics"]["auc"]) for role, row in by_role.items()
    }
    r1d_anchor_auc = float(r1d_gate.get("aucs", {}).get("equivariant", math.nan))
    margins = {
        "true_minus_r1d_anchor": aucs["true"] - r1d_anchor_auc,
        "true_minus_corrupted": aucs["true"] - aucs["corrupted"],
        "true_minus_independent": aucs["true"] - aucs["independent"],
    }
    research_checks = {
        "true_auc_at_least_0p520": aucs["true"] >= 0.520,
        "true_within_r1d_anchor_tolerance": margins["true_minus_r1d_anchor"]
        >= -0.005,
        "true_exceeds_corrupted_by_0p005": margins["true_minus_corrupted"]
        >= 0.005,
        "true_exceeds_independent_by_0p005": margins["true_minus_independent"]
        >= 0.005,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_r2a_protocol_invalid"
        next_action = "repair the R2a protocol or artifacts before interpretation"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_runtime_spn_r2a_seed0_supported"
        next_action = (
            "repeat the same three-control GIFT gate at 8192/class and 10 epochs; "
            "do not add PRESENT or seed1 until that same-cipher gate passes"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_r2a_topology_attribution_not_supported"
        next_action = (
            "redesign the runtime linear-topology interaction locally while preserving "
            "the E4-equivariant frontend; do not scale this candidate"
        )
    return {
        "run_id": run_id,
        "cipher": "GIFT-64",
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "reference_aucs": {"r1d_equivariant": r1d_anchor_auc},
        "margins": margins,
        "thresholds": {
            "true_auc": 0.520,
            "r1d_anchor_tolerance": 0.005,
            "control_margin": 0.005,
        },
        "claim_scope": (
            "GIFT-64 seed0 2048/class runtime topology-attribution calibration only; "
            "not multi-seed, multi-cipher, formal, or paper-scale evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "run PRESENT",
            "run seed1",
            "increase beyond the conditional 8192/class gate",
            "remote scale-up",
            "claim stable topology superiority",
        ],
    }


def adjudicate_runtime_spn_r2d_sbox_scale(
    *,
    run_id: str,
    candidate_rows: list[dict[str, Any]],
    r2c_rows: list[dict[str, Any]],
    r2c_gate: dict[str, Any],
    r1d_gate: dict[str, Any],
) -> dict[str, Any]:
    model = "gift64_runtime_e4_equivariant_true"
    candidate_matches = [row for row in candidate_rows if row.get("model") == model]
    baseline_matches = [row for row in r2c_rows if row.get("model") == model]
    if len(candidate_matches) != 1 or len(baseline_matches) != 1:
        raise ValueError("R2d requires one candidate and one R2c true-topology row")
    candidate = candidate_matches[0]
    baseline = baseline_matches[0]
    static_fields = (
        "cipher",
        "rounds",
        "seed",
        "samples_per_class",
        "dataset_label_mode",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "train_key",
        "validation_key",
    )
    training_fields = (
        "epochs",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
        "dataset_cache_root",
    )
    candidate_options = candidate.get("training", {}).get("model_options", {})
    baseline_options = baseline.get("training", {}).get("model_options", {})
    candidate_scale = float(candidate_options.get("sbox_context_scale", math.nan))
    baseline_scale = float(baseline_options.get("sbox_context_scale", 1.0))
    protocol_checks = {
        "source_r2c_was_valid_hold": r2c_gate.get("status") == "hold"
        and r2c_gate.get("decision")
        == "innovation1_runtime_spn_r2a_topology_attribution_not_supported"
        and all(r2c_gate.get("protocol_checks", {}).values()),
        "source_r1d_supported_equivariant_backbone": r1d_gate.get("status")
        == "pass"
        and r1d_gate.get("decision")
        == "innovation1_runtime_spn_equivariant_e4_backbone_supported",
        "one_true_topology_candidate": len(candidate_rows) == 1,
        "same_data_protocol": all(
            candidate.get(field) == baseline.get(field) for field in static_fields
        ),
        "same_training_protocol": all(
            candidate.get("training", {}).get(field)
            == baseline.get("training", {}).get(field)
            for field in training_fields
        ),
        "only_sbox_scale_option_changed": {
            key: value
            for key, value in candidate_options.items()
            if key != "sbox_context_scale"
        }
        == {
            key: value
            for key, value in baseline_options.items()
            if key != "sbox_context_scale"
        },
        "preregistered_nonzero_scale": math.isclose(candidate_scale, 0.1)
        and candidate_scale > 0.0
        and math.isclose(baseline_scale, 1.0),
        "same_parameter_geometry": (
            int(candidate.get("parameter_count", -1)),
            int(candidate.get("trainable_parameter_count", -1)),
        )
        == (
            int(baseline.get("parameter_count", -1)),
            int(baseline.get("trainable_parameter_count", -1)),
        ),
        "runtime_bit_order_adapter_recorded": candidate.get("input_bit_order")
        == "project_msb_to_runtime_lsb",
        "finite_auc_metrics": math.isfinite(
            float(candidate.get("metrics", {}).get("auc", math.nan))
        ),
        "disk_backed_dataset_reused": candidate.get("training", {}).get(
            "train_dataset_storage"
        )
        == "disk"
        and candidate.get("training", {}).get("validation_dataset_storage")
        == "disk",
    }
    candidate_auc = float(candidate.get("metrics", {}).get("auc", math.nan))
    baseline_auc = float(baseline.get("metrics", {}).get("auc", math.nan))
    anchor_auc = float(r1d_gate.get("aucs", {}).get("equivariant", math.nan))
    margins = {
        "candidate_minus_scale1_baseline": candidate_auc - baseline_auc,
        "candidate_minus_r1d_anchor": candidate_auc - anchor_auc,
    }
    research_checks = {
        "candidate_auc_at_least_0p520": candidate_auc >= 0.520,
        "candidate_within_r1d_anchor_tolerance": margins[
            "candidate_minus_r1d_anchor"
        ]
        >= -0.005,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_sbox_scale_protocol_invalid"
        next_action = "repair the R2d protocol before interpreting the calibration"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_runtime_spn_sbox_scale_calibration_supported"
        next_action = (
            "rerun the full R2c true/corrupted/no-topology matrix with the frozen "
            "0.1 S-box context scale at the same 2048/class seed0 budget"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_sbox_scale_calibration_not_supported"
        next_action = (
            "stop S-box scale tuning and redesign the runtime fusion locally; do not "
            "scale, add seeds, or relax the anchor tolerance"
        )
    return {
        "run_id": run_id,
        "cipher": "GIFT-64",
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": {
            "scale_0p1_candidate": candidate_auc,
            "scale_1p0_baseline": baseline_auc,
            "r1d_equivariant_anchor": anchor_auc,
        },
        "margins": margins,
        "thresholds": {"candidate_auc": 0.520, "r1d_anchor_tolerance": 0.005},
        "claim_scope": (
            "GIFT-64 seed0 2048/class S-box residual calibration only; not full "
            "topology attribution, multi-seed, multi-cipher, formal, or paper-scale evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "run PRESENT",
            "run seed1",
            "increase samples or epochs",
            "remote scale-up",
            "claim stable topology superiority",
        ],
    }
