from __future__ import annotations

import math
from typing import Any


def adjudicate_runtime_spn_present_transfer(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    expected_seed: int = 0,
) -> dict[str, Any]:
    expected_models = {
        "true": "present_runtime_e4_equivariant_true",
        "corrupted": "present_runtime_e4_equivariant_corrupted",
        "independent": "present_runtime_e4_equivariant_independent",
    }
    by_model = {str(row.get("model")): row for row in rows}
    missing = [model for model in expected_models.values() if model not in by_model]
    if missing:
        raise ValueError(f"missing RTG1-T1 rows: {missing}")
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
        "model_options",
    )
    protocol_checks = {
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
        "frozen_transfer_scale": all(
            row.get("cipher") == "PRESENT-80"
            and row.get("rounds") == 7
            and row.get("seed") == expected_seed
            and row.get("samples_per_class") == 2048
            and row.get("pairs_per_sample") == 16
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
            and row.get("training", {}).get("epochs") == 5
            for row in by_role.values()
        ),
        "raw_ciphertext_pair_input": all(
            row.get("feature_encoding") == "ciphertext_pair_bits"
            for row in by_role.values()
        ),
        "official_mcnd_sample_organization": all(
            row.get("sample_structure") == "zhang_wang_case2_official_mcnd"
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
    margins = {
        "true_minus_corrupted": aucs["true"] - aucs["corrupted"],
        "true_minus_independent": aucs["true"] - aucs["independent"],
    }
    research_checks = {
        "true_auc_at_least_0p520": aucs["true"] >= 0.520,
        "true_exceeds_corrupted_by_0p005": margins["true_minus_corrupted"]
        >= 0.005,
        "true_exceeds_independent_by_0p005": margins["true_minus_independent"]
        >= 0.005,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_present_transfer_protocol_invalid"
        next_action = "repair only the failed T1 protocol check before interpretation"
    elif all(research_checks.values()):
        status = "pass"
        decision = (
            f"innovation1_runtime_spn_present_transfer_seed{expected_seed}_supported"
        )
        next_action = (
            "repeat the identical local PRESENT T1 matrix at seed1; do not scale"
            if expected_seed == 0
            else "audit two-seed cross-cipher stability before any scale decision"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_present_transfer_not_supported"
        next_action = (
            "stop PRESENT replication and scale-up; diagnose the failed signal or "
            "topology-control gate without changing T1 after seeing the result"
        )
    return {
        "run_id": run_id,
        "cipher": "PRESENT-80",
        "seed": expected_seed,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": margins,
        "thresholds": {"true_auc": 0.520, "control_margin": 0.005},
        "claim_scope": (
            f"PRESENT-80 r7 seed{expected_seed} 2048/class runtime-topology transfer "
            "diagnostic only; not multi-seed, formal, paper-scale, SOTA, or a "
            "Zhang/Wang reproduction"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "remote scale-up",
            "increase samples or epochs",
            "claim stable cross-cipher topology superiority",
        ],
    }


__all__ = ["adjudicate_runtime_spn_present_transfer"]
