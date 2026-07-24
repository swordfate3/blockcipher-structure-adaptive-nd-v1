from __future__ import annotations

import math
from typing import Any


MODELS = {
    "true": "skinny64_runtime_e4_equivariant_true",
    "corrupted": "skinny64_runtime_e4_equivariant_corrupted",
    "independent": "skinny64_runtime_e4_equivariant_independent",
}


def adjudicate_runtime_spn_skinny_attribution(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    expected_seed: int,
) -> dict[str, Any]:
    by_model = {str(row.get("model")): row for row in rows}
    by_role = {
        role: by_model[model]
        for role, model in MODELS.items()
        if model in by_model
    }
    reference = by_role.get("true", {})
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
        "input_difference",
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
        "three_runtime_controls_complete": (
            len(rows) == 3 and set(by_model) == set(MODELS.values())
        ),
        "same_data_protocol": len(by_role) == 3
        and all(
            all(row.get(field) == reference.get(field) for field in static_fields)
            for row in by_role.values()
        ),
        "same_training_protocol": len(by_role) == 3
        and all(
            all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in by_role.values()
        ),
        "frozen_t2c_scale_and_task": len(by_role) == 3
        and all(
            row.get("cipher") == "SKINNY-64/64"
            and row.get("rounds") == 7
            and row.get("seed") == expected_seed
            and row.get("samples_per_class") == 2048
            and row.get("pairs_per_sample") == 4
            and row.get("difference_profile") == "skinny64_gohr2022_single_key"
            and row.get("input_difference") == 0x2000
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
            and row.get("training", {}).get("epochs") == 5
            for row in by_role.values()
        ),
        "strict_encrypted_random_plaintext_negatives": len(by_role) == 3
        and all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in by_role.values()
        ),
        "raw_independent_ciphertext_pairs": len(by_role) == 3
        and all(
            row.get("feature_encoding") == "ciphertext_pair_bits"
            and row.get("sample_structure") == "independent_pairs"
            for row in by_role.values()
        ),
        "equal_parameter_geometry": len(by_role) == 3
        and len(
            {
                (
                    int(row.get("parameter_count", -1)),
                    int(row.get("trainable_parameter_count", -1)),
                )
                for row in by_role.values()
            }
        )
        == 1,
        "runtime_bit_order_adapter_recorded": len(by_role) == 3
        and all(
            row.get("input_bit_order") == "project_msb_to_runtime_lsb"
            for row in by_role.values()
        ),
        "disk_backed_datasets": len(by_role) == 3
        and all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in by_role.values()
        ),
        "finite_auc_metrics": len(by_role) == 3
        and all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in by_role.values()
        ),
    }
    aucs = {
        role: float(row.get("metrics", {}).get("auc", math.nan))
        for role, row in by_role.items()
    }
    margins = {
        "true_minus_corrupted": aucs.get("true", math.nan)
        - aucs.get("corrupted", math.nan),
        "true_minus_independent": aucs.get("true", math.nan)
        - aucs.get("independent", math.nan),
    }
    research_checks = {
        "true_auc_at_least_0p55": aucs.get("true", 0.0) >= 0.55,
        "true_exceeds_corrupted_by_0p005": margins["true_minus_corrupted"] >= 0.005,
        "true_exceeds_independent_by_0p005": margins["true_minus_independent"] >= 0.005,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_skinny_attribution_protocol_invalid"
        next_action = "repair only the failed T2-C protocol check before interpretation"
    elif all(research_checks.values()):
        status = "pass"
        decision = f"innovation1_runtime_spn_skinny_attribution_seed{expected_seed}_supported"
        next_action = (
            "run the identical seed1 T2-C matrix"
            if expected_seed == 0
            else "synthesize the two-seed P-layer and general-GF(2) runtime-topology evidence"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_skinny_attribution_not_supported"
        next_action = (
            "stop T2-C and general-GF(2) scale-up; diagnose the failed signal or "
            "topology-control gate without changing the frozen protocol"
        )
    return {
        "run_id": run_id,
        "cipher": "SKINNY-64/64",
        "seed": expected_seed,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": margins,
        "thresholds": {"true_auc": 0.55, "control_margin": 0.005},
        "claim_scope": (
            f"SKINNY-64/64 r7 seed{expected_seed} local 2048/class general-GF(2) "
            "runtime-topology attribution diagnostic only; not formal scale, paper "
            "reproduction, attack, SOTA, or universal-SPN evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "remote scale-up",
            "increase samples, pairs, or epochs",
            "claim stable general-GF(2) topology superiority before both seeds pass",
        ],
    }


__all__ = ["MODELS", "adjudicate_runtime_spn_skinny_attribution"]
