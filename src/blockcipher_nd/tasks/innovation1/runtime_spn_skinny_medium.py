from __future__ import annotations

import math
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_attribution import MODELS


SAMPLES_PER_CLASS = 65_536
TRAIN_ROWS = 131_072
VALIDATION_ROWS = 65_536
SIGNAL_FLOOR = 0.55
CONTROL_MARGIN = 0.005


def adjudicate_runtime_spn_skinny_medium(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    expected_seed: int,
) -> dict[str, Any]:
    if expected_seed not in {0, 1}:
        raise ValueError("RTG2-A supports only seed0 and conditional seed1")

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
        "batch_size",
        "train_eval_interval",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
        "model_options",
        "dataset_cache_root",
        "dataset_cache_chunk_size",
        "dataset_cache_workers",
    )
    cache_roots = {
        str(row.get("training", {}).get("dataset_cache_root") or "")
        for row in by_role.values()
    }
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
        "frozen_rtg2a_scale_and_task": len(by_role) == 3
        and all(
            row.get("cipher") == "SKINNY-64/64"
            and row.get("rounds") == 7
            and row.get("seed") == expected_seed
            and row.get("samples_per_class") == SAMPLES_PER_CLASS
            and row.get("pairs_per_sample") == 4
            and row.get("difference_profile") == "skinny64_gohr2022_single_key"
            and row.get("input_difference") == 0x2000
            and row.get("training", {}).get("train_rows") == TRAIN_ROWS
            and row.get("training", {}).get("validation_rows") == VALIDATION_ROWS
            and row.get("training", {}).get("epochs") == 5
            and row.get("training", {}).get("batch_size") == 64
            and row.get("training", {}).get("train_eval_interval") == 1
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
        "disk_backed_parameter_matched_cache": len(by_role) == 3
        and len(cache_roots) == 1
        and next(iter(cache_roots), "").startswith(
            "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        )
        and all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            and row.get("training", {}).get("dataset_cache_chunk_size") == 1024
            and row.get("training", {}).get("dataset_cache_workers") == 1
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
        "true_auc_at_least_0p55": aucs.get("true", 0.0) >= SIGNAL_FLOOR,
        "true_exceeds_corrupted_by_0p005": (
            margins["true_minus_corrupted"] >= CONTROL_MARGIN
        ),
        "true_exceeds_independent_by_0p005": (
            margins["true_minus_independent"] >= CONTROL_MARGIN
        ),
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_rtg2a_skinny_medium_protocol_invalid"
        next_action = "repair only the failed RTG2-A protocol check; do not interpret or rerun at larger scale"
    elif all(research_checks.values()):
        status = "pass"
        decision = f"innovation1_rtg2a_skinny_medium_seed{expected_seed}_supported"
        next_action = (
            "run the identical conditional seed1 RTG2-A matrix"
            if expected_seed == 0
            else "synthesize the two-seed medium evidence before considering 262144/class"
        )
    else:
        status = "hold"
        decision = "innovation1_rtg2a_skinny_medium_not_supported"
        next_action = (
            "stop RTG2-A scale-up and audit whether the local margin was sample variance or a "
            "training-dynamics mismatch; do not rescue it by changing the frozen protocol"
        )

    blocked_actions = [
        "claim formal, paper-scale, attack, SOTA, breakthrough, or universal-SPN evidence",
        "change difference, pairs, epochs, negatives, topology corruption, or model geometry",
        "add DDT or trail features, switch to related keys, or launch a broad cipher matrix",
        "launch 262144/class before both RTG2-A medium seeds pass",
    ]
    if expected_seed == 0 and status != "pass":
        blocked_actions.append("launch seed1 after a seed0 hold or protocol failure")

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
        "thresholds": {
            "true_auc": SIGNAL_FLOOR,
            "control_margin": CONTROL_MARGIN,
        },
        "claim_scope": (
            f"SKINNY-64/64 r7 seed{expected_seed} remote medium 65536/class general-GF(2) "
            "runtime-topology replication; architecture/protocol diagnostic only, not formal "
            "scale, paper reproduction, attack, SOTA, breakthrough, or universal-SPN evidence"
        ),
        "next_action": next_action,
        "blocked_actions": blocked_actions,
    }


__all__ = [
    "CONTROL_MARGIN",
    "SAMPLES_PER_CLASS",
    "SIGNAL_FLOOR",
    "TRAIN_ROWS",
    "VALIDATION_ROWS",
    "adjudicate_runtime_spn_skinny_medium",
]
