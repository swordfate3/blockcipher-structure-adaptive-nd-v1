from __future__ import annotations

import math
from typing import Any


MODELS = {
    "correct": "runtime_spn_e4_equivariant_true",
    "corrupted": "runtime_spn_e4_equivariant_corrupted",
    "no_topology": "runtime_spn_e4_equivariant_independent",
}
SEEDS = (0, 1)
TRUE_AUC_FLOOR = 0.520
CONTROL_MARGIN = 0.005


def adjudicate_runtime_spn_dialga_d1(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (int(row.get("seed", -1)), str(row.get("model"))): row for row in rows
    }
    expected_keys = {(seed, model) for seed in SEEDS for model in MODELS.values()}
    groups = {
        seed: {
            role: by_key.get((seed, model), {}) for role, model in MODELS.items()
        }
        for seed in SEEDS
    }
    reference = groups[0]["correct"]
    flat_rows = [groups[seed][role] for seed in SEEDS for role in MODELS]

    static_fields = (
        "cipher",
        "cipher_key",
        "rounds",
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
        "input_bits",
        "pair_bits",
    )
    protocol_checks = {
        "six_rows_two_seeds_complete": len(rows) == 6
        and set(by_key) == expected_keys,
        "same_data_protocol": all(
            row
            and all(row.get(field) == reference.get(field) for field in static_fields)
            for row in flat_rows
        ),
        "same_training_protocol": all(
            row
            and all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in flat_rows
        ),
        "frozen_d1_task_and_scale": all(
            row.get("cipher") == "Dialga-128"
            and row.get("cipher_key") == "dialga128"
            and row.get("rounds") == 4
            and row.get("samples_per_class") == 2048
            and row.get("pairs_per_sample") == 4
            and row.get("input_difference") == 0x40
            and row.get("train_key") == 0
            and row.get("validation_key") == int("11" * 32, 16)
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
            and row.get("training", {}).get("epochs") == 10
            and row.get("training", {}).get("input_bits") == 1024
            and row.get("training", {}).get("pair_bits") == 256
            for row in flat_rows
        ),
        "strict_encrypted_random_plaintext_negatives": all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in flat_rows
        ),
        "raw_independent_ciphertext_pairs": all(
            row.get("feature_encoding") == "ciphertext_pair_bits"
            and row.get("sample_structure") == "independent_pairs"
            for row in flat_rows
        ),
        "equal_parameter_geometry": len(
            {
                (
                    int(row.get("parameter_count", -1)),
                    int(row.get("trainable_parameter_count", -1)),
                )
                for row in flat_rows
            }
        )
        == 1,
        "runtime_descriptor_contract": _runtime_descriptor_contract(groups),
        "runtime_topology_controls_distinct": _topology_controls_distinct(groups),
        "best_validation_checkpoints_restored": all(
            row.get("training", {}).get("checkpoint_metric") == "val_auc"
            and row.get("training", {}).get("restore_best_checkpoint") is True
            and row.get("training", {}).get("selected_checkpoint") == "best"
            for row in flat_rows
        ),
        "complete_best_checkpoint_histories": all(
            _history_matches_result(row) for row in flat_rows
        ),
        "disk_backed_datasets": all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in flat_rows
        ),
        "finite_auc_metrics": all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in flat_rows
        ),
    }

    aucs = {
        f"seed{seed}": {
            role: float(groups[seed][role].get("metrics", {}).get("auc", math.nan))
            for role in MODELS
        }
        for seed in SEEDS
    }
    margins = {
        seed_key: {
            "correct_minus_corrupted": values["correct"] - values["corrupted"],
            "correct_minus_no_topology": values["correct"]
            - values["no_topology"],
        }
        for seed_key, values in aucs.items()
    }
    research_checks = {
        f"seed{seed}_correct_auc_at_least_0p520": aucs[f"seed{seed}"]["correct"]
        >= TRUE_AUC_FLOOR
        for seed in SEEDS
    }
    research_checks.update(
        {
            f"seed{seed}_correct_exceeds_corrupted_by_0p005": margins[f"seed{seed}"][
                "correct_minus_corrupted"
            ]
            >= CONTROL_MARGIN
            for seed in SEEDS
        }
    )
    research_checks.update(
        {
            f"seed{seed}_correct_exceeds_no_topology_by_0p005": margins[f"seed{seed}"][
                "correct_minus_no_topology"
            ]
            >= CONTROL_MARGIN
            for seed in SEEDS
        }
    )

    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_dialga_runtime_e4_d1_protocol_invalid"
        next_action = "repair only the failed frozen protocol check before interpretation"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_dialga_runtime_e4_d1_two_seed_supported"
        next_action = (
            "freeze both correct best checkpoints and run same-checkpoint swaps to "
            "corrupted and no-topology Dialga structures before any scale increase"
        )
    else:
        status = "hold"
        decision = "innovation1_dialga_runtime_e4_d1_not_supported"
        next_action = (
            "keep the Runtime-E4 architecture fixed and run only a tiny Dialga input-"
            "difference screen if all roles are near chance; otherwise diagnose the "
            "failed topology control without increasing samples"
        )

    return {
        "run_id": run_id,
        "cipher": "Dialga-128",
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": margins,
        "thresholds": {
            "correct_auc": TRUE_AUC_FLOOR,
            "control_margin": CONTROL_MARGIN,
        },
        "claim_scope": (
            "Dialga-128 prefix-r4 two-seed local 2048/class runtime-topology "
            "diagnostic only; not formal scale, attack, paper reproduction, SOTA, or "
            "universal-SPN evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "remote scale-up",
            "increase samples, pairs, epochs, or rounds before the D1 gate",
            "claim topology attribution from correct AUC without both controls",
            "claim a Dialga attack or formal cross-cipher result",
        ],
    }


def _runtime_descriptor_contract(
    groups: dict[int, dict[str, dict[str, Any]]],
) -> bool:
    expected_modes = {
        "correct": "true",
        "corrupted": "corrupted",
        "no_topology": "independent",
    }
    descriptor_hashes: set[str] = set()
    for seed in SEEDS:
        for role in MODELS:
            row = groups[seed][role]
            descriptor_hashes.add(str(row.get("runtime_structure_descriptor_sha256")))
            if not (
                row.get("runtime_structure_descriptor_name")
                == "Dialga-128 20-round heterogeneous runtime SPN structure"
                and str(row.get("runtime_structure_descriptor_path", "")).endswith(
                    "configs/runtime/spn/dialga128.json"
                )
                and row.get("runtime_structure_round_start") == 2
                and row.get("runtime_structure_available_rounds") == 20
                and row.get("runtime_structure_loaded_rounds") == 2
                and row.get("runtime_structure_unique_transition_count") == 2
                and row.get("runtime_structure_homogeneous") is False
                and row.get("runtime_structure_mode") == expected_modes[role]
                and row.get("runtime_structure_window_control") == "full"
                and len(row.get("runtime_structure_transition_sha256s", [])) == 2
                and len(str(row.get("runtime_structure_window_sha256", ""))) == 64
            ):
                return False
    return len(descriptor_hashes) == 1 and len(next(iter(descriptor_hashes), "")) == 64


def _topology_controls_distinct(
    groups: dict[int, dict[str, dict[str, Any]]],
) -> bool:
    role_hashes: dict[str, set[str]] = {role: set() for role in MODELS}
    for seed in SEEDS:
        for role in MODELS:
            role_hashes[role].add(
                str(groups[seed][role].get("runtime_structure_window_sha256", ""))
            )
    return (
        all(len(values) == 1 for values in role_hashes.values())
        and role_hashes["correct"] == role_hashes["no_topology"]
        and role_hashes["correct"] != role_hashes["corrupted"]
    )


def _history_matches_result(row: dict[str, Any]) -> bool:
    history = row.get("history")
    if not isinstance(history, list) or len(history) != 10:
        return False
    try:
        best_auc = max(float(epoch["val_auc"]) for epoch in history)
        result_auc = float(row["metrics"]["auc"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(best_auc) and abs(best_auc - result_auc) <= 1e-12


__all__ = [
    "CONTROL_MARGIN",
    "MODELS",
    "SEEDS",
    "TRUE_AUC_FLOOR",
    "adjudicate_runtime_spn_dialga_d1",
]
