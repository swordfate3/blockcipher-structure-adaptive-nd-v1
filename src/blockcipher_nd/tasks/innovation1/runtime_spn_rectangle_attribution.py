from __future__ import annotations

import math
from typing import Any


EXPECTED_SEEDS = (0, 1)
MODELS = {
    "true": "runtime_spn_e4_equivariant_true",
    "corrupted": "runtime_spn_e4_equivariant_corrupted",
    "independent": "runtime_spn_e4_equivariant_independent",
}
PROFILE = "rectangle80_weng_repo_best_trail_r6"
INPUT_DIFFERENCE = 0x0000002100010020
DESCRIPTOR_SHA256 = "904241dc1d42470188b5ed6a1c080a24191433cfc065f8838cdbe06ba2a2a4cd"
TRUE_AUC_FLOOR = 0.55
CONTROL_MARGIN = 0.005
MEDIUM_SAMPLES_PER_CLASS = 65_536
MEDIUM_TRAIN_ROWS = 131_072
MEDIUM_VALIDATION_ROWS = 65_536
RUNTIME_E4_PARAMETER_COUNT = 442_466
RUNTIME_E4_MODEL_OPTIONS = {
    "runtime_structure_path": "configs/runtime/spn/rectangle64.json",
    "runtime_rounds": 2,
    "processor_steps": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "sbox_context_mode": "late_pair",
}


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _history_replays_selected_checkpoint(row: dict[str, Any]) -> bool:
    history = row.get("history")
    training = row.get("training")
    metrics = row.get("metrics")
    if (
        not isinstance(history, list)
        or len(history) != 5
        or not isinstance(training, dict)
        or not isinstance(metrics, dict)
    ):
        return False
    for expected_epoch, epoch_row in enumerate(history, start=1):
        if not isinstance(epoch_row, dict):
            return False
        if not _finite_number(epoch_row.get("epoch")) or int(
            float(epoch_row["epoch"])
        ) != expected_epoch:
            return False
        if not all(
            _finite_number(epoch_row.get(metric))
            for metric in ("train_loss", "train_auc", "val_loss", "val_auc")
        ):
            return False
    best = max(history, key=lambda epoch_row: float(epoch_row["val_auc"]))
    best_checkpoint_metric = training.get("best_checkpoint_metric")
    result_auc = metrics.get("auc")
    return (
        training.get("epochs_ran") == 5
        and training.get("best_epoch") == int(best["epoch"])
        and training.get("checkpoint_metric") == "val_auc"
        and training.get("restore_best_checkpoint") is True
        and training.get("selected_checkpoint") == "best"
        and training.get("stopped_epoch") == 0
        and _finite_number(best_checkpoint_metric)
        and _finite_number(result_auc)
        and math.isclose(
            float(best_checkpoint_metric),
            float(best["val_auc"]),
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        and math.isclose(
            float(result_auc),
            float(best["val_auc"]),
            rel_tol=0.0,
            abs_tol=1e-7,
        )
    )


def adjudicate_runtime_spn_rectangle_attribution(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (int(row.get("seed", -1)), str(row.get("model", ""))): row
        for row in rows
    }
    expected_keys = {
        (seed, model)
        for seed in EXPECTED_SEEDS
        for model in MODELS.values()
    }
    reference = by_key.get((0, MODELS["true"]), {})
    static_fields = (
        "cipher",
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
        "model_options",
    )
    expected_options = {
        "runtime_structure_path": "configs/runtime/spn/rectangle64.json",
        "runtime_rounds": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": "late_pair",
    }
    complete = len(rows) == 6 and len(by_key) == 6 and set(by_key) == expected_keys
    protocol_checks = {
        "two_seed_three_control_matrix_complete": complete,
        "same_data_protocol_except_seed": complete
        and all(
            all(row.get(field) == reference.get(field) for field in static_fields)
            for row in by_key.values()
        ),
        "same_training_protocol": complete
        and all(
            all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in by_key.values()
        ),
        "frozen_rct1_scale_and_task": complete
        and all(
            row.get("cipher") == "RECTANGLE-80"
            and row.get("cipher_key") == "rectangle80"
            and row.get("rounds") == 6
            and row.get("seed") in EXPECTED_SEEDS
            and row.get("samples_per_class") == 2048
            and row.get("pairs_per_sample") == 4
            and row.get("difference_profile") == PROFILE
            and row.get("difference_member") == 0
            and row.get("input_difference") == INPUT_DIFFERENCE
            and row.get("train_key") == 0
            and row.get("validation_key") == 0x11111111111111111111
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
            and row.get("training", {}).get("epochs") == 5
            and row.get("training", {}).get("model_options") == expected_options
            for row in by_key.values()
        ),
        "strict_encrypted_random_plaintext_negatives": complete
        and all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in by_key.values()
        ),
        "raw_independent_ciphertext_pairs": complete
        and all(
            row.get("feature_encoding") == "ciphertext_pair_bits"
            and row.get("sample_structure") == "independent_pairs"
            and row.get("training", {}).get("input_bits") == 512
            for row in by_key.values()
        ),
        "exact_rectangle_descriptor": complete
        and all(
            row.get("runtime_structure_descriptor_name")
            == "RECTANGLE-80 runtime SPN structure"
            and str(row.get("runtime_structure_descriptor_path", "")).replace(
                "\\", "/"
            ).endswith("configs/runtime/spn/rectangle64.json")
            and row.get("runtime_structure_descriptor_sha256") == DESCRIPTOR_SHA256
            and row.get("runtime_structure_loaded_rounds") == 2
            and row.get("runtime_structure_unique_transition_count") == 1
            and row.get("runtime_structure_homogeneous") is True
            for row in by_key.values()
        ),
        "runtime_control_modes_recorded": complete
        and all(
            by_key[(seed, model)].get("runtime_structure_mode") == role
            for seed in EXPECTED_SEEDS
            for role, model in MODELS.items()
        ),
        "corrupted_topology_is_distinct": complete
        and all(
            by_key[(seed, MODELS["true"])].get("runtime_structure_window_sha256")
            != by_key[(seed, MODELS["corrupted"])].get(
                "runtime_structure_window_sha256"
            )
            for seed in EXPECTED_SEEDS
        ),
        "equal_parameter_geometry": complete
        and len(
            {
                (
                    int(row.get("parameter_count", -1)),
                    int(row.get("trainable_parameter_count", -1)),
                )
                for row in by_key.values()
            }
        )
        == 1,
        "runtime_bit_order_adapter_recorded": complete
        and all(
            row.get("input_bit_order") == "project_msb_to_runtime_lsb"
            for row in by_key.values()
        ),
        "disk_backed_datasets": complete
        and all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in by_key.values()
        ),
        "finite_auc_metrics": complete
        and all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in by_key.values()
        ),
    }

    aucs = {
        str(seed): {
            role: float(
                by_key.get((seed, model), {}).get("metrics", {}).get("auc", math.nan)
            )
            for role, model in MODELS.items()
        }
        for seed in EXPECTED_SEEDS
    }
    margins = {
        str(seed): {
            "true_minus_corrupted": (
                aucs[str(seed)]["true"] - aucs[str(seed)]["corrupted"]
            ),
            "true_minus_independent": (
                aucs[str(seed)]["true"] - aucs[str(seed)]["independent"]
            ),
        }
        for seed in EXPECTED_SEEDS
    }
    research_checks = {
        str(seed): {
            "true_auc_at_least_0p55": aucs[str(seed)]["true"] >= TRUE_AUC_FLOOR,
            "true_exceeds_corrupted_by_0p005": (
                margins[str(seed)]["true_minus_corrupted"] >= CONTROL_MARGIN
            ),
            "true_exceeds_independent_by_0p005": (
                margins[str(seed)]["true_minus_independent"] >= CONTROL_MARGIN
            ),
        }
        for seed in EXPECTED_SEEDS
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_rectangle_attribution_protocol_invalid"
        next_action = "repair only the failed RCT1 protocol check and rerun the frozen matrix"
    elif all(
        all(seed_checks.values()) for seed_checks in research_checks.values()
    ):
        status = "pass"
        decision = "innovation1_runtime_spn_rectangle_noncontiguous_attribution_supported"
        next_action = (
            "freeze a remote 65536/class seed0 confirmation with the identical data, "
            "model and three controls when the active remote lane is available"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_rectangle_noncontiguous_attribution_not_supported"
        next_action = (
            "do not scale; inspect the failed per-seed signal/control gate and rank one "
            "cell-orientation or topology-identifiability redesign"
        )
    return {
        "run_id": run_id,
        "cipher": "RECTANGLE-80",
        "seeds": list(EXPECTED_SEEDS),
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": margins,
        "thresholds": {
            "true_auc": TRUE_AUC_FLOOR,
            "control_margin": CONTROL_MARGIN,
        },
        "claim_scope": (
            "RECTANGLE-80 r6 local 2048/class non-contiguous-cell RuntimeE4 "
            "topology attribution diagnostic only; not formal scale, paper "
            "reproduction, attack, SOTA, or universal-SPN evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "remote scale-up before both local seeds pass",
            "increase samples, pairs, epochs, or change the frozen difference",
            "claim cross-cipher weight reuse or universal SPN support from RCT1",
        ],
    }


def adjudicate_runtime_spn_rectangle_medium(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    expected_seed: int,
) -> dict[str, Any]:
    if expected_seed not in {0, 1}:
        raise ValueError("RECTANGLE RCT2 supports seed0 and conditional seed1")

    by_model = {str(row.get("model", "")): row for row in rows}
    complete = len(rows) == 3 and set(by_model) == set(MODELS.values())
    by_role = {
        role: by_model.get(model, {}) for role, model in MODELS.items()
    }
    reference = by_role["true"]
    static_fields = (
        "cipher",
        "cipher_key",
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
        "batch_size",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_eval_interval",
        "train_rows",
        "validation_rows",
        "model_options",
        "dataset_cache_root",
        "dataset_cache_chunk_size",
        "dataset_cache_workers",
    )
    cache_roots = {
        str(row.get("training", {}).get("dataset_cache_root", ""))
        for row in by_role.values()
    }
    cache_root = next(iter(cache_roots), "")
    protocol_checks = {
        "three_runtime_controls_complete": complete,
        "same_data_protocol": complete
        and all(
            all(row.get(field) == reference.get(field) for field in static_fields)
            for row in by_role.values()
        ),
        "same_training_protocol": complete
        and all(
            all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in by_role.values()
        ),
        "frozen_rct2_scale_and_task": complete
        and all(
            row.get("cipher") == "RECTANGLE-80"
            and row.get("cipher_key") == "rectangle80"
            and row.get("rounds") == 6
            and row.get("seed") == expected_seed
            and row.get("samples_per_class") == MEDIUM_SAMPLES_PER_CLASS
            and row.get("pairs_per_sample") == 4
            and row.get("difference_profile") == PROFILE
            and row.get("difference_member") == 0
            and row.get("input_difference") == INPUT_DIFFERENCE
            and row.get("train_key") == 0
            and row.get("validation_key") == 0x11111111111111111111
            and row.get("training", {}).get("train_rows") == MEDIUM_TRAIN_ROWS
            and row.get("training", {}).get("validation_rows")
            == MEDIUM_VALIDATION_ROWS
            and row.get("training", {}).get("epochs") == 5
            and row.get("training", {}).get("batch_size") == 64
            and row.get("training", {}).get("train_eval_interval") == 1
            and row.get("training", {}).get("model_options")
            == RUNTIME_E4_MODEL_OPTIONS
            for row in by_role.values()
        ),
        "strict_encrypted_random_plaintext_negatives": complete
        and all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in by_role.values()
        ),
        "raw_independent_ciphertext_pairs": complete
        and all(
            row.get("feature_encoding") == "ciphertext_pair_bits"
            and row.get("sample_structure") == "independent_pairs"
            and row.get("training", {}).get("input_bits") == 512
            and row.get("training", {}).get("pair_bits") == 128
            for row in by_role.values()
        ),
        "exact_rectangle_descriptor": complete
        and all(
            row.get("runtime_structure_descriptor_name")
            == "RECTANGLE-80 runtime SPN structure"
            and str(row.get("runtime_structure_descriptor_path", "")).replace(
                "\\", "/"
            ).endswith("configs/runtime/spn/rectangle64.json")
            and row.get("runtime_structure_descriptor_sha256") == DESCRIPTOR_SHA256
            and row.get("runtime_structure_loaded_rounds") == 2
            and row.get("runtime_structure_unique_transition_count") == 1
            and row.get("runtime_structure_homogeneous") is True
            for row in by_role.values()
        ),
        "runtime_control_modes_recorded": complete
        and all(
            by_role[role].get("runtime_structure_mode") == role
            for role in MODELS
        ),
        "corrupted_topology_is_distinct": complete
        and by_role["true"].get("runtime_structure_window_sha256")
        != by_role["corrupted"].get("runtime_structure_window_sha256"),
        "equal_parameter_geometry": complete
        and all(
            row.get("parameter_count") == RUNTIME_E4_PARAMETER_COUNT
            and row.get("trainable_parameter_count")
            == RUNTIME_E4_PARAMETER_COUNT
            for row in by_role.values()
        ),
        "runtime_bit_order_adapter_recorded": complete
        and all(
            row.get("input_bit_order") == "project_msb_to_runtime_lsb"
            for row in by_role.values()
        ),
        "disk_backed_remote_cache": complete
        and len(cache_roots) == 1
        and cache_root.replace("/", "\\").lower().startswith("g:\\lxy\\")
        and all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            and row.get("training", {}).get("dataset_cache_chunk_size") == 1024
            and row.get("training", {}).get("dataset_cache_workers") == 1
            for row in by_role.values()
        ),
        "validation_geometry": complete
        and all(
            row.get("validation", {}).get("samples_per_class") == 32_768
            and row.get("validation", {}).get("samples_total")
            == MEDIUM_VALIDATION_ROWS
            and row.get("validation", {}).get("positive_rows") == 32_768
            and row.get("validation", {}).get("negative_rows") == 32_768
            and row.get("validation", {}).get("negative_mode")
            == "encrypted_random_plaintexts"
            for row in by_role.values()
        ),
        "five_epoch_best_checkpoint_replay": complete
        and all(
            _history_replays_selected_checkpoint(row)
            for row in by_role.values()
        ),
        "finite_auc_metrics": complete
        and all(
            _finite_number(row.get("metrics", {}).get("auc"))
            for row in by_role.values()
        ),
    }
    aucs = {
        role: float(by_role[role].get("metrics", {}).get("auc", math.nan))
        for role in MODELS
    }
    margins = {
        "true_minus_corrupted": aucs["true"] - aucs["corrupted"],
        "true_minus_independent": aucs["true"] - aucs["independent"],
    }
    research_checks = {
        "true_auc_at_least_0p55": aucs["true"] >= TRUE_AUC_FLOOR,
        "true_exceeds_corrupted_by_0p005": (
            margins["true_minus_corrupted"] >= CONTROL_MARGIN
        ),
        "true_exceeds_independent_by_0p005": (
            margins["true_minus_independent"] >= CONTROL_MARGIN
        ),
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_rct2_rectangle_medium_protocol_invalid"
        next_action = "repair only the failed evidence check and rerun the frozen seed"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_rct2_rectangle_medium_seed_passed"
        next_action = (
            "run the identical 65536/class seed1 confirmation after visual QA and "
            "verified result retrieval"
            if expected_seed == 0
            else "adjudicate the two distinct 65536/class seeds before any larger run"
        )
    else:
        status = "hold"
        decision = "innovation1_rct2_rectangle_medium_signal_not_confirmed"
        next_action = (
            "stop the scale ladder and diagnose the failed signal/control margin "
            "without increasing samples, pairs, epochs, or rounds"
        )
    return {
        "run_id": run_id,
        "cipher": "RECTANGLE-80",
        "phase": "rct2",
        "seed": expected_seed,
        "samples_per_class": MEDIUM_SAMPLES_PER_CLASS,
        "train_rows": MEDIUM_TRAIN_ROWS,
        "validation_rows": MEDIUM_VALIDATION_ROWS,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": margins,
        "thresholds": {
            "true_auc": TRUE_AUC_FLOOR,
            "control_margin": CONTROL_MARGIN,
        },
        "claim_scope": (
            "RECTANGLE-80 r6 single-seed 65536/class medium RuntimeE4 topology "
            "confirmation; not formal scale, paper reproduction, attack, SOTA, "
            "or universal-SPN evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "call 65536/class formal training or paper-scale evidence",
            "launch seed1 before verified seed0 gate and visual QA",
            "increase samples, pairs, epochs, or rounds as a rescue",
        ],
    }


__all__ = [
    "CONTROL_MARGIN",
    "DESCRIPTOR_SHA256",
    "EXPECTED_SEEDS",
    "INPUT_DIFFERENCE",
    "MEDIUM_SAMPLES_PER_CLASS",
    "MEDIUM_TRAIN_ROWS",
    "MEDIUM_VALIDATION_ROWS",
    "MODELS",
    "PROFILE",
    "RUNTIME_E4_MODEL_OPTIONS",
    "RUNTIME_E4_PARAMETER_COUNT",
    "TRUE_AUC_FLOOR",
    "adjudicate_runtime_spn_rectangle_attribution",
    "adjudicate_runtime_spn_rectangle_medium",
]
