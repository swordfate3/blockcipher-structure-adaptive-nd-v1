from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import load_bound_state
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_neural_attribution_k1ai import (
    AUC_FLOOR,
    CIPHER_ROUND_WINDOW_START,
    DIFFERENCE_PROFILE,
    EXPECTED_BATCH_SIZE,
    EXPECTED_EPOCHS,
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_PAIRS,
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_TRAIN_SAMPLES_PER_CLASS,
    FRESH_SPLITS,
    INPUT_DIFFERENCE,
    RUNTIME_ROUNDS,
    RUNTIME_ROUND_START,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_midori64_canonical_walsh_k1an_2048_seed6_seed7_20260729"
K1AK_RUN_ID = "i1_uknit_family_midori64_sbox_transition_k1ak_2048_seed6_seed7_20260729"
K1AM_RUN_ID = (
    "i1_uknit_family_midori64_semantic_contrast_k1am_2048_seed6_seed7_20260729"
)
K1AK_DECISION = (
    "innovation1_uknit_family_midori64_k1ak_sbox_transition_discrimination_failed"
)
K1AM_DECISION = (
    "innovation1_uknit_family_midori64_k1am_"
    "semantic_preference_imposed_substitute_unresolved"
)
EXPECTED_SOURCE_DIGESTS = {
    "k1ak_gate": "a8cd9de68a7b4e43a4c8f0793e31cbf8ce87f090c35be6f6821cab282e927f8f",
    "k1ak_validation": (
        "2d64a4e27b39a65fda5b44b217226fabb78a954d843573b47abbe34e0070e419"
    ),
    "k1ak_controls": (
        "3b667435eb6c91dfb1c828953e834e9556dedf16c5054b4e70ded1d598e6e04e"
    ),
    "k1ak_dataset_manifest": (
        "5525a28f099a21bcca09aafbe05498f0f7951e22e171eaac6db055c174ff35bc"
    ),
    "k1am_gate": "eda28b0116560d3d1fc5f4dcdcd3859e48ff671a5c04844b42d5494b34458178",
    "k1am_validation": (
        "988f425ab6d2c1f9ccda2b6f1d509a1cf636e7e1150d6a0e8042798eb05fab0a"
    ),
    "k1am_results": (
        "1b67df6adb9ae265cdb1c2dadcc9abf1fbf8be965d2e30d3f250e8106bd7731c"
    ),
    "k1am_controls": (
        "686a642e536a2ac5a1c4e09e1569c6f8d76f1c70d4c060a8a18c5663b4f4f904"
    ),
}
CONTROL_MODELS = {
    "correct_structure": "runtime_spn_ct_k1an_walsh_transition_true",
    "wrong_sbox": "runtime_spn_ct_k1an_walsh_transition_wrong_sbox",
    "transition_branch_off": "runtime_spn_ct_k1an_walsh_transition_branch_off",
}
MODEL_TO_CONDITION = {model: condition for condition, model in CONTROL_MODELS.items()}
CONTROL_CONDITIONS = tuple(CONTROL_MODELS)
EXPECTED_PARAMETER_COUNT = 131_876
EXPECTED_OPTIMIZER_STEPS = 640
EXPECTED_TRAINING_ROWS = len(EXPECTED_SEEDS) * len(CONTROL_CONDITIONS)
EXPECTED_EVALUATION_ROWS = (
    len(EXPECTED_SEEDS) * len(EXPECTED_SPLITS) * len(CONTROL_CONDITIONS)
)
ANCHOR_RETENTION_MARGIN = -0.010
SEMANTIC_MARGIN = 0.005
BRANCH_MARGIN = 0.005
CANONICAL_WALSH_FEATURES = 64
TOPOLOGY_CORRUPTION_SEED = 20260729


def build_k1an_control(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int = 512,
) -> torch.nn.Module:
    if condition not in CONTROL_MODELS:
        raise ValueError("unknown K1-AN condition")
    return build_model(
        CONTROL_MODELS[condition],
        input_bits=input_bits,
        hidden_bits=32,
        pair_bits=128,
        structure="SPN",
        model_options=deepcopy(dict(task["model_options"])),
    )


def task_map(
    tasks: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for task in tasks:
        condition = MODEL_TO_CONDITION.get(str(task.get("model_key")))
        if condition is None:
            continue
        key = (int(task["seed"]), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-AN task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-AN task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == EXPECTED_TRAINING_ROWS
        and set(mapped) == expected_condition_keys()
        and all(
            task.get("cipher_key") == "midori64"
            and int(task.get("rounds", -1)) == 4
            and int(task.get("seed", -1)) == seed
            and task.get("model_key") == CONTROL_MODELS[condition]
            and int(task.get("samples_per_class", -1))
            == EXPECTED_TRAIN_SAMPLES_PER_CLASS
            and task.get("validation_samples_total") is None
            and int(task.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and int(task.get("input_difference", -1)) == INPUT_DIFFERENCE
            and task.get("difference_profile") == DIFFERENCE_PROFILE
            and task.get("feature_encoding") == "ciphertext_pair_bits"
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("sample_structure") == "independent_pairs"
            and int(task.get("key_rotation_interval", -1)) == 0
            and task.get("loss") == "mse"
            and task.get("optimizer") == "adam"
            and float(task.get("learning_rate", math.nan)) == 1e-4
            and float(task.get("weight_decay", math.nan)) == 1e-5
            and task.get("lr_scheduler") == "none"
            and task.get("checkpoint_metric") == "val_auc"
            and task.get("restore_best_checkpoint") is True
            and int(task.get("target_epochs", -1)) == EXPECTED_EPOCHS
            and _model_options_frozen(task.get("model_options", {}))
            for (seed, condition), task in mapped.items()
        )
    )


def _model_options_frozen(options: Mapping[str, Any]) -> bool:
    try:
        return (
            options.get("runtime_structure_path") == "configs/runtime/spn/midori64.json"
            and int(options.get("runtime_round_start", -1)) == RUNTIME_ROUND_START
            and int(options.get("runtime_rounds", -1)) == RUNTIME_ROUNDS
            and int(options.get("cipher_round_window_start", -1))
            == CIPHER_ROUND_WINDOW_START
            and int(options.get("pair_embedding_dim", -1)) == 128
            and float(options.get("dropout", math.nan)) == 0.0
            and float(options.get("residual_gate_initial_effective", math.nan)) == 0.05
            and float(options.get("transition_gate_initial_effective", math.nan))
            == 0.05
            and int(options.get("canonical_walsh_features", -1))
            == CANONICAL_WALSH_FEATURES
            and int(options.get("active_cell", -1)) == 8
            and int(options.get("active_bit_role", -1)) == 1
            and int(str(options.get("input_difference_hex", "0")), 0)
            == INPUT_DIFFERENCE
            and int(options.get("topology_corruption_seed", -1))
            == TOPOLOGY_CORRUPTION_SEED
        )
    except (TypeError, ValueError):
        return False


def build_control_checks(tasks: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    mapped = task_map(tasks)
    models: dict[tuple[int, str], torch.nn.Module] = {}
    with torch.random.fork_rng():
        for key, task in mapped.items():
            torch.manual_seed(key[0])
            models[key] = build_k1an_control(task=task, condition=key[1])

    parameter_counts = {
        int(model_metadata(model)["trainable_parameter_count"])
        for model in models.values()
    }
    geometries = {
        tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())
        for model in models.values()
    }
    initial_hashes = {
        key: tensor_mapping_sha256(model.state_dict()) for key, model in models.items()
    }
    correct = models[(EXPECTED_SEEDS[0], "correct_structure")]
    wrong = models[(EXPECTED_SEEDS[0], "wrong_sbox")]
    branch_off = models[(EXPECTED_SEEDS[0], "transition_branch_off")]
    forbidden_parameters = ("transition_encoder", "transition_projection", "walsh")
    return {
        "all_models_parameter_count_exact": parameter_counts
        == {EXPECTED_PARAMETER_COUNT},
        "all_models_state_dict_geometry_identical": len(geometries) == 1,
        "same_seed_initial_tensor_hash_exact": all(
            len({initial_hashes[(seed, condition)] for condition in CONTROL_CONDITIONS})
            == 1
            for seed in EXPECTED_SEEDS
        ),
        "distinct_seed_initializations": (
            initial_hashes[(EXPECTED_SEEDS[0], "correct_structure")]
            != initial_hashes[(EXPECTED_SEEDS[1], "correct_structure")]
        ),
        "no_trainable_walsh_or_transition_encoder": all(
            not any(token in name for token in forbidden_parameters)
            for model in models.values()
            for name, _ in model.named_parameters()
        ),
        "runtime_cell_partition_identical": all(
            torch.equal(
                model.runtime_structure.cell_membership,
                correct.runtime_structure.cell_membership,
            )
            and torch.equal(
                model.runtime_structure.bit_role,
                correct.runtime_structure.bit_role,
            )
            for model in (wrong, branch_off)
        ),
        "wrong_sbox_changes_only_sbox": (
            not torch.equal(
                wrong.runtime_structure.sbox_truth_bits,
                correct.runtime_structure.sbox_truth_bits,
            )
            and torch.equal(
                wrong.runtime_structure.linear_matrices,
                correct.runtime_structure.linear_matrices,
            )
            and wrong.apply_sboxes is True
        ),
        "branch_off_changes_only_forward_use": (
            torch.equal(
                branch_off.runtime_structure.sbox_truth_bits,
                correct.runtime_structure.sbox_truth_bits,
            )
            and torch.equal(
                branch_off.runtime_structure.linear_matrices,
                correct.runtime_structure.linear_matrices,
            )
            and correct.transition_branch_enabled is True
            and branch_off.transition_branch_enabled is False
        ),
        "canonical_basis_exact_and_shared": (
            all(
                model.canonical_walsh_features_per_stage == 64
                for model in models.values()
            )
            and all(
                len(model.canonical_walsh_mask_pairs) == 64 for model in models.values()
            )
            and len({model.canonical_walsh_fingerprint for model in models.values()})
            == 1
        ),
        "runtime_transition_semantics_identifiable": (
            correct.sbox_transition_semantics_sha256
            != wrong.sbox_transition_semantics_sha256
            and correct.sbox_transition_semantics_sha256
            == branch_off.sbox_transition_semantics_sha256
        ),
        "no_cipher_or_absolute_cell_identity": all(
            model.uses_cipher_identity is False
            and model.uses_absolute_cell_or_bit_identity is False
            and model.uses_runtime_native_cell_slots is False
            for model in models.values()
        ),
    }


def source_binding_checks(
    *,
    k1ak_gate: Mapping[str, Any],
    k1ak_validation: Mapping[str, Any],
    k1ak_controls: Sequence[Mapping[str, Any]],
    dataset_manifest: Sequence[Mapping[str, Any]],
    k1am_gate: Mapping[str, Any],
    k1am_validation: Mapping[str, Any],
    k1am_results: Sequence[Mapping[str, Any]],
    k1am_controls: Sequence[Mapping[str, Any]],
    source_digests: Mapping[str, str],
) -> dict[str, bool]:
    expected_datasets = expected_dataset_keys()
    anchor_keys = {
        (int(row.get("seed", -1)), str(row.get("split")))
        for row in k1ak_controls
        if row.get("condition") == "correct_structure"
    }
    return {
        "source_artifact_digests_exact": dict(source_digests)
        == EXPECTED_SOURCE_DIGESTS,
        "k1ak_exact_hold": (
            k1ak_gate.get("run_id") == K1AK_RUN_ID
            and k1ak_gate.get("status") == "hold"
            and k1ak_gate.get("decision") == K1AK_DECISION
            and not k1ak_gate.get("failed_protocol_checks")
        ),
        "k1ak_validation_exact_pass": (
            k1ak_validation.get("run_id") == K1AK_RUN_ID
            and k1ak_validation.get("status") == "pass"
            and not k1ak_validation.get("errors")
        ),
        "six_k1ak_correct_anchor_rows": anchor_keys == expected_datasets,
        "six_bound_dataset_rows": (
            len(dataset_manifest) == len(expected_datasets)
            and {
                (int(row.get("seed", -1)), str(row.get("split")))
                for row in dataset_manifest
            }
            == expected_datasets
            and all(
                int(row.get("cell", -1)) == 8
                and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
                and int(row.get("rounds", -1)) == 4
                and row.get("cache_payloads_present") is True
                for row in dataset_manifest
            )
        ),
        "k1am_exact_hold": (
            k1am_gate.get("run_id") == K1AM_RUN_ID
            and k1am_gate.get("status") == "hold"
            and k1am_gate.get("decision") == K1AM_DECISION
            and not k1am_gate.get("failed_protocol_checks")
            and k1am_gate.get("remote_scale") == "no"
        ),
        "k1am_validation_exact_pass": (
            k1am_validation.get("run_id") == K1AM_RUN_ID
            and k1am_validation.get("status") == "pass"
            and not k1am_validation.get("errors")
            and int(k1am_validation.get("training_rows", -1)) == 4
            and int(k1am_validation.get("evaluation_rows", -1)) == 36
        ),
        "four_k1am_training_rows": len(k1am_results) == 4,
        "thirty_six_k1am_evaluation_rows": len(k1am_controls) == 36,
    }


def evaluate_k1an_panel(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    datasets: Mapping[tuple[int, str], DifferentialDataset],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    tasks_by_key = task_map(tasks)
    trained = training_map(training_rows)
    checkpoints = checkpoint_map(checkpoint_manifest)
    if set(datasets) != expected_dataset_keys():
        raise ValueError("K1-AN requires six seed6/7 cell8 datasets")

    rows: list[dict[str, Any]] = []
    for seed, condition in sorted(expected_condition_keys()):
        task = tasks_by_key[(seed, condition)]
        source = trained[(seed, condition)]
        checkpoint_path = Path(str(source["training"]["checkpoint_output"]))
        state, checkpoint_sha = load_bound_state(
            checkpoint_path,
            checkpoints[(seed, condition)],
        )
        state_sha = tensor_mapping_sha256(state)
        for split in EXPECTED_SPLITS:
            dataset = datasets[(seed, split)]
            model = build_k1an_control(
                task=task,
                condition=condition,
                input_bits=int(dataset.features.shape[1]),
            )
            model.load_state_dict(state, strict=True)
            if tensor_mapping_sha256(model.state_dict()) != state_sha:
                raise ValueError("K1-AN strict checkpoint load changed learned state")
            probabilities = predict_binary_probabilities(
                model,
                dataset,
                batch_size=EXPECTED_BATCH_SIZE,
                device=device,
            )
            labels = np.asarray(dataset.labels, dtype=np.float32)
            rows.append(
                {
                    "run_id": RUN_ID,
                    "cipher_key": "midori64",
                    "rounds": 4,
                    "seed": seed,
                    "condition": condition,
                    "model": CONTROL_MODELS[condition],
                    "split": split,
                    "rows": int(dataset.features.shape[0]),
                    "auc": binary_auc(labels, probabilities),
                    "dataset_sha256": differential_dataset_sha256(dataset),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha,
                    "state_dict_sha256": state_sha,
                    "composition_sha256": model.composition_sha256,
                    "sbox_transition_semantics_sha256": (
                        model.sbox_transition_semantics_sha256
                    ),
                    "canonical_walsh_fingerprint": model.canonical_walsh_fingerprint,
                    "canonical_walsh_features_per_stage": (
                        model.canonical_walsh_features_per_stage
                    ),
                    "transition_branch_enabled": model.transition_branch_enabled,
                    "residual_gate": float(
                        torch.tanh(model.backbone.residual_gate.detach())
                    ),
                    "transition_gate": float(
                        torch.tanh(model.backbone.transition_gate.detach())
                    ),
                    "strict_state_dict_load": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )
    return rows


def adjudicate_k1an(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    anchor_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
    control_checks: Mapping[str, bool],
    cache_checks: Mapping[str, bool],
) -> dict[str, Any]:
    trained = training_map(training_rows, fail_closed=False)
    evaluated = evaluation_map(evaluation_rows)
    checkpoints = checkpoint_map(checkpoint_manifest, fail_closed=False)
    anchors = anchor_map(anchor_rows)
    seed_results = {
        str(seed): {
            split: split_result(evaluated, anchors, seed, split)
            for split in EXPECTED_SPLITS
        }
        for seed in EXPECTED_SEEDS
    }
    protocol_checks = {
        **dict(source_checks),
        **dict(control_checks),
        **dict(cache_checks),
        "candidate_protocol_frozen": candidate_protocol_frozen(tasks),
        "six_training_rows_complete": set(trained) == expected_condition_keys(),
        "six_checkpoints_complete": set(checkpoints) == expected_condition_keys(),
        "training_protocol_frozen": training_protocol_frozen(training_rows),
        "same_seed_training_initialization_exact": same_seed_training_initialization(
            trained
        ),
        "eighteen_evaluation_rows_complete": (
            len(evaluation_rows) == EXPECTED_EVALUATION_ROWS
            and set(evaluated) == expected_evaluation_keys()
        ),
        "evaluation_rows_zero_training": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and row.get("strict_state_dict_load") is True
            for row in evaluation_rows
        ),
        "split_row_counts_exact": all(
            int(row.get("rows", -1))
            == (
                EXPECTED_TRAIN_ROWS
                if row.get("split") == "train_seen"
                else EXPECTED_HOLDOUT_ROWS
            )
            for row in evaluation_rows
        ),
        "same_dataset_per_seed_split": same_dataset_per_seed_split(evaluated),
        "runtime_interventions_exact": runtime_interventions_exact(evaluated),
        "cross_key_auc_replays_training": all(
            abs(
                float(evaluated[(seed, "cross_key_validation", condition)]["auc"])
                - float(trained[(seed, condition)]["metrics"]["auc"])
            )
            <= 1e-6
            for seed in EXPECTED_SEEDS
            for condition in CONTROL_CONDITIONS
            if (seed, condition) in trained
        ),
        "finite_metrics": all(_finite(row.get("auc")) for row in evaluation_rows),
    }

    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        for split in FRESH_SPLITS:
            result = seed_results[str(seed)][split]
            prefix = f"seed{seed}_{split}"
            research_checks[f"{prefix}_correct_auc_floor"] = (
                result["correct_auc"] >= AUC_FLOOR
            )
            research_checks[f"{prefix}_retains_k1ak_correct_anchor"] = (
                result["correct_minus_k1ak_anchor"] >= ANCHOR_RETENTION_MARGIN
            )
            research_checks[f"{prefix}_beats_wrong_sbox"] = (
                result["correct_minus_wrong_sbox"] >= SEMANTIC_MARGIN
            )
            research_checks[f"{prefix}_beats_transition_branch_off"] = (
                result["correct_minus_transition_branch_off"] >= BRANCH_MARGIN
            )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    anchor_pass = all(
        research_checks[f"seed{seed}_{split}_correct_auc_floor"]
        and research_checks[f"seed{seed}_{split}_retains_k1ak_correct_anchor"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    semantic_pass = all(
        research_checks[f"seed{seed}_{split}_beats_wrong_sbox"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    branch_pass = all(
        research_checks[f"seed{seed}_{split}_beats_transition_branch_off"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_midori64_k1an_protocol_invalid"
        next_action = "repair only the failed K1-AN binding and rerun unchanged"
    elif research_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_midori64_k1an_"
            "canonical_walsh_transition_supported"
        )
        next_action = (
            "retain the fixed Walsh representation and run one unchanged uKNIT-BC "
            "or Dialga family-transfer attribution panel before any scale"
        )
    elif anchor_pass and branch_pass and not semantic_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1an_"
            "independent_wrong_sbox_substitute_unresolved"
        )
        next_action = (
            "stop single-cipher semantic regularization and preregister a shared-weight "
            "uKNIT-BC, Midori64 and Dialga identifiability experiment"
        )
    elif not branch_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1an_"
            "canonical_transition_signal_not_supported"
        )
        next_action = (
            "discard the canonical Walsh residual and retain only the K1-AK/K1-AL "
            "same-checkpoint causal evidence"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1an_"
            "canonical_representation_too_restrictive"
        )
        next_action = (
            "discard the fixed Walsh representation and return to the K1-AK anchor "
            "before another representation hypothesis"
        )

    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "thresholds": {
            "correct_auc": AUC_FLOOR,
            "correct_minus_k1ak_anchor": ANCHOR_RETENTION_MARGIN,
            "correct_minus_wrong_sbox": SEMANTIC_MARGIN,
            "correct_minus_transition_branch_off": BRANCH_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class Midori64 r4 fixed-budget canonical Walsh "
            "transition diagnostic; not formal scale, attack, SOTA, family transfer, "
            "arbitrary-SPN, or ceiling evidence"
        ),
        "blocked_actions": [
            "remote scale or family transfer before K1-AN passes",
            "more pairs, samples, epochs, seeds, rounds, positions, or width",
            "Walsh coefficient scans, DDT/trail inputs, or MoE inside K1-AN",
        ],
    }


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    mapped = training_map(rows, fail_closed=False)
    return (
        len(rows) == EXPECTED_TRAINING_ROWS
        and set(mapped) == expected_condition_keys()
        and all(
            row.get("model") == CONTROL_MODELS[condition]
            and int(row.get("trainable_parameter_count", -1))
            == EXPECTED_PARAMETER_COUNT
            and int(row.get("rounds", -1)) == 4
            and int(row.get("seed", -1)) == seed
            and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
            and row.get("difference_profile") == DIFFERENCE_PROFILE
            and int(row.get("samples_per_class", -1))
            == EXPECTED_TRAIN_SAMPLES_PER_CLASS
            and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("initialization", {}).get("kind") == "scratch"
            and row.get("initialization", {}).get("strict_state_dict_load") is False
            and isinstance(
                row.get("initialization", {}).get("initial_state_sha256"), str
            )
            and int(row.get("training", {}).get("batch_size", -1))
            == EXPECTED_BATCH_SIZE
            and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
            and int(row.get("training", {}).get("epochs_ran", -1)) == EXPECTED_EPOCHS
            and int(row.get("training", {}).get("optimizer_state_step_after", -1))
            == EXPECTED_OPTIMIZER_STEPS
            and row.get("training", {}).get("optimizer") == "adam"
            and row.get("training", {}).get("loss") == "mse"
            and row.get("training", {}).get("checkpoint_metric") == "val_auc"
            and row.get("training", {}).get("selected_checkpoint") == "best"
            and int(row.get("training", {}).get("samples_total", -1))
            == EXPECTED_TRAIN_ROWS
            and int(row.get("validation", {}).get("samples_total", -1))
            == EXPECTED_HOLDOUT_ROWS
            and len(row.get("history", [])) == EXPECTED_EPOCHS
            for (seed, condition), row in mapped.items()
        )
    )


def same_seed_training_initialization(
    rows: Mapping[tuple[int, str], Mapping[str, Any]],
) -> bool:
    return all(
        len(
            {
                rows[(seed, condition)]
                .get("initialization", {})
                .get("initial_state_sha256")
                for condition in CONTROL_CONDITIONS
                if (seed, condition) in rows
            }
        )
        == 1
        for seed in EXPECTED_SEEDS
    ) and (
        rows[(EXPECTED_SEEDS[0], "correct_structure")]
        .get("initialization", {})
        .get("initial_state_sha256")
        != rows[(EXPECTED_SEEDS[1], "correct_structure")]
        .get("initialization", {})
        .get("initial_state_sha256")
    )


def training_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        condition = MODEL_TO_CONDITION.get(str(row.get("model")))
        if condition is None:
            continue
        key = (int(row["seed"]), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-AN training row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-AN training panel is incomplete")
    return mapped


def checkpoint_map(
    manifest: Mapping[str, Any],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in manifest.get("entries", []):
        condition = str(row.get("condition"))
        if condition not in CONTROL_CONDITIONS:
            continue
        key = (int(row["seed"]), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-AN checkpoint: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-AN checkpoint panel is incomplete")
    return mapped


def evaluation_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["split"]), str(row["condition"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-AN evaluation row: {key}")
        mapped[key] = row
    return mapped


def anchor_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped = {
        (int(row["seed"]), str(row["split"])): row
        for row in rows
        if row.get("condition") == "correct_structure"
    }
    if set(mapped) != expected_dataset_keys():
        raise ValueError("K1-AN requires six K1-AK correct anchor rows")
    return mapped


def split_result(
    rows: Mapping[tuple[int, str, str], Mapping[str, Any]],
    anchors: Mapping[tuple[int, str], Mapping[str, Any]],
    seed: int,
    split: str,
) -> dict[str, float]:
    aucs = {
        condition: float(rows[(seed, split, condition)]["auc"])
        for condition in CONTROL_CONDITIONS
    }
    correct = aucs["correct_structure"]
    anchor = float(anchors[(seed, split)]["auc"])
    return {
        "correct_auc": correct,
        "k1ak_correct_anchor_auc": anchor,
        "wrong_sbox_auc": aucs["wrong_sbox"],
        "transition_branch_off_auc": aucs["transition_branch_off"],
        "correct_minus_k1ak_anchor": correct - anchor,
        "correct_minus_wrong_sbox": correct - aucs["wrong_sbox"],
        "correct_minus_transition_branch_off": (
            correct - aucs["transition_branch_off"]
        ),
        "residual_gate": float(
            rows[(seed, split, "correct_structure")]["residual_gate"]
        ),
        "transition_gate": float(
            rows[(seed, split, "correct_structure")]["transition_gate"]
        ),
    }


def same_dataset_per_seed_split(
    rows: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> bool:
    return all(
        len(
            {
                rows[(seed, split, condition)].get("dataset_sha256")
                for condition in CONTROL_CONDITIONS
            }
        )
        == 1
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
    )


def runtime_interventions_exact(
    rows: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> bool:
    if set(rows) != expected_evaluation_keys():
        return False
    return all(
        rows[(seed, split, "correct_structure")].get("composition_sha256")
        != rows[(seed, split, "wrong_sbox")].get("composition_sha256")
        and rows[(seed, split, "correct_structure")].get(
            "sbox_transition_semantics_sha256"
        )
        != rows[(seed, split, "wrong_sbox")].get("sbox_transition_semantics_sha256")
        and rows[(seed, split, "correct_structure")].get("composition_sha256")
        == rows[(seed, split, "transition_branch_off")].get("composition_sha256")
        and rows[(seed, split, "correct_structure")].get(
            "sbox_transition_semantics_sha256"
        )
        == rows[(seed, split, "transition_branch_off")].get(
            "sbox_transition_semantics_sha256"
        )
        and rows[(seed, split, "correct_structure")].get("canonical_walsh_fingerprint")
        == rows[(seed, split, "wrong_sbox")].get("canonical_walsh_fingerprint")
        == rows[(seed, split, "transition_branch_off")].get(
            "canonical_walsh_fingerprint"
        )
        and rows[(seed, split, "correct_structure")].get("transition_branch_enabled")
        is True
        and rows[(seed, split, "wrong_sbox")].get("transition_branch_enabled") is True
        and rows[(seed, split, "transition_branch_off")].get(
            "transition_branch_enabled"
        )
        is False
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
    )


def expected_condition_keys() -> set[tuple[int, str]]:
    return {
        (seed, condition) for seed in EXPECTED_SEEDS for condition in CONTROL_CONDITIONS
    }


def expected_dataset_keys() -> set[tuple[int, str]]:
    return {(seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS}


def expected_evaluation_keys() -> set[tuple[int, str, str]]:
    return {
        (seed, split, condition)
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in CONTROL_CONDITIONS
    }


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "ANCHOR_RETENTION_MARGIN",
    "BRANCH_MARGIN",
    "CANONICAL_WALSH_FEATURES",
    "CONTROL_CONDITIONS",
    "CONTROL_MODELS",
    "EXPECTED_EVALUATION_ROWS",
    "EXPECTED_OPTIMIZER_STEPS",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_SOURCE_DIGESTS",
    "EXPECTED_TRAINING_ROWS",
    "MODEL_TO_CONDITION",
    "RUN_ID",
    "SEMANTIC_MARGIN",
    "adjudicate_k1an",
    "build_control_checks",
    "build_k1an_control",
    "candidate_protocol_frozen",
    "evaluate_k1an_panel",
    "expected_condition_keys",
    "expected_dataset_keys",
    "source_binding_checks",
]
