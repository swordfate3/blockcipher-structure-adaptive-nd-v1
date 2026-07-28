from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
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
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_sbox_transition_k1ak import (
    EXPECTED_PARAMETER_COUNT,
    RUN_ID as K1AK_RUN_ID,
    build_k1ak_control,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_transition_causal_k1al import (
    RUN_ID as K1AL_RUN_ID,
    TransitionBranchOffWrapper,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = (
    "i1_uknit_family_midori64_semantic_contrast_"
    "k1am_2048_seed6_seed7_20260729"
)
K1AK_DECISION = (
    "innovation1_uknit_family_midori64_k1ak_"
    "sbox_transition_discrimination_failed"
)
K1AL_DECISION = (
    "innovation1_uknit_family_midori64_k1al_"
    "transition_and_sbox_causal_use_supported"
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
    "k1al_gate": "481cf6c90c281766e891c9a04de28d82cdf2b5051abbb570e83e81b0d5a433c2",
    "k1al_validation": (
        "f58f38e72b8b27bcd2ac75502265526bf049c79ee7756ed70501db9670a15a65"
    ),
    "k1al_results": (
        "a7a28643e76a36456143c7e112641fe8c2890eda9b892241b51b8f60dc463ce5"
    ),
}
ORIENTATIONS = ("correct_oriented", "swapped_orientation")
ORIENTATION_MODELS = {
    "correct_oriented": "runtime_spn_ct_k1ak_sbox_transition_true",
    "swapped_orientation": "runtime_spn_ct_k1ak_sbox_transition_wrong_sbox",
}
MODEL_TO_ORIENTATION = {model: name for name, model in ORIENTATION_MODELS.items()}
ORIENTATION_OPTIONS = {
    "correct_oriented": "correct_vs_wrong",
    "swapped_orientation": "wrong_vs_correct",
}
PRIMARY_CONDITIONS = {
    "correct_oriented": "correct_structure",
    "swapped_orientation": "wrong_sbox",
}
COUNTERFACTUAL_CONDITIONS = {
    "correct_oriented": "wrong_sbox",
    "swapped_orientation": "correct_structure",
}
EVALUATION_CONDITIONS = (
    "correct_runtime",
    "wrong_sbox_same_checkpoint",
    "transition_branch_off_same_checkpoint",
)
CONTRAST_SCALE = 0.25
CONTRAST_MARGIN = 0.02
SEMANTIC_MARGIN = 0.005
ORIENTATION_MARGIN = 0.005
INDEPENDENT_WRONG_MARGIN = 0.005
BRANCH_MARGIN = 0.005
ANCHOR_RETENTION_MARGIN = -0.010
PROBABILITY_DELTA_FLOOR = 1e-6
EXPECTED_OPTIMIZER_STEPS = 640
EXPECTED_TRAINING_ROWS = len(EXPECTED_SEEDS) * len(ORIENTATIONS)
EXPECTED_EVALUATION_ROWS = (
    len(EXPECTED_SEEDS)
    * len(ORIENTATIONS)
    * len(EXPECTED_SPLITS)
    * len(EVALUATION_CONDITIONS)
)


def task_map(
    tasks: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for task in tasks:
        orientation = MODEL_TO_ORIENTATION.get(str(task.get("model_key")))
        if orientation is None:
            continue
        key = (int(task["seed"]), orientation)
        if key in mapped:
            raise ValueError(f"duplicate K1-AM task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_training_keys():
        raise ValueError("K1-AM task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == EXPECTED_TRAINING_ROWS
        and set(mapped) == expected_training_keys()
        and all(
            task.get("cipher_key") == "midori64"
            and int(task.get("rounds", -1)) == 4
            and int(task.get("seed", -1)) == seed
            and task.get("model_key") == ORIENTATION_MODELS[orientation]
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
            and _model_options_frozen(task.get("model_options", {}), orientation)
            for (seed, orientation), task in mapped.items()
        )
    )


def _model_options_frozen(options: Mapping[str, Any], orientation: str) -> bool:
    try:
        return (
            options.get("runtime_structure_path") == "configs/runtime/spn/midori64.json"
            and int(options.get("runtime_round_start", -1)) == RUNTIME_ROUND_START
            and int(options.get("runtime_rounds", -1)) == RUNTIME_ROUNDS
            and int(options.get("cipher_round_window_start", -1))
            == CIPHER_ROUND_WINDOW_START
            and int(options.get("pair_embedding_dim", -1)) == 128
            and int(options.get("transition_value_dim", -1)) == 20
            and int(options.get("virtual_projection_slots", -1)) == 16
            and int(options.get("active_cell", -1)) == 8
            and int(options.get("active_bit_role", -1)) == 1
            and int(str(options.get("input_difference_hex", "0")), 0)
            == INPUT_DIFFERENCE
            and int(options.get("topology_corruption_seed", -1)) == 20260729
            and options.get("semantic_contrast_orientation")
            == ORIENTATION_OPTIONS[orientation]
            and float(options.get("semantic_contrast_scale", math.nan))
            == CONTRAST_SCALE
            and float(options.get("semantic_contrast_margin", math.nan))
            == CONTRAST_MARGIN
        )
    except (TypeError, ValueError):
        return False


def build_k1am_model(
    *,
    task: Mapping[str, Any],
    orientation: str,
    input_bits: int = 512,
) -> torch.nn.Module:
    if orientation not in ORIENTATIONS:
        raise ValueError("unknown K1-AM orientation")
    torch.manual_seed(int(task["seed"]))
    primary = build_k1ak_control(
        task=task,
        condition=PRIMARY_CONDITIONS[orientation],
        input_bits=input_bits,
    )
    counterfactual = build_k1ak_control(
        task=task,
        condition=COUNTERFACTUAL_CONDITIONS[orientation],
        input_bits=input_bits,
    )
    options = task["model_options"]
    primary.configure_semantic_contrast(
        orientation=ORIENTATION_OPTIONS[orientation],
        counterfactual_structure=counterfactual.runtime_structure,
        counterfactual_apply_sboxes=counterfactual.apply_sboxes,
        scale=float(options["semantic_contrast_scale"]),
        margin=float(options["semantic_contrast_margin"]),
    )
    return primary


def build_model_checks(tasks: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    mapped = task_map(tasks)
    models = {
        key: build_k1am_model(task=task, orientation=key[1])
        for key, task in mapped.items()
    }
    parameter_counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    }
    geometries = {
        tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())
        for model in models.values()
    }
    initial_hashes = {
        key: tensor_mapping_sha256(model.state_dict()) for key, model in models.items()
    }
    return {
        "four_models_parameter_count_exact": parameter_counts
        == {EXPECTED_PARAMETER_COUNT},
        "four_models_state_geometry_identical": len(geometries) == 1,
        "paired_initialization_exact_within_seed": all(
            initial_hashes[(seed, "correct_oriented")]
            == initial_hashes[(seed, "swapped_orientation")]
            for seed in EXPECTED_SEEDS
        ),
        "distinct_seed_initializations": len(
            {initial_hashes[(seed, "correct_oriented")] for seed in EXPECTED_SEEDS}
        )
        == len(EXPECTED_SEEDS),
        "contrast_configuration_exact": all(
            model.semantic_contrast_orientation == ORIENTATION_OPTIONS[key[1]]
            and model.semantic_contrast_scale == CONTRAST_SCALE
            and model.semantic_contrast_margin == CONTRAST_MARGIN
            and model.semantic_counterfactual_structure is not None
            for key, model in models.items()
        ),
        "no_new_trainable_semantic_state": all(
            not any("semantic" in name for name, _parameter in model.named_parameters())
            for model in models.values()
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
    k1al_gate: Mapping[str, Any],
    k1al_validation: Mapping[str, Any],
    k1al_results: Sequence[Mapping[str, Any]],
    source_digests: Mapping[str, str],
) -> dict[str, bool]:
    expected_dataset_keys = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    k1ak_keys = {
        (int(row.get("seed", -1)), str(row.get("split")), str(row.get("condition")))
        for row in k1ak_controls
    }
    k1al_keys = {
        (int(row.get("seed", -1)), str(row.get("split")), str(row.get("condition")))
        for row in k1al_results
    }
    return {
        "source_artifact_digests_exact": dict(source_digests)
        == EXPECTED_SOURCE_DIGESTS,
        "k1ak_exact_hold": (
            k1ak_gate.get("run_id") == K1AK_RUN_ID
            and k1ak_gate.get("status") == "hold"
            and k1ak_gate.get("decision") == K1AK_DECISION
            and k1ak_gate.get("remote_scale") == "no"
            and not k1ak_gate.get("failed_protocol_checks")
        ),
        "k1ak_validation_exact_pass": (
            k1ak_validation.get("run_id") == K1AK_RUN_ID
            and k1ak_validation.get("status") == "pass"
            and not k1ak_validation.get("errors")
        ),
        "twenty_four_k1ak_anchor_rows": (
            len(k1ak_controls) == 24
            and k1ak_keys
            == {
                (seed, split, condition)
                for seed in EXPECTED_SEEDS
                for split in EXPECTED_SPLITS
                for condition in (
                    "correct_structure",
                    "wrong_sbox",
                    "corrupted_linear",
                    "no_structure",
                )
            }
        ),
        "six_bound_dataset_rows": (
            len(dataset_manifest) == 6
            and {
                (int(row.get("seed", -1)), str(row.get("split")))
                for row in dataset_manifest
            }
            == expected_dataset_keys
            and all(
                int(row.get("cell", -1)) == 8
                and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
                and int(row.get("rounds", -1)) == 4
                and row.get("cache_payloads_present") is True
                for row in dataset_manifest
            )
        ),
        "k1al_exact_pass": (
            k1al_gate.get("run_id") == K1AL_RUN_ID
            and k1al_gate.get("status") == "pass"
            and k1al_gate.get("decision") == K1AL_DECISION
            and k1al_gate.get("remote_scale") == "no"
            and not k1al_gate.get("failed_protocol_checks")
            and not k1al_gate.get("failed_research_checks")
        ),
        "k1al_validation_exact_pass": (
            k1al_validation.get("run_id") == K1AL_RUN_ID
            and k1al_validation.get("status") == "pass"
            and not k1al_validation.get("errors")
            and int(k1al_validation.get("optimizer_steps", -1)) == 0
        ),
        "eighteen_k1al_causal_rows": (
            len(k1al_results) == 18
            and k1al_keys
            == {
                (seed, split, condition)
                for seed in EXPECTED_SEEDS
                for split in EXPECTED_SPLITS
                for condition in EVALUATION_CONDITIONS
            }
        ),
    }


def evaluate_k1am_panel(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    datasets: Mapping[tuple[int, str], DifferentialDataset],
    batch_size: int = EXPECTED_BATCH_SIZE,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    tasks_by_key = task_map(tasks)
    trained = training_map(training_rows)
    checkpoints = checkpoint_map(checkpoint_manifest)
    if set(datasets) != expected_dataset_keys():
        raise ValueError("K1-AM requires all six bound Midori64 datasets")

    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for orientation in ORIENTATIONS:
            key = (seed, orientation)
            task = tasks_by_key[key]
            trained_row = trained[key]
            checkpoint_row = checkpoints[key]
            checkpoint_path = Path(str(checkpoint_row["path"]))
            state, checkpoint_sha256 = load_bound_state(
                checkpoint_path,
                checkpoint_row,
            )
            state_sha256 = tensor_mapping_sha256(state)
            for split in EXPECTED_SPLITS:
                dataset = datasets[(seed, split)]
                labels = np.asarray(dataset.labels, dtype=np.float32)
                dataset_sha256 = differential_dataset_sha256(dataset)
                correct = build_k1ak_control(
                    task=task,
                    condition="correct_structure",
                    input_bits=int(dataset.features.shape[1]),
                )
                wrong = build_k1ak_control(
                    task=task,
                    condition="wrong_sbox",
                    input_bits=int(dataset.features.shape[1]),
                )
                branch_off_base = build_k1ak_control(
                    task=task,
                    condition="correct_structure",
                    input_bits=int(dataset.features.shape[1]),
                )
                base_models = {
                    "correct_runtime": correct,
                    "wrong_sbox_same_checkpoint": wrong,
                    "transition_branch_off_same_checkpoint": branch_off_base,
                }
                for model in base_models.values():
                    model.load_state_dict(state, strict=True)
                    if tensor_mapping_sha256(model.state_dict()) != state_sha256:
                        raise ValueError("K1-AM strict load changed selected state")
                audit_models: dict[str, torch.nn.Module] = {
                    "correct_runtime": correct,
                    "wrong_sbox_same_checkpoint": wrong,
                    "transition_branch_off_same_checkpoint": (
                        TransitionBranchOffWrapper(branch_off_base)
                    ),
                }
                probabilities = {
                    condition: predict_binary_probabilities(
                        model,
                        dataset,
                        batch_size=batch_size,
                        device=device,
                    )
                    for condition, model in audit_models.items()
                }
                correct_probabilities = probabilities["correct_runtime"]
                for condition in EVALUATION_CONDITIONS:
                    model = base_models[condition]
                    values = probabilities[condition]
                    delta = np.abs(correct_probabilities - values)
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "seed": seed,
                            "orientation": orientation,
                            "split": split,
                            "condition": condition,
                            "primary_condition": PRIMARY_CONDITIONS[orientation],
                            "cipher_key": "midori64",
                            "rounds": 4,
                            "auc": binary_auc(labels, values),
                            "correct_minus_condition_auc": (
                                0.0
                                if condition == "correct_runtime"
                                else binary_auc(labels, correct_probabilities)
                                - binary_auc(labels, values)
                            ),
                            "max_abs_probability_delta_from_correct": float(
                                delta.max()
                            ),
                            "mean_abs_probability_delta_from_correct": float(
                                delta.mean()
                            ),
                            "probability_sha256": hashlib.sha256(
                                values.astype(np.float32, copy=False).tobytes()
                            ).hexdigest(),
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_sha256": checkpoint_sha256,
                            "checkpoint_selected": checkpoint_row.get(
                                "selected_checkpoint"
                            ),
                            "state_dict_sha256": state_sha256,
                            "initial_state_sha256": trained_row.get(
                                "initial_state_sha256"
                            ),
                            "dataset_sha256": dataset_sha256,
                            "composition_sha256": model.composition_sha256,
                            "sbox_transition_semantics_sha256": (
                                model.sbox_transition_semantics_sha256
                            ),
                            "transition_branch_enabled": condition
                            != "transition_branch_off_same_checkpoint",
                            "rows": int(dataset.features.shape[0]),
                            "input_bits": int(dataset.features.shape[1]),
                            "pairs_per_sample": int(
                                dataset.metadata["pairs_per_sample"]
                            ),
                            "input_difference": int(
                                dataset.metadata["input_difference"]
                            ),
                            "negative_mode": dataset.metadata["negative_mode"],
                            "sample_structure": dataset.metadata["sample_structure"],
                            "parameter_count": sum(
                                parameter.numel() for parameter in model.parameters()
                            ),
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                            "epochs": 0,
                        }
                    )
    return rows


def adjudicate_k1am(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    k1ak_controls: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
    model_checks: Mapping[str, bool],
) -> dict[str, Any]:
    trained = training_map(training_rows, fail_closed=False)
    evaluated = evaluation_map(evaluation_rows)
    checkpoints = checkpoint_map(checkpoint_manifest, fail_closed=False)
    anchors = k1ak_anchor_map(k1ak_controls)
    seed_results = {
        str(seed): {
            split: split_result(evaluated, anchors, seed, split)
            for split in EXPECTED_SPLITS
        }
        for seed in EXPECTED_SEEDS
    }
    protocol_checks = {
        **dict(source_checks),
        **dict(model_checks),
        "four_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "four_training_rows_complete": (
            len(training_rows) == EXPECTED_TRAINING_ROWS
            and set(trained) == expected_training_keys()
        ),
        "training_protocol_frozen": training_protocol_frozen(training_rows),
        "paired_training_initialization_exact": all(
            trained[(seed, "correct_oriented")].get("initial_state_sha256")
            == trained[(seed, "swapped_orientation")].get("initial_state_sha256")
            for seed in EXPECTED_SEEDS
            if (seed, "correct_oriented") in trained
            and (seed, "swapped_orientation") in trained
        ),
        "four_checkpoint_manifest_entries": (
            len(checkpoint_manifest.get("entries", [])) == EXPECTED_TRAINING_ROWS
            and set(checkpoints) == expected_training_keys()
        ),
        "thirty_six_evaluation_rows_complete": (
            len(evaluation_rows) == EXPECTED_EVALUATION_ROWS
            and set(evaluated) == expected_evaluation_keys()
        ),
        "evaluation_rows_zero_training": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and int(row.get("epochs", -1)) == 0
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
        "same_state_and_dataset_per_orientation_split": all(
            len(
                {
                    (
                        evaluated[(seed, orientation, split, condition)].get(
                            "checkpoint_sha256"
                        ),
                        evaluated[(seed, orientation, split, condition)].get(
                            "state_dict_sha256"
                        ),
                        evaluated[(seed, orientation, split, condition)].get(
                            "dataset_sha256"
                        ),
                    )
                    for condition in EVALUATION_CONDITIONS
                }
            )
            == 1
            for seed in EXPECTED_SEEDS
            for orientation in ORIENTATIONS
            for split in EXPECTED_SPLITS
        ),
        "runtime_interventions_exact": runtime_interventions_exact(evaluated),
        "cross_key_primary_auc_replays_training": all(
            abs(
                float(
                    evaluated[
                        (
                            seed,
                            orientation,
                            "cross_key_validation",
                            (
                                "correct_runtime"
                                if orientation == "correct_oriented"
                                else "wrong_sbox_same_checkpoint"
                            ),
                        )
                    ]["auc"]
                )
                - float(trained[(seed, orientation)]["metrics"]["auc"])
            )
            <= 1e-6
            for seed in EXPECTED_SEEDS
            for orientation in ORIENTATIONS
            if (seed, orientation) in trained
        ),
        "finite_metrics": all(
            _finite(row.get(field))
            for row in evaluation_rows
            for field in (
                "auc",
                "correct_minus_condition_auc",
                "max_abs_probability_delta_from_correct",
                "mean_abs_probability_delta_from_correct",
            )
        ),
    }

    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        for split in FRESH_SPLITS:
            result = seed_results[str(seed)][split]
            prefix = f"seed{seed}_{split}"
            research_checks[f"{prefix}_correct_auc_floor"] = (
                result["candidate_correct_auc"] >= AUC_FLOOR
            )
            research_checks[f"{prefix}_retains_k1ak_correct_anchor"] = (
                result["candidate_minus_k1ak_correct"] >= ANCHOR_RETENTION_MARGIN
            )
            research_checks[f"{prefix}_beats_k1ak_independent_wrong"] = (
                result["candidate_minus_k1ak_wrong"] >= INDEPENDENT_WRONG_MARGIN
            )
            research_checks[f"{prefix}_beats_swapped_primary"] = (
                result["candidate_minus_swapped_primary"] >= ORIENTATION_MARGIN
            )
            research_checks[f"{prefix}_beats_wrong_same_checkpoint"] = (
                result["candidate_minus_wrong_same_checkpoint"] >= SEMANTIC_MARGIN
            )
            research_checks[f"{prefix}_beats_transition_off"] = (
                result["candidate_minus_transition_off"] >= BRANCH_MARGIN
            )
            research_checks[f"{prefix}_wrong_changes_predictions"] = (
                result["wrong_max_probability_delta"] > PROBABILITY_DELTA_FLOOR
            )
            research_checks[f"{prefix}_transition_off_changes_predictions"] = (
                result["transition_off_max_probability_delta"]
                > PROBABILITY_DELTA_FLOOR
            )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    anchor_pass = all(
        research_checks[f"seed{seed}_{split}_correct_auc_floor"]
        and research_checks[f"seed{seed}_{split}_retains_k1ak_correct_anchor"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    causal_pass = all(
        research_checks[f"seed{seed}_{split}_beats_wrong_same_checkpoint"]
        and research_checks[f"seed{seed}_{split}_beats_transition_off"]
        and research_checks[f"seed{seed}_{split}_wrong_changes_predictions"]
        and research_checks[f"seed{seed}_{split}_transition_off_changes_predictions"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    substitute_pass = all(
        research_checks[f"seed{seed}_{split}_beats_k1ak_independent_wrong"]
        and research_checks[f"seed{seed}_{split}_beats_swapped_primary"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_midori64_k1am_protocol_invalid"
        next_action = "repair only the failed K1-AM binding and rerun unchanged"
    elif research_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_midori64_k1am_"
            "paired_semantic_contrast_supported"
        )
        next_action = (
            "retain the K1-AK representation and paired semantic objective, then "
            "run one same-protocol uKNIT-BC or Dialga transfer attribution panel"
        )
    elif anchor_pass and causal_pass and not substitute_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1am_"
            "semantic_preference_imposed_substitute_unresolved"
        )
        next_action = (
            "discard this objective and audit one shared-normalization or "
            "representation-level identifiability constraint without tuning the "
            "contrast weight or margin"
        )
    elif not anchor_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1am_"
            "semantic_contrast_destroys_anchor_signal"
        )
        next_action = (
            "discard the paired objective and return to the unmodified K1-AK "
            "optimization path before another representation hypothesis"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1am_"
            "semantic_contrast_causal_retention_failed"
        )
        next_action = (
            "discard the objective and locate whether S-box or transition-branch "
            "causality was lost before another trained model"
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
            "candidate_minus_k1ak_correct": ANCHOR_RETENTION_MARGIN,
            "candidate_minus_k1ak_independent_wrong": INDEPENDENT_WRONG_MARGIN,
            "candidate_minus_swapped_primary": ORIENTATION_MARGIN,
            "candidate_minus_wrong_same_checkpoint": SEMANTIC_MARGIN,
            "candidate_minus_transition_off": BRANCH_MARGIN,
            "max_probability_delta": PROBABILITY_DELTA_FLOOR,
            "semantic_contrast_scale": CONTRAST_SCALE,
            "semantic_contrast_margin": CONTRAST_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class Midori64 r4 fixed-budget paired semantic-"
            "contrast diagnostic; not formal scale, attack, SOTA, family transfer, "
            "arbitrary-SPN, or ceiling evidence"
        ),
        "blocked_actions": [
            "remote scale or family transfer from K1-AM",
            "contrast weight/margin sweep or more data, pairs, epochs, seeds, rounds",
            "DDT/trail inputs, width changes, or MoE",
        ],
    }


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    mapped = training_map(rows, fail_closed=False)
    return (
        len(rows) == EXPECTED_TRAINING_ROWS
        and set(mapped) == expected_training_keys()
        and all(
            int(row.get("trainable_parameter_count", -1))
            == EXPECTED_PARAMETER_COUNT
            and int(row.get("rounds", -1)) == 4
            and int(row.get("seed", -1)) == seed
            and row.get("orientation") == orientation
            and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
            and int(row.get("samples_per_class", -1))
            == EXPECTED_TRAIN_SAMPLES_PER_CLASS
            and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("semantic_contrast_orientation")
            == ORIENTATION_OPTIONS[orientation]
            and float(row.get("semantic_contrast_scale", math.nan))
            == CONTRAST_SCALE
            and float(row.get("semantic_contrast_margin", math.nan))
            == CONTRAST_MARGIN
            and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
            and int(row.get("training", {}).get("epochs_ran", -1))
            == EXPECTED_EPOCHS
            and int(row.get("training", {}).get("batch_size", -1))
            == EXPECTED_BATCH_SIZE
            and int(row.get("training", {}).get("optimizer_steps", -1))
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
            and all(
                _finite(epoch.get("train_auxiliary_loss"))
                and float(epoch.get("train_auxiliary_loss", 0.0)) >= 0.0
                and _finite(epoch.get("train_semantic_loss_gap"))
                for epoch in row.get("history", [])
            )
            and any(
                float(epoch.get("train_auxiliary_loss", 0.0)) > 0.0
                for epoch in row.get("history", [])
            )
            for (seed, orientation), row in mapped.items()
        )
    )


def training_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        orientation = str(row.get("orientation"))
        if orientation not in ORIENTATIONS:
            continue
        key = (int(row["seed"]), orientation)
        if key in mapped:
            raise ValueError(f"duplicate K1-AM training row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_training_keys():
        raise ValueError("K1-AM training panel is incomplete")
    return mapped


def checkpoint_map(
    manifest: Mapping[str, Any],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in manifest.get("entries", []):
        orientation = str(row.get("orientation"))
        if orientation not in ORIENTATIONS:
            continue
        key = (int(row["seed"]), orientation)
        if key in mapped:
            raise ValueError(f"duplicate K1-AM checkpoint: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_training_keys():
        raise ValueError("K1-AM checkpoint panel is incomplete")
    return mapped


def evaluation_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["orientation"]),
            str(row["split"]),
            str(row["condition"]),
        )
        if key in mapped:
            raise ValueError(f"duplicate K1-AM evaluation row: {key}")
        mapped[key] = row
    return mapped


def k1ak_anchor_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped = {
        (int(row["seed"]), str(row["split"]), str(row["condition"])): row
        for row in rows
    }
    required = {
        (seed, split, condition)
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in ("correct_structure", "wrong_sbox")
    }
    if not required.issubset(mapped):
        raise ValueError("K1-AM requires K1-AK correct and wrong-S-box anchors")
    return mapped


def split_result(
    rows: Mapping[tuple[int, str, str, str], Mapping[str, Any]],
    anchors: Mapping[tuple[int, str, str], Mapping[str, Any]],
    seed: int,
    split: str,
) -> dict[str, float]:
    candidate_correct = rows[
        (seed, "correct_oriented", split, "correct_runtime")
    ]
    candidate_wrong = rows[
        (seed, "correct_oriented", split, "wrong_sbox_same_checkpoint")
    ]
    candidate_off = rows[
        (
            seed,
            "correct_oriented",
            split,
            "transition_branch_off_same_checkpoint",
        )
    ]
    swapped_primary = rows[
        (seed, "swapped_orientation", split, "wrong_sbox_same_checkpoint")
    ]
    candidate_auc = float(candidate_correct["auc"])
    wrong_auc = float(candidate_wrong["auc"])
    off_auc = float(candidate_off["auc"])
    swapped_auc = float(swapped_primary["auc"])
    k1ak_correct = float(anchors[(seed, split, "correct_structure")]["auc"])
    k1ak_wrong = float(anchors[(seed, split, "wrong_sbox")]["auc"])
    return {
        "candidate_correct_auc": candidate_auc,
        "candidate_wrong_same_checkpoint_auc": wrong_auc,
        "candidate_transition_off_auc": off_auc,
        "swapped_primary_auc": swapped_auc,
        "k1ak_correct_anchor_auc": k1ak_correct,
        "k1ak_independent_wrong_auc": k1ak_wrong,
        "candidate_minus_k1ak_correct": candidate_auc - k1ak_correct,
        "candidate_minus_k1ak_wrong": candidate_auc - k1ak_wrong,
        "candidate_minus_swapped_primary": candidate_auc - swapped_auc,
        "candidate_minus_wrong_same_checkpoint": candidate_auc - wrong_auc,
        "candidate_minus_transition_off": candidate_auc - off_auc,
        "wrong_max_probability_delta": float(
            candidate_wrong["max_abs_probability_delta_from_correct"]
        ),
        "transition_off_max_probability_delta": float(
            candidate_off["max_abs_probability_delta_from_correct"]
        ),
    }


def runtime_interventions_exact(
    rows: Mapping[tuple[int, str, str, str], Mapping[str, Any]],
) -> bool:
    if set(rows) != expected_evaluation_keys():
        return False
    return all(
        rows[(seed, orientation, split, "correct_runtime")].get(
            "composition_sha256"
        )
        == rows[
            (
                seed,
                orientation,
                split,
                "transition_branch_off_same_checkpoint",
            )
        ].get("composition_sha256")
        and rows[(seed, orientation, split, "correct_runtime")].get(
            "sbox_transition_semantics_sha256"
        )
        == rows[
            (
                seed,
                orientation,
                split,
                "transition_branch_off_same_checkpoint",
            )
        ].get("sbox_transition_semantics_sha256")
        and rows[(seed, orientation, split, "correct_runtime")].get(
            "composition_sha256"
        )
        != rows[(seed, orientation, split, "wrong_sbox_same_checkpoint")].get(
            "composition_sha256"
        )
        and rows[
            (
                seed,
                orientation,
                split,
                "transition_branch_off_same_checkpoint",
            )
        ].get("transition_branch_enabled")
        is False
        for seed in EXPECTED_SEEDS
        for orientation in ORIENTATIONS
        for split in EXPECTED_SPLITS
    )


def expected_training_keys() -> set[tuple[int, str]]:
    return {(seed, orientation) for seed in EXPECTED_SEEDS for orientation in ORIENTATIONS}


def expected_dataset_keys() -> set[tuple[int, str]]:
    return {(seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS}


def expected_evaluation_keys() -> set[tuple[int, str, str, str]]:
    return {
        (seed, orientation, split, condition)
        for seed in EXPECTED_SEEDS
        for orientation in ORIENTATIONS
        for split in EXPECTED_SPLITS
        for condition in EVALUATION_CONDITIONS
    }


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "ANCHOR_RETENTION_MARGIN",
    "BRANCH_MARGIN",
    "CONTRAST_MARGIN",
    "CONTRAST_SCALE",
    "EVALUATION_CONDITIONS",
    "EXPECTED_EVALUATION_ROWS",
    "EXPECTED_OPTIMIZER_STEPS",
    "EXPECTED_SOURCE_DIGESTS",
    "EXPECTED_TRAINING_ROWS",
    "INDEPENDENT_WRONG_MARGIN",
    "ORIENTATIONS",
    "ORIENTATION_MARGIN",
    "ORIENTATION_MODELS",
    "ORIENTATION_OPTIONS",
    "PRIMARY_CONDITIONS",
    "RUN_ID",
    "SEMANTIC_MARGIN",
    "adjudicate_k1am",
    "build_k1am_model",
    "build_model_checks",
    "candidate_protocol_frozen",
    "checkpoint_map",
    "evaluate_k1am_panel",
    "expected_dataset_keys",
    "expected_training_keys",
    "source_binding_checks",
    "task_map",
    "training_protocol_frozen",
]
