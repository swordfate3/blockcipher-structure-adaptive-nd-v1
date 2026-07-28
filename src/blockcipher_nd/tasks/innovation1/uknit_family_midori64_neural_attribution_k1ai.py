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
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_difference_position_k1ah import (
    CONFIRMATION_KEYS,
    RUN_ID as K1AH_RUN_ID,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_midori64_neural_attribution_k1ai_2048_seed6_seed7_20260729"
K1AH_DECISION = "innovation1_uknit_family_midori64_k1ah_confirmed_r4_position_supported"
EXPECTED_SOURCE_DIGESTS = {
    "gate": "5fb101cd892dcedb849e7a4745996fc9fced8d9450b0449c8f206b53cc786708",
    "dataset_manifest": (
        "6e7351a132518baa0942431d132d164fd2ef01fe6c12bb75af7fd96b96a7d1c8"
    ),
    "validation": "e081af654348ee97d65d62756d668d367976150d44449c7fe3e598c7f8f67fb9",
}
CONTROL_MODELS = {
    "correct_structure": "runtime_spn_ct_k1aa_virtual_slot_histogram_true",
    "wrong_sbox": "runtime_spn_ct_k1aa_virtual_slot_histogram_wrong_sbox",
    "corrupted_linear": ("runtime_spn_ct_k1aa_virtual_slot_histogram_corrupted_linear"),
    "no_structure": "runtime_spn_ct_k1aa_virtual_slot_histogram_none",
}
MODEL_TO_CONDITION = {model: condition for condition, model in CONTROL_MODELS.items()}
CONTROL_CONDITIONS = tuple(CONTROL_MODELS)
EXPECTED_SEEDS = (6, 7)
EXPECTED_SPLITS = ("train_seen", "same_key_fresh", "cross_key_validation")
FRESH_SPLITS = ("same_key_fresh", "cross_key_validation")
INPUT_DIFFERENCE = 0x0000000400000000
DIFFERENCE_PROFILE = "midori64_k1ah_cell8_r4"
EXPECTED_TRAIN_SAMPLES_PER_CLASS = 2048
EXPECTED_HOLDOUT_SAMPLES_PER_CLASS = 1024
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_HOLDOUT_ROWS = 2048
EXPECTED_PAIRS = 4
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
EXPECTED_PARAMETER_COUNT = 214_316
EXPECTED_TRAINING_ROWS = len(EXPECTED_SEEDS) * len(CONTROL_CONDITIONS)
EXPECTED_EVALUATION_ROWS = (
    len(EXPECTED_SEEDS) * len(CONTROL_CONDITIONS) * len(EXPECTED_SPLITS)
)
RUNTIME_ROUND_START = 0
RUNTIME_ROUNDS = 2
CIPHER_ROUND_WINDOW_START = 2
TOPOLOGY_CORRUPTION_SEED = 20260729
AUC_FLOOR = 0.550
NO_STRUCTURE_MARGIN = 0.010
SEMANTIC_MARGIN = 0.005
REPLAY_TOLERANCE = 1e-6


def build_k1ai_control(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int = 512,
) -> torch.nn.Module:
    if condition not in CONTROL_MODELS:
        raise ValueError("unknown K1-AI condition")
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
            raise ValueError(f"duplicate K1-AI task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-AI task matrix is incomplete")
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
            and int(task.get("train_key", -1)) == CONFIRMATION_KEYS[seed][0]
            and int(task.get("validation_key", -1)) == CONFIRMATION_KEYS[seed][1]
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
            and int(options.get("histogram_value_dim", -1)) == 8
            and int(options.get("virtual_projection_slots", -1)) == 16
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
    models = {
        key: build_k1ai_control(task=task, condition=key[1])
        for key, task in mapped.items()
    }
    parameter_counts = {
        int(model_metadata(model)["trainable_parameter_count"])
        for model in models.values()
    }
    geometries = {
        tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())
        for model in models.values()
    }
    correct = models[(EXPECTED_SEEDS[0], "correct_structure")]
    wrong = models[(EXPECTED_SEEDS[0], "wrong_sbox")]
    corrupted = models[(EXPECTED_SEEDS[0], "corrupted_linear")]
    none = models[(EXPECTED_SEEDS[0], "no_structure")]
    correct_structure = correct.runtime_structure
    wrong_structure = wrong.runtime_structure
    corrupted_structure = corrupted.runtime_structure
    none_structure = none.runtime_structure
    identity = torch.eye(correct_structure.block_bits, dtype=torch.uint8).unsqueeze(0)
    identity = identity.repeat(correct_structure.rounds, 1, 1)
    shared_partition = all(
        torch.equal(
            model.runtime_structure.cell_membership, correct_structure.cell_membership
        )
        and torch.equal(model.runtime_structure.bit_role, correct_structure.bit_role)
        for model in (wrong, corrupted, none)
    )
    return {
        "all_models_parameter_count_exact": parameter_counts
        == {EXPECTED_PARAMETER_COUNT},
        "all_models_state_dict_geometry_identical": len(geometries) == 1,
        "runtime_cell_partition_identical": shared_partition,
        "wrong_sbox_changes_only_sbox": (
            not torch.equal(
                wrong_structure.sbox_truth_bits,
                correct_structure.sbox_truth_bits,
            )
            and torch.equal(
                wrong_structure.linear_matrices,
                correct_structure.linear_matrices,
            )
            and wrong.apply_sboxes is True
        ),
        "corrupted_linear_changes_only_linear": (
            torch.equal(
                corrupted_structure.sbox_truth_bits,
                correct_structure.sbox_truth_bits,
            )
            and not torch.equal(
                corrupted_structure.linear_matrices,
                correct_structure.linear_matrices,
            )
            and corrupted.apply_sboxes is True
        ),
        "no_structure_is_identity_without_sboxes": (
            torch.equal(none_structure.linear_matrices, identity)
            and none.apply_sboxes is False
        ),
        "correct_midori_window_homogeneous": (
            correct_structure.is_homogeneous
            and correct_structure.unique_transition_count == 1
            and torch.equal(
                correct_structure.linear_matrices,
                correct_structure.linear_matrices.flip(0),
            )
            and torch.equal(
                correct_structure.sbox_truth_bits,
                correct_structure.sbox_truth_bits.flip(0),
            )
        ),
        "reversed_control_unavailable": (
            "reversed_linear" not in CONTROL_CONDITIONS
            and not any("reversed" in model for model in CONTROL_MODELS.values())
        ),
        "all_semantic_fingerprints_distinct": len(
            {model.composition_sha256 for model in (correct, wrong, corrupted, none)}
        )
        == len(CONTROL_CONDITIONS),
    }


def source_binding_checks(
    *,
    gate: Mapping[str, Any],
    validation: Mapping[str, Any],
    source_digests: Mapping[str, str],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    expected_manifest_keys = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    observed_manifest_keys = {
        (int(row.get("seed", -1)), str(row.get("split"))) for row in manifest_rows
    }
    return {
        "k1ah_source_digests_exact": dict(source_digests) == EXPECTED_SOURCE_DIGESTS,
        "k1ah_gate_exact_pass": (
            gate.get("run_id") == K1AH_RUN_ID
            and gate.get("status") == "pass"
            and gate.get("decision") == K1AH_DECISION
            and 8 in gate.get("confirmed_cells", [])
            and 8 in gate.get("selection", {}).get("selected_cells", [])
            and bool(gate.get("protocol_checks"))
            and all(gate.get("protocol_checks", {}).values())
        ),
        "k1ah_validation_exact_pass": (
            validation.get("run_id") == K1AH_RUN_ID
            and validation.get("status") == "pass"
            and not validation.get("errors")
        ),
        "six_cell8_confirmation_caches_exact": (
            len(manifest_rows) == len(expected_manifest_keys)
            and observed_manifest_keys == expected_manifest_keys
            and all(
                row.get("run_id") == K1AH_RUN_ID
                and row.get("phase") == "confirmation"
                and int(row.get("cell", -1)) == 8
                and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
                and int(row.get("rounds", -1)) == 4
                and row.get("cache_payloads_present") is True
                and int(row.get("rows", -1))
                == (
                    EXPECTED_TRAIN_ROWS
                    if row.get("split") == "train_seen"
                    else EXPECTED_HOLDOUT_ROWS
                )
                for row in manifest_rows
            )
        ),
    }


def evaluate_k1ai_panel(
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
        raise ValueError("K1-AI requires six seed6/7 cell8 datasets")

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
            model = build_k1ai_control(
                task=task,
                condition=condition,
                input_bits=int(dataset.features.shape[1]),
            )
            model.load_state_dict(state, strict=True)
            if tensor_mapping_sha256(model.state_dict()) != state_sha:
                raise ValueError("K1-AI strict checkpoint load changed learned state")
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
                    "runtime_structure_mode": model.runtime_structure_mode,
                    "residual_gate": float(
                        torch.tanh(model.backbone.residual_gate.detach())
                    ),
                    "histogram_gate": float(
                        torch.tanh(model.backbone.histogram_gate.detach())
                    ),
                    "strict_state_dict_load": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )
    return rows


def adjudicate_k1ai(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    control_checks: Mapping[str, bool],
    cache_checks: Mapping[str, bool],
) -> dict[str, Any]:
    trained = training_map(training_rows, fail_closed=False)
    evaluated = evaluation_map(evaluation_rows)
    checkpoints = checkpoint_map(checkpoint_manifest, fail_closed=False)
    seed_results = {
        str(seed): {
            split: split_result(evaluated, seed, split) for split in EXPECTED_SPLITS
        }
        for seed in EXPECTED_SEEDS
    }
    protocol_checks = {
        **dict(source_checks),
        **dict(control_checks),
        **dict(cache_checks),
        "eight_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "eight_training_rows_complete": (
            len(training_rows) == EXPECTED_TRAINING_ROWS
            and set(trained) == expected_condition_keys()
        ),
        "training_protocol_frozen": training_protocol_frozen(training_rows),
        "eight_checkpoint_manifest_entries": (
            len(checkpoint_manifest.get("entries", [])) == EXPECTED_TRAINING_ROWS
            and set(checkpoints) == expected_condition_keys()
        ),
        "twenty_four_evaluation_rows_complete": (
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
        "semantic_fingerprints_distinct": all(
            len(
                {
                    evaluated[(seed, split, condition)].get("composition_sha256")
                    for condition in CONTROL_CONDITIONS
                }
            )
            == len(CONTROL_CONDITIONS)
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ),
        "cross_key_auc_replays_training_result": all(
            abs(
                float(evaluated[(seed, "cross_key_validation", condition)]["auc"])
                - float(trained[(seed, condition)]["metrics"]["auc"])
            )
            <= REPLAY_TOLERANCE
            for seed, condition in expected_condition_keys()
            if (seed, condition) in trained
            and (seed, "cross_key_validation", condition) in evaluated
        ),
        "finite_metrics": all(
            math.isfinite(float(row.get("auc", math.nan)))
            and 0.0 <= float(row.get("auc", math.nan)) <= 1.0
            for row in evaluation_rows
        ),
    }
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        for split in FRESH_SPLITS:
            result = seed_results[str(seed)][split]
            prefix = f"seed{seed}_{split}"
            research_checks[f"{prefix}_correct_auc_floor"] = (
                result["correct_auc"] >= AUC_FLOOR
            )
            research_checks[f"{prefix}_beats_no_structure"] = (
                result["correct_minus_no_structure"] >= NO_STRUCTURE_MARGIN
            )
            research_checks[f"{prefix}_beats_wrong_sbox"] = (
                result["correct_minus_wrong_sbox"] >= SEMANTIC_MARGIN
            )
            research_checks[f"{prefix}_beats_corrupted_linear"] = (
                result["correct_minus_corrupted_linear"] >= SEMANTIC_MARGIN
            )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    correct_signal = all(
        seed_results[str(seed)][split]["correct_auc"] >= AUC_FLOOR
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    same_key_pass = all(
        research_checks.get(f"seed{seed}_same_key_fresh_{suffix}", False)
        for seed in EXPECTED_SEEDS
        for suffix in (
            "correct_auc_floor",
            "beats_no_structure",
            "beats_wrong_sbox",
            "beats_corrupted_linear",
        )
    )
    cross_key_pass = all(
        research_checks.get(f"seed{seed}_cross_key_validation_{suffix}", False)
        for seed in EXPECTED_SEEDS
        for suffix in (
            "correct_auc_floor",
            "beats_no_structure",
            "beats_wrong_sbox",
            "beats_corrupted_linear",
        )
    )

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_midori64_k1ai_protocol_invalid"
        next_action = (
            "repair only the failed K1-AI source, control, cache, checkpoint, or "
            "metric binding and rerun the unchanged eight-row matrix"
        )
    elif research_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_midori64_k1ai_"
            "neural_structure_attribution_supported"
        )
        next_action = (
            "preregister a remote 65536/class Midori64 r4 medium diagnostic using "
            "the correct model and only the strongest necessary controls with "
            "disk-backed cache/progress/resume; do not call it formal evidence"
        )
    elif correct_signal:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1ai_"
            "signal_learned_structure_attribution_not_supported"
        )
        next_action = (
            "keep the K1-AH cell8 data route, reject attribution to the current "
            "K1-AA structure interaction, and audit the shortcut shared with the "
            "strongest failed control before another architecture or scale"
        )
    elif same_key_pass and not cross_key_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1ai_key_specific_neural_attribution"
        )
        next_action = (
            "hold scale and preregister one same-budget key-invariance change while "
            "keeping the Midori64 cell8 data and four controls frozen"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_midori64_k1ai_neural_signal_not_supported"
        next_action = (
            "localize the first stage where the confirmed K1-AH statistic is lost "
            "inside the K1-AA model before adding capacity, pairs, data, MoE, or "
            "another architecture family"
        )

    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
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
            "correct_auc_floor": AUC_FLOOR,
            "correct_minus_no_structure": NO_STRUCTURE_MARGIN,
            "correct_minus_semantic_control": SEMANTIC_MARGIN,
        },
        "remote_scale": "yes" if protocol_valid and research_pass else "no",
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class Midori64 r4 cell8 independently trained "
            "neural structure-attribution diagnostic; not formal scale, attack, "
            "SOTA, family transfer, arbitrary-SPN, or Midori64 ceiling evidence"
        ),
        "blocked_actions": [
            "more pairs, samples, seeds, epochs, width, positions, bit roles, or rounds inside K1-AI",
            "MoE, trail/DDT input, another cipher, or changed negative samples",
            "remote scale unless every seed and fresh-split K1-AI research gate passes",
        ],
    }


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_TRAINING_ROWS and all(
        row.get("model") in MODEL_TO_CONDITION
        and row.get("trainable_parameter_count") == EXPECTED_PARAMETER_COUNT
        and int(row.get("rounds", -1)) == 4
        and int(row.get("seed", -1)) in EXPECTED_SEEDS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("difference_profile") == DIFFERENCE_PROFILE
        and int(row.get("samples_per_class", -1)) == EXPECTED_TRAIN_SAMPLES_PER_CLASS
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and int(row.get("training", {}).get("batch_size", -1)) == EXPECTED_BATCH_SIZE
        and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
        and int(row.get("training", {}).get("epochs_ran", -1)) == EXPECTED_EPOCHS
        and row.get("training", {}).get("checkpoint_metric") == "val_auc"
        and row.get("training", {}).get("selected_checkpoint") == "best"
        and int(row.get("training", {}).get("samples_total", -1)) == EXPECTED_TRAIN_ROWS
        and int(row.get("validation", {}).get("samples_total", -1))
        == EXPECTED_HOLDOUT_ROWS
        for row in rows
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
            raise ValueError(f"duplicate K1-AI training row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-AI training panel is incomplete")
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
            raise ValueError(f"duplicate K1-AI checkpoint: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-AI checkpoint panel is incomplete")
    return mapped


def evaluation_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["split"]), str(row["condition"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-AI evaluation row: {key}")
        mapped[key] = row
    return mapped


def split_result(
    rows: Mapping[tuple[int, str, str], Mapping[str, Any]],
    seed: int,
    split: str,
) -> dict[str, float]:
    aucs = {
        condition: float(rows[(seed, split, condition)]["auc"])
        for condition in CONTROL_CONDITIONS
    }
    correct = aucs["correct_structure"]
    return {
        "correct_auc": correct,
        "wrong_sbox_auc": aucs["wrong_sbox"],
        "corrupted_linear_auc": aucs["corrupted_linear"],
        "no_structure_auc": aucs["no_structure"],
        "correct_minus_wrong_sbox": correct - aucs["wrong_sbox"],
        "correct_minus_corrupted_linear": correct - aucs["corrupted_linear"],
        "correct_minus_no_structure": correct - aucs["no_structure"],
        "residual_gate": float(
            rows[(seed, split, "correct_structure")]["residual_gate"]
        ),
        "histogram_gate": float(
            rows[(seed, split, "correct_structure")]["histogram_gate"]
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


__all__ = [
    "AUC_FLOOR",
    "CONTROL_CONDITIONS",
    "CONTROL_MODELS",
    "DIFFERENCE_PROFILE",
    "EXPECTED_BATCH_SIZE",
    "EXPECTED_EPOCHS",
    "EXPECTED_EVALUATION_ROWS",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_SEEDS",
    "EXPECTED_SOURCE_DIGESTS",
    "EXPECTED_SPLITS",
    "EXPECTED_TRAINING_ROWS",
    "FRESH_SPLITS",
    "INPUT_DIFFERENCE",
    "MODEL_TO_CONDITION",
    "NO_STRUCTURE_MARGIN",
    "RUN_ID",
    "SEMANTIC_MARGIN",
    "adjudicate_k1ai",
    "build_control_checks",
    "build_k1ai_control",
    "candidate_protocol_frozen",
    "evaluate_k1ai_panel",
    "expected_condition_keys",
    "expected_dataset_keys",
    "source_binding_checks",
]
