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
    EXPECTED_SOURCE_DIGESTS as K1AH_SOURCE_DIGESTS,
    EXPECTED_SPLITS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_TRAIN_SAMPLES_PER_CLASS,
    FRESH_SPLITS,
    INPUT_DIFFERENCE,
    NO_STRUCTURE_MARGIN,
    RUNTIME_ROUNDS,
    RUNTIME_ROUND_START,
    SEMANTIC_MARGIN,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_midori64_sbox_transition_k1ak_2048_seed6_seed7_20260729"
K1AI_RUN_ID = (
    "i1_uknit_family_midori64_neural_attribution_k1ai_2048_seed6_seed7_20260729"
)
K1AJ_RUN_ID = "i1_uknit_family_midori64_same_checkpoint_k1aj_replay_fix_20260729"
K1AI_DECISION = (
    "innovation1_uknit_family_midori64_k1ai_"
    "signal_learned_structure_attribution_not_supported"
)
K1AJ_DECISION = (
    "innovation1_uknit_family_midori64_k1aj_diffusion_causal_sbox_discrimination_failed"
)
EXPECTED_SOURCE_DIGESTS = {
    "k1ai_gate": "5f7eca268a26a9f3d3fdf746a0e9beae4552b156c1a832a7f81f02457d32803d",
    "k1ai_validation": "a901d807da281762acbba30d960fc787dedd5df0981ed77499d09cf0589e370e",
    "k1ai_controls": "f2e6a9ba34821f3acd1ccc787befb465ceca4e9f9f90ca58bbc62ca5d87092de",
    "k1ai_dataset_manifest": "5525a28f099a21bcca09aafbe05498f0f7951e22e171eaac6db055c174ff35bc",
    "k1aj_gate": "03c04e0fafc71d3cf947b0d4855d533f17b5b2dec75e697885599c0312f85f6c",
    "k1aj_validation": "ed673e80ab397f5dadba7bf842f4894ba550406e66ffa057c32f5b422f05db9e",
}
CONTROL_MODELS = {
    "correct_structure": "runtime_spn_ct_k1ak_sbox_transition_true",
    "wrong_sbox": "runtime_spn_ct_k1ak_sbox_transition_wrong_sbox",
    "corrupted_linear": "runtime_spn_ct_k1ak_sbox_transition_corrupted_linear",
    "no_structure": "runtime_spn_ct_k1ak_sbox_transition_none",
}
MODEL_TO_CONDITION = {model: condition for condition, model in CONTROL_MODELS.items()}
CONTROL_CONDITIONS = tuple(CONTROL_MODELS)
EXPECTED_PARAMETER_COUNT = 219_320
ANCHOR_PARAMETER_COUNT = 214_316
PARAMETER_RATIO_LIMIT = 1.025
EXPECTED_TRAINING_ROWS = len(EXPECTED_SEEDS) * len(CONTROL_CONDITIONS)
EXPECTED_EVALUATION_ROWS = (
    len(EXPECTED_SEEDS) * len(EXPECTED_SPLITS) * len(CONTROL_CONDITIONS)
)
ANCHOR_RETENTION_MARGIN = -0.010
TOPOLOGY_CORRUPTION_SEED = 20260729


def build_k1ak_control(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int = 512,
) -> torch.nn.Module:
    if condition not in CONTROL_MODELS:
        raise ValueError("unknown K1-AK condition")
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
            raise ValueError(f"duplicate K1-AK task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-AK task matrix is incomplete")
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
            and int(options.get("transition_value_dim", -1)) == 20
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
        key: build_k1ak_control(task=task, condition=key[1])
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
    identity = torch.eye(correct.runtime_structure.block_bits, dtype=torch.uint8)
    identity = identity.unsqueeze(0).repeat(correct.runtime_structure.rounds, 1, 1)
    return {
        "all_models_parameter_count_exact": parameter_counts
        == {EXPECTED_PARAMETER_COUNT},
        "parameter_budget_within_1p025_anchor": (
            max(parameter_counts, default=math.inf)
            <= ANCHOR_PARAMETER_COUNT * PARAMETER_RATIO_LIMIT
        ),
        "all_models_state_dict_geometry_identical": len(geometries) == 1,
        "runtime_cell_partition_identical": all(
            torch.equal(
                model.runtime_structure.cell_membership,
                correct.runtime_structure.cell_membership,
            )
            and torch.equal(
                model.runtime_structure.bit_role,
                correct.runtime_structure.bit_role,
            )
            for model in (wrong, corrupted, none)
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
        "corrupted_linear_changes_only_linear": (
            torch.equal(
                corrupted.runtime_structure.sbox_truth_bits,
                correct.runtime_structure.sbox_truth_bits,
            )
            and not torch.equal(
                corrupted.runtime_structure.linear_matrices,
                correct.runtime_structure.linear_matrices,
            )
            and corrupted.apply_sboxes is True
        ),
        "no_structure_is_identity_without_sboxes": (
            torch.equal(none.runtime_structure.linear_matrices, identity)
            and none.apply_sboxes is False
        ),
        "no_cipher_or_absolute_cell_identity": all(
            model.uses_cipher_identity is False
            and model.uses_absolute_cell_or_bit_identity is False
            and model.uses_runtime_native_cell_slots is False
            for model in models.values()
        ),
        "all_transition_fingerprints_distinct": len(
            {
                model.sbox_transition_semantics_sha256
                for model in (correct, wrong, corrupted, none)
            }
        )
        == len(CONTROL_CONDITIONS),
    }


def source_binding_checks(
    *,
    k1ai_gate: Mapping[str, Any],
    k1ai_validation: Mapping[str, Any],
    k1ai_controls: Sequence[Mapping[str, Any]],
    dataset_manifest: Sequence[Mapping[str, Any]],
    k1aj_gate: Mapping[str, Any],
    k1aj_validation: Mapping[str, Any],
    source_digests: Mapping[str, str],
) -> dict[str, bool]:
    expected_dataset_keys = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    correct_anchor_keys = {
        (int(row.get("seed", -1)), str(row.get("split")))
        for row in k1ai_controls
        if row.get("condition") == "correct_structure"
    }
    return {
        "source_artifact_digests_exact": dict(source_digests)
        == EXPECTED_SOURCE_DIGESTS,
        "k1ai_exact_hold": (
            k1ai_gate.get("run_id") == K1AI_RUN_ID
            and k1ai_gate.get("status") == "hold"
            and k1ai_gate.get("decision") == K1AI_DECISION
            and not k1ai_gate.get("failed_protocol_checks")
            and k1ai_gate.get("remote_scale") == "no"
        ),
        "k1ai_validation_exact_pass": (
            k1ai_validation.get("run_id") == K1AI_RUN_ID
            and k1ai_validation.get("status") == "pass"
            and not k1ai_validation.get("errors")
        ),
        "six_k1ai_correct_anchor_rows": (
            correct_anchor_keys == expected_dataset_keys
            and sum(
                row.get("condition") == "correct_structure" for row in k1ai_controls
            )
            == 6
        ),
        "six_k1ah_dataset_rows_exact": (
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
        "k1aj_exact_hold": (
            k1aj_gate.get("run_id") == K1AJ_RUN_ID
            and k1aj_gate.get("status") == "hold"
            and k1aj_gate.get("decision") == K1AJ_DECISION
            and not k1aj_gate.get("failed_protocol_checks")
            and k1aj_gate.get("remote_scale") == "no"
        ),
        "k1aj_validation_exact_pass": (
            k1aj_validation.get("run_id") == K1AJ_RUN_ID
            and k1aj_validation.get("status") == "pass"
            and not k1aj_validation.get("errors")
            and int(k1aj_validation.get("optimizer_steps", -1)) == 0
        ),
        "k1ah_source_chain_bound": bool(K1AH_SOURCE_DIGESTS),
    }


def evaluate_k1ak_panel(
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
        raise ValueError("K1-AK requires six seed6/7 cell8 datasets")

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
            model = build_k1ak_control(
                task=task,
                condition=condition,
                input_bits=int(dataset.features.shape[1]),
            )
            model.load_state_dict(state, strict=True)
            if tensor_mapping_sha256(model.state_dict()) != state_sha:
                raise ValueError("K1-AK strict checkpoint load changed learned state")
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


def adjudicate_k1ak(
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
        "transition_fingerprints_distinct": all(
            len(
                {
                    evaluated[(seed, split, condition)].get(
                        "sbox_transition_semantics_sha256"
                    )
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
            <= 1e-6
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
            research_checks[f"{prefix}_retains_k1ai_anchor"] = (
                result["correct_minus_anchor"] >= ANCHOR_RETENTION_MARGIN
            )
            research_checks[f"{prefix}_beats_wrong_sbox"] = (
                result["correct_minus_wrong_sbox"] >= SEMANTIC_MARGIN
            )
            research_checks[f"{prefix}_beats_corrupted_linear"] = (
                result["correct_minus_corrupted_linear"] >= SEMANTIC_MARGIN
            )
            research_checks[f"{prefix}_beats_no_structure"] = (
                result["correct_minus_no_structure"] >= NO_STRUCTURE_MARGIN
            )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    sbox_pass = all(
        research_checks.get(f"seed{seed}_{split}_beats_wrong_sbox", False)
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    anchor_pass = all(
        research_checks.get(f"seed{seed}_{split}_retains_k1ai_anchor", False)
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    diffusion_pass = all(
        research_checks.get(f"seed{seed}_{split}_beats_corrupted_linear", False)
        and research_checks.get(f"seed{seed}_{split}_beats_no_structure", False)
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    signal_pass = all(
        research_checks.get(f"seed{seed}_{split}_correct_auc_floor", False)
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_midori64_k1ak_protocol_invalid"
        next_action = "repair only the failed K1-AK binding and rerun unchanged"
    elif research_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_midori64_k1ak_sbox_transition_residual_supported"
        )
        next_action = (
            "retain the transition readout and run one same-protocol "
            "uKNIT-BC/Dialga family-transfer attribution panel before scale"
        )
    elif signal_pass and diffusion_pass and not sbox_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1ak_"
            "sbox_transition_discrimination_failed"
        )
        next_action = (
            "discard the transition readout and run a zero-training transition-"
            "branch tap audit before another architecture"
        )
    elif sbox_pass and not anchor_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1ak_"
            "sbox_supported_anchor_retention_failed"
        )
        next_action = (
            "hold the branch and inspect fusion/gate optimization at the same "
            "capacity before any scale"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1ak_"
            "signal_or_diffusion_retention_failed"
        )
        next_action = (
            "discard the transition readout and identify the first failed "
            "signal/diffusion tap without changing data or scale"
        )

    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "thresholds": {
            "correct_auc": AUC_FLOOR,
            "correct_minus_k1ai_anchor": ANCHOR_RETENTION_MARGIN,
            "correct_minus_wrong_sbox": SEMANTIC_MARGIN,
            "correct_minus_corrupted_linear": SEMANTIC_MARGIN,
            "correct_minus_no_structure": NO_STRUCTURE_MARGIN,
            "parameter_ratio_limit": PARAMETER_RATIO_LIMIT,
        },
        "remote_scale": "no",
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class Midori64 r4 cell8 fixed-budget S-box "
            "transition-readout diagnostic; not formal scale, attack, SOTA, "
            "family transfer, arbitrary-SPN, or ceiling evidence"
        ),
        "blocked_actions": [
            "remote scale from K1-AK",
            "more pairs, samples, epochs, seeds, positions, rounds, width, or MoE",
            "DDT/trail inputs or averaging failed seeds/splits",
        ],
    }


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_TRAINING_ROWS and all(
        row.get("model") in MODEL_TO_CONDITION
        and int(row.get("trainable_parameter_count", row.get("parameter_count", -1)))
        == EXPECTED_PARAMETER_COUNT
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
            raise ValueError(f"duplicate K1-AK training row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-AK training panel is incomplete")
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
            raise ValueError(f"duplicate K1-AK checkpoint: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-AK checkpoint panel is incomplete")
    return mapped


def evaluation_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["split"]), str(row["condition"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-AK evaluation row: {key}")
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
        raise ValueError("K1-AK requires six K1-AI correct anchor rows")
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
        "k1ai_anchor_auc": anchor,
        "wrong_sbox_auc": aucs["wrong_sbox"],
        "corrupted_linear_auc": aucs["corrupted_linear"],
        "no_structure_auc": aucs["no_structure"],
        "correct_minus_anchor": correct - anchor,
        "correct_minus_wrong_sbox": correct - aucs["wrong_sbox"],
        "correct_minus_corrupted_linear": correct - aucs["corrupted_linear"],
        "correct_minus_no_structure": correct - aucs["no_structure"],
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
    "ANCHOR_RETENTION_MARGIN",
    "CONTROL_CONDITIONS",
    "CONTROL_MODELS",
    "EXPECTED_EVALUATION_ROWS",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_SOURCE_DIGESTS",
    "EXPECTED_TRAINING_ROWS",
    "MODEL_TO_CONDITION",
    "RUN_ID",
    "adjudicate_k1ak",
    "build_control_checks",
    "build_k1ak_control",
    "candidate_protocol_frozen",
    "evaluate_k1ak_panel",
    "expected_condition_keys",
    "expected_dataset_keys",
    "source_binding_checks",
]
