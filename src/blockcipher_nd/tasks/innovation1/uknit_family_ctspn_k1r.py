from __future__ import annotations

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
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    load_bound_state,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import (
    CONTROL_MODELS,
    EXPECTED_PARAMETER_COUNT,
    build_k1n_control,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1q import (
    CONFIRMATION_KEYS,
    RUN_ID as K1Q_RUN_ID,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_ctspn_cell11_neural_attribution_k1r_2048_seed3_seed4_20260728"
K1Q_DECISION = (
    "innovation1_uknit_family_ctspn_k1q_confirmed_r5_difference_position_supported"
)
EXPECTED_SOURCE_DIGESTS = {
    "gate": "1af79fa865736635d40f729fe6621e677a4378e64c6779fc449756ae48609f8b",
    "dataset_manifest": (
        "16d9549df5d1a6b2d88fd95e10ceec484e6f5443bd774f11d0f7d68dc85494f2"
    ),
    "validation": ("25b59f9b0eeab8eb894c4b3a40513437306a2c660f0c68f4ab478260689d8059"),
}
EXPECTED_SEEDS = (3, 4)
EXPECTED_SPLITS = ("train_seen", "same_key_fresh", "cross_key_validation")
FRESH_SPLITS = ("same_key_fresh", "cross_key_validation")
CONTROL_CONDITIONS = (
    "exact_composition",
    "wrong_sbox_semantics",
    "no_sbox_composition",
    "no_topology",
)
MODEL_TO_CONDITION = {
    CONTROL_MODELS[condition]: condition for condition in CONTROL_CONDITIONS
}
INPUT_DIFFERENCE = 0x0000400000000000
DIFFERENCE_PROFILE = "uknit64_k1q_cell11_r5"
EXPECTED_TRAIN_SAMPLES_PER_CLASS = 2048
EXPECTED_HOLDOUT_SAMPLES_PER_CLASS = 1024
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_HOLDOUT_ROWS = 2048
EXPECTED_PAIRS = 4
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
EXPECTED_TRAINING_ROWS = len(EXPECTED_SEEDS) * len(CONTROL_CONDITIONS)
EXPECTED_EVALUATION_ROWS = (
    len(EXPECTED_SEEDS) * len(CONTROL_CONDITIONS) * len(EXPECTED_SPLITS)
)
AUC_FLOOR = 0.550
NO_TOPOLOGY_MARGIN = 0.010
SEMANTIC_MARGIN = 0.005
REPLAY_TOLERANCE = 1e-12


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
            raise ValueError(f"duplicate K1-R task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-R task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == EXPECTED_TRAINING_ROWS
        and set(mapped) == expected_condition_keys()
        and all(
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == 5
            and int(task.get("seed", -1)) == seed
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
            and task.get("checkpoint_metric") == "val_auc"
            and task.get("restore_best_checkpoint") is True
            and int(task.get("target_epochs", -1)) == EXPECTED_EPOCHS
            and int(task.get("model_options", {}).get("runtime_round_start", -1)) == 3
            and int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
            and int(task.get("model_options", {}).get("active_cell", -1)) == 11
            and int(task.get("model_options", {}).get("active_bit_role", -1)) == 1
            and int(
                str(task.get("model_options", {}).get("input_difference_hex", "0")),
                0,
            )
            == INPUT_DIFFERENCE
            for (seed, _), task in mapped.items()
        )
    )


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
        "k1q_source_digests_exact": dict(source_digests) == EXPECTED_SOURCE_DIGESTS,
        "k1q_gate_exact_pass": (
            gate.get("run_id") == K1Q_RUN_ID
            and gate.get("status") == "pass"
            and gate.get("decision") == K1Q_DECISION
            and 11 in gate.get("confirmed_cells", [])
            and bool(gate.get("protocol_checks"))
            and all(gate.get("protocol_checks", {}).values())
        ),
        "k1q_validation_exact_pass": (
            validation.get("run_id") == K1Q_RUN_ID
            and validation.get("status") == "pass"
            and not validation.get("errors")
        ),
        "six_cell11_confirmation_caches_exact": (
            len(manifest_rows) == len(expected_manifest_keys)
            and observed_manifest_keys == expected_manifest_keys
            and all(
                row.get("run_id") == K1Q_RUN_ID
                and row.get("phase") == "confirmation"
                and int(row.get("cell", -1)) == 11
                and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
                and int(row.get("rounds", -1)) == 5
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


def evaluate_k1r_panel(
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
        raise ValueError("K1-R requires six seed3/4 cell11 datasets")

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
            model = build_k1n_control(
                task=task,
                condition=condition,
                input_bits=int(dataset.features.shape[1]),
            )
            model.load_state_dict(state, strict=True)
            if tensor_mapping_sha256(model.state_dict()) != state_sha:
                raise ValueError("K1-R strict checkpoint load changed learned state")
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
                    "cipher_key": "uknit64",
                    "rounds": 5,
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
                    "effective_gate": float(
                        torch.tanh(model.backbone.residual_gate.detach())
                    ),
                    "strict_state_dict_load": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )
    return rows


def adjudicate_k1r(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    source_checks: Mapping[str, bool],
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
            research_checks[f"{prefix}_exact_auc_floor"] = (
                result["exact_auc"] >= AUC_FLOOR
            )
            research_checks[f"{prefix}_beats_no_topology"] = (
                result["exact_minus_no_topology"] >= NO_TOPOLOGY_MARGIN
            )
            research_checks[f"{prefix}_beats_wrong_sbox"] = (
                result["exact_minus_wrong_sbox_semantics"] >= SEMANTIC_MARGIN
            )
            research_checks[f"{prefix}_beats_no_sbox"] = (
                result["exact_minus_no_sbox_composition"] >= SEMANTIC_MARGIN
            )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    exact_signal = all(
        seed_results[str(seed)][split]["exact_auc"] >= AUC_FLOOR
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    same_key_pass = all(
        research_checks.get(f"seed{seed}_same_key_fresh_{suffix}", False)
        for seed in EXPECTED_SEEDS
        for suffix in (
            "exact_auc_floor",
            "beats_no_topology",
            "beats_wrong_sbox",
            "beats_no_sbox",
        )
    )
    cross_key_pass = all(
        research_checks.get(f"seed{seed}_cross_key_validation_{suffix}", False)
        for seed in EXPECTED_SEEDS
        for suffix in (
            "exact_auc_floor",
            "beats_no_topology",
            "beats_wrong_sbox",
            "beats_no_sbox",
        )
    )

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1r_protocol_invalid"
        next_action = (
            "repair only the failed K1-R source, cache, checkpoint, or metric binding "
            "and rerun the unchanged eight-row matrix"
        )
    elif research_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1r_"
            "cell11_neural_structure_attribution_supported"
        )
        next_action = (
            "preregister K1-S as a remote 65536/class medium diagnostic using only "
            "exact composition and the strongest necessary controls, with disk-backed "
            "cache/progress/resume; do not call it formal or paper-scale evidence"
        )
    elif exact_signal:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1r_"
            "cell11_signal_learned_structure_attribution_not_supported"
        )
        next_action = (
            "keep the K1-Q cell11 data route, reject attribution to the current K1-N "
            "structure interaction, and preregister one local structure-interaction "
            "redesign against the strongest failed control before any scale"
        )
    elif same_key_pass and not cross_key_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1r_cell11_key_specific_neural_attribution"
        )
        next_action = (
            "hold scale and preregister one same-budget key-invariance change while "
            "keeping the cell11 data and control matrix frozen"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1r_cell11_neural_signal_not_supported"
        )
        next_action = (
            "treat access to the K1-Q five-stage statistic as the bottleneck and run "
            "one local representation-access audit before another architecture or scale"
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
            "exact_auc_floor": AUC_FLOOR,
            "exact_minus_no_topology": NO_TOPOLOGY_MARGIN,
            "exact_minus_semantic_control": SEMANTIC_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class uKNIT r5 cell11 independently trained neural "
            "structure-attribution diagnostic; not formal scale, attack, SOTA, family "
            "transfer, arbitrary-SPN, or uKNIT ceiling evidence"
        ),
        "blocked_actions": [
            "cell0, more positions or bit roles, pairs, samples, seeds, epochs, or width inside K1-R",
            "MoE, DDT/trail input, architecture switching, or changed negative samples",
            "remote scale unless every seed and fresh-split K1-R research gate passes",
        ],
    }


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_TRAINING_ROWS and all(
        row.get("model") in MODEL_TO_CONDITION
        and row.get("trainable_parameter_count") == EXPECTED_PARAMETER_COUNT
        and int(row.get("rounds", -1)) == 5
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
            raise ValueError(f"duplicate K1-R training row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-R training panel is incomplete")
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
            raise ValueError(f"duplicate K1-R checkpoint: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_condition_keys():
        raise ValueError("K1-R checkpoint panel is incomplete")
    return mapped


def evaluation_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["split"]), str(row["condition"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-R evaluation row: {key}")
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
    exact = aucs["exact_composition"]
    return {
        "exact_auc": exact,
        "wrong_sbox_semantics_auc": aucs["wrong_sbox_semantics"],
        "no_sbox_composition_auc": aucs["no_sbox_composition"],
        "no_topology_auc": aucs["no_topology"],
        "exact_minus_wrong_sbox_semantics": (exact - aucs["wrong_sbox_semantics"]),
        "exact_minus_no_sbox_composition": (exact - aucs["no_sbox_composition"]),
        "exact_minus_no_topology": exact - aucs["no_topology"],
        "effective_gate": float(
            rows[(seed, split, "exact_composition")]["effective_gate"]
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
    "NO_TOPOLOGY_MARGIN",
    "RUN_ID",
    "SEMANTIC_MARGIN",
    "adjudicate_k1r",
    "candidate_protocol_frozen",
    "evaluate_k1r_panel",
    "source_binding_checks",
]
