from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1_readiness import (
    ANCHOR_MODEL,
    CANDIDATE_MODEL,
    CORRUPTED_MODEL,
    INDEPENDENT_MODEL,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_ctspn_linear_schedule_k1_2048_seed0_seed1_20260727"
EXPECTED_CIPHERS = ("uknit64", "dialga128")
EXPECTED_SEEDS = (0, 1)
SOURCE_ROLES = ("anchor", "candidate")
CONTROL_CONDITIONS = (
    "correct_ordered",
    "repeat_last",
    "rotated",
    "corrupted",
    "no_topology",
)
EXPECTED_TRAINING_ROWS = 8
EXPECTED_CONTROL_ROWS = 40
EXPECTED_BATCH_SIZE = 64
EXPECTED_EPOCHS = 10
EXPECTED_TRAIN_SAMPLES_PER_CLASS = 2048
EXPECTED_VALIDATION_SAMPLES_PER_CLASS = 1024
EXPECTED_PAIRS_PER_SAMPLE = 4
EXPECTED_PARAMETER_COUNTS = {
    "anchor": 442466,
    "candidate": 438702,
}
AUC_FLOORS = {
    "uknit64": 0.520,
    "dialga128": 0.550,
}
MARGIN_FLOOR = 0.005
SOURCE_AUC_REPLAY_TOLERANCE = 1e-12
PROBABILITY_DELTA_FLOOR = 0.0


def evaluate_frozen_control_panel(
    *,
    task_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    validation_datasets: Mapping[tuple[str, int], DifferentialDataset],
    batch_size: int = EXPECTED_BATCH_SIZE,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Evaluate both trained roles under five controls without optimizer steps."""
    if batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError(f"K1 batch_size must be {EXPECTED_BATCH_SIZE}")
    tasks = _task_map(task_rows)
    sources = _training_row_map(training_rows)
    if set(tasks) != _expected_source_keys() or set(sources) != _expected_source_keys():
        raise ValueError("K1 requires exactly two trained roles for both ciphers and seeds")
    if set(validation_datasets) != {
        (cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS
    }:
        raise ValueError("K1 validation datasets must cover both ciphers and seeds")

    rows: list[dict[str, Any]] = []
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            dataset = validation_datasets[(cipher, seed)]
            dataset_sha256 = differential_dataset_sha256(dataset)
            _validate_validation_dataset(dataset, cipher)
            for source_role in SOURCE_ROLES:
                key = (cipher, seed, source_role)
                task = tasks[key]
                source = sources[key]
                checkpoint_path = _checkpoint_path(source)
                payload = _load_checkpoint(checkpoint_path)
                checkpoint_sha256 = file_sha256(checkpoint_path)
                state_dict_sha256 = tensor_mapping_sha256(payload["state_dict"])
                _validate_checkpoint_source(payload, source, source_role)
                probabilities: dict[str, np.ndarray] = {}
                metadata: dict[str, dict[str, Any]] = {}
                for condition in CONTROL_CONDITIONS:
                    model = _build_control_model(
                        task=task,
                        source_role=source_role,
                        condition=condition,
                        input_bits=int(dataset.features.shape[1]),
                    )
                    model.load_state_dict(payload["state_dict"], strict=True)
                    if tensor_mapping_sha256(model.state_dict()) != state_dict_sha256:
                        raise ValueError("K1 control strict load changed learned state")
                    probabilities[condition] = predict_binary_probabilities(
                        model,
                        dataset,
                        batch_size=batch_size,
                        device=device,
                    )
                    metadata[condition] = _control_metadata(model, condition)

                labels = np.asarray(dataset.labels, dtype=np.float32)
                aucs = {
                    condition: binary_auc(labels, probabilities[condition])
                    for condition in CONTROL_CONDITIONS
                }
                reference = probabilities["correct_ordered"]
                source_auc = float(source["metrics"]["auc"])
                for condition in CONTROL_CONDITIONS:
                    current = probabilities[condition]
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "cipher": source["cipher"],
                            "cipher_key": cipher,
                            "rounds": int(source["rounds"]),
                            "seed": seed,
                            "source_role": source_role,
                            "condition": condition,
                            "model": str(source["model"]),
                            "auc": aucs[condition],
                            "source_auc": source_auc,
                            "correct_minus_source_auc": (
                                aucs["correct_ordered"] - source_auc
                            ),
                            "correct_minus_condition_auc": (
                                0.0
                                if condition == "correct_ordered"
                                else aucs["correct_ordered"] - aucs[condition]
                            ),
                            "max_abs_probability_delta_from_correct": float(
                                np.max(np.abs(reference - current))
                            ),
                            "mean_abs_probability_delta_from_correct": float(
                                np.mean(np.abs(reference - current))
                            ),
                            "mean_probability": float(current.mean()),
                            "probability_sha256": hashlib.sha256(
                                current.tobytes()
                            ).hexdigest(),
                            "dataset_sha256": dataset_sha256,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_sha256": checkpoint_sha256,
                            "state_dict_sha256": state_dict_sha256,
                            "checkpoint_selected": payload["metadata"].get(
                                "selected_checkpoint"
                            ),
                            "checkpoint_metric": payload["metadata"].get(
                                "checkpoint_metric"
                            ),
                            "samples_total": int(len(dataset.labels)),
                            "samples_per_class": int(
                                dataset.metadata["samples_per_class"]
                            ),
                            "pairs_per_sample": int(
                                dataset.metadata["pairs_per_sample"]
                            ),
                            "negative_mode": dataset.metadata["negative_mode"],
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                            **metadata[condition],
                        }
                    )
    return rows


def adjudicate_ctspn_k1(
    *,
    run_id: str,
    task_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    control_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    controls = list(control_rows)
    tasks_valid = _task_panel_valid(task_rows)
    training = _training_row_map(training_rows, fail_closed=False)
    grouped = _control_groups(controls)
    expected_keys = _expected_source_keys()
    complete = all(
        len(grouped[key].get(condition, ())) == 1
        for key in expected_keys
        for condition in CONTROL_CONDITIONS
    )
    seed_results = {
        cipher: {
            str(seed): _seed_result(training, grouped, cipher, seed)
            for seed in EXPECTED_SEEDS
        }
        for cipher in EXPECTED_CIPHERS
    }
    protocol_checks = {
        "frozen_eight_row_plan": tasks_valid,
        "eight_training_rows_complete": (
            len(training_rows) == EXPECTED_TRAINING_ROWS
            and set(training) == expected_keys
        ),
        "forty_frozen_control_rows_complete": (
            len(controls) == EXPECTED_CONTROL_ROWS
            and set(grouped) == expected_keys
            and complete
            and all(
                set(grouped[key]) == set(CONTROL_CONDITIONS)
                for key in expected_keys
            )
        ),
        "training_protocol_frozen": _training_protocol_valid(training_rows),
        "training_rows_match_frozen_plan": _training_rows_match_tasks(
            task_rows, training_rows
        ),
        "selected_best_auc_checkpoints": all(
            row.get("checkpoint_selected") == "best"
            and row.get("checkpoint_metric") == "val_auc"
            for row in controls
        ),
        "same_checkpoint_per_source_panel": complete
        and _same_field_per_source(grouped, "checkpoint_sha256"),
        "same_state_dict_per_source_panel": complete
        and _same_field_per_source(grouped, "state_dict_sha256"),
        "same_validation_rows_per_cipher_seed": complete
        and _same_dataset_per_cipher_seed(grouped),
        "strict_state_dict_load": all(
            row.get("strict_state_dict_load") is True for row in controls
        ),
        "no_control_training_or_optimizer_steps": all(
            row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in controls
        ),
        "correct_auc_replays_training_row": complete
        and all(
            abs(
                float(grouped[key]["correct_ordered"][0].get(
                    "correct_minus_source_auc", math.inf
                ))
            )
            <= SOURCE_AUC_REPLAY_TOLERANCE
            for key in expected_keys
        ),
        "control_contract_exact": complete
        and all(_control_contract(grouped[key]) for key in expected_keys),
        "finite_metrics": all(_control_row_finite(row) for row in controls),
    }
    research_checks: dict[str, bool] = {}
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            result = seed_results[cipher][str(seed)]
            prefix = f"{cipher}_seed{seed}"
            research_checks[f"{prefix}_candidate_auc_floor"] = bool(
                result["candidate_auc"] is not None
                and result["candidate_auc"] >= AUC_FLOORS[cipher]
            )
            research_checks[f"{prefix}_candidate_beats_anchor"] = bool(
                result["candidate_minus_anchor"] is not None
                and result["candidate_minus_anchor"] >= MARGIN_FLOOR
            )
            for condition in CONTROL_CONDITIONS[1:]:
                research_checks[f"{prefix}_candidate_beats_{condition}"] = bool(
                    result[f"candidate_minus_{condition}"] is not None
                    and result[f"candidate_minus_{condition}"] >= MARGIN_FLOOR
                )
                research_checks[f"{prefix}_{condition}_changes_probabilities"] = bool(
                    result[f"candidate_{condition}_probability_delta"] is not None
                    and result[f"candidate_{condition}_probability_delta"]
                    > PROBABILITY_DELTA_FLOOR
                )

    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_uknit_family_ctspn_k1_protocol_invalid"
        next_action = (
            "Repair only the failed K1 execution, checkpoint, shared-validation, or "
            "control binding and rerun the frozen diagnostic unchanged."
        )
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1_linear_schedule_supported"
        next_action = (
            "Retain CT-SPN canonical linear-schedule fusion and preregister K2 as "
            "one exact MANTIS S-box composition hypothesis at the same local budget."
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1_linear_schedule_not_supported"
        next_action = _hold_next_action(seed_results, research_checks)

    return {
        "run_id": run_id,
        "task": "innovation1_uknit_family_ctspn_k1_linear_schedule",
        "status": status,
        "decision": decision,
        "thresholds": {
            "auc_floors": AUC_FLOORS,
            "candidate_margin": MARGIN_FLOOR,
            "source_auc_replay_tolerance": SOURCE_AUC_REPLAY_TOLERANCE,
            "probability_delta_strictly_greater_than": PROBABILITY_DELTA_FLOOR,
        },
        "seed_results": seed_results,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": sorted(
            key for key, passed in protocol_checks.items() if not passed
        ),
        "failed_research_checks": sorted(
            key for key, passed in research_checks.items() if not passed
        ),
        "claim_scope": (
            "uKNIT-BC prefix-r5 and Dialga-128 prefix-r4 two-seed, 2048/class "
            "local CT-SPN diagnostic with frozen-checkpoint controls; not formal "
            "scale, attack, SOTA, arbitrary-SPN, or MSX evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "remote scale-up or mechanical sample increase from K1",
            "starting K2 unless every per-cipher per-seed K1 gate passes",
            "adding learned MoE, DDT, trail, partial-decryption, or guessed-key features",
            "using a macro average to hide a failed cipher or seed",
            "including generalized-Feistel MSX in the CT-SPN claim",
        ],
    }


def differential_dataset_sha256(dataset: DifferentialDataset) -> str:
    digest = hashlib.sha256()
    for name, values in (("features", dataset.features), ("labels", dataset.labels)):
        array = np.asarray(values)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _build_control_model(
    *,
    task: Mapping[str, Any],
    source_role: str,
    condition: str,
    input_bits: int,
) -> torch.nn.Module:
    if source_role not in SOURCE_ROLES or condition not in CONTROL_CONDITIONS:
        raise ValueError("unknown K1 source role or control condition")
    options = deepcopy(dict(task["model_options"]))
    options["runtime_structure_window_control"] = "full"
    options["topology_corruption_seed"] = 20260727
    if source_role == "candidate":
        options["canonical_schedule_control"] = "ordered"
        model_key = CANDIDATE_MODEL
        if condition == "repeat_last":
            options["runtime_structure_window_control"] = "repeat_last"
        elif condition == "rotated":
            options["canonical_schedule_control"] = "rotated"
        elif condition == "corrupted":
            model_key = CORRUPTED_MODEL
        elif condition == "no_topology":
            model_key = INDEPENDENT_MODEL
    else:
        model_key = ANCHOR_MODEL
        if condition == "repeat_last":
            options["runtime_structure_window_control"] = "repeat_last"
        elif condition == "rotated":
            options["runtime_structure_window_control"] = "rotated"
        elif condition == "corrupted":
            model_key = "runtime_spn_e4_equivariant_corrupted"
        elif condition == "no_topology":
            model_key = "runtime_spn_e4_equivariant_independent"
    block_bits = 64 if task["cipher_key"] == "uknit64" else 128
    return build_model(
        model_key,
        input_bits=input_bits,
        hidden_bits=64,
        pair_bits=2 * block_bits,
        structure="SPN",
        model_options=options,
    )


def _control_metadata(model: torch.nn.Module, condition: str) -> dict[str, Any]:
    metadata = model_metadata(model)
    payload = {
        "relation_mode": getattr(model, "relation_mode", None),
        "runtime_structure_mode": getattr(model, "runtime_structure_mode", None),
        "runtime_structure_window_control": getattr(
            model, "runtime_structure_window_control", None
        ),
        "runtime_structure_window_sha256": getattr(
            model, "runtime_structure_window_sha256", None
        ),
        "canonical_schedule_control": getattr(
            model, "canonical_schedule_control", None
        ),
        "canonical_factor_manifest_sha256": getattr(
            model, "canonical_factor_manifest_sha256", None
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **metadata,
        "relation_mode": payload["relation_mode"],
        "control_fingerprint_sha256": fingerprint,
    }


def _task_map(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for task in tasks:
        role = _source_role(str(task.get("model_key", "")))
        key = (str(task.get("cipher_key", "")), int(task.get("seed", -1)), role)
        if key in result:
            raise ValueError(f"duplicate K1 task: {key}")
        result[key] = task
    return result


def _training_row_map(
    rows: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        try:
            role = _source_role(str(row.get("model", "")))
            key = (str(row.get("cipher_key", "")), int(row.get("seed", -1)), role)
            if key in result:
                raise ValueError(f"duplicate K1 training row: {key}")
            result[key] = row
        except ValueError:
            if fail_closed:
                raise
    return result


def _source_role(model_key: str) -> str:
    if model_key == ANCHOR_MODEL:
        return "anchor"
    if model_key == CANDIDATE_MODEL:
        return "candidate"
    raise ValueError(f"unexpected K1 source model: {model_key}")


def _expected_source_keys() -> set[tuple[str, int, str]]:
    return {
        (cipher, seed, role)
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
        for role in SOURCE_ROLES
    }


def _checkpoint_path(source: Mapping[str, Any]) -> Path:
    path = Path(str(source.get("training", {}).get("checkpoint_output", "")))
    if not path.is_file():
        raise ValueError(f"missing K1 checkpoint: {path}")
    return path


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("K1 checkpoint payload must be a mapping")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("K1 checkpoint must contain state_dict")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("K1 checkpoint must contain metadata")
    return payload


def _validate_checkpoint_source(
    payload: Mapping[str, Any], source: Mapping[str, Any], source_role: str
) -> None:
    metadata = payload["metadata"]
    training = source.get("training", {})
    expected_parameters = EXPECTED_PARAMETER_COUNTS[source_role]
    if (
        metadata.get("selected_checkpoint") != "best"
        or metadata.get("checkpoint_metric") != "val_auc"
        or training.get("selected_checkpoint") != "best"
        or training.get("checkpoint_metric") != "val_auc"
        or int(source.get("trainable_parameter_count", -1)) != expected_parameters
    ):
        raise ValueError("K1 checkpoint does not match selected best-AUC source row")


def _validate_validation_dataset(dataset: DifferentialDataset, cipher: str) -> None:
    expected_input_bits = 512 if cipher == "uknit64" else 1024
    metadata = dataset.metadata
    if (
        len(dataset.labels) != 2 * EXPECTED_VALIDATION_SAMPLES_PER_CLASS
        or int(dataset.features.shape[1]) != expected_input_bits
        or metadata.get("samples_per_class")
        != EXPECTED_VALIDATION_SAMPLES_PER_CLASS
        or metadata.get("pairs_per_sample") != EXPECTED_PAIRS_PER_SAMPLE
        or metadata.get("negative_mode") != "encrypted_random_plaintexts"
        or metadata.get("sample_structure") != "independent_pairs"
    ):
        raise ValueError("K1 validation dataset does not match the frozen protocol")


def _task_panel_valid(tasks: Sequence[Mapping[str, Any]]) -> bool:
    try:
        mapped = _task_map(tasks)
    except (TypeError, ValueError):
        return False
    if len(tasks) != EXPECTED_TRAINING_ROWS or set(mapped) != _expected_source_keys():
        return False
    cipher_contracts = {
        "uknit64": {
            "rounds": 5,
            "runtime_round_start": 3,
            "train_key": 0,
            "validation_key": int("1" * 32, 16),
        },
        "dialga128": {
            "rounds": 4,
            "runtime_round_start": 2,
            "train_key": 0,
            "validation_key": int("1" * 64, 16),
        },
    }
    for (cipher, _seed, role), task in mapped.items():
        contract = cipher_contracts[cipher]
        options = task.get("model_options", {})
        if (
            task.get("rounds") != contract["rounds"]
            or task.get("samples_per_class") != EXPECTED_TRAIN_SAMPLES_PER_CLASS
            or task.get("pairs_per_sample") != EXPECTED_PAIRS_PER_SAMPLE
            or task.get("input_difference") != 0x40
            or task.get("negative_mode") != "encrypted_random_plaintexts"
            or task.get("sample_structure") != "independent_pairs"
            or task.get("train_key") != contract["train_key"]
            or task.get("validation_key") != contract["validation_key"]
            or task.get("loss") != "mse"
            or task.get("optimizer") != "adam"
            or task.get("learning_rate") != 0.0001
            or task.get("weight_decay") != 0.00001
            or task.get("checkpoint_metric") != "val_auc"
            or task.get("restore_best_checkpoint") is not True
            or task.get("target_epochs") != EXPECTED_EPOCHS
            or options.get("runtime_round_start")
            != contract["runtime_round_start"]
            or options.get("runtime_rounds") != 2
            or options.get("processor_steps") != 2
            or options.get("pair_embedding_dim") != 128
            or options.get("runtime_structure_window_control") != "full"
        ):
            return False
        if role == "anchor" and (
            options.get("round_window_mode") != "recurrent_window"
            or options.get("sbox_context_scale") != 0.0
            or options.get("cell_input_mode") != "state_triplet"
        ):
            return False
        if role == "candidate" and (
            options.get("canonical_schedule_control") != "ordered"
            or options.get("temporal_hidden_dim") != 76
        ):
            return False
    return True


def _training_rows_match_tasks(
    tasks: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> bool:
    try:
        task_map = _task_map(tasks)
        row_map = _training_row_map(rows)
    except (TypeError, ValueError):
        return False
    if set(task_map) != set(row_map):
        return False
    for key, task in task_map.items():
        row = row_map[key]
        training = row.get("training", {})
        if (
            row.get("rounds") != task.get("rounds")
            or row.get("samples_per_class") != task.get("samples_per_class")
            or row.get("pairs_per_sample") != task.get("pairs_per_sample")
            or row.get("input_difference") != task.get("input_difference")
            or row.get("negative_mode") != task.get("negative_mode")
            or row.get("sample_structure") != task.get("sample_structure")
            or row.get("train_key") != task.get("train_key")
            or row.get("validation_key") != task.get("validation_key")
            or training.get("model_options") != task.get("model_options")
        ):
            return False
    return True


def _training_protocol_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows) != EXPECTED_TRAINING_ROWS:
        return False
    try:
        return all(
            row.get("samples_per_class") == EXPECTED_TRAIN_SAMPLES_PER_CLASS
            and row.get("pairs_per_sample") == EXPECTED_PAIRS_PER_SAMPLE
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("sample_structure") == "independent_pairs"
            and row.get("training", {}).get("batch_size") == EXPECTED_BATCH_SIZE
            and row.get("training", {}).get("epochs") == EXPECTED_EPOCHS
            and row.get("training", {}).get("loss") == "mse"
            and row.get("training", {}).get("optimizer") == "adam"
            and row.get("training", {}).get("learning_rate") == 0.0001
            and row.get("training", {}).get("weight_decay") == 0.00001
            and row.get("training", {}).get("checkpoint_metric") == "val_auc"
            and row.get("training", {}).get("restore_best_checkpoint") is True
            and row.get("training", {}).get("selected_checkpoint") == "best"
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
            and row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            and row.get("trainable_parameter_count")
            == EXPECTED_PARAMETER_COUNTS[_source_role(str(row.get("model", "")))]
            for row in rows
        )
    except ValueError:
        return False


def _control_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[
        tuple[str, int, str], dict[str, list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (
            str(row.get("cipher_key", "")),
            int(row.get("seed", -1)),
            str(row.get("source_role", "")),
        )
        grouped[key][str(row.get("condition", ""))].append(row)
    return grouped


def _same_field_per_source(
    grouped: Mapping[
        tuple[str, int, str], Mapping[str, Sequence[Mapping[str, Any]]]
    ],
    field: str,
) -> bool:
    return all(
        len({grouped[key][condition][0].get(field) for condition in CONTROL_CONDITIONS})
        == 1
        for key in _expected_source_keys()
    )


def _same_dataset_per_cipher_seed(
    grouped: Mapping[
        tuple[str, int, str], Mapping[str, Sequence[Mapping[str, Any]]]
    ],
) -> bool:
    return all(
        len(
            {
                grouped[(cipher, seed, role)][condition][0].get("dataset_sha256")
                for role in SOURCE_ROLES
                for condition in CONTROL_CONDITIONS
            }
        )
        == 1
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
    )


def _control_contract(
    group: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    rows = {condition: group[condition][0] for condition in CONTROL_CONDITIONS}
    source_role = str(rows["correct_ordered"].get("source_role"))
    expected_window_controls = {
        "correct_ordered": "full",
        "repeat_last": "repeat_last",
        "rotated": "full" if source_role == "candidate" else "rotated",
        "corrupted": "full",
        "no_topology": "full",
    }
    if any(
        rows[condition].get("runtime_structure_window_control")
        != expected_window_controls[condition]
        for condition in CONTROL_CONDITIONS
    ):
        return False
    if rows["no_topology"].get("relation_mode") != "independent":
        return False
    if any(
        rows[condition].get("relation_mode") != "true"
        for condition in CONTROL_CONDITIONS[:-1]
    ):
        return False
    if len(
        {
            rows[condition].get("control_fingerprint_sha256")
            for condition in CONTROL_CONDITIONS
        }
    ) != len(CONTROL_CONDITIONS):
        return False
    if (
        rows["repeat_last"].get("runtime_structure_homogeneous") is not True
        or rows["correct_ordered"].get("runtime_structure_homogeneous") is True
    ):
        return False
    if source_role == "candidate":
        return (
            rows["correct_ordered"].get("canonical_schedule_control") == "ordered"
            and rows["rotated"].get("canonical_schedule_control") == "rotated"
        )
    return rows["rotated"].get("runtime_structure_window_control") == "rotated"


def _control_row_finite(row: Mapping[str, Any]) -> bool:
    return all(
        isinstance(row.get(field), (int, float))
        and math.isfinite(float(row[field]))
        for field in (
            "auc",
            "source_auc",
            "correct_minus_source_auc",
            "correct_minus_condition_auc",
            "max_abs_probability_delta_from_correct",
            "mean_abs_probability_delta_from_correct",
            "mean_probability",
        )
    )


def _seed_result(
    training: Mapping[tuple[str, int, str], Mapping[str, Any]],
    grouped: Mapping[
        tuple[str, int, str], Mapping[str, Sequence[Mapping[str, Any]]]
    ],
    cipher: str,
    seed: int,
) -> dict[str, float | None]:
    candidate = grouped.get((cipher, seed, "candidate"), {})
    anchor = grouped.get((cipher, seed, "anchor"), {})
    candidate_auc = _condition_float(candidate, "correct_ordered", "auc")
    anchor_auc = _condition_float(anchor, "correct_ordered", "auc")
    result: dict[str, float | None] = {
        "candidate_auc": candidate_auc,
        "anchor_auc": anchor_auc,
        "candidate_minus_anchor": _difference(candidate_auc, anchor_auc),
    }
    for condition in CONTROL_CONDITIONS[1:]:
        control_auc = _condition_float(candidate, condition, "auc")
        result[f"candidate_{condition}_auc"] = control_auc
        result[f"candidate_minus_{condition}"] = _difference(
            candidate_auc, control_auc
        )
        result[f"candidate_{condition}_probability_delta"] = _condition_float(
            candidate,
            condition,
            "max_abs_probability_delta_from_correct",
        )
    source = training.get((cipher, seed, "candidate"))
    source_auc = None if source is None else source.get("metrics", {}).get("auc")
    result["source_training_auc"] = (
        float(source_auc) if isinstance(source_auc, (int, float)) else None
    )
    return result


def _condition_float(
    group: Mapping[str, Sequence[Mapping[str, Any]]], condition: str, field: str
) -> float | None:
    rows = group.get(condition, ())
    if len(rows) != 1:
        return None
    value = rows[0].get(field)
    return float(value) if isinstance(value, (int, float)) else None


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _hold_next_action(
    seed_results: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    checks: Mapping[str, bool],
) -> str:
    uknit_failed = any(
        not passed for name, passed in checks.items() if name.startswith("uknit64_")
    )
    dialga_failed = any(
        not passed for name, passed in checks.items() if name.startswith("dialga128_")
    )
    control_failed = any(
        not passed
        for name, passed in checks.items()
        if "beats_repeat_last" in name
        or "beats_rotated" in name
        or "beats_corrupted" in name
        or "beats_no_topology" in name
    )
    if control_failed:
        return (
            "Hold CT-SPN: the frozen candidate did not attribute its AUC to the "
            "correct ordered schedule in every cipher and seed. Inspect canonical "
            "edge/transition alignment only; do not add capacity or scale."
        )
    if uknit_failed and not dialga_failed:
        return (
            "Hold K1 and inspect uKNIT canonical edge/transition alignment; do not "
            "add capacity, samples, or a learned expert."
        )
    if dialga_failed and not uknit_failed:
        return (
            "Hold the two-cipher class claim and audit whether 128-bit edge pooling "
            "loses byte-local Dialga information; keep the data budget fixed."
        )
    _ = seed_results
    return (
        "Hold K1 as an unstable or weak local diagnostic and redesign the canonical "
        "transition interaction at 2048/class before any scale-up."
    )


__all__ = [
    "AUC_FLOORS",
    "CONTROL_CONDITIONS",
    "EXPECTED_BATCH_SIZE",
    "EXPECTED_CONTROL_ROWS",
    "EXPECTED_TRAINING_ROWS",
    "MARGIN_FLOOR",
    "RUN_ID",
    "SOURCE_ROLES",
    "adjudicate_ctspn_k1",
    "differential_dataset_sha256",
    "evaluate_frozen_control_panel",
    "file_sha256",
    "tensor_mapping_sha256",
]
