from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


EXPECTED_SEEDS = (0, 1)
EXPECTED_CONDITIONS = ("correct", "corrupted", "no_topology")
CORRUPTION_SEED = 20260725
AUC_FLOOR = 0.520
MARGIN_FLOOR = 0.005
PROBABILITY_DELTA_FLOOR = 1e-6
PARAMETER_COUNT = 442466
VALIDATION_KEY = int("11" * 32, 16)
SOURCE_DECISION = "innovation1_dialga_runtime_e4_d1_two_seed_supported"
FROZEN_MODEL_OPTIONS: dict[str, object] = {
    "runtime_structure_path": "configs/runtime/spn/dialga128.json",
    "runtime_round_start": 2,
    "runtime_rounds": 2,
    "processor_steps": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "sbox_context_mode": "edge_gate",
    "cell_input_mode": "state_triplet",
    "round_window_mode": "recurrent_window",
    "runtime_structure_window_control": "full",
}


def evaluate_same_checkpoint_dialga(
    *,
    seed: int,
    model_options: dict[str, Any],
    checkpoint_path: Path,
    dataset: DifferentialDataset,
    correct_structure: RuntimeSpnStructure,
    corrupted_structure: RuntimeSpnStructure,
    source_auc: float,
    checkpoint_sha256: str,
    feature_sha256: str,
    label_sha256: str,
    metadata_sha256: str,
    source_results_sha256: str,
    source_gate_sha256: str,
    descriptor_name: str,
    descriptor_path: str,
    descriptor_sha256: str,
    source_descriptor_sha256: str,
    batch_size: int = 256,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unexpected D2 seed: {seed}")
    if model_options != FROZEN_MODEL_OPTIONS:
        raise ValueError("source checkpoint model options do not match frozen D2")
    metadata = dict(dataset.metadata)
    _validate_dataset_metadata(metadata, seed)

    model = build_model(
        "runtime_spn_e4_equivariant_true",
        input_bits=int(dataset.features.shape[1]),
        hidden_bits=64,
        pair_bits=256,
        structure="SPN",
        model_options=model_options,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("D1 checkpoint must contain a state_dict")
    checkpoint_metadata = payload.get("metadata")
    if not isinstance(checkpoint_metadata, dict):
        raise ValueError("D1 checkpoint must contain metadata")
    if (
        checkpoint_metadata.get("selected_checkpoint") != "best"
        or checkpoint_metadata.get("seed") != seed
    ):
        raise ValueError("D1 checkpoint is not the expected seed-specific best model")
    model.load_state_dict(payload["state_dict"], strict=True)
    parameter_count = sum(value.numel() for value in model.parameters())

    conditions = {
        "correct": (correct_structure, "true"),
        "corrupted": (corrupted_structure, "true"),
        "no_topology": (correct_structure, "independent"),
    }
    probabilities: dict[str, np.ndarray] = {}
    for condition, (runtime_structure, relation_mode) in conditions.items():
        model.runtime_structure = runtime_structure
        model.relation_mode = relation_mode
        model.mapping_mode = relation_mode
        probabilities[condition] = predict_binary_probabilities(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
        )

    labels = np.asarray(dataset.labels, dtype=np.float32)
    aucs = {
        condition: binary_auc(labels, values)
        for condition, values in probabilities.items()
    }
    reference = probabilities["correct"]

    return [
        {
            "seed": seed,
            "condition": condition,
            "cipher": "Dialga-128",
            "rounds": 4,
            "auc": aucs[condition],
            "source_auc": float(source_auc),
            "correct_minus_condition_auc": (
                0.0 if condition == "correct" else aucs["correct"] - aucs[condition]
            ),
            "max_abs_probability_delta_from_correct": float(
                np.max(np.abs(reference - probabilities[condition]))
            ),
            "mean_abs_probability_delta_from_correct": float(
                np.mean(np.abs(reference - probabilities[condition]))
            ),
            "mean_probability": float(probabilities[condition].mean()),
            "probability_sha256": hashlib.sha256(
                probabilities[condition].tobytes()
            ).hexdigest(),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_selected": checkpoint_metadata.get("selected_checkpoint"),
            "checkpoint_reported_seed": checkpoint_metadata.get("seed"),
            "checkpoint_best_metric": checkpoint_metadata.get(
                "best_checkpoint_metric"
            ),
            "strict_state_dict_load": True,
            "feature_sha256": feature_sha256,
            "label_sha256": label_sha256,
            "metadata_sha256": metadata_sha256,
            "source_results_sha256": source_results_sha256,
            "source_gate_sha256": source_gate_sha256,
            "source_d1_verified": True,
            "source_d1_decision": SOURCE_DECISION,
            "descriptor_name": descriptor_name,
            "descriptor_path": descriptor_path,
            "descriptor_sha256": descriptor_sha256,
            "source_descriptor_sha256": source_descriptor_sha256,
            "descriptor_round_start": 2,
            "descriptor_loaded_rounds": 2,
            "runtime_structure_mode": condition,
            "relation_mode": relation_mode,
            "runtime_structure_window_sha256": runtime_structure.window_sha256(),
            "runtime_structure_transition_sha256s": list(
                runtime_structure.transition_sha256s()
            ),
            "runtime_structure_unique_transition_count": (
                runtime_structure.unique_transition_count
            ),
            "runtime_intervention_sha256": _intervention_sha256(
                condition=condition,
                relation_mode=relation_mode,
                structure=runtime_structure,
            ),
            "samples_total": int(len(dataset.labels)),
            "validation_seed": metadata.get("seed"),
            "input_bits": int(dataset.features.shape[1]),
            "pair_bits": metadata.get("pair_bits"),
            "pairs_per_sample": metadata.get("pairs_per_sample"),
            "input_difference": metadata.get("input_difference"),
            "negative_mode": metadata.get("negative_mode"),
            "sample_structure": metadata.get("sample_structure"),
            "validation_key": VALIDATION_KEY,
            "parameter_count": parameter_count,
            "model_options": model_options,
            "training_performed": False,
        }
        for condition, (runtime_structure, relation_mode) in conditions.items()
    ]


def adjudicate_same_checkpoint_dialga(
    *,
    run_id: str,
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row.get("seed", -1))][str(row.get("condition"))].append(row)

    complete = all(
        len(grouped[seed].get(condition, ())) == 1
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    )
    seed_results = {
        str(seed): _seed_result(grouped[seed]) for seed in EXPECTED_SEEDS
    }
    protocol_checks = {
        "six_rows_complete": len(rows) == 6,
        "two_seed_three_condition_panel": complete
        and set(grouped) == set(EXPECTED_SEEDS),
        "same_checkpoint_within_seed": complete
        and _same_seed_field(grouped, "checkpoint_sha256"),
        "same_features_within_seed": complete
        and _same_seed_field(grouped, "feature_sha256"),
        "same_labels_within_seed": complete
        and _same_seed_field(grouped, "label_sha256"),
        "same_metadata_within_seed": complete
        and _same_seed_field(grouped, "metadata_sha256"),
        "distinct_seed_checkpoints": complete
        and len(
            {
                grouped[seed]["correct"][0].get("checkpoint_sha256")
                for seed in EXPECTED_SEEDS
            }
        )
        == 2,
        "selected_seed_specific_best_checkpoints": all(
            row.get("checkpoint_selected") == "best"
            and row.get("checkpoint_reported_seed") == row.get("seed")
            and _finite(row.get("checkpoint_best_metric"))
            for row in rows
        ),
        "strict_state_dict_load": all(
            row.get("strict_state_dict_load") is True for row in rows
        ),
        "verified_d1_source": all(
            row.get("source_d1_verified") is True
            and row.get("source_d1_decision") == SOURCE_DECISION
            for row in rows
        ),
        "same_source_provenance": len(rows) == 6
        and len({row.get("source_results_sha256") for row in rows}) == 1
        and len({row.get("source_gate_sha256") for row in rows}) == 1,
        "source_sha256_fields_present": all(
            _is_sha256(row.get(field))
            for row in rows
            for field in ("source_results_sha256", "source_gate_sha256")
        ),
        "exact_descriptor_window": complete
        and _descriptor_contract(grouped),
        "exact_structure_interventions": complete
        and _intervention_contract(grouped),
        "frozen_model_geometry": len(rows) == 6
        and {row.get("parameter_count") for row in rows} == {PARAMETER_COUNT}
        and all(row.get("model_options") == FROZEN_MODEL_OPTIONS for row in rows),
        "frozen_validation_protocol": all(
            row.get("cipher") == "Dialga-128"
            and row.get("rounds") == 4
            and row.get("samples_total") == 2048
            and row.get("validation_seed") == 10000 + int(row.get("seed", -1))
            and row.get("input_bits") == 1024
            and row.get("pair_bits") == 256
            and row.get("pairs_per_sample") == 4
            and row.get("input_difference") == 0x40
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("sample_structure") == "independent_pairs"
            and row.get("validation_key") == VALIDATION_KEY
            for row in rows
        ),
        "correct_auc_reproduces_d1": all(
            result["correct_auc"] is not None
            and result["source_auc"] is not None
            and abs(result["correct_auc"] - result["source_auc"]) <= 1e-12
            for result in seed_results.values()
        ),
        "finite_metrics": all(
            _finite(row.get("auc"))
            and _finite(row.get("source_auc"))
            and _finite(row.get("mean_probability"))
            and _finite(row.get("max_abs_probability_delta_from_correct"))
            and _finite(row.get("mean_abs_probability_delta_from_correct"))
            for row in rows
        ),
        "no_training_performed": all(
            row.get("training_performed") is False for row in rows
        ),
        "all_artifact_sha256_fields_present": all(
            _is_sha256(row.get(field))
            for row in rows
            for field in (
                "checkpoint_sha256",
                "feature_sha256",
                "label_sha256",
                "metadata_sha256",
                "descriptor_sha256",
                "source_descriptor_sha256",
                "runtime_structure_window_sha256",
                "runtime_intervention_sha256",
                "probability_sha256",
            )
        ),
    }

    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        result = seed_results[str(seed)]
        research_checks[f"seed{seed}_correct_auc_at_least_0p520"] = bool(
            result["correct_auc"] is not None and result["correct_auc"] >= AUC_FLOOR
        )
        research_checks[f"seed{seed}_beats_corrupted_by_0p005"] = bool(
            result["correct_minus_corrupted_auc"] is not None
            and result["correct_minus_corrupted_auc"] >= MARGIN_FLOOR
        )
        research_checks[f"seed{seed}_beats_no_topology_by_0p005"] = bool(
            result["correct_minus_no_topology_auc"] is not None
            and result["correct_minus_no_topology_auc"] >= MARGIN_FLOOR
        )
        research_checks[f"seed{seed}_corrupted_probabilities_change"] = bool(
            result["corrupted_probability_delta"] is not None
            and result["corrupted_probability_delta"] > PROBABILITY_DELTA_FLOOR
        )
        research_checks[f"seed{seed}_no_topology_probabilities_change"] = bool(
            result["no_topology_probability_delta"] is not None
            and result["no_topology_probability_delta"] > PROBABILITY_DELTA_FLOOR
        )

    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_dialga_runtime_e4_d2_protocol_invalid"
        next_action = (
            "repair only the inference audit or source binding without changing "
            "checkpoints, caches, structures, or thresholds"
        )
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_dialga_runtime_e4_d2_functional_topology_use_supported"
        next_action = (
            "preregister one same-budget Dialga prefix-r5 adjacent-window replication; "
            "do not increase samples first"
        )
    else:
        status = "hold"
        decision = "innovation1_dialga_runtime_e4_d2_functional_topology_use_not_supported"
        next_action = (
            "retain only the supported D1 training-time signal and redesign the "
            "runtime interaction locally without scale-up"
        )

    return {
        "run_id": run_id,
        "task": "innovation1_dialga_runtime_e4_d2_same_checkpoint",
        "cipher": "Dialga-128",
        "status": status,
        "decision": decision,
        "thresholds": {
            "correct_auc": AUC_FLOOR,
            "auc_margin": MARGIN_FLOOR,
            "probability_delta": PROBABILITY_DELTA_FLOOR,
            "source_auc_tolerance": 1e-12,
        },
        "seed_results": seed_results,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "claim_scope": (
            "Dialga-128 prefix-r4 two-seed same-checkpoint inference-only runtime-"
            "topology audit; no training, formal scale, attack, paper reproduction, "
            "SOTA, cross-cipher, or universal-SPN claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "retrain or reselect checkpoints inside D2",
            "increase samples, pairs, epochs, rounds, seeds, or keys",
            "launch remote GPU scale",
            "add DDT, trail, partial-decryption, or guessed-key features",
        ],
    }


def _validate_dataset_metadata(metadata: dict[str, Any], seed: int) -> None:
    expected = {
        "cipher": "Dialga-128",
        "rounds": 4,
        "seed": 10000 + seed,
        "samples_total": 2048,
        "samples_per_class": 1024,
        "input_bits": 1024,
        "pair_bits": 256,
        "pairs_per_sample": 4,
        "input_difference": 0x40,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
    }
    if any(metadata.get(field) != value for field, value in expected.items()):
        raise ValueError("D1 validation cache does not match frozen D2 protocol")


def _descriptor_contract(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
) -> bool:
    rows = [
        grouped[seed][condition][0]
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    ]
    return (
        len({row.get("descriptor_sha256") for row in rows}) == 1
        and all(
            row.get("descriptor_name")
            == "Dialga-128 20-round heterogeneous runtime SPN structure"
            and str(row.get("descriptor_path", "")).endswith(
                "configs/runtime/spn/dialga128.json"
            )
            and row.get("descriptor_sha256")
            == row.get("source_descriptor_sha256")
            and row.get("descriptor_round_start") == 2
            and row.get("descriptor_loaded_rounds") == 2
            and row.get("runtime_structure_unique_transition_count") == 2
            and len(row.get("runtime_structure_transition_sha256s", [])) == 2
            for row in rows
        )
    )


def _intervention_contract(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
) -> bool:
    expected_modes = {
        "correct": ("true", "correct"),
        "corrupted": ("true", "corrupted"),
        "no_topology": ("independent", "no_topology"),
    }
    for seed in EXPECTED_SEEDS:
        rows = {condition: grouped[seed][condition][0] for condition in EXPECTED_CONDITIONS}
        if any(
            rows[condition].get("relation_mode") != relation_mode
            or rows[condition].get("runtime_structure_mode") != structure_mode
            for condition, (relation_mode, structure_mode) in expected_modes.items()
        ):
            return False
        if not (
            rows["correct"].get("runtime_structure_window_sha256")
            == rows["no_topology"].get("runtime_structure_window_sha256")
            != rows["corrupted"].get("runtime_structure_window_sha256")
        ):
            return False
        if len(
            {rows[condition].get("runtime_intervention_sha256") for condition in EXPECTED_CONDITIONS}
        ) != 3:
            return False
    return all(
        grouped[0][condition][0].get(field)
        == grouped[1][condition][0].get(field)
        for condition in EXPECTED_CONDITIONS
        for field in (
            "runtime_structure_window_sha256",
            "runtime_intervention_sha256",
        )
    )


def _seed_result(group: dict[str, list[dict[str, Any]]]) -> dict[str, float | None]:
    values = {
        condition: rows[0] if len(rows) == 1 else None
        for condition, rows in (
            (condition, group.get(condition, []))
            for condition in EXPECTED_CONDITIONS
        )
    }
    correct_auc = _row_float(values["correct"], "auc")
    corrupted_auc = _row_float(values["corrupted"], "auc")
    no_topology_auc = _row_float(values["no_topology"], "auc")
    return {
        "correct_auc": correct_auc,
        "corrupted_auc": corrupted_auc,
        "no_topology_auc": no_topology_auc,
        "source_auc": _row_float(values["correct"], "source_auc"),
        "correct_minus_corrupted_auc": _difference(correct_auc, corrupted_auc),
        "correct_minus_no_topology_auc": _difference(correct_auc, no_topology_auc),
        "corrupted_probability_delta": _row_float(
            values["corrupted"], "max_abs_probability_delta_from_correct"
        ),
        "no_topology_probability_delta": _row_float(
            values["no_topology"], "max_abs_probability_delta_from_correct"
        ),
    }


def _same_seed_field(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
    field: str,
) -> bool:
    return all(
        len(
            {
                grouped[seed][condition][0].get(field)
                for condition in EXPECTED_CONDITIONS
            }
        )
        == 1
        for seed in EXPECTED_SEEDS
    )


def _intervention_sha256(
    *,
    condition: str,
    relation_mode: str,
    structure: RuntimeSpnStructure,
) -> str:
    payload = {
        "condition": condition,
        "relation_mode": relation_mode,
        "structure_window_sha256": structure.window_sha256(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _row_float(row: dict[str, Any] | None, field: str) -> float | None:
    value = row.get(field) if row is not None else None
    return float(value) if _finite(value) else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "FROZEN_MODEL_OPTIONS",
    "adjudicate_same_checkpoint_dialga",
    "evaluate_same_checkpoint_dialga",
]
