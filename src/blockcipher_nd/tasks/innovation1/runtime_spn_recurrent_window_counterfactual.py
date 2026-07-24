from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


EXPECTED_SEEDS = (0, 1)
EXPECTED_CONDITIONS = (
    "full_correct",
    "repeat_last",
    "corrupted",
    "no_topology",
)
U3_RUN_ID = (
    "i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_20260725"
)
U3_DECISION = "innovation1_runtime_spn_recurrent_window_two_seed_supported"
U3_PLAN_SHA256 = "060805c3e1e6793aa11b3e9758ddef738d646c77df596150032c8486b7bbd87f"
MARGIN_FLOOR = 0.005
PROBABILITY_DELTA_FLOOR = 1e-6
SOURCE_AUC_REPLAY_TOLERANCE = 1e-12
PARAMETER_COUNT = 442_466
FROZEN_MODEL_OPTIONS = {
    "runtime_structure_path": "configs/runtime/spn/uknit64.json",
    "runtime_round_start": 3,
    "runtime_rounds": 2,
    "processor_steps": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "sbox_context_mode": "edge_gate",
    "cell_input_mode": "state_triplet",
    "round_window_mode": "recurrent_window",
    "runtime_structure_window_control": "full",
}


def adjudicate_window_counterfactual_source(
    *,
    run_id: str,
    persisted_gate: dict[str, Any],
    replayed_gate: dict[str, Any],
    validation: dict[str, Any],
    plan_validation: dict[str, Any],
    result_rows_count: int,
    candidate_rows_valid: bool,
    candidate_checkpoints_exist: bool,
    visual_qa_passed: bool,
    plan_sha256: str,
) -> dict[str, Any]:
    protocol_checks = {
        "u3_run_identity": persisted_gate.get("run_id") == U3_RUN_ID,
        "u3_gate_recomputed_exactly": persisted_gate == replayed_gate,
        "u3_two_seed_pass": persisted_gate.get("status") == "pass"
        and persisted_gate.get("decision") == U3_DECISION,
        "u3_protocol_checks_all_true": _all_true(
            persisted_gate.get("protocol_checks")
        ),
        "u3_research_checks_all_true": _all_true(
            persisted_gate.get("research_checks")
        ),
        "u3_validation_matches_gate": validation.get("status") == "pass"
        and validation.get("run_id") == U3_RUN_ID
        and validation.get("checks") == persisted_gate.get("protocol_checks"),
        "u3_plan_validation_passed": plan_validation.get("status") == "pass"
        and plan_validation.get("expected_rows") == 10
        and plan_validation.get("result_rows") == 10
        and plan_validation.get("errors") == [],
        "u3_ten_result_rows": result_rows_count == 10,
        "u3_candidate_rows_valid": candidate_rows_valid,
        "u3_candidate_checkpoints_exist": candidate_checkpoints_exist,
        "u3_visual_qa_passed": visual_qa_passed,
        "u3_frozen_plan_sha256": plan_sha256 == U3_PLAN_SHA256,
    }
    authorized = all(protocol_checks.values())
    source_is_valid_hold = bool(
        persisted_gate == replayed_gate
        and persisted_gate.get("status") == "hold"
        and persisted_gate.get("decision")
        == "innovation1_runtime_spn_recurrent_window_not_supported"
        and validation.get("status") == "pass"
        and plan_validation.get("status") == "pass"
    )
    if authorized:
        status = "pass"
        decision = "innovation1_runtime_spn_window_u4_execution_authorized"
        next_action = "run the frozen eight-row same-checkpoint window panel"
    elif source_is_valid_hold:
        status = "hold"
        decision = "innovation1_runtime_spn_window_u4_stopped_by_u3"
        next_action = "do not run U4; redesign the recurrent interaction locally"
    else:
        status = "fail"
        decision = "innovation1_runtime_spn_window_u4_source_invalid"
        next_action = "repair only the missing or mismatched U3 source evidence"
    return {
        "run_id": run_id,
        "task": "innovation1_runtime_spn_window_u4_source_authorization",
        "status": status,
        "decision": decision,
        "execution_authorized": authorized,
        "protocol_checks": protocol_checks,
        "next_action": next_action,
    }


def evaluate_same_checkpoint_window_panel(
    *,
    seed: int,
    source_row: dict[str, Any],
    checkpoint_path: Path,
    dataset: DifferentialDataset,
    correct_structure: RuntimeSpnStructure,
    checkpoint_sha256: str,
    feature_path: Path,
    feature_sha256: str,
    label_path: Path,
    label_sha256: str,
    metadata_path: Path,
    metadata_sha256: str,
    descriptor_sha256: str,
    source_hashes: dict[str, str],
    batch_size: int = 256,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    _validate_candidate_source_row(source_row, seed)
    _validate_validation_dataset(dataset)
    model_options = dict(source_row["training"]["model_options"])
    model = build_model(
        "runtime_spn_e4_equivariant_true",
        input_bits=int(dataset.features.shape[1]),
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=model_options,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("U3 candidate checkpoint must contain a state_dict")
    checkpoint_metadata = payload.get("metadata")
    if not isinstance(checkpoint_metadata, dict):
        raise ValueError("U3 candidate checkpoint must contain metadata")
    if checkpoint_metadata.get("selected_checkpoint") != "best":
        raise ValueError("U3 candidate checkpoint must be the selected best checkpoint")
    model.load_state_dict(payload["state_dict"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != PARAMETER_COUNT:
        raise ValueError("U3 candidate checkpoint geometry does not match U4")

    structures = {
        "full_correct": correct_structure,
        "repeat_last": correct_structure.repeat_last_transition(),
        "corrupted": correct_structure.corrupted(20260724),
        "no_topology": correct_structure,
    }
    relation_modes = {
        "full_correct": "true",
        "repeat_last": "true",
        "corrupted": "true",
        "no_topology": "independent",
    }
    probabilities: dict[str, np.ndarray] = {}
    condition_metadata: dict[str, dict[str, Any]] = {}
    for condition in EXPECTED_CONDITIONS:
        structure = structures[condition]
        relation_mode = relation_modes[condition]
        model.runtime_structure = structure
        model.relation_mode = relation_mode
        model.mapping_mode = relation_mode
        probabilities[condition] = predict_binary_probabilities(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
        )
        transition_sha256s = structure.transition_sha256s()
        window_sha256 = structure.window_sha256()
        condition_metadata[condition] = {
            "relation_mode": relation_mode,
            "runtime_structure_transition_sha256s": list(transition_sha256s),
            "runtime_structure_window_sha256": window_sha256,
            "runtime_structure_unique_transition_count": len(
                set(transition_sha256s)
            ),
            "runtime_structure_homogeneous": len(set(transition_sha256s)) == 1,
            "intervention_sha256": _intervention_sha256(
                window_sha256,
                relation_mode,
            ),
        }

    labels = np.asarray(dataset.labels, dtype=np.float32)
    aucs = {
        condition: binary_auc(labels, probabilities[condition])
        for condition in EXPECTED_CONDITIONS
    }
    reference_probabilities = probabilities["full_correct"]
    source_auc = float(source_row["metrics"]["auc"])
    return [
        {
            "seed": seed,
            "condition": condition,
            **condition_metadata[condition],
            "auc": aucs[condition],
            "full_correct_minus_condition_auc": (
                0.0
                if condition == "full_correct"
                else aucs["full_correct"] - aucs[condition]
            ),
            "max_abs_probability_delta_from_full": float(
                np.max(
                    np.abs(reference_probabilities - probabilities[condition])
                )
            ),
            "mean_probability": float(probabilities[condition].mean()),
            "probability_sha256": hashlib.sha256(
                probabilities[condition].tobytes()
            ).hexdigest(),
            "source_candidate_auc": source_auc,
            "full_correct_minus_source_auc": aucs["full_correct"] - source_auc,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_selected": checkpoint_metadata.get("selected_checkpoint"),
            "feature_path": str(feature_path),
            "feature_sha256": feature_sha256,
            "label_path": str(label_path),
            "label_sha256": label_sha256,
            "metadata_path": str(metadata_path),
            "metadata_sha256": metadata_sha256,
            "descriptor_sha256": descriptor_sha256,
            "descriptor_round_start": 3,
            "descriptor_loaded_rounds": 2,
            "source_results_sha256": source_hashes["results"],
            "source_gate_sha256": source_hashes["gate"],
            "source_validation_sha256": source_hashes["validation"],
            "source_plan_validation_sha256": source_hashes["plan_validation"],
            "source_visual_qa_verified": True,
            "samples_total": int(len(dataset.labels)),
            "input_bits": int(dataset.features.shape[1]),
            "pairs_per_sample": 4,
            "parameter_count": parameter_count,
            "model_options": model_options,
            "negative_mode": "encrypted_random_plaintexts",
            "strict_state_dict_load": True,
            "training_performed": False,
        }
        for condition in EXPECTED_CONDITIONS
    ]


def adjudicate_same_checkpoint_window_panel(
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
    seed_results = {str(seed): _seed_result(grouped[seed]) for seed in EXPECTED_SEEDS}
    protocol_checks = {
        "eight_rows_complete": len(rows) == 8,
        "two_seed_four_condition_panel": complete
        and set(grouped) == set(EXPECTED_SEEDS)
        and all(set(grouped[seed]) == set(EXPECTED_CONDITIONS) for seed in EXPECTED_SEEDS),
        "same_checkpoint_within_seed": complete
        and _same_seed_field(grouped, "checkpoint_sha256"),
        "same_features_within_seed": complete
        and _same_seed_field(grouped, "feature_sha256"),
        "same_labels_within_seed": complete
        and _same_seed_field(grouped, "label_sha256"),
        "same_metadata_within_seed": complete
        and _same_seed_field(grouped, "metadata_sha256"),
        "independent_seed_checkpoints": complete
        and _seed_reference_field(grouped, 0, "checkpoint_sha256")
        != _seed_reference_field(grouped, 1, "checkpoint_sha256"),
        "independent_seed_features": complete
        and _seed_reference_field(grouped, 0, "feature_sha256")
        != _seed_reference_field(grouped, 1, "feature_sha256"),
        "shared_u3_source_provenance": complete
        and all(
            len({row.get(field) for row in rows}) == 1
            for field in (
                "source_results_sha256",
                "source_gate_sha256",
                "source_validation_sha256",
                "source_plan_validation_sha256",
                "descriptor_sha256",
            )
        ),
        "selected_best_checkpoints": all(
            row.get("checkpoint_selected") == "best" for row in rows
        ),
        "strict_state_dict_load": all(
            row.get("strict_state_dict_load") is True for row in rows
        ),
        "frozen_model_geometry": len(rows) == 8
        and {row.get("parameter_count") for row in rows} == {PARAMETER_COUNT}
        and all(row.get("model_options") == FROZEN_MODEL_OPTIONS for row in rows),
        "frozen_validation_protocol": all(
            row.get("samples_total") == 2048
            and row.get("input_bits") == 512
            and row.get("pairs_per_sample") == 4
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("descriptor_round_start") == 3
            and row.get("descriptor_loaded_rounds") == 2
            for row in rows
        ),
        "full_auc_replays_u3_candidate": complete
        and all(
            abs(
                float(grouped[seed]["full_correct"][0].get("full_correct_minus_source_auc", math.inf))
            )
            <= SOURCE_AUC_REPLAY_TOLERANCE
            for seed in EXPECTED_SEEDS
        ),
        "window_interventions_exact": complete
        and all(_window_contract(grouped[seed]) for seed in EXPECTED_SEEDS),
        "interventions_seed_invariant": complete
        and all(
            grouped[0][condition][0].get("intervention_sha256")
            == grouped[1][condition][0].get("intervention_sha256")
            for condition in EXPECTED_CONDITIONS
        ),
        "finite_metrics": all(
            _finite(row.get("auc"))
            and _finite(row.get("mean_probability"))
            and _finite(row.get("max_abs_probability_delta_from_full"))
            for row in rows
        ),
        "sha256_provenance_complete": all(
            _is_sha256(row.get(field))
            for row in rows
            for field in (
                "checkpoint_sha256",
                "feature_sha256",
                "label_sha256",
                "metadata_sha256",
                "descriptor_sha256",
                "source_results_sha256",
                "source_gate_sha256",
                "source_validation_sha256",
                "source_plan_validation_sha256",
                "runtime_structure_window_sha256",
                "intervention_sha256",
                "probability_sha256",
            )
        ),
        "source_visual_qa_verified": all(
            row.get("source_visual_qa_verified") is True for row in rows
        ),
        "no_training_performed": all(
            row.get("training_performed") is False for row in rows
        ),
    }
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        result = seed_results[str(seed)]
        for condition in ("repeat_last", "corrupted", "no_topology"):
            research_checks[
                f"seed{seed}_full_beats_{condition}_by_0p005"
            ] = _at_least(result[f"full_minus_{condition}"], MARGIN_FLOOR)
            research_checks[
                f"seed{seed}_{condition}_changes_probabilities"
            ] = _greater_than(
                result[f"{condition}_probability_delta"],
                PROBABILITY_DELTA_FLOOR,
            )

    protocol_passed = all(protocol_checks.values())
    research_passed = all(research_checks.values())
    if not protocol_passed:
        status = "fail"
        decision = "innovation1_runtime_spn_window_same_checkpoint_protocol_invalid"
        next_action = "repair source replay or artifact loading without changing thresholds"
    elif research_passed:
        status = "pass"
        decision = (
            "innovation1_runtime_spn_window_same_checkpoint_attribution_supported"
        )
        next_action = (
            "preregister one cross-cipher same-backbone checkpoint-reuse gate; "
            "do not scale uKNIT samples or epochs"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_runtime_spn_window_same_checkpoint_attribution_not_supported"
        )
        next_action = (
            "stop recurrent-window scale-up and redesign the failed local structure "
            "interaction against the frozen U3/U4 anchors"
        )
    return {
        "run_id": run_id,
        "task": "innovation1_runtime_spn_window_same_checkpoint_u4",
        "cipher": "uKNIT-BC",
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "seed_results": seed_results,
        "thresholds": {
            "full_minus_each_control_auc": MARGIN_FLOOR,
            "control_probability_delta_strictly_greater_than": PROBABILITY_DELTA_FLOOR,
            "source_auc_replay_absolute_tolerance": SOURCE_AUC_REPLAY_TOLERANCE,
        },
        "claim_scope": (
            "uKNIT-BC prefix-r5 two-seed same-checkpoint inference-time window "
            "attribution audit; no training, scale, attack, SOTA, breakthrough, "
            "or cross-cipher transfer claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "train separate U4 control models",
            "regenerate the U3 validation data",
            "tune thresholds after reading U4 metrics",
            "increase uKNIT samples, epochs, or pair count",
            "add DDT, trail, related-key, or partial-decryption features",
        ],
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_candidate_source_row(row: dict[str, Any], seed: int) -> None:
    training = row.get("training")
    validation = row.get("validation")
    if not isinstance(training, dict) or not isinstance(validation, dict):
        raise ValueError("U3 candidate row is missing training or validation metadata")
    if not (
        seed in EXPECTED_SEEDS
        and row.get("seed") == seed
        and row.get("cipher_key") == "uknit64"
        and row.get("rounds") == 5
        and row.get("model") == "runtime_spn_e4_equivariant_true"
        and row.get("samples_per_class") == 2048
        and row.get("pairs_per_sample") == 4
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and validation.get("samples_per_class") == 1024
        and validation.get("samples_total") == 2048
        and training.get("selected_checkpoint") == "best"
        and training.get("restore_best_checkpoint") is True
        and training.get("validation_rows") == 2048
        and training.get("model_options") == FROZEN_MODEL_OPTIONS
        and _finite(row.get("metrics", {}).get("auc"))
    ):
        raise ValueError("U3 candidate row does not match the frozen U4 source contract")


def _validate_validation_dataset(dataset: DifferentialDataset) -> None:
    features = np.asarray(dataset.features)
    labels = np.asarray(dataset.labels)
    if features.shape != (2048, 512) or labels.shape != (2048,):
        raise ValueError("U3 validation cache must contain 2048 rows of 512 bits")
    unique, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(unique, np.array([0, 1])) or not np.array_equal(
        counts, np.array([1024, 1024])
    ):
        raise ValueError("U3 validation labels must contain 1024 rows per class")


def _window_contract(seed_rows: dict[str, list[dict[str, Any]]]) -> bool:
    rows = {condition: seed_rows[condition][0] for condition in EXPECTED_CONDITIONS}
    full = rows["full_correct"]
    repeated = rows["repeat_last"]
    corrupted = rows["corrupted"]
    independent = rows["no_topology"]
    full_transitions = full.get("runtime_structure_transition_sha256s")
    repeated_transitions = repeated.get("runtime_structure_transition_sha256s")
    return bool(
        all(
            rows[condition].get("relation_mode")
            == ("independent" if condition == "no_topology" else "true")
            for condition in EXPECTED_CONDITIONS
        )
        and isinstance(full_transitions, list)
        and len(full_transitions) == 2
        and len(set(full_transitions)) == 2
        and full.get("runtime_structure_homogeneous") is False
        and isinstance(repeated_transitions, list)
        and len(repeated_transitions) == 2
        and len(set(repeated_transitions)) == 1
        and repeated.get("runtime_structure_homogeneous") is True
        and repeated_transitions[-1] == full_transitions[-1]
        and full.get("runtime_structure_window_sha256")
        == independent.get("runtime_structure_window_sha256")
        and full.get("runtime_structure_window_sha256")
        != repeated.get("runtime_structure_window_sha256")
        and full.get("runtime_structure_window_sha256")
        != corrupted.get("runtime_structure_window_sha256")
        and len(
            {rows[condition].get("intervention_sha256") for condition in EXPECTED_CONDITIONS}
        )
        == 4
    )


def _seed_result(seed_rows: dict[str, list[dict[str, Any]]]) -> dict[str, float | None]:
    aucs = {
        condition: _single_float(seed_rows.get(condition, ()), "auc")
        for condition in EXPECTED_CONDITIONS
    }
    full = aucs["full_correct"]
    result: dict[str, float | None] = {
        f"{condition}_auc": value for condition, value in aucs.items()
    }
    for condition in ("repeat_last", "corrupted", "no_topology"):
        control = aucs[condition]
        result[f"full_minus_{condition}"] = (
            full - control if full is not None and control is not None else None
        )
        result[f"{condition}_probability_delta"] = _single_float(
            seed_rows.get(condition, ()),
            "max_abs_probability_delta_from_full",
        )
    return result


def _single_float(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    rows = list(rows)
    if len(rows) != 1 or not _finite(rows[0].get(field)):
        return None
    return float(rows[0][field])


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


def _seed_reference_field(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
    seed: int,
    field: str,
) -> object:
    return grouped[seed]["full_correct"][0].get(field)


def _intervention_sha256(window_sha256: str, relation_mode: str) -> str:
    payload = json.dumps(
        {"window_sha256": window_sha256, "relation_mode": relation_mode},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _all_true(value: object) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        item is True for item in value.values()
    )


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _at_least(value: object, threshold: float) -> bool:
    return _finite(value) and float(value) >= threshold


def _greater_than(value: object, threshold: float) -> bool:
    return _finite(value) and float(value) > threshold


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "EXPECTED_CONDITIONS",
    "FROZEN_MODEL_OPTIONS",
    "U3_PLAN_SHA256",
    "U3_RUN_ID",
    "adjudicate_same_checkpoint_window_panel",
    "adjudicate_window_counterfactual_source",
    "evaluate_same_checkpoint_window_panel",
    "file_sha256",
]
