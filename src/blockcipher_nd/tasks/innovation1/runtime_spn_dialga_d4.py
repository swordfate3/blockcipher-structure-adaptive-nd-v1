from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d2 import (
    FROZEN_MODEL_OPTIONS,
    PARAMETER_COUNT,
    VALIDATION_KEY,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


EXPECTED_SEEDS = (0, 1)
EXPECTED_CONDITIONS = ("r4_w2", "r4_w3", "r5_w2", "r5_w3")
CONDITION_SPEC = {
    "r4_w2": ("d1_r4", 4, 2),
    "r4_w3": ("d1_r4", 4, 3),
    "r5_w2": ("d3_r5", 5, 2),
    "r5_w3": ("d3_r5", 5, 3),
}
D1_DECISION = "innovation1_dialga_runtime_e4_d1_two_seed_supported"
D3_DECISION = "innovation1_dialga_runtime_e4_d3_adjacent_window_not_replicated"
RETENTION_FRACTION = 0.5
SOURCE_AUC_TOLERANCE = 1e-12


def evaluate_factorial_dialga(
    *,
    seed: int,
    model_options: dict[str, Any],
    checkpoint_path: Path,
    datasets: dict[str, DifferentialDataset],
    dataset_hashes: dict[str, dict[str, str]],
    structures: dict[int, RuntimeSpnStructure],
    anchor_auc: float,
    checkpoint_sha256: str,
    source_hashes: dict[str, str],
    descriptor_name: str,
    descriptor_path: str,
    descriptor_sha256: str,
    batch_size: int = 256,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unexpected D4 seed: {seed}")
    if model_options != FROZEN_MODEL_OPTIONS:
        raise ValueError("source checkpoint model options do not match frozen D4")
    if set(datasets) != {"d1_r4", "d3_r5"}:
        raise ValueError("D4 requires exact D1-r4 and D3-r5 validation datasets")
    if set(structures) != {2, 3}:
        raise ValueError("D4 requires runtime windows starting at rounds 2 and 3")
    for data_source, expected_rounds in (("d1_r4", 4), ("d3_r5", 5)):
        _validate_dataset_metadata(
            dict(datasets[data_source].metadata), seed, expected_rounds
        )

    model = build_model(
        "runtime_spn_e4_equivariant_true",
        input_bits=1024,
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

    probabilities: dict[str, np.ndarray] = {}
    aucs: dict[str, float] = {}
    for condition in EXPECTED_CONDITIONS:
        data_source, _, runtime_round_start = CONDITION_SPEC[condition]
        model.runtime_structure = structures[runtime_round_start]
        model.relation_mode = "true"
        model.mapping_mode = "true"
        probability = predict_binary_probabilities(
            model,
            datasets[data_source],
            batch_size=batch_size,
            device=device,
        )
        probabilities[condition] = probability
        aucs[condition] = binary_auc(
            np.asarray(datasets[data_source].labels, dtype=np.float32), probability
        )

    anchor_probability = probabilities["r4_w2"]
    return [
        _result_row(
            seed=seed,
            condition=condition,
            auc=aucs[condition],
            anchor_auc=anchor_auc,
            probabilities=probabilities[condition],
            anchor_probability=anchor_probability,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_metadata=checkpoint_metadata,
            dataset_hashes=dataset_hashes,
            structure=structures[CONDITION_SPEC[condition][2]],
            source_hashes=source_hashes,
            descriptor_name=descriptor_name,
            descriptor_path=descriptor_path,
            descriptor_sha256=descriptor_sha256,
            parameter_count=parameter_count,
            model_options=model_options,
        )
        for condition in EXPECTED_CONDITIONS
    ]


def adjudicate_factorial_dialga(
    *, run_id: str, rows: Iterable[dict[str, Any]]
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
    flat_rows = [
        grouped[seed][condition][0]
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
        if len(grouped[seed].get(condition, ())) == 1
    ]
    protocol_checks = {
        "eight_rows_complete": len(rows) == 8,
        "two_seed_four_condition_panel": complete
        and set(grouped) == set(EXPECTED_SEEDS),
        "same_checkpoint_within_seed": complete
        and _same_seed_field(grouped, "checkpoint_sha256"),
        "distinct_seed_checkpoints": complete
        and len(
            {
                grouped[seed]["r4_w2"][0].get("checkpoint_sha256")
                for seed in EXPECTED_SEEDS
            }
        )
        == 2,
        "selected_seed_specific_best_checkpoints": all(
            row.get("checkpoint_selected") == "best"
            and row.get("checkpoint_reported_seed") == row.get("seed")
            and _finite(row.get("checkpoint_best_metric"))
            for row in flat_rows
        ),
        "strict_state_dict_load": len(flat_rows) == 8
        and all(row.get("strict_state_dict_load") is True for row in flat_rows),
        "verified_d1_and_d3_sources": len(flat_rows) == 8
        and all(
            row.get("source_d1_verified") is True
            and row.get("source_d3_verified") is True
            and row.get("source_d1_decision") == D1_DECISION
            and row.get("source_d3_decision") == D3_DECISION
            for row in flat_rows
        ),
        "same_source_provenance": len(flat_rows) == 8
        and all(
            len({row.get(field) for row in flat_rows}) == 1
            for field in (
                "d1_results_sha256",
                "d1_gate_sha256",
                "d3_results_sha256",
                "d3_gate_sha256",
            )
        ),
        "source_sha256_fields_present": len(flat_rows) == 8
        and all(
            _is_sha256(row.get(field))
            for row in flat_rows
            for field in (
                "d1_results_sha256",
                "d1_gate_sha256",
                "d3_results_sha256",
                "d3_gate_sha256",
            )
        ),
        "exact_condition_contract": complete and _condition_contract(grouped),
        "exact_descriptor_windows": complete and _descriptor_contract(grouped),
        "paired_dataset_hashes": complete and _dataset_hash_contract(grouped),
        "frozen_model_geometry": len(flat_rows) == 8
        and {row.get("parameter_count") for row in flat_rows} == {PARAMETER_COUNT}
        and all(row.get("model_options") == FROZEN_MODEL_OPTIONS for row in flat_rows),
        "frozen_validation_protocol": len(flat_rows) == 8
        and all(_row_has_frozen_validation_protocol(row) for row in flat_rows),
        "d1_anchor_reproduced": all(
            result["r4_w2_auc"] is not None
            and result["source_anchor_auc"] is not None
            and abs(result["r4_w2_auc"] - result["source_anchor_auc"])
            <= SOURCE_AUC_TOLERANCE
            for result in seed_results.values()
        ),
        "finite_metrics": len(flat_rows) == 8
        and all(
            _finite(row.get(field))
            for row in flat_rows
            for field in (
                "auc",
                "source_anchor_auc",
                "auc_delta_from_r4_w2",
                "mean_probability",
                "max_abs_probability_delta_from_r4_w2",
                "mean_abs_probability_delta_from_r4_w2",
            )
        ),
        "no_training_or_data_generation": len(flat_rows) == 8
        and all(
            row.get("training_performed") is False
            and row.get("data_generation_performed") is False
            for row in flat_rows
        ),
        "artifact_sha256_fields_present": len(flat_rows) == 8
        and all(
            _is_sha256(row.get(field))
            for row in flat_rows
            for field in (
                "checkpoint_sha256",
                "feature_sha256",
                "label_sha256",
                "metadata_sha256",
                "descriptor_sha256",
                "runtime_structure_window_sha256",
                "probability_sha256",
            )
        ),
    }

    research_checks = _research_checks(seed_results)
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_dialga_runtime_e4_d4_protocol_invalid"
        diagnosis = "protocol_invalid"
        next_action = (
            "repair only the D4 source binding or inference implementation; keep "
            "all checkpoints, caches, windows, and thresholds frozen"
        )
    else:
        status = "pass"
        diagnosis, decision, next_action = _diagnose(research_checks)

    return {
        "run_id": run_id,
        "task": "innovation1_dialga_runtime_e4_d4_factorial",
        "cipher": "Dialga-128",
        "status": status,
        "decision": decision,
        "diagnosis": diagnosis,
        "thresholds": {
            "chance_auc": 0.5,
            "anchor_excess_retention_fraction": RETENTION_FRACTION,
            "source_auc_tolerance": SOURCE_AUC_TOLERANCE,
        },
        "seed_results": seed_results,
        "factor_effects": {
            seed: {
                key: value
                for key, value in result.items()
                if key.endswith("_effect") or key == "interaction"
            }
            for seed, result in seed_results.items()
        },
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "claim_scope": (
            "Dialga-128 D1-checkpoint local inference-only 2x2 audit over existing "
            "prefix-r4/r5 validation caches and runtime round_start 2/3; no new "
            "training, data, formal scale, attack, SOTA, or universal-SPN claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "retrain or reselect checkpoints inside D4",
            "generate new data or increase samples, pairs, epochs, rounds, seeds, or keys",
            "launch remote GPU scale before acting on the isolated factor",
            "combine an input-difference change with a network redesign",
        ],
    }


def _result_row(
    *,
    seed: int,
    condition: str,
    auc: float,
    anchor_auc: float,
    probabilities: np.ndarray,
    anchor_probability: np.ndarray,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_metadata: dict[str, Any],
    dataset_hashes: dict[str, dict[str, str]],
    structure: RuntimeSpnStructure,
    source_hashes: dict[str, str],
    descriptor_name: str,
    descriptor_path: str,
    descriptor_sha256: str,
    parameter_count: int,
    model_options: dict[str, Any],
) -> dict[str, Any]:
    data_source, data_rounds, runtime_round_start = CONDITION_SPEC[condition]
    hashes = dataset_hashes[data_source]
    return {
        "seed": seed,
        "condition": condition,
        "cipher": "Dialga-128",
        "data_source": data_source,
        "data_rounds": data_rounds,
        "runtime_round_start": runtime_round_start,
        "runtime_loaded_rounds": 2,
        "auc": auc,
        "source_anchor_auc": float(anchor_auc),
        "auc_delta_from_r4_w2": auc - float(anchor_auc),
        "chance_excess_retention_ratio": _retention_ratio(auc, anchor_auc),
        "max_abs_probability_delta_from_r4_w2": float(
            np.max(np.abs(anchor_probability - probabilities))
        ),
        "mean_abs_probability_delta_from_r4_w2": float(
            np.mean(np.abs(anchor_probability - probabilities))
        ),
        "mean_probability": float(probabilities.mean()),
        "probability_sha256": hashlib.sha256(probabilities.tobytes()).hexdigest(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_selected": checkpoint_metadata.get("selected_checkpoint"),
        "checkpoint_reported_seed": checkpoint_metadata.get("seed"),
        "checkpoint_best_metric": checkpoint_metadata.get("best_checkpoint_metric"),
        "strict_state_dict_load": True,
        "feature_sha256": hashes["feature_sha256"],
        "label_sha256": hashes["label_sha256"],
        "metadata_sha256": hashes["metadata_sha256"],
        **source_hashes,
        "source_d1_verified": True,
        "source_d3_verified": True,
        "source_d1_decision": D1_DECISION,
        "source_d3_decision": D3_DECISION,
        "descriptor_name": descriptor_name,
        "descriptor_path": descriptor_path,
        "descriptor_sha256": descriptor_sha256,
        "runtime_structure_mode": "correct",
        "relation_mode": "true",
        "runtime_structure_window_sha256": structure.window_sha256(),
        "runtime_structure_transition_sha256s": list(structure.transition_sha256s()),
        "runtime_structure_unique_transition_count": structure.unique_transition_count,
        "samples_total": 2048,
        "validation_seed": 10000 + seed,
        "input_bits": 1024,
        "pair_bits": 256,
        "pairs_per_sample": 4,
        "input_difference": 0x40,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "validation_key": VALIDATION_KEY,
        "parameter_count": parameter_count,
        "model_options": model_options,
        "training_performed": False,
        "data_generation_performed": False,
    }


def _seed_result(group: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    aucs = {
        condition: _row_float(
            group.get(condition, [None])[0]
            if len(group.get(condition, ())) == 1
            else None,
            "auc",
        )
        for condition in EXPECTED_CONDITIONS
    }
    anchor = aucs["r4_w2"]
    result: dict[str, Any] = {
        f"{condition}_auc": auc for condition, auc in aucs.items()
    }
    anchor_row = group["r4_w2"][0] if len(group.get("r4_w2", ())) == 1 else None
    result["source_anchor_auc"] = _row_float(anchor_row, "source_anchor_auc")
    result["retention_threshold"] = _retention_threshold(anchor)
    for condition in EXPECTED_CONDITIONS[1:]:
        result[f"{condition}_retention_ratio"] = _retention_ratio(
            aucs[condition], anchor
        )
    result.update(
        {
            "data_at_w2_effect": _difference(aucs["r5_w2"], aucs["r4_w2"]),
            "data_at_w3_effect": _difference(aucs["r5_w3"], aucs["r4_w3"]),
            "window_at_r4_effect": _difference(aucs["r4_w3"], aucs["r4_w2"]),
            "window_at_r5_effect": _difference(aucs["r5_w3"], aucs["r5_w2"]),
        }
    )
    left = result["window_at_r5_effect"]
    right = result["window_at_r4_effect"]
    result["interaction"] = _difference(left, right)
    return result


def _research_checks(seed_results: dict[str, dict[str, Any]]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        result = seed_results[str(seed)]
        anchor = result["r4_w2_auc"]
        threshold = result["retention_threshold"]
        checks[f"seed{seed}_anchor_above_chance"] = bool(
            anchor is not None and anchor > 0.5
        )
        for condition in EXPECTED_CONDITIONS[1:]:
            value = result[f"{condition}_auc"]
            checks[f"seed{seed}_{condition}_retains_half_anchor_excess"] = bool(
                value is not None and threshold is not None and value >= threshold
            )
        b = result["r4_w3_auc"]
        c = result["r5_w2_auc"]
        d = result["r5_w3_auc"]
        joint_anchor = max(value for value in (b, c) if value is not None)
        joint_threshold = _retention_threshold(joint_anchor)
        checks[f"seed{seed}_r5_w3_retains_half_best_single_factor_excess"] = bool(
            d is not None and joint_threshold is not None and d >= joint_threshold
        )
    return checks


def _diagnose(checks: dict[str, bool]) -> tuple[str, str, str]:
    b = [
        checks[f"seed{seed}_r4_w3_retains_half_anchor_excess"]
        for seed in EXPECTED_SEEDS
    ]
    c = [
        checks[f"seed{seed}_r5_w2_retains_half_anchor_excess"]
        for seed in EXPECTED_SEEDS
    ]
    d = [
        checks[f"seed{seed}_r5_w3_retains_half_best_single_factor_excess"]
        for seed in EXPECTED_SEEDS
    ]
    if all(b) and not any(c) and not any(d):
        return (
            "fifth_round_data_signal_loss",
            "innovation1_dialga_runtime_e4_d4_data_depth_isolated",
            "keep Runtime-E4 frozen and run one tiny same-budget prefix-r5 input-difference screen with the exact D4 controls; do not redesign the network or scale samples yet",
        )
    if not any(b) and all(c):
        return (
            "runtime_window_incompatibility",
            "innovation1_dialga_runtime_e4_d4_window_incompatibility_isolated",
            "implement an independently useful representation with residual or gated topology messages, then compare it against frozen Runtime-E4 on the same D1/D3 caches",
        )
    if not any(b) and not any(c):
        return (
            "both_data_and_window_degrade",
            "innovation1_dialga_runtime_e4_d4_both_factors_degrade",
            "run separate single-variable local experiments: first a fixed-network r5 input-difference screen, then a fixed-data residual topology processor; do not combine or scale them",
        )
    if all(b) and all(c) and not any(d):
        return (
            "joint_data_window_interaction",
            "innovation1_dialga_runtime_e4_d4_joint_interaction_isolated",
            "redesign the runtime processor with a residual/gated topology path and retest only the r5_w3 interaction cell against the same-budget anchor",
        )
    if all(b) and all(c) and all(d):
        return (
            "frozen_transfer_supported",
            "innovation1_dialga_runtime_e4_d4_frozen_transfer_supported",
            "audit D3 optimization and initialization under the frozen protocol because the D1 checkpoint retains both factors; do not increase data before locating the training instability",
        )
    return (
        "mixed_seed_factor_response",
        "innovation1_dialga_runtime_e4_d4_mixed_seed_response",
        "do not select a data or network route yet; preregister one independent validation-key replication of the same frozen 2x2 audit",
    )


def _condition_contract(grouped: dict[int, dict[str, list[dict[str, Any]]]]) -> bool:
    return all(
        row.get("data_source") == data_source
        and row.get("data_rounds") == data_rounds
        and row.get("runtime_round_start") == runtime_round_start
        and row.get("runtime_loaded_rounds") == 2
        and row.get("runtime_structure_mode") == "correct"
        and row.get("relation_mode") == "true"
        for seed in EXPECTED_SEEDS
        for condition, (
            data_source,
            data_rounds,
            runtime_round_start,
        ) in CONDITION_SPEC.items()
        for row in (grouped[seed][condition][0],)
    )


def _descriptor_contract(grouped: dict[int, dict[str, list[dict[str, Any]]]]) -> bool:
    rows = [
        grouped[seed][condition][0]
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    ]
    if not (
        len({row.get("descriptor_sha256") for row in rows}) == 1
        and all(
            row.get("descriptor_name")
            == "Dialga-128 20-round heterogeneous runtime SPN structure"
            and str(row.get("descriptor_path", "")).endswith(
                "configs/runtime/spn/dialga128.json"
            )
            and row.get("runtime_structure_unique_transition_count") == 2
            and len(row.get("runtime_structure_transition_sha256s", ())) == 2
            for row in rows
        )
    ):
        return False
    hashes_by_window = {
        start: {
            row.get("runtime_structure_window_sha256")
            for row in rows
            if row.get("runtime_round_start") == start
        }
        for start in (2, 3)
    }
    return all(len(values) == 1 for values in hashes_by_window.values()) and (
        hashes_by_window[2] != hashes_by_window[3]
    )


def _dataset_hash_contract(grouped: dict[int, dict[str, list[dict[str, Any]]]]) -> bool:
    for seed in EXPECTED_SEEDS:
        rows = {
            condition: grouped[seed][condition][0] for condition in EXPECTED_CONDITIONS
        }
        for left, right in (("r4_w2", "r4_w3"), ("r5_w2", "r5_w3")):
            if any(
                rows[left].get(field) != rows[right].get(field)
                for field in ("feature_sha256", "label_sha256", "metadata_sha256")
            ):
                return False
        if rows["r4_w2"].get("feature_sha256") == rows["r5_w2"].get("feature_sha256"):
            return False
        if rows["r4_w2"].get("metadata_sha256") == rows["r5_w2"].get("metadata_sha256"):
            return False
        if rows["r4_w2"].get("label_sha256") != rows["r5_w2"].get("label_sha256"):
            return False
    return True


def _row_has_frozen_validation_protocol(row: dict[str, Any]) -> bool:
    expected = CONDITION_SPEC.get(str(row.get("condition")))
    return bool(
        expected
        and row.get("cipher") == "Dialga-128"
        and row.get("data_rounds") == expected[1]
        and row.get("samples_total") == 2048
        and row.get("validation_seed") == 10000 + int(row.get("seed", -1))
        and row.get("input_bits") == 1024
        and row.get("pair_bits") == 256
        and row.get("pairs_per_sample") == 4
        and row.get("input_difference") == 0x40
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == "independent_pairs"
        and row.get("validation_key") == VALIDATION_KEY
    )


def _validate_dataset_metadata(
    metadata: dict[str, Any], seed: int, rounds: int
) -> None:
    expected = {
        "cipher": "Dialga-128",
        "rounds": rounds,
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
        raise ValueError(f"Dialga r{rounds} cache does not match frozen D4 protocol")


def _same_seed_field(
    grouped: dict[int, dict[str, list[dict[str, Any]]]], field: str
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


def _retention_threshold(anchor: float | None) -> float | None:
    if anchor is None or not math.isfinite(anchor) or anchor <= 0.5:
        return None
    return 0.5 + RETENTION_FRACTION * (anchor - 0.5)


def _retention_ratio(value: float | None, anchor: float | None) -> float:
    if value is None or anchor is None or anchor <= 0.5:
        return math.nan
    return (value - 0.5) / (anchor - 0.5)


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _row_float(row: dict[str, Any] | None, field: str) -> float | None:
    value = row.get(field) if row is not None else None
    return float(value) if _finite(value) else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "CONDITION_SPEC",
    "EXPECTED_CONDITIONS",
    "RETENTION_FRACTION",
    "adjudicate_factorial_dialga",
    "evaluate_factorial_dialga",
]
