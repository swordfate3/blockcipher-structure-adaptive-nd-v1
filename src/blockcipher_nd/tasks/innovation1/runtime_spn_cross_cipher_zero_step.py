from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_counterfactual import (
    file_sha256,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


EXPECTED_SEEDS = (0, 1)
EXPECTED_CONDITIONS = (
    "true_source_true_target",
    "corrupted_source_true_target",
    "true_source_corrupted_target",
    "true_source_no_target",
)
SOURCE_MODELS = {
    "true": "gift64_runtime_e4_equivariant_true",
    "corrupted": "gift64_runtime_e4_equivariant_corrupted",
}
TARGET_MODELS = {
    "true": "skinny64_runtime_e4_equivariant_true",
    "corrupted": "skinny64_runtime_e4_equivariant_corrupted",
    "independent": "skinny64_runtime_e4_equivariant_independent",
}
FROZEN_MODEL_OPTIONS = {
    "dropout": 0.0,
    "pair_embedding_dim": 128,
    "processor_steps": 2,
    "sbox_context_mode": "late_pair",
}
PARAMETER_COUNT = 442_466
TARGET_VALIDATION_KEY = 0x1111111111111111
AUC_FLOOR = 0.52
MARGIN_FLOOR = 0.005
PROBABILITY_DELTA_FLOOR = 1e-6


def evaluate_zero_step_seed(
    *,
    seed: int,
    source_rows: list[dict[str, Any]],
    dataset: DifferentialDataset,
    feature_path: Path,
    label_path: Path,
    metadata_path: Path,
    target_results_path: Path,
    target_validation_key: int,
    batch_size: int = 256,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unsupported seed: {seed}")
    _validate_target_dataset(seed, dataset)
    if target_validation_key != TARGET_VALIDATION_KEY:
        raise ValueError("target validation key does not match X1")
    sources = {
        role: _source_row(source_rows, seed, model)
        for role, model in SOURCE_MODELS.items()
    }
    for row in sources.values():
        _validate_source_row(row, seed)
    options = dict(sources["true"]["training"]["model_options"])
    if options != FROZEN_MODEL_OPTIONS:
        raise ValueError("true source checkpoint model options do not match X1")
    if dict(sources["corrupted"]["training"]["model_options"]) != options:
        raise ValueError("source checkpoint model options differ")

    source_payloads: dict[str, dict[str, Any]] = {}
    source_paths: dict[str, Path] = {}
    for role, row in sources.items():
        checkpoint_path = Path(row["training"]["checkpoint_output"])
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("state_dict"), dict
        ):
            raise ValueError("source checkpoint must contain a state_dict")
        source_payloads[role] = payload
        source_paths[role] = checkpoint_path

    panel = (
        ("true_source_true_target", "true", "true"),
        ("corrupted_source_true_target", "corrupted", "true"),
        ("true_source_corrupted_target", "true", "corrupted"),
        ("true_source_no_target", "true", "independent"),
    )
    probabilities: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for condition, source_role, target_mode in panel:
        model = build_model(
            TARGET_MODELS[target_mode],
            input_bits=int(dataset.features.shape[1]),
            hidden_bits=64,
            pair_bits=128,
            structure="SPN",
            model_options=options,
        )
        model.load_state_dict(source_payloads[source_role]["state_dict"], strict=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != PARAMETER_COUNT:
            raise ValueError("target adapter parameter count does not match X1")
        probabilities[condition] = predict_binary_probabilities(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
        )
        target_structure_sha256 = _structure_sha256(model.runtime_structure)
        target_relation_mode = str(model.relation_mode)
        metadata[condition] = {
            "source_role": source_role,
            "target_mode": target_mode,
            "checkpoint_path": str(source_paths[source_role]),
            "checkpoint_sha256": file_sha256(source_paths[source_role]),
            "checkpoint_selected": source_payloads[source_role]
            .get("metadata", {})
            .get("selected_checkpoint"),
            "target_structure_sha256": target_structure_sha256,
            "target_relation_mode": target_relation_mode,
            "target_intervention_sha256": _intervention_sha256(
                target_structure_sha256,
                target_relation_mode,
            ),
            "parameter_count": parameter_count,
        }

    labels = np.asarray(dataset.labels, dtype=np.float32)
    aucs = {
        condition: binary_auc(labels, values)
        for condition, values in probabilities.items()
    }
    reference = probabilities["true_source_true_target"]
    feature_sha256 = file_sha256(feature_path)
    label_sha256 = file_sha256(label_path)
    metadata_sha256 = file_sha256(metadata_path)
    target_results_sha256 = file_sha256(target_results_path)
    return [
        {
            "seed": seed,
            "condition": condition,
            **metadata[condition],
            "auc": aucs[condition],
            "candidate_minus_condition_auc": (
                0.0
                if condition == "true_source_true_target"
                else aucs["true_source_true_target"] - aucs[condition]
            ),
            "max_abs_probability_delta_from_candidate": float(
                np.max(np.abs(reference - probabilities[condition]))
            ),
            "mean_probability": float(probabilities[condition].mean()),
            "probability_sha256": hashlib.sha256(
                probabilities[condition].tobytes()
            ).hexdigest(),
            "feature_path": str(feature_path),
            "feature_sha256": feature_sha256,
            "label_path": str(label_path),
            "label_sha256": label_sha256,
            "metadata_path": str(metadata_path),
            "metadata_sha256": metadata_sha256,
            "target_results_path": str(target_results_path),
            "target_results_sha256": target_results_sha256,
            "target_validation_key": target_validation_key,
            "source_protocol_verified": True,
            "target_dataset_metadata_verified": True,
            "samples_total": int(len(dataset.labels)),
            "input_bits": int(dataset.features.shape[1]),
            "pairs_per_sample": 4,
            "source_cipher": "GIFT-64",
            "source_rounds": 6,
            "target_cipher": "SKINNY-64/64",
            "target_rounds": 7,
            "target_difference": 0x2000,
            "negative_mode": "encrypted_random_plaintexts",
            "model_options": options,
            "strict_state_dict_load": True,
            "training_performed": False,
        }
        for condition in EXPECTED_CONDITIONS
    ]


def adjudicate_zero_step_panel(
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
        and set(grouped) == set(EXPECTED_SEEDS),
        "same_target_features_within_seed": complete
        and _same_seed_field(grouped, "feature_sha256"),
        "same_target_labels_within_seed": complete
        and _same_seed_field(grouped, "label_sha256"),
        "same_target_metadata_within_seed": complete
        and _same_seed_field(grouped, "metadata_sha256"),
        "same_target_result_provenance_within_seed": complete
        and _same_seed_field(grouped, "target_results_sha256"),
        "true_checkpoint_shared_across_target_controls": complete
        and all(
            len(
                {
                    grouped[seed][condition][0].get("checkpoint_sha256")
                    for condition in (
                        "true_source_true_target",
                        "true_source_corrupted_target",
                        "true_source_no_target",
                    )
                }
            )
            == 1
            for seed in EXPECTED_SEEDS
        ),
        "source_checkpoint_control_is_distinct": complete
        and all(
            grouped[seed]["true_source_true_target"][0].get("checkpoint_sha256")
            != grouped[seed]["corrupted_source_true_target"][0].get("checkpoint_sha256")
            for seed in EXPECTED_SEEDS
        ),
        "target_structure_modes_are_exact": complete
        and all(_target_modes_exact(grouped[seed]) for seed in EXPECTED_SEEDS),
        "target_structures_are_seed_invariant": complete
        and all(
            grouped[0][condition][0].get("target_structure_sha256")
            == grouped[1][condition][0].get("target_structure_sha256")
            for condition in EXPECTED_CONDITIONS
        ),
        "target_interventions_are_seed_invariant": complete
        and all(
            grouped[0][condition][0].get("target_intervention_sha256")
            == grouped[1][condition][0].get("target_intervention_sha256")
            for condition in EXPECTED_CONDITIONS
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
        "frozen_target_protocol": all(
            row.get("samples_total") == 2048
            and row.get("input_bits") == 512
            and row.get("pairs_per_sample") == 4
            and row.get("source_cipher") == "GIFT-64"
            and row.get("source_rounds") == 6
            and row.get("target_cipher") == "SKINNY-64/64"
            and row.get("target_rounds") == 7
            and row.get("target_difference") == 0x2000
            and row.get("target_validation_key") == TARGET_VALIDATION_KEY
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("source_protocol_verified") is True
            and row.get("target_dataset_metadata_verified") is True
            for row in rows
        ),
        "finite_metrics": all(
            _finite(row.get("auc"))
            and _finite(row.get("mean_probability"))
            and _finite(row.get("max_abs_probability_delta_from_candidate"))
            for row in rows
        ),
        "no_training_performed": all(
            row.get("training_performed") is False for row in rows
        ),
        "sha256_fields_present": all(
            _is_sha256(row.get(field))
            for row in rows
            for field in (
                "checkpoint_sha256",
                "feature_sha256",
                "label_sha256",
                "metadata_sha256",
                "target_results_sha256",
                "target_structure_sha256",
                "target_intervention_sha256",
                "probability_sha256",
            )
        ),
    }

    research_checks: dict[str, bool] = {}
    sensitivity_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        result = seed_results[str(seed)]
        research_checks[f"seed{seed}_candidate_auc_at_least_0p52"] = bool(
            result["candidate_auc"] is not None and result["candidate_auc"] >= AUC_FLOOR
        )
        for control in ("source", "target", "no_topology"):
            research_checks[f"seed{seed}_beats_{control}_by_0p005"] = bool(
                result[f"candidate_minus_{control}_auc"] is not None
                and result[f"candidate_minus_{control}_auc"] >= MARGIN_FLOOR
            )
            sensitivity_checks[f"seed{seed}_{control}_probabilities_change"] = bool(
                result[f"{control}_probability_delta"] is not None
                and result[f"{control}_probability_delta"] > PROBABILITY_DELTA_FLOOR
            )

    if not all(protocol_checks.values()) or not all(sensitivity_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_zero_step_protocol_or_intervention_invalid"
        next_action = "repair only the inference audit; do not change checkpoints, caches, or thresholds"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_runtime_spn_zero_step_topology_use_supported"
        next_action = (
            "after RTG2-B adjudication, rank one new local runtime-shared adaptation "
            "hypothesis against the frozen historical E4 anchor"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_runtime_spn_topology_sensitive_not_zero_step_discriminative"
        )
        next_action = "retain topology sensitivity only; do not train, scale, or claim zero-shot transfer"

    return {
        "run_id": run_id,
        "task": "innovation1_runtime_spn_gift_to_skinny_zero_step_topology_x1",
        "status": status,
        "decision": decision,
        "thresholds": {
            "candidate_auc": AUC_FLOOR,
            "auc_margin": MARGIN_FLOOR,
            "probability_delta": PROBABILITY_DELTA_FLOOR,
        },
        "seed_results": seed_results,
        "protocol_checks": protocol_checks,
        "sensitivity_checks": sensitivity_checks,
        "research_checks": research_checks,
        "claim_scope": (
            "GIFT-to-SKINNY same-checkpoint zero-step runtime-topology diagnostic; "
            "no target training, adaptation, scale, attack, SOTA, or breakthrough claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "launch remote GPU work from X1",
            "add target training, samples, seeds, epochs, or calibration",
            "reuse X1 as a rescue for RTG2-B",
            "claim general cross-SPN zero-shot transfer unless every frozen gate passes",
        ],
    }


def _source_row(rows: list[dict[str, Any]], seed: int, model: str) -> dict[str, Any]:
    matches = [
        row for row in rows if row.get("seed") == seed and row.get("model") == model
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one source row for seed={seed} model={model}, got {len(matches)}"
        )
    return matches[0]


def validate_target_result_rows(rows: list[dict[str, Any]], seed: int) -> int:
    if len(rows) != 3:
        raise ValueError(f"expected three target attribution rows, got {len(rows)}")
    expected_models = set(TARGET_MODELS.values())
    if {row.get("model") for row in rows} != expected_models:
        raise ValueError("target attribution models do not match X1")
    expected = {
        "cipher": "SKINNY-64/64",
        "rounds": 7,
        "seed": seed,
        "pairs_per_sample": 4,
        "input_difference": 0x2000,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "structure": "SPN",
        "validation_key": TARGET_VALIDATION_KEY,
    }
    for row in rows:
        if any(row.get(field) != value for field, value in expected.items()):
            raise ValueError("target attribution protocol does not match X1")
        training = row.get("training", {})
        validation = row.get("validation", {})
        if (
            training.get("input_bits") != 512
            or training.get("validation_rows") != 2048
            or training.get("model_options") != FROZEN_MODEL_OPTIONS
            or validation.get("samples_total") != 2048
            or validation.get("samples_per_class") != 1024
        ):
            raise ValueError("target attribution validation geometry does not match X1")
    return TARGET_VALIDATION_KEY


def _validate_source_row(row: dict[str, Any], seed: int) -> None:
    expected = {
        "cipher": "GIFT-64",
        "rounds": 6,
        "seed": seed,
        "pairs_per_sample": 4,
        "input_difference": 0x40,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "structure": "SPN",
    }
    if any(row.get(field) != value for field, value in expected.items()):
        raise ValueError("source checkpoint protocol does not match X1")
    training = row.get("training", {})
    validation = row.get("validation", {})
    if (
        training.get("input_bits") != 512
        or training.get("validation_rows") != 2048
        or training.get("selected_checkpoint") != "best"
        or validation.get("samples_total") != 2048
        or validation.get("samples_per_class") != 1024
    ):
        raise ValueError("source checkpoint validation geometry does not match X1")


def _validate_target_dataset(seed: int, dataset: DifferentialDataset) -> None:
    expected = {
        "cipher": "SKINNY-64/64",
        "rounds": 7,
        "seed": 10000 + seed,
        "input_difference": 0x2000,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "samples_total": 2048,
        "samples_per_class": 1024,
        "positive_rows": 1024,
        "negative_rows": 1024,
        "input_bits": 512,
        "structure": "SPN",
    }
    metadata = dataset.metadata
    if any(metadata.get(field) != value for field, value in expected.items()):
        raise ValueError("target validation cache metadata does not match X1")
    if dataset.features.shape != (2048, 512) or dataset.labels.shape != (2048,):
        raise ValueError("target validation arrays do not match X1")


def _seed_result(group: dict[str, list[dict[str, Any]]]) -> dict[str, float | None]:
    values = {
        condition: rows[0] if len(rows) == 1 else None
        for condition, rows in (
            (condition, group.get(condition, [])) for condition in EXPECTED_CONDITIONS
        )
    }
    candidate = _row_float(values["true_source_true_target"], "auc")
    source = _row_float(values["corrupted_source_true_target"], "auc")
    target = _row_float(values["true_source_corrupted_target"], "auc")
    no_topology = _row_float(values["true_source_no_target"], "auc")
    return {
        "candidate_auc": candidate,
        "corrupted_source_auc": source,
        "corrupted_target_auc": target,
        "no_topology_auc": no_topology,
        "candidate_minus_source_auc": _difference(candidate, source),
        "candidate_minus_target_auc": _difference(candidate, target),
        "candidate_minus_no_topology_auc": _difference(candidate, no_topology),
        "source_probability_delta": _row_float(
            values["corrupted_source_true_target"],
            "max_abs_probability_delta_from_candidate",
        ),
        "target_probability_delta": _row_float(
            values["true_source_corrupted_target"],
            "max_abs_probability_delta_from_candidate",
        ),
        "no_topology_probability_delta": _row_float(
            values["true_source_no_target"],
            "max_abs_probability_delta_from_candidate",
        ),
    }


def _target_modes_exact(group: dict[str, list[dict[str, Any]]]) -> bool:
    expected = {
        "true_source_true_target": ("true", "true"),
        "corrupted_source_true_target": ("true", "true"),
        "true_source_corrupted_target": ("corrupted", "true"),
        "true_source_no_target": ("independent", "independent"),
    }
    try:
        structure_hashes = {
            condition: group[condition][0]["target_structure_sha256"]
            for condition in EXPECTED_CONDITIONS
        }
        intervention_hashes = {
            condition: group[condition][0]["target_intervention_sha256"]
            for condition in EXPECTED_CONDITIONS
        }
    except (IndexError, KeyError):
        return False
    return (
        all(
            (
                group[condition][0].get("target_mode"),
                group[condition][0].get("target_relation_mode"),
            )
            == modes
            for condition, modes in expected.items()
        )
        and structure_hashes["true_source_true_target"]
        == structure_hashes["corrupted_source_true_target"]
        == structure_hashes["true_source_no_target"]
        and structure_hashes["true_source_true_target"]
        != structure_hashes["true_source_corrupted_target"]
        and intervention_hashes["true_source_true_target"]
        == intervention_hashes["corrupted_source_true_target"]
        and len(
            {
                intervention_hashes["true_source_true_target"],
                intervention_hashes["true_source_corrupted_target"],
                intervention_hashes["true_source_no_target"],
            }
        )
        == 3
    )


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


def _structure_sha256(structure: Any) -> str:
    digest = hashlib.sha256()
    for field in (
        "cell_membership",
        "bit_role",
        "sbox_truth_bits",
        "linear_matrices",
        "inverse_linear_matrices",
    ):
        tensor = getattr(structure, field).detach().cpu().contiguous()
        digest.update(field.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _intervention_sha256(structure_sha256: str, relation_mode: str) -> str:
    return hashlib.sha256(
        f"{structure_sha256}:{relation_mode}".encode("ascii")
    ).hexdigest()


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _row_float(row: dict[str, Any] | None, field: str) -> float | None:
    if row is None or not _finite(row.get(field)):
        return None
    return float(row[field])


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "EXPECTED_CONDITIONS",
    "FROZEN_MODEL_OPTIONS",
    "TARGET_VALIDATION_KEY",
    "adjudicate_zero_step_panel",
    "evaluate_zero_step_seed",
    "validate_target_result_rows",
]
