from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_counterfactual import (
    file_sha256,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


EXPECTED_SEEDS = (0, 1)
EXPECTED_CONDITIONS = ("correct_delta_u", "shuffled_delta_u", "delta_v_identity")
MARGIN_FLOOR = 0.005
PROBABILITY_DELTA_FLOOR = 1e-6


class _QueryOverrideModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        query_input_mode: str,
        query_structure: RuntimeSpnStructure | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.query_input_mode = query_input_mode
        self.query_structure = query_structure

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(
            features,
            query_input_mode=self.query_input_mode,
            query_structure=self.query_structure,
        )


def evaluate_same_checkpoint_delta_query(
    *,
    seed: int,
    model_options: dict[str, Any],
    checkpoint_path: Path,
    dataset: DifferentialDataset,
    correct_structure: RuntimeSpnStructure,
    shuffled_structure: RuntimeSpnStructure,
    checkpoint_sha256: str,
    feature_sha256: str,
    label_sha256: str,
    descriptor_sha256: str,
    batch_size: int = 256,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    if model_options.get("cell_input_mode") != "state_triplet_delta_u_query":
        raise ValueError("source checkpoint must use the delta-U query candidate")
    if model_options.get("sbox_context_mode") != "edge_gate":
        raise ValueError("source checkpoint must use the frozen edge gate")
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
        raise ValueError("source checkpoint must contain a state_dict")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.runtime_structure = correct_structure
    parameter_count = sum(value.numel() for value in model.parameters())

    conditions = {
        "correct_delta_u": ("delta_u", None),
        "shuffled_delta_u": ("delta_u", shuffled_structure),
        "delta_v_identity": ("delta_v", None),
    }
    probabilities: dict[str, np.ndarray] = {}
    for condition, (query_mode, query_structure) in conditions.items():
        wrapped = _QueryOverrideModel(
            model,
            query_input_mode=query_mode,
            query_structure=query_structure,
        )
        probabilities[condition] = predict_binary_probabilities(
            wrapped,
            dataset,
            batch_size=batch_size,
            device=device,
        )

    labels = np.asarray(dataset.labels, dtype=np.float32)
    aucs = {
        condition: binary_auc(labels, values)
        for condition, values in probabilities.items()
    }
    reference = probabilities["correct_delta_u"]
    checkpoint_metadata = payload.get("metadata", {})
    sbox_hashes = {
        "correct_delta_u": _tensor_sha256(correct_structure.sbox_truth_bits),
        "shuffled_delta_u": _tensor_sha256(shuffled_structure.sbox_truth_bits),
        "delta_v_identity": "not_used",
    }
    return [
        {
            "seed": seed,
            "condition": condition,
            "auc": aucs[condition],
            "reference_minus_condition_auc": (
                0.0
                if condition == "correct_delta_u"
                else aucs["correct_delta_u"] - aucs[condition]
            ),
            "max_abs_probability_delta_from_reference": float(
                np.max(np.abs(reference - probabilities[condition]))
            ),
            "mean_probability": float(probabilities[condition].mean()),
            "probability_sha256": hashlib.sha256(
                probabilities[condition].tobytes()
            ).hexdigest(),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_selected": checkpoint_metadata.get("selected_checkpoint"),
            "feature_sha256": feature_sha256,
            "label_sha256": label_sha256,
            "descriptor_sha256": descriptor_sha256,
            "query_sbox_truth_sha256": sbox_hashes[condition],
            "descriptor_round_start": 2,
            "descriptor_loaded_rounds": 2,
            "samples_total": int(len(dataset.labels)),
            "input_bits": int(dataset.features.shape[1]),
            "parameter_count": parameter_count,
            "training_performed": False,
            "main_structure_mode": "correct",
            "sbox_context_mode": model_options.get("sbox_context_mode"),
            "cell_input_mode": model_options.get("cell_input_mode"),
        }
        for condition in EXPECTED_CONDITIONS
    ]


def adjudicate_same_checkpoint_delta_query(
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
        "selected_best_checkpoints": all(
            row.get("checkpoint_selected") == "best" for row in rows
        ),
        "exact_descriptor_window": len(rows) == 6
        and len({row.get("descriptor_sha256") for row in rows}) == 1
        and all(
            row.get("descriptor_round_start") == 2
            and row.get("descriptor_loaded_rounds") == 2
            for row in rows
        ),
        "frozen_validation_size": all(
            row.get("samples_total") == 2048 and row.get("input_bits") == 512
            for row in rows
        ),
        "equal_parameter_geometry": len(rows) == 6
        and {row.get("parameter_count") for row in rows} == {458850},
        "query_only_contract": all(
            row.get("main_structure_mode") == "correct"
            and row.get("sbox_context_mode") == "edge_gate"
            and row.get("cell_input_mode") == "state_triplet_delta_u_query"
            for row in rows
        ),
        "distinct_query_ownership": complete
        and all(
            grouped[seed]["correct_delta_u"][0].get("query_sbox_truth_sha256")
            != grouped[seed]["shuffled_delta_u"][0].get(
                "query_sbox_truth_sha256"
            )
            and grouped[seed]["delta_v_identity"][0].get(
                "query_sbox_truth_sha256"
            )
            == "not_used"
            for seed in EXPECTED_SEEDS
        ),
        "finite_metrics": all(
            _finite(row.get("auc"))
            and _finite(row.get("mean_probability"))
            and _finite(row.get("max_abs_probability_delta_from_reference"))
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
                "descriptor_sha256",
                "probability_sha256",
            )
        ),
    }
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        result = seed_results[str(seed)]
        research_checks[f"seed{seed}_beats_shuffled_by_0p005"] = bool(
            result["correct_minus_shuffled_auc"] is not None
            and result["correct_minus_shuffled_auc"] >= MARGIN_FLOOR
        )
        research_checks[f"seed{seed}_beats_identity_by_0p005"] = bool(
            result["correct_minus_identity_auc"] is not None
            and result["correct_minus_identity_auc"] >= MARGIN_FLOOR
        )
        research_checks[f"seed{seed}_shuffled_probabilities_change"] = bool(
            result["shuffled_probability_delta"] is not None
            and result["shuffled_probability_delta"] > PROBABILITY_DELTA_FLOOR
        )
        research_checks[f"seed{seed}_identity_probabilities_change"] = bool(
            result["identity_probability_delta"] is not None
            and result["identity_probability_delta"] > PROBABILITY_DELTA_FLOOR
        )

    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_uknit_delta_query_counterfactual_protocol_invalid"
        next_action = "repair the inference-only audit without changing sources or thresholds"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_uknit_delta_u_same_checkpoint_use_supported"
        next_action = (
            "preregister one same-budget uKNIT transition-window replication; "
            "do not scale samples yet"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_delta_u_training_distribution_only"
        next_action = (
            "close the delta-U query route without scale-up and retain U2-C as the "
            "uKNIT state-input anchor"
        )

    return {
        "run_id": run_id,
        "task": "innovation1_uknit_runtime_e4_delta_u_query_u2g_counterfactual",
        "cipher": "uKNIT-BC",
        "status": status,
        "decision": decision,
        "thresholds": {
            "auc_margin": MARGIN_FLOOR,
            "probability_delta": PROBABILITY_DELTA_FLOOR,
        },
        "seed_results": seed_results,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "claim_scope": (
            "uKNIT-BC U2-F same-checkpoint query-use audit only; no training, "
            "formal scale, attack, cross-cipher, SOTA, or breakthrough claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "train or select another checkpoint inside U2-G",
            "increase samples, epochs, pairs, or seeds",
            "launch remote GPU scale",
            "add DDT, trail, guessed-key, or partial-decryption features",
        ],
    }


def _seed_result(group: dict[str, list[dict[str, Any]]]) -> dict[str, float | None]:
    values: dict[str, dict[str, Any] | None] = {
        condition: rows[0] if len(rows) == 1 else None
        for condition, rows in (
            (condition, group.get(condition, []))
            for condition in EXPECTED_CONDITIONS
        )
    }
    reference = values["correct_delta_u"]
    shuffled = values["shuffled_delta_u"]
    identity = values["delta_v_identity"]
    reference_auc = _row_float(reference, "auc")
    shuffled_auc = _row_float(shuffled, "auc")
    identity_auc = _row_float(identity, "auc")
    return {
        "correct_delta_u_auc": reference_auc,
        "shuffled_delta_u_auc": shuffled_auc,
        "delta_v_identity_auc": identity_auc,
        "correct_minus_shuffled_auc": (
            reference_auc - shuffled_auc
            if reference_auc is not None and shuffled_auc is not None
            else None
        ),
        "correct_minus_identity_auc": (
            reference_auc - identity_auc
            if reference_auc is not None and identity_auc is not None
            else None
        ),
        "shuffled_probability_delta": _row_float(
            shuffled, "max_abs_probability_delta_from_reference"
        ),
        "identity_probability_delta": _row_float(
            identity, "max_abs_probability_delta_from_reference"
        ),
    }


def _same_seed_field(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
    field: str,
) -> bool:
    return all(
        len({grouped[seed][condition][0].get(field) for condition in EXPECTED_CONDITIONS})
        == 1
        for seed in EXPECTED_SEEDS
    )


def _row_float(row: dict[str, Any] | None, field: str) -> float | None:
    value = row.get(field) if row is not None else None
    return float(value) if _finite(value) else None


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "adjudicate_same_checkpoint_delta_query",
    "evaluate_same_checkpoint_delta_query",
    "file_sha256",
]
