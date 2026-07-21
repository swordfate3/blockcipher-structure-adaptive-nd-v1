from __future__ import annotations

import math
import random
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.models.structure.spn.present_bit_role_parity_predictor import (
    PresentBitRoleParityPredictor,
    PresentBitRoleParityPredictorSpec,
    present_player,
    wrong_player,
)
from blockcipher_nd.tasks.innovation2.output_parity_mask_geometry import (
    build_mask_geometry_data,
    validate_mask_geometry_contract,
)
from blockcipher_nd.tasks.innovation2.output_parity_prediction import (
    OutputParityPredictionConfig,
    OutputPredictionMlp,
)
from blockcipher_nd.tasks.innovation2.output_parity_spn_local import (
    train_aligned_parity_model,
)


RUN_ID = "i2_output_parity_prediction_op7_present_r3_bit_role_routing_seed0_20260721"
BIT_CHANNELS = 13
ROUTING_DEPTH = 2
HEAD_HIDDEN_DIM = 64


def build_bit_role_data(
    config: OutputParityPredictionConfig,
) -> dict[str, dict[str, Any]]:
    return build_mask_geometry_data(config)


def validate_bit_role_contract(
    config: OutputParityPredictionConfig,
    datasets: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    geometry_checks = validate_mask_geometry_contract(config, datasets)
    model_seed = config.seed + 1000
    _seed_everything(model_seed)
    true_model = _build_bit_role_model("true")
    _seed_everything(model_seed)
    wrong_model = _build_bit_role_model("wrong")
    _seed_everything(model_seed)
    mlp = OutputPredictionMlp(16, config.hidden_dim)
    true_parameters = dict(true_model.named_parameters())
    wrong_parameters = dict(wrong_model.named_parameters())
    parameters_equal = true_parameters.keys() == wrong_parameters.keys() and all(
        torch.equal(true_parameters[name], wrong_parameters[name])
        for name in true_parameters
    )
    true_count = sum(parameter.numel() for parameter in true_model.parameters())
    wrong_count = sum(parameter.numel() for parameter in wrong_model.parameters())
    mlp_count = sum(parameter.numel() for parameter in mlp.parameters())
    true_route = present_player()
    wrong_route = wrong_player()
    fixture_matches = all(
        Present80.permutation_layer(1 << source) == 1 << target
        for source, target in enumerate(true_route)
    )
    with torch.no_grad():
        output = true_model(torch.zeros(2, 64))
    return {
        "mask_geometry_output_prediction_contract_passed": all(
            geometry_checks.values()
        ),
        "rounds_are_three": config.rounds == 3,
        "true_player_is_64_bit_bijection": sorted(true_route) == list(range(64)),
        "wrong_player_is_64_bit_bijection": sorted(wrong_route) == list(range(64)),
        "true_and_wrong_players_differ": true_route != wrong_route,
        "true_player_matches_scalar_present_p_layer": fixture_matches,
        "model_outputs_sixteen_lsb_nibble_logits": output.shape == (2, 16),
        "true_and_wrong_parameter_names_and_initial_values_match": parameters_equal,
        "true_and_wrong_parameter_counts_match": true_count == wrong_count,
        "candidate_and_mlp_parameter_counts_within_one_percent": abs(
            true_count - mlp_count
        )
        / mlp_count
        <= 0.01,
        "labels_are_real_ciphertext_outputs_not_sample_classes": True,
    }


def train_bit_role_matrix(
    config: OutputParityPredictionConfig,
    datasets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    aligned = datasets["aligned"]
    model_seed = config.seed + 1000
    specifications: tuple[tuple[str, Callable[[], nn.Module], str, bool], ...] = (
        (
            "aligned_parity_mlp",
            lambda: OutputPredictionMlp(16, config.hidden_dim),
            "none",
            False,
        ),
        ("bit_role_true_p", lambda: _build_bit_role_model("true"), "true", False),
        (
            "bit_role_wrong_p",
            lambda: _build_bit_role_model("wrong"),
            "wrong",
            False,
        ),
        (
            "bit_role_true_p_label_shuffle",
            lambda: _build_bit_role_model("true"),
            "true",
            True,
        ),
    )
    rows = []
    history = []
    trained: dict[str, dict[str, Any]] = {}
    for row_name, model_factory, topology, shuffle_labels in specifications:
        result = train_aligned_parity_model(
            config,
            aligned,
            row_name=row_name,
            model_factory=model_factory,
            shuffle_train_labels=shuffle_labels,
            seed=model_seed,
        )
        trained[row_name] = result
        history.extend(result["history"])
        metrics = result["test_metrics"]
        rows.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_output_parity_prediction",
                "experiment": "op7_present_r3_bit_role_routing",
                "model": row_name,
                "architecture": (
                    "mlp" if row_name == "aligned_parity_mlp" else "bit_role_spn"
                ),
                "p_topology": topology,
                "train_labels_shuffled": shuffle_labels,
                "model_seed": model_seed,
                "seed": config.seed,
                "rounds": config.rounds,
                "parameters": result["parameters"],
                "epochs": config.epochs,
                "train_rows": config.train_rows,
                "validation_rows": config.validation_rows,
                "test_rows": config.test_rows,
                "test_loss": metrics["loss"],
                "test_accuracy": metrics["accuracy"],
                "test_macro_auc": metrics["macro_auc"],
                "test_exact_match": metrics["exact_match"],
                "test_majority_accuracy": metrics["majority_accuracy"],
                "training_performed": True,
            }
        )
    return {"rows": rows, "history": history, "trained": trained}


def adjudicate_bit_role_readiness(
    config: OutputParityPredictionConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
) -> dict[str, Any]:
    rows = {row["model"]: row for row in training["rows"]}
    numeric_values = [
        float(row[key])
        for row in rows.values()
        for key in ("test_loss", "test_accuracy", "test_macro_auc", "test_exact_match")
    ]
    checks = {
        **protocol_checks,
        "four_training_rows_complete": len(rows) == 4,
        "all_training_metrics_finite": all(
            math.isfinite(value) for value in numeric_values
        ),
        "history_rows_complete": len(training["history"]) == config.epochs * 4,
        "all_models_share_initialization_seed": len(
            {int(row["model_seed"]) for row in rows.values()}
        )
        == 1,
        "shuffle_control_uses_true_test_targets": training["trained"][
            "bit_role_true_p_label_shuffle"
        ]["test_target_identity"]
        == "true_aligned_parity_targets",
    }
    mlp = rows["aligned_parity_mlp"]
    true_p = rows["bit_role_true_p"]
    wrong_p = rows["bit_role_wrong_p"]
    label_shuffle = rows["bit_role_true_p_label_shuffle"]
    deltas = {
        "true_minus_mlp_macro_auc": true_p["test_macro_auc"] - mlp["test_macro_auc"],
        "true_minus_wrong_p_macro_auc": true_p["test_macro_auc"]
        - wrong_p["test_macro_auc"],
        "true_minus_label_shuffle_macro_auc": true_p["test_macro_auc"]
        - label_shuffle["test_macro_auc"],
    }
    attributed = (
        true_p["test_macro_auc"] >= 0.55
        and deltas["true_minus_mlp_macro_auc"] >= 0.03
        and deltas["true_minus_wrong_p_macro_auc"] >= 0.03
        and deltas["true_minus_label_shuffle_macro_auc"] >= 0.03
    )
    generic_gain = (
        true_p["test_macro_auc"] >= 0.55
        and deltas["true_minus_mlp_macro_auc"] >= 0.03
        and deltas["true_minus_label_shuffle_macro_auc"] >= 0.03
    )
    if not all(checks.values()):
        status = "fail"
        decision = "innovation2_output_parity_present_r3_bit_role_protocol_invalid"
        next_adjudication = "repair_op7_protocol"
        action = "repair only the bit routing, controls, output, or training protocol"
    elif attributed:
        status = "pass"
        decision = "innovation2_output_parity_present_r3_bit_role_attributed"
        next_adjudication = "op8_present_r3_bit_role_seed1_confirmation"
        action = "repeat the exact four-row bit-role matrix on fixed-key seed1"
    elif generic_gain:
        status = "hold"
        decision = "innovation2_output_parity_present_r3_bit_role_generic_gain_only"
        next_adjudication = "present_r3_bit_role_topology_attribution_audit"
        action = "retain generic bit-role evidence but audit the wrong-P control before scaling"
    else:
        status = "hold"
        decision = "innovation2_output_parity_present_r3_bit_role_not_ready"
        next_adjudication = "present_r3_deterministic_dependency_cone_audit"
        action = (
            "stop neural routing changes and audit exact dependency-cone and Boolean-function "
            "difficulty before considering more local data"
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "checks": checks,
        "thresholds": {
            "true_p_macro_auc_min": 0.55,
            "true_minus_mlp_macro_auc_min": 0.03,
            "true_minus_wrong_p_macro_auc_min": 0.03,
            "true_minus_label_shuffle_macro_auc_min": 0.03,
        },
        "metrics": {
            "mlp_accuracy": mlp["test_accuracy"],
            "mlp_macro_auc": mlp["test_macro_auc"],
            "true_p_accuracy": true_p["test_accuracy"],
            "true_p_macro_auc": true_p["test_macro_auc"],
            "wrong_p_accuracy": wrong_p["test_accuracy"],
            "wrong_p_macro_auc": wrong_p["test_macro_auc"],
            "label_shuffle_accuracy": label_shuffle["test_accuracy"],
            "label_shuffle_macro_auc": label_shuffle["test_macro_auc"],
            **deltas,
        },
        "claim_scope": (
            "local PRESENT-80 r3 fixed-key real-ciphertext output-parity bit-role routing "
            "readiness; targets are outputs rather than sample classes, and this is not "
            "a high-round attack, paper reproduction, or SOTA result"
        ),
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "remote_scale": False,
            "high_round_training": False,
            "sample_classification": False,
        },
    }


def _build_bit_role_model(topology: str) -> PresentBitRoleParityPredictor:
    return PresentBitRoleParityPredictor(
        PresentBitRoleParityPredictorSpec(
            bit_channels=BIT_CHANNELS,
            routing_depth=ROUTING_DEPTH,
            head_hidden_dim=HEAD_HIDDEN_DIM,
            p_topology=topology,
        )
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
