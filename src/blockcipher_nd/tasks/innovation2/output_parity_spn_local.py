from __future__ import annotations

import math
import random
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from blockcipher_nd.models.structure.spn.present_output_parity_predictor import (
    PresentOutputParityPredictor,
    PresentOutputParityPredictorSpec,
    msb_token_logits_to_lsb_outputs,
    plaintext_bits_to_msb_nibble_tokens,
)
from blockcipher_nd.tasks.innovation2.output_parity_mask_geometry import (
    build_mask_geometry_data,
    validate_mask_geometry_contract,
)
from blockcipher_nd.tasks.innovation2.output_parity_prediction import (
    OutputParityPredictionConfig,
    OutputPredictionMlp,
    multilabel_metrics,
)


RUN_ID = "i2_output_parity_prediction_op6_present_r3_spn_local_readiness_seed0_20260721"
TOKEN_DIM = 28
MIXER_DEPTH = 2


def build_spn_local_data(
    config: OutputParityPredictionConfig,
) -> dict[str, Any]:
    return build_mask_geometry_data(config)


def validate_spn_local_contract(
    config: OutputParityPredictionConfig,
    datasets: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    geometry_checks = validate_mask_geometry_contract(config, datasets)
    model_seed = config.seed + 1000
    _seed_everything(model_seed)
    true_model = _build_spn_model("true")
    _seed_everything(model_seed)
    shuffled_model = _build_spn_model("shuffled")
    _seed_everything(model_seed)
    mlp = OutputPredictionMlp(16, config.hidden_dim)
    true_parameters = dict(true_model.named_parameters())
    shuffled_parameters = dict(shuffled_model.named_parameters())
    parameters_equal = true_parameters.keys() == shuffled_parameters.keys() and all(
        torch.equal(true_parameters[name], shuffled_parameters[name])
        for name in true_parameters
    )
    true_count = sum(parameter.numel() for parameter in true_model.parameters())
    shuffled_count = sum(parameter.numel() for parameter in shuffled_model.parameters())
    mlp_count = sum(parameter.numel() for parameter in mlp.parameters())
    true_sources = true_model.mixer_blocks[0].p_sources
    shuffled_sources = shuffled_model.mixer_blocks[0].p_sources
    fixture_bits = torch.arange(64, dtype=torch.float32).reshape(1, 64)
    tokens = plaintext_bits_to_msb_nibble_tokens(fixture_bits)
    fixture_logits = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    lsb_outputs = msb_token_logits_to_lsb_outputs(fixture_logits)
    return {
        "mask_geometry_output_prediction_contract_passed": all(
            geometry_checks.values()
        ),
        "rounds_are_three": config.rounds == 3,
        "msb_token_first_is_plaintext_nibble15": torch.equal(
            tokens[0, 0], torch.tensor([60.0, 61.0, 62.0, 63.0])
        ),
        "msb_token_last_is_plaintext_nibble0": torch.equal(
            tokens[0, 15], torch.tensor([0.0, 1.0, 2.0, 3.0])
        ),
        "output_order_reverses_to_lsb_first": torch.equal(
            lsb_outputs, torch.arange(15, -1, -1, dtype=torch.float32).reshape(1, 16)
        ),
        "true_and_shuffled_topologies_differ": not torch.equal(
            true_sources, shuffled_sources
        ),
        "true_and_shuffled_parameter_names_and_initial_values_match": parameters_equal,
        "true_and_shuffled_parameter_counts_match": true_count == shuffled_count,
        "candidate_and_mlp_parameter_counts_within_one_percent": abs(
            true_count - mlp_count
        )
        / mlp_count
        <= 0.01,
        "labels_are_real_ciphertext_outputs_not_sample_classes": True,
    }


def train_spn_local_matrix(
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
        ("spn_local_true_p", lambda: _build_spn_model("true"), "true", False),
        (
            "spn_local_shuffled_p",
            lambda: _build_spn_model("shuffled"),
            "shuffled",
            False,
        ),
        (
            "spn_local_true_p_label_shuffle",
            lambda: _build_spn_model("true"),
            "true",
            True,
        ),
    )
    trained: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    rows = []
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
                "experiment": "op6_present_r3_spn_local_readiness",
                "model": row_name,
                "architecture": (
                    "mlp" if row_name == "aligned_parity_mlp" else "spn_local"
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


def adjudicate_spn_local_readiness(
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
            "spn_local_true_p_label_shuffle"
        ]["test_target_identity"]
        == "true_aligned_parity_targets",
    }
    mlp = rows["aligned_parity_mlp"]
    true_p = rows["spn_local_true_p"]
    shuffled_p = rows["spn_local_shuffled_p"]
    label_shuffle = rows["spn_local_true_p_label_shuffle"]
    deltas = {
        "true_minus_mlp_macro_auc": true_p["test_macro_auc"] - mlp["test_macro_auc"],
        "true_minus_shuffled_p_macro_auc": true_p["test_macro_auc"]
        - shuffled_p["test_macro_auc"],
        "true_minus_label_shuffle_macro_auc": true_p["test_macro_auc"]
        - label_shuffle["test_macro_auc"],
    }
    attributed = (
        true_p["test_macro_auc"] >= 0.55
        and deltas["true_minus_mlp_macro_auc"] >= 0.03
        and deltas["true_minus_shuffled_p_macro_auc"] >= 0.03
        and deltas["true_minus_label_shuffle_macro_auc"] >= 0.03
    )
    generic_local_gain = (
        true_p["test_macro_auc"] >= 0.55
        and deltas["true_minus_mlp_macro_auc"] >= 0.03
        and deltas["true_minus_label_shuffle_macro_auc"] >= 0.03
    )
    if not all(checks.values()):
        status = "fail"
        decision = "innovation2_output_parity_present_r3_spn_local_protocol_invalid"
        next_adjudication = "repair_op6_protocol"
        action = "repair only the output, token-order, topology-control, or training protocol"
    elif attributed:
        status = "pass"
        decision = "innovation2_output_parity_present_r3_spn_local_attributed"
        next_adjudication = "op7_present_r3_spn_local_seed1_confirmation"
        action = "repeat the same four-row matrix on independent fixed-key seed1"
    elif generic_local_gain:
        status = "hold"
        decision = "innovation2_output_parity_present_r3_spn_local_generic_gain_only"
        next_adjudication = "present_r3_exact_bit_role_routing_audit"
        action = (
            "retain only generic local-representation evidence and audit exact P-layer "
            "bit-role routing before another neural run"
        )
    else:
        status = "hold"
        decision = "innovation2_output_parity_present_r3_spn_local_not_ready"
        next_adjudication = "present_r3_exact_bit_level_spn_routing"
        action = (
            "stop nibble-adjacency scaling and design exact bit-level S-box/P-layer routing "
            "with unchanged real ciphertext-output targets"
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "checks": checks,
        "thresholds": {
            "true_p_macro_auc_min": 0.55,
            "true_minus_mlp_macro_auc_min": 0.03,
            "true_minus_shuffled_p_macro_auc_min": 0.03,
            "true_minus_label_shuffle_macro_auc_min": 0.03,
        },
        "metrics": {
            "mlp_accuracy": mlp["test_accuracy"],
            "mlp_macro_auc": mlp["test_macro_auc"],
            "true_p_accuracy": true_p["test_accuracy"],
            "true_p_macro_auc": true_p["test_macro_auc"],
            "shuffled_p_accuracy": shuffled_p["test_accuracy"],
            "shuffled_p_macro_auc": shuffled_p["test_macro_auc"],
            "label_shuffle_accuracy": label_shuffle["test_accuracy"],
            "label_shuffle_macro_auc": label_shuffle["test_macro_auc"],
            **deltas,
        },
        "claim_scope": (
            "local PRESENT-80 r3 fixed-key ciphertext-output parity architecture readiness; "
            "the 0/1 targets are real ciphertext outputs rather than sample classes, and "
            "this is not a high-round attack, paper reproduction, or SOTA result"
        ),
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "remote_scale": False,
            "high_round_training": False,
            "sample_classification": False,
        },
    }


def _build_spn_model(topology: str) -> PresentOutputParityPredictor:
    return PresentOutputParityPredictor(
        PresentOutputParityPredictorSpec(
            token_dim=TOKEN_DIM,
            mixer_depth=MIXER_DEPTH,
            p_topology=topology,
        )
    )


def train_aligned_parity_model(
    config: OutputParityPredictionConfig,
    split_data: dict[str, Any],
    *,
    row_name: str,
    model_factory: Callable[[], nn.Module],
    shuffle_train_labels: bool,
    seed: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    model = model_factory().to(config.device)
    train_targets = split_data["train"].parity_targets.copy()
    if shuffle_train_labels:
        rng = np.random.default_rng(960_000 + seed)
        train_targets = train_targets[rng.permutation(len(train_targets))]
    generator = torch.Generator().manual_seed(970_000 + seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(split_data["train"].features),
            torch.from_numpy(train_targets),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_cells = 0
        for features, labels in loader:
            features = features.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * labels.numel()
            total_cells += labels.numel()
        validation_probabilities = _predict(
            model,
            split_data["validation"].features,
            config.batch_size,
            config.device,
        )
        validation_metrics = multilabel_metrics(
            validation_probabilities,
            split_data["validation"].parity_targets,
        )
        history.append(
            {
                "run_id": config.run_id,
                "model": row_name,
                "epoch": epoch,
                "train_loss": total_loss / max(1, total_cells),
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_auc": validation_metrics["macro_auc"],
            }
        )
    test_probabilities = _predict(
        model, split_data["test"].features, config.batch_size, config.device
    )
    return {
        "history": history,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "test_metrics": multilabel_metrics(
            test_probabilities, split_data["test"].parity_targets
        ),
        "test_target_identity": "true_aligned_parity_targets",
    }


def _predict(
    model: nn.Module, features: np.ndarray, batch_size: int, device: str
) -> np.ndarray:
    model.eval()
    outputs = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
