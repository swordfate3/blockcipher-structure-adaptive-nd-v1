from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    SelectedOutputResidualCnn,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
    SelectedOutputBitHeadConfig,
    SelectedOutputMlp,
    prepare_selected_output_data,
    validate_selected_output_contract,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID_PREFIX = "i2_output_prediction_ope1_present_r3_r4_selected8_parity"
SELECTED8_PARITY_MASK = SELECTED_MSB_INDICES
SHIFTED_CONTROL_MASK = (1, 3, 9, 11, 33, 35, 41, 43)
MODEL_SPECS = (
    ("selected8_mlp_true_output", "selected8", "mlp", False),
    ("selected8_parity_mlp_true_output", "selected_parity", "mlp", False),
    (
        "selected8_parity_rescnn_true_output",
        "selected_parity",
        "rescnn",
        False,
    ),
    (
        "control8_parity_rescnn_true_output",
        "control_parity",
        "rescnn",
        False,
    ),
    (
        "selected8_parity_rescnn_label_shuffle",
        "selected_parity",
        "rescnn",
        True,
    ),
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class Selected8ParityConfig:
    run_id: str = f"{RUN_ID_PREFIX}_r3_smoke_seed1_20260722"
    mode: str = "smoke"
    rounds: int = 3
    seed: int = 1
    train_rows: int = 64
    test_rows: int = 64
    hidden_dim: int = 256
    rescnn_channels: int = 36
    rescnn_blocks: int = 10
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    data_chunk_rows: int = 32
    selected_msb_indices: tuple[int, ...] = SELECTED_MSB_INDICES
    control_msb_indices: tuple[int, ...] = SHIFTED_CONTROL_MASK
    maximum_parameter_gap: float = 0.005
    minimum_r3_auc: float = 0.520
    minimum_r3_accuracy_margin: float = 0.005
    minimum_r3_shuffle_margin: float = 0.010
    minimum_r4_auc: float = 0.510
    minimum_r4_accuracy_margin: float = 0.005
    minimum_r4_shuffle_margin: float = 0.005
    minimum_r4_geometry_margin: float = 0.005
    minimum_r4_derived_margin: float = 0.005
    minimum_r4_component_margin: float = 0.002
    minimum_r4_mlp_margin: float = 0.002
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "diagnostic"}:
            raise ValueError("invalid OPE1 mode")
        if self.rounds not in {3, 4}:
            raise ValueError("OPE1 is frozen to PRESENT rounds three and four")
        if self.seed != 1:
            raise ValueError("OPE1 is frozen to the second fixed-key seed")
        if self.selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("OPE1 selected positions must match OP10 through OPD1")
        if self.control_msb_indices != SHIFTED_CONTROL_MASK:
            raise ValueError("OPE1 shifted geometry control is frozen")
        if min(
            self.train_rows,
            self.test_rows,
            self.hidden_dim,
            self.rescnn_channels,
            self.rescnn_blocks,
            self.epochs,
            self.batch_size,
            self.data_chunk_rows,
        ) <= 0:
            raise ValueError("row, model, epoch, batch, and chunk values must be positive")

    @classmethod
    def diagnostic(
        cls,
        *,
        rounds: int,
        run_id: str | None = None,
        device: str = "cpu",
    ) -> Selected8ParityConfig:
        return cls(
            run_id=run_id
            or f"{RUN_ID_PREFIX}_r{rounds}_diagnostic_seed1_20260722",
            mode="diagnostic",
            rounds=rounds,
            train_rows=4096,
            test_rows=4096,
            epochs=10,
            batch_size=128,
            data_chunk_rows=1024,
            device=device,
        )


def prepare_selected8_parity_data(
    config: Selected8ParityConfig,
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    base = SelectedOutputBitHeadConfig(
        run_id=config.run_id,
        mode="smoke",
        rounds=config.rounds,
        seed=config.seed,
        train_rows=config.train_rows,
        test_rows=config.test_rows,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        data_chunk_rows=config.data_chunk_rows,
        selected_msb_indices=config.selected_msb_indices,
        device=config.device,
    )
    return prepare_selected_output_data(base, output_root, progress=progress)


def validate_selected8_parity_contract(
    config: Selected8ParityConfig,
    data: dict[str, Any],
) -> dict[str, bool]:
    base = SelectedOutputBitHeadConfig(
        run_id=config.run_id,
        mode="smoke",
        rounds=config.rounds,
        seed=config.seed,
        train_rows=config.train_rows,
        test_rows=config.test_rows,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        data_chunk_rows=config.data_chunk_rows,
        selected_msb_indices=config.selected_msb_indices,
        device=config.device,
    )
    checks = validate_selected_output_contract(base, data)
    counts = parameter_counts(config)
    selected = config.selected_msb_indices
    control = config.control_msb_indices
    checks.update(
        {
            "candidate_is_xor_of_all_eight_frozen_bits": selected
            == SELECTED8_PARITY_MASK,
            "shifted_control_has_same_weight": len(control) == len(selected) == 8,
            "shifted_control_preserves_relative_geometry": all(
                shifted == bit + 1 for bit, shifted in zip(selected, control, strict=True)
            ),
            "candidate_and_control_coordinates_are_disjoint": set(selected).isdisjoint(
                control
            ),
            "single_output_mlp_and_rescnn_are_parameter_matched": abs(
                counts["parity_mlp"] - counts["parity_rescnn"]
            )
            / counts["parity_mlp"]
            <= config.maximum_parameter_gap,
            "five_frozen_model_rows": len(MODEL_SPECS) == 5,
            "labels_are_true_output_parity_not_sample_classes": True,
        }
    )
    return checks


def train_selected8_parity_matrix(
    config: Selected8ParityConfig,
    data: dict[str, Any],
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    train_features = np.array(data["features"][: config.train_rows], copy=True)
    test_features = np.array(data["features"][config.train_rows :], copy=True)
    train_full = np.array(data["full_targets"][: config.train_rows], copy=True)
    test_full = np.array(data["full_targets"][config.train_rows :], copy=True)
    selected_columns = np.asarray(config.selected_msb_indices, dtype=np.int64)
    train_selected = train_full[:, selected_columns]
    test_selected = test_full[:, selected_columns]
    train_parity = parity_targets(train_full, config.selected_msb_indices)
    test_parity = parity_targets(test_full, config.selected_msb_indices)
    train_control = parity_targets(train_full, config.control_msb_indices)
    test_control = parity_targets(test_full, config.control_msb_indices)
    targets = {
        "selected8": (train_selected, test_selected),
        "selected_parity": (train_parity, test_parity),
        "control_parity": (train_control, test_control),
    }
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    scores: dict[str, np.ndarray] = {}
    for model_name, target_mode, architecture, shuffle_labels in MODEL_SPECS:
        train_targets, test_targets = targets[target_mode]
        train_targets = np.array(train_targets, copy=True)
        if shuffle_labels:
            permutation = np.random.default_rng(1_420_000 + config.seed).permutation(
                config.train_rows
            )
            train_targets = train_targets[permutation]
        result = _train_one_model(
            config,
            model_name=model_name,
            target_mode=target_mode,
            architecture=architecture,
            train_features=train_features,
            train_targets=train_targets,
            test_features=test_features,
            test_targets=test_targets,
            output_root=output_root,
            progress=progress,
        )
        scores[model_name] = result.pop("scores")
        rows.extend(result["rows"])
        summaries.append(result["summary"])
        history.extend(result["history"])
        checkpoints.append(result["checkpoint"])
    _attach_comparisons(
        rows,
        selected_scores=scores["selected8_mlp_true_output"],
        selected_targets=test_selected,
        parity_targets_array=test_parity[:, 0],
        selected_bits=config.selected_msb_indices,
    )
    return {
        "rows": rows,
        "summaries": summaries,
        "history": history,
        "checkpoints": checkpoints,
    }


def adjudicate_selected8_parity(
    config: Selected8ParityConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
    *,
    r3_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    indexed = {str(row["model"]): row for row in training["rows"] if "msb_index" not in row}
    candidate = indexed["selected8_parity_rescnn_true_output"]
    r3_checks = {
        "candidate_auc_at_least_0_520": float(candidate["auc"])
        >= config.minimum_r3_auc,
        "accuracy_margin_at_least_0_005": float(
            candidate["accuracy_minus_majority"]
        )
        >= config.minimum_r3_accuracy_margin,
        "candidate_minus_shuffle_at_least_0_010": float(
            candidate["auc_minus_shuffle"]
        )
        >= config.minimum_r3_shuffle_margin,
    }
    r4_checks = {
        "candidate_auc_at_least_0_510": float(candidate["auc"])
        >= config.minimum_r4_auc,
        "accuracy_margin_at_least_0_005": float(
            candidate["accuracy_minus_majority"]
        )
        >= config.minimum_r4_accuracy_margin,
        "candidate_minus_shuffle_at_least_0_005": float(
            candidate["auc_minus_shuffle"]
        )
        >= config.minimum_r4_shuffle_margin,
        "candidate_minus_geometry_at_least_0_005": float(
            candidate["auc_minus_geometry"]
        )
        >= config.minimum_r4_geometry_margin,
        "candidate_minus_derived_at_least_0_005": float(
            candidate["auc_minus_derived"]
        )
        >= config.minimum_r4_derived_margin,
        "candidate_minus_best_component_at_least_0_002": float(
            candidate["auc_minus_best_component"]
        )
        >= config.minimum_r4_component_margin,
        "candidate_minus_mlp_at_least_0_002": float(candidate["auc_minus_mlp"])
        >= config.minimum_r4_mlp_margin,
    }
    execution_checks = {
        "five_models_complete": len(training["summaries"]) == 5,
        "twelve_result_rows_complete": len(training["rows"]) == 12,
        "history_rows_complete": len(training["history"]) == config.epochs * 5,
        "five_checkpoint_hashes_present": len(training["checkpoints"]) == 5
        and all(row.get("sha256") for row in training["checkpoints"]),
        "all_metrics_are_finite": all(
            math.isfinite(float(row[field]))
            for row in training["rows"]
            for field in (
                "threshold_accuracy",
                "majority_accuracy",
                "accuracy_minus_majority",
                "auc",
                "mse",
            )
        ),
        "candidate_has_all_comparisons": all(
            field in candidate
            for field in (
                "mlp_auc",
                "geometry_auc",
                "shuffle_auc",
                "derived_auc",
                "best_component_auc",
            )
        ),
    }
    source_r3_valid = config.rounds == 3 or (
        isinstance(r3_gate, dict)
        and r3_gate.get("status") == "pass"
        and r3_gate.get("decision")
        == "innovation2_selected8_parity_r3_calibrated"
        and all(r3_gate.get("protocol_checks", {}).values())
        and all(r3_gate.get("execution_checks", {}).values())
    )
    all_valid = all(protocol_checks.values()) and all(execution_checks.values())
    if not all_valid:
        status = "fail"
        decision = "innovation2_selected8_parity_protocol_invalid"
        action = "repair only data, target, control, metric, cache, checkpoint, or plotting protocol"
        remote_scale = False
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_selected8_parity_local_readiness_passed"
        action = "run the frozen 4096/4096 and 10-epoch local diagnostic"
        remote_scale = False
    elif config.rounds == 3 and all(r3_checks.values()):
        status = "pass"
        decision = "innovation2_selected8_parity_r3_calibrated"
        action = "run the unchanged r4 local diagnostic with this r3 gate as authority"
        remote_scale = False
    elif config.rounds == 4 and source_r3_valid and all(r4_checks.values()):
        status = "pass"
        decision = "innovation2_selected8_parity_r4_local_supported"
        action = "prepare a separate A6000 2^17/2^16 and 100-epoch formal plan; do not launch before its readiness audit"
        remote_scale = False
    else:
        status = "hold"
        decision = (
            "innovation2_selected8_parity_r3_not_calibrated"
            if config.rounds == 3
            else "innovation2_selected8_parity_r4_not_supported"
        )
        action = "stop this selected8 full-parity route without remote scale, mask search, more epochs, more models, or higher rounds"
        remote_scale = False
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "source_r3_gate_valid": source_r3_valid,
        "metrics": {
            "rounds": config.rounds,
            "candidate_auc": float(candidate["auc"]),
            "candidate_accuracy_minus_majority": float(
                candidate["accuracy_minus_majority"]
            ),
            "mlp_auc": float(candidate["mlp_auc"]),
            "geometry_auc": float(candidate["geometry_auc"]),
            "shuffle_auc": float(candidate["shuffle_auc"]),
            "derived_auc": float(candidate["derived_auc"]),
            "best_component_auc": float(candidate["best_component_auc"]),
            "candidate_minus_mlp_auc": float(candidate["auc_minus_mlp"]),
            "candidate_minus_geometry_auc": float(candidate["auc_minus_geometry"]),
            "candidate_minus_shuffle_auc": float(candidate["auc_minus_shuffle"]),
            "candidate_minus_derived_auc": float(candidate["auc_minus_derived"]),
            "candidate_minus_best_component_auc": float(
                candidate["auc_minus_best_component"]
            ),
            "r3_calibration_checks": r3_checks,
            "r4_feasibility_checks": r4_checks,
        },
        "claim_scope": (
            f"local seed1 fixed-key PRESENT r{config.rounds} direct prediction of the XOR of all eight frozen ciphertext output bits"
            "; not sample classification, integral balance, formal scale, broad cross-key statistics, higher-round evidence, or SOTA"
        ),
        "next_action": {
            "action": action,
            "remote_scale": remote_scale,
            "sample_classification": False,
            "target": "xor_of_eight_preregistered_true_ciphertext_output_bits",
        },
    }


def parity_targets(full_targets: np.ndarray, bits: tuple[int, ...]) -> np.ndarray:
    parity = np.bitwise_xor.reduce(
        np.asarray(full_targets[:, bits], dtype=np.uint8), axis=1
    )
    return parity.astype(np.float32).reshape(-1, 1)


def parameter_counts(config: Selected8ParityConfig) -> dict[str, int]:
    return {
        "selected8_mlp": _count(SelectedOutputMlp(config.hidden_dim, output_bits=8)),
        "parity_mlp": _count(SelectedOutputMlp(config.hidden_dim, output_bits=1)),
        "parity_rescnn": _count(
            SelectedOutputResidualCnn(
                channels=config.rescnn_channels,
                blocks=config.rescnn_blocks,
                output_bits=1,
            )
        ),
    }


def serializable_config(config: Selected8ParityConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["selected_msb_indices"] = list(config.selected_msb_indices)
    payload["control_msb_indices"] = list(config.control_msb_indices)
    return payload


def _train_one_model(
    config: Selected8ParityConfig,
    *,
    model_name: str,
    target_mode: str,
    architecture: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    output_root: Path,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    _seed_everything(1_410_000 + config.seed)
    output_bits = int(train_targets.shape[1])
    model: nn.Module
    if architecture == "mlp":
        model = SelectedOutputMlp(config.hidden_dim, output_bits=output_bits)
    elif architecture == "rescnn":
        model = SelectedOutputResidualCnn(
            channels=config.rescnn_channels,
            blocks=config.rescnn_blocks,
            output_bits=output_bits,
        )
    else:
        raise ValueError(f"unknown OPE1 architecture: {architecture}")
    model.to(config.device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    model_root = output_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    latest_path = model_root / f"{model_name}_latest.pt"
    final_path = model_root / f"{model_name}_final.pt"
    config_hash = _training_config_hash(config, model_name)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if latest_path.exists():
        checkpoint = torch.load(
            latest_path, map_location=config.device, weights_only=False
        )
        if checkpoint.get("config_hash") != config_hash:
            raise ValueError(f"checkpoint config mismatch for {model_name}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
    feature_tensor = torch.from_numpy(train_features)
    target_tensor = torch.from_numpy(train_targets)
    for epoch in range(start_epoch, config.epochs + 1):
        loader = DataLoader(
            TensorDataset(feature_tensor, target_tensor),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(
                1_430_000 + config.seed + epoch
            ),
        )
        model.train()
        total_loss = 0.0
        total_cells = 0
        for features, targets in loader:
            features = features.to(config.device)
            targets = targets.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * targets.numel()
            total_cells += targets.numel()
        history_row = {
            "run_id": config.run_id,
            "model": model_name,
            "epoch": epoch,
            "train_mse": total_loss / max(1, total_cells),
        }
        history.append(history_row)
        torch.save(
            {
                "config_hash": config_hash,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            },
            latest_path,
        )
        if progress is not None:
            progress("epoch_done", history_row)
    scores = _predict_raw(
        model, test_features, batch_size=config.batch_size, device=config.device
    )
    if target_mode == "selected8":
        rows = [
            {
                **_binary_metrics(scores[:, index], test_targets[:, index]),
                "model": model_name,
                "architecture": architecture,
                "target_kind": "preregistered_true_ciphertext_output_bit",
                "sample_classification": False,
                "msb_index": bit,
                "integer_bit": 63 - bit,
            }
            for index, bit in enumerate(config.selected_msb_indices)
        ]
    else:
        mask = (
            config.selected_msb_indices
            if target_mode == "selected_parity"
            else config.control_msb_indices
        )
        rows = [
            {
                **_binary_metrics(scores[:, 0], test_targets[:, 0]),
                "model": model_name,
                "architecture": architecture,
                "target_kind": "true_ciphertext_eight_bit_output_parity",
                "sample_classification": False,
                "mask_bits": list(mask),
                "mask_weight": 8,
                "test_target_identity": "true_ciphertext_output_parity",
            }
        ]
    torch.save(
        {
            "config_hash": config_hash,
            "epoch": config.epochs,
            "model_state": model.state_dict(),
        },
        final_path,
    )
    return {
        "rows": rows,
        "summary": {
            "model": model_name,
            "architecture": architecture,
            "target_mode": target_mode,
            "output_bits": output_bits,
            "parameters": _count(model),
            "train_labels_shuffled": model_name.endswith("label_shuffle"),
            "mean_auc": _mean_field(rows, "auc"),
            "mean_accuracy_margin": _mean_field(rows, "accuracy_minus_majority"),
        },
        "history": history,
        "checkpoint": {
            "model": model_name,
            "path": str(final_path.relative_to(output_root)),
            "sha256": _sha256(final_path),
            "config_hash": config_hash,
        },
        "scores": scores,
    }


def _attach_comparisons(
    rows: list[dict[str, Any]],
    *,
    selected_scores: np.ndarray,
    selected_targets: np.ndarray,
    parity_targets_array: np.ndarray,
    selected_bits: tuple[int, ...],
) -> None:
    indexed = {
        str(row["model"]): row for row in rows if "msb_index" not in row
    }
    selected_rows = {
        int(row["msb_index"]): row
        for row in rows
        if row["model"] == "selected8_mlp_true_output"
    }
    candidate = indexed["selected8_parity_rescnn_true_output"]
    component_scores = np.asarray(selected_scores, dtype=np.float64)
    clipped = np.clip(component_scores, 0.0, 1.0)
    derived_scores = (
        1.0 - np.prod(1.0 - 2.0 * clipped, axis=1, dtype=np.float64)
    ) / 2.0
    derived = _binary_metrics(derived_scores, parity_targets_array)
    mlp = indexed["selected8_parity_mlp_true_output"]
    geometry = indexed["control8_parity_rescnn_true_output"]
    shuffled = indexed["selected8_parity_rescnn_label_shuffle"]
    best_component_auc = max(float(selected_rows[bit]["auc"]) for bit in selected_bits)
    candidate.update(
        {
            "mlp_auc": float(mlp["auc"]),
            "geometry_auc": float(geometry["auc"]),
            "shuffle_auc": float(shuffled["auc"]),
            "derived_auc": float(derived["auc"]),
            "derived_threshold_accuracy": float(derived["threshold_accuracy"]),
            "derived_clip_rate": float(
                np.mean((component_scores < 0.0) | (component_scores > 1.0))
            ),
            "best_component_auc": best_component_auc,
            "auc_minus_mlp": float(candidate["auc"]) - float(mlp["auc"]),
            "auc_minus_geometry": float(candidate["auc"])
            - float(geometry["auc"]),
            "auc_minus_shuffle": float(candidate["auc"]) - float(shuffled["auc"]),
            "auc_minus_derived": float(candidate["auc"]) - float(derived["auc"]),
            "auc_minus_best_component": float(candidate["auc"])
            - best_component_auc,
            "component_target_replay": bool(
                np.array_equal(
                    parity_targets(selected_targets, tuple(range(8)))[:, 0],
                    parity_targets_array,
                )
            ),
        }
    )


def _binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    bit_scores = np.asarray(scores, dtype=np.float64)
    bit_labels = np.asarray(labels, dtype=np.float64)
    predictions = bit_scores >= 0.5
    prevalence = float(np.mean(bit_labels))
    majority = max(prevalence, 1.0 - prevalence)
    accuracy = float(np.mean(predictions == bit_labels))
    return {
        "threshold_accuracy": accuracy,
        "majority_accuracy": majority,
        "accuracy_minus_majority": accuracy - majority,
        "auc": float(binary_auc(bit_labels, bit_scores)),
        "mse": float(np.mean(np.square(bit_scores - bit_labels))),
    }


def _predict_raw(
    model: nn.Module,
    features: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _training_config_hash(config: Selected8ParityConfig, model_name: str) -> str:
    payload = {**serializable_config(config), "model_name": model_name}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _mean_field(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "MODEL_SPECS",
    "RUN_ID_PREFIX",
    "SELECTED8_PARITY_MASK",
    "SHIFTED_CONTROL_MASK",
    "Selected8ParityConfig",
    "adjudicate_selected8_parity",
    "parameter_counts",
    "parity_targets",
    "prepare_selected8_parity_data",
    "serializable_config",
    "train_selected8_parity_matrix",
    "validate_selected8_parity_contract",
]
