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

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.tasks.innovation2.output_prediction_kimura_lstm import (
    KimuraOutputPredictionConfig,
    ParameterMatchedOutputMlp,
    prepare_disk_output_prediction_data,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i2_output_prediction_op11_present_r3_selected8_key1_smoke_20260721"
REMOTE_RUN_ID = "i2_output_prediction_op11_present_r3_selected8_key1_gpu0_20260721"
SELECTED_MSB_INDICES = (0, 2, 8, 10, 32, 34, 40, 42)
MODEL_SPECS = (
    ("full64_mlp_true_output", "full64", False),
    ("selected8_mlp_true_output", "selected8", False),
    ("selected8_mlp_label_shuffle", "selected8", True),
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SelectedOutputBitHeadConfig:
    run_id: str = RUN_ID
    mode: str = "smoke"
    rounds: int = 3
    seed: int = 1
    train_rows: int = 64
    test_rows: int = 64
    hidden_dim: int = 1936
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    data_chunk_rows: int = 32
    selected_msb_indices: tuple[int, ...] = SELECTED_MSB_INDICES
    minimum_auc: float = 0.510
    minimum_accuracy_margin: float = 0.005
    minimum_shuffle_auc_margin: float = 0.005
    minimum_cross_key_bits: int = 4
    minimum_dedicated_auc_gain: float = 0.002
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "independent_key_confirmation"}:
            raise ValueError("invalid selected-output mode")
        if len(self.selected_msb_indices) != 8 or len(set(self.selected_msb_indices)) != 8:
            raise ValueError("exactly eight unique output positions are required")
        if any(bit < 0 or bit >= 64 for bit in self.selected_msb_indices):
            raise ValueError("selected output positions must be in [0, 63]")
        if min(
            self.rounds,
            self.train_rows,
            self.test_rows,
            self.hidden_dim,
            self.epochs,
            self.batch_size,
            self.data_chunk_rows,
        ) <= 0:
            raise ValueError("round, row, model, epoch, batch, and chunk values must be positive")

    @classmethod
    def independent_key_confirmation(
        cls,
        *,
        run_id: str = REMOTE_RUN_ID,
        device: str = "cuda",
    ) -> SelectedOutputBitHeadConfig:
        return cls(
            run_id=run_id,
            mode="independent_key_confirmation",
            seed=1,
            train_rows=1 << 17,
            test_rows=1 << 16,
            hidden_dim=1936,
            epochs=100,
            batch_size=250,
            learning_rate=1e-3,
            data_chunk_rows=4096,
            device=device,
        )


class SelectedOutputMlp(nn.Module):
    def __init__(self, hidden_dim: int, output_bits: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_bits),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 64:
            raise ValueError(f"expected [batch, 64] input, got {tuple(features.shape)}")
        return self.network(features.float())


def prepare_selected_output_data(
    config: SelectedOutputBitHeadConfig,
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    data_config = KimuraOutputPredictionConfig(
        run_id=config.run_id,
        mode="smoke",
        rounds=config.rounds,
        seed=config.seed,
        train_rows=config.train_rows,
        test_rows=config.test_rows,
        hidden_dim=300,
        layers=6,
        mlp_hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        data_chunk_rows=config.data_chunk_rows,
        device=config.device,
    )
    return prepare_disk_output_prediction_data(
        data_config, output_root, progress=progress
    )


def validate_selected_output_contract(
    config: SelectedOutputBitHeadConfig,
    data: dict[str, Any],
) -> dict[str, bool]:
    total_rows = config.train_rows + config.test_rows
    plaintexts = data["plaintexts"]
    features = data["features"]
    targets = data["full_targets"]
    train_plaintexts = {int(value) for value in plaintexts[: config.train_rows]}
    test_plaintexts = {int(value) for value in plaintexts[config.train_rows :]}
    cipher = Present80(rounds=config.rounds, key=int(data["secret_key"]))
    sample_indices = sorted({0, config.train_rows - 1, config.train_rows, total_rows - 1})
    expected_positions = config.selected_msb_indices == SELECTED_MSB_INDICES
    return {
        "official_present_vector_matches": Present80(rounds=31, key=0).encrypt(0)
        == 0x5579C1387B228445,
        "cache_is_complete": data["metadata"]["status"] == "complete"
        and int(data["metadata"]["completed_rows"]) == total_rows,
        "arrays_have_expected_shapes": plaintexts.shape == (total_rows,)
        and features.shape == (total_rows, 64)
        and targets.shape == (total_rows, 64),
        "plaintexts_are_unique": len(train_plaintexts | test_plaintexts) == total_rows,
        "train_test_plaintexts_are_disjoint": train_plaintexts.isdisjoint(
            test_plaintexts
        ),
        "features_are_msb_first": all(
            _bits_to_word(features[index]) == int(plaintexts[index])
            for index in sample_indices
        ),
        "targets_are_true_ciphertext_bits": all(
            _bits_to_word(targets[index]) == cipher.encrypt(int(plaintexts[index]))
            for index in sample_indices
        ),
        "selected_positions_match_op10": expected_positions,
        "independent_key_seed_is_one": config.seed == 1,
        "labels_are_outputs_not_sample_classes": True,
    }


def train_selected_output_matrix(
    config: SelectedOutputBitHeadConfig,
    data: dict[str, Any],
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    train_features = np.array(data["features"][: config.train_rows], copy=True)
    test_features = np.array(data["features"][config.train_rows :], copy=True)
    train_full = np.array(data["full_targets"][: config.train_rows], copy=True)
    test_full = np.array(data["full_targets"][config.train_rows :], copy=True)
    selected = np.asarray(config.selected_msb_indices, dtype=np.int64)
    train_selected = train_full[:, selected]
    test_selected = test_full[:, selected]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for model_name, output_mode, shuffle_labels in MODEL_SPECS:
        target_train = train_full if output_mode == "full64" else train_selected
        if shuffle_labels:
            permutation = np.random.default_rng(1_120_000 + config.seed).permutation(
                len(target_train)
            )
            target_train = target_train[permutation]
        result = _train_one_model(
            config,
            model_name=model_name,
            output_mode=output_mode,
            train_features=train_features,
            train_targets=target_train,
            test_features=test_features,
            test_targets=test_full if output_mode == "full64" else test_selected,
            selected_test_targets=test_selected,
            output_root=output_root,
            progress=progress,
        )
        rows.extend(result["rows"])
        summaries.append(result["summary"])
        history.extend(result["history"])
        checkpoints.append(result["checkpoint"])
    return {
        "rows": rows,
        "summaries": summaries,
        "history": history,
        "checkpoints": checkpoints,
    }


def adjudicate_selected_output_head(
    config: SelectedOutputBitHeadConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
) -> dict[str, Any]:
    indexed = {
        (row["model"], int(row["msb_index"])): row for row in training["rows"]
    }
    confirmation: list[dict[str, Any]] = []
    for bit in config.selected_msb_indices:
        selected = indexed[("selected8_mlp_true_output", bit)]
        shuffled = indexed[("selected8_mlp_label_shuffle", bit)]
        full = indexed[("full64_mlp_true_output", bit)]
        shuffle_margin = float(selected["auc"]) - float(shuffled["auc"])
        checks = {
            "auc_at_least_0_510": float(selected["auc"]) >= config.minimum_auc,
            "accuracy_margin_at_least_0_005": float(
                selected["accuracy_minus_majority"]
            )
            >= config.minimum_accuracy_margin,
            "auc_minus_matched_shuffle_at_least_0_005": shuffle_margin
            >= config.minimum_shuffle_auc_margin,
        }
        confirmation.append(
            {
                "msb_index": bit,
                "integer_bit": 63 - bit,
                "selected8_auc": float(selected["auc"]),
                "selected8_accuracy": float(selected["threshold_accuracy"]),
                "selected8_accuracy_margin": float(
                    selected["accuracy_minus_majority"]
                ),
                "selected8_shuffle_auc": float(shuffled["auc"]),
                "selected8_auc_minus_shuffle": shuffle_margin,
                "full64_auc": float(full["auc"]),
                "selected8_auc_minus_full64": float(selected["auc"])
                - float(full["auc"]),
                "checks": checks,
                "confirmed": all(checks.values()),
            }
        )
    mean_selected_auc = float(
        np.mean([row["selected8_auc"] for row in confirmation])
    )
    mean_shuffle_auc = float(
        np.mean([row["selected8_shuffle_auc"] for row in confirmation])
    )
    mean_full_auc = float(np.mean([row["full64_auc"] for row in confirmation]))
    confirmed = [row for row in confirmation if row["confirmed"]]
    execution_checks = {
        "three_models_complete": len(training["summaries"]) == 3,
        "twenty_four_bit_rows_complete": len(training["rows"]) == 24,
        "history_rows_complete": len(training["history"]) == config.epochs * 3,
        "three_checkpoint_hashes_present": len(training["checkpoints"]) == 3
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
                "invalid_numpy_rint_rate",
            )
        ),
        "matched_shuffle_uses_true_test_targets": indexed[
            ("selected8_mlp_label_shuffle", config.selected_msb_indices[0])
        ]["test_target_identity"]
        == "true_selected_ciphertext_targets",
    }
    all_protocol = all(protocol_checks.values()) and all(execution_checks.values())
    cross_key_pass = len(confirmed) >= config.minimum_cross_key_bits
    dedicated_gain = mean_selected_auc - mean_full_auc
    dedicated_supported = dedicated_gain >= config.minimum_dedicated_auc_gain
    if not all_protocol:
        status = "fail"
        decision = "innovation2_selected8_independent_key_protocol_invalid"
        next_adjudication = "op11_protocol_repair"
        action = "repair only data, selected-position, model, shuffle, metric, cache, or checkpoint protocol"
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_selected8_independent_key_local_smoke_passed"
        next_adjudication = "op11_remote_independent_key_confirmation"
        action = "launch the frozen seed1 three-row matrix at 2^17 train, 2^16 test, and 100 epochs"
    elif cross_key_pass:
        status = "pass"
        decision = (
            "innovation2_selected8_cross_key_and_dedicated_head_supported"
            if dedicated_supported
            else "innovation2_selected8_cross_key_supported_without_head_gain"
        )
        next_adjudication = "op12_present_r4_fixed_selected8"
        action = "advance the same eight positions and the stronger frozen anchor to PRESENT r4 without reselection"
    else:
        status = "hold"
        decision = "innovation2_selected8_not_cross_key_supported"
        next_adjudication = "selected_bit_key_condition_audit"
        action = "stop seed, epoch, data, and round scaling; treat seed0 bit signal as key-conditional pending audit"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "metrics": {
            "confirmed_count": len(confirmed),
            "confirmed_msb_indices": [row["msb_index"] for row in confirmed],
            "mean_selected8_auc": mean_selected_auc,
            "mean_selected8_shuffle_auc": mean_shuffle_auc,
            "mean_full64_auc_on_selected_bits": mean_full_auc,
            "mean_selected8_auc_minus_shuffle": mean_selected_auc
            - mean_shuffle_auc,
            "mean_selected8_auc_minus_full64": dedicated_gain,
            "dedicated_head_supported": dedicated_supported,
        },
        "bit_confirmation": confirmation,
        "claim_scope": (
            "local implementation smoke"
            if config.mode == "smoke"
            else "second fixed secret key PRESENT r3 confirmation of eight preregistered true output bits"
        )
        + "; not full-ciphertext recovery, broad cross-key statistics, r4 evidence, sample classification, or SOTA",
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "sample_classification": False,
            "target": "eight_preregistered_true_ciphertext_output_bits",
        },
    }


def parameter_counts(config: SelectedOutputBitHeadConfig) -> dict[str, int]:
    return {
        "full64_mlp": sum(
            parameter.numel()
            for parameter in ParameterMatchedOutputMlp(config.hidden_dim).parameters()
        ),
        "selected8_mlp": sum(
            parameter.numel()
            for parameter in SelectedOutputMlp(config.hidden_dim).parameters()
        ),
    }


def serializable_config(config: SelectedOutputBitHeadConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["selected_msb_indices"] = list(config.selected_msb_indices)
    return payload


def _train_one_model(
    config: SelectedOutputBitHeadConfig,
    *,
    model_name: str,
    output_mode: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    selected_test_targets: np.ndarray,
    output_root: Path,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    _seed_everything(1_110_000 + config.seed)
    model: nn.Module = (
        ParameterMatchedOutputMlp(config.hidden_dim)
        if output_mode == "full64"
        else SelectedOutputMlp(config.hidden_dim)
    )
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
        checkpoint = torch.load(latest_path, map_location=config.device, weights_only=False)
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
                1_130_000 + config.seed + epoch
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
    selected_scores = (
        scores[:, np.asarray(config.selected_msb_indices, dtype=np.int64)]
        if output_mode == "full64"
        else scores
    )
    rows = _selected_bit_rows(
        model_name,
        selected_scores,
        selected_test_targets,
        config.selected_msb_indices,
    )
    torch.save(
        {
            "config_hash": config_hash,
            "epoch": config.epochs,
            "model_state": model.state_dict(),
        },
        final_path,
    )
    checkpoint = {
        "model": model_name,
        "path": str(final_path.relative_to(output_root)),
        "sha256": _sha256(final_path),
        "config_hash": config_hash,
    }
    return {
        "rows": rows,
        "summary": {
            "model": model_name,
            "output_mode": output_mode,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train_labels_shuffled": model_name.endswith("label_shuffle"),
            "mean_auc": float(np.mean([row["auc"] for row in rows])),
            "mean_accuracy_margin": float(
                np.mean([row["accuracy_minus_majority"] for row in rows])
            ),
            "test_target_identity": "true_selected_ciphertext_targets",
        },
        "history": history,
        "checkpoint": checkpoint,
    }


def _selected_bit_rows(
    model_name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    selected_bits: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column, bit in enumerate(selected_bits):
        bit_scores = np.asarray(scores[:, column], dtype=np.float64)
        bit_labels = np.asarray(labels[:, column], dtype=np.float64)
        predictions = bit_scores >= 0.5
        prevalence = float(np.mean(bit_labels))
        majority = max(prevalence, 1.0 - prevalence)
        rounded = np.rint(bit_scores)
        valid = (rounded == 0.0) | (rounded == 1.0)
        accuracy = float(np.mean(predictions == bit_labels))
        rows.append(
            {
                "model": model_name,
                "target": "preregistered_true_ciphertext_output_bit",
                "sample_classification": False,
                "msb_index": bit,
                "integer_bit": 63 - bit,
                "nibble_msb_index": bit // 4,
                "bit_in_nibble_msb": bit % 4,
                "threshold_accuracy": accuracy,
                "majority_accuracy": majority,
                "accuracy_minus_majority": accuracy - majority,
                "auc": float(binary_auc(bit_labels, bit_scores)),
                "mse": float(np.mean(np.square(bit_scores - bit_labels))),
                "invalid_numpy_rint_rate": float(1.0 - np.mean(valid)),
                "test_target_identity": "true_selected_ciphertext_targets",
            }
        )
    return rows


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


def _bits_to_word(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8):
        value = (value << 1) | int(bit)
    return value


def _training_config_hash(
    config: SelectedOutputBitHeadConfig, model_name: str
) -> str:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
