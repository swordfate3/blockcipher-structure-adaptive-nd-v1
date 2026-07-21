from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i2_output_parity_prediction_readiness_present_r1_seed0_20260721"
MASKS = tuple(0xF << (4 * nibble) for nibble in range(16))


@dataclass(frozen=True)
class OutputParityPredictionConfig:
    run_id: str = RUN_ID
    rounds: int = 1
    seed: int = 0
    train_rows: int = 4096
    validation_rows: int = 1024
    test_rows: int = 2048
    hidden_dim: int = 128
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")
        if min(self.train_rows, self.validation_rows, self.test_rows) <= 0:
            raise ValueError("split row counts must be positive")
        if min(self.hidden_dim, self.epochs, self.batch_size) <= 0:
            raise ValueError("hidden_dim, epochs, and batch_size must be positive")


@dataclass(frozen=True)
class OutputPredictionSplit:
    plaintexts: np.ndarray
    features: np.ndarray
    full_targets: np.ndarray
    parity_targets: np.ndarray


class OutputPredictionMlp(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(features))


def generate_output_prediction_data(
    config: OutputParityPredictionConfig,
    masks: tuple[int, ...] = MASKS,
) -> dict[str, Any]:
    key_rng = random.Random(910_000 + config.seed)
    secret_key = key_rng.getrandbits(80)
    total_rows = config.train_rows + config.validation_rows + config.test_rows
    plaintexts = _unique_uint64_values(total_rows, seed=920_000 + config.seed)
    cipher = Present80(rounds=config.rounds, key=secret_key)
    ciphertexts = np.asarray(
        [cipher.encrypt(int(plaintext)) for plaintext in plaintexts],
        dtype=np.uint64,
    )
    features = _words_to_bits(plaintexts)
    full_targets = _words_to_bits(ciphertexts)
    parity_targets = parity_targets_from_words(ciphertexts, masks)
    boundaries = (
        config.train_rows,
        config.train_rows + config.validation_rows,
        total_rows,
    )
    train_end, validation_end, _ = boundaries
    return {
        "secret_key": secret_key,
        "ciphertexts": ciphertexts,
        "masks": masks,
        "train": OutputPredictionSplit(
            plaintexts[:train_end],
            features[:train_end],
            full_targets[:train_end],
            parity_targets[:train_end],
        ),
        "validation": OutputPredictionSplit(
            plaintexts[train_end:validation_end],
            features[train_end:validation_end],
            full_targets[train_end:validation_end],
            parity_targets[train_end:validation_end],
        ),
        "test": OutputPredictionSplit(
            plaintexts[validation_end:],
            features[validation_end:],
            full_targets[validation_end:],
            parity_targets[validation_end:],
        ),
    }


def validate_output_prediction_contract(
    config: OutputParityPredictionConfig,
    data: dict[str, Any],
) -> dict[str, bool]:
    train = data["train"]
    validation = data["validation"]
    test = data["test"]
    train_plaintexts = {int(value) for value in train.plaintexts}
    validation_plaintexts = {int(value) for value in validation.plaintexts}
    test_plaintexts = {int(value) for value in test.plaintexts}
    cipher = Present80(rounds=config.rounds, key=int(data["secret_key"]))
    split_shapes = all(
        split.features.shape == (rows, 64)
        and split.full_targets.shape == (rows, 64)
        and split.parity_targets.shape == (rows, 16)
        for split, rows in (
            (train, config.train_rows),
            (validation, config.validation_rows),
            (test, config.test_rows),
        )
    )
    labels_nonconstant = all(
        bool(np.all((split.parity_targets.sum(axis=0) > 0)))
        and bool(np.all((split.parity_targets.sum(axis=0) < len(split.parity_targets))))
        for split in (train, validation, test)
    )
    scalar_replay = all(
        cipher.encrypt(int(plaintext)) == int(ciphertext)
        for plaintext, ciphertext in zip(
            np.concatenate(
                (train.plaintexts, validation.plaintexts, test.plaintexts)
            ),
            data["ciphertexts"],
            strict=True,
        )
    )
    parity_replay = all(
        np.array_equal(
            split.parity_targets,
            parity_targets_from_full_bits(split.full_targets, data["masks"]),
        )
        for split in (train, validation, test)
    )
    return {
        "official_present_vector_matches": Present80(rounds=31, key=0).encrypt(0)
        == 0x5579C1387B228445,
        "single_fixed_secret_key": isinstance(data["secret_key"], int),
        "split_shapes_match": split_shapes,
        "plaintext_splits_are_disjoint": train_plaintexts.isdisjoint(
            validation_plaintexts
        )
        and train_plaintexts.isdisjoint(test_plaintexts)
        and validation_plaintexts.isdisjoint(test_plaintexts),
        "all_plaintexts_are_unique": len(
            train_plaintexts | validation_plaintexts | test_plaintexts
        )
        == config.train_rows + config.validation_rows + config.test_rows,
        "input_is_plaintext_bits_only": all(
            np.array_equal(split.features, _words_to_bits(split.plaintexts))
            for split in (train, validation, test)
        ),
        "full_targets_replay_scalar_present": scalar_replay,
        "parity_targets_equal_full_output_xor": parity_replay,
        "sixteen_distinct_nibble_masks": len(set(data["masks"])) == 16,
        "nibble_masks_cover_each_output_bit_once": sum(data["masks"])
        == (1 << 64) - 1,
        "every_split_mask_has_both_labels": labels_nonconstant,
        "no_sample_classification_label": True,
    }


def train_output_prediction_matrix(
    config: OutputParityPredictionConfig,
    data: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    trained: dict[str, dict[str, Any]] = {}
    for row_index, (row_name, target_kind, shuffle_train_labels) in enumerate(
        (
            ("full_output_mlp", "full", False),
            ("direct_parity_mlp", "parity", False),
            ("direct_parity_label_shuffle", "parity", True),
        )
    ):
        result = train_output_prediction_row(
            config,
            data,
            row_name=row_name,
            target_kind=target_kind,
            shuffle_train_labels=shuffle_train_labels,
            seed=config.seed + row_index * 1000,
        )
        trained[row_name] = result
        history.extend(result["history"])
    full_result = trained["full_output_mlp"]
    full_probabilities = full_result["test_probabilities"]
    derived_probabilities = parity_probabilities_from_bit_probabilities(
        full_probabilities, data["masks"]
    )
    derived_metrics = multilabel_metrics(
        derived_probabilities, data["test"].parity_targets
    )
    for row_name, target_kind, _shuffle in (
        ("full_output_mlp", "full", False),
        ("direct_parity_mlp", "parity", False),
        ("direct_parity_label_shuffle", "parity", True),
    ):
        result = trained[row_name]
        metrics = result["test_metrics"]
        row = {
            "run_id": config.run_id,
            "task": "innovation2_output_parity_prediction",
            "model": row_name,
            "target_kind": target_kind,
            "seed": config.seed,
            "rounds": config.rounds,
            "secret_key_scope": "single_fixed_unknown_key",
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
        if row_name == "full_output_mlp":
            row.update(
                {
                    "derived_parity_accuracy": derived_metrics["accuracy"],
                    "derived_parity_macro_auc": derived_metrics["macro_auc"],
                    "derived_parity_exact_match": derived_metrics["exact_match"],
                }
            )
        else:
            row.update(
                {
                    "parity_accuracy": metrics["accuracy"],
                    "parity_macro_auc": metrics["macro_auc"],
                    "parity_exact_match": metrics["exact_match"],
                }
            )
        rows.append(row)
    return {
        "rows": rows,
        "history": history,
        "derived_parity_metrics": derived_metrics,
        "trained": trained,
    }


def adjudicate_output_prediction_readiness(
    config: OutputParityPredictionConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
) -> dict[str, Any]:
    rows = training["rows"]
    numeric_values = [
        float(row[key])
        for row in rows
        for key in ("test_loss", "test_accuracy", "test_macro_auc", "test_exact_match")
    ]
    checks = {
        **protocol_checks,
        "three_training_rows_complete": len(rows) == 3,
        "all_training_metrics_finite": all(math.isfinite(value) for value in numeric_values),
        "history_rows_complete": len(training["history"]) == config.epochs * 3,
        "shuffle_control_uses_true_test_targets": training["trained"][
            "direct_parity_label_shuffle"
        ]["test_target_identity"]
        == "true_parity_targets",
    }
    direct = next(row for row in rows if row["model"] == "direct_parity_mlp")
    shuffled = next(
        row for row in rows if row["model"] == "direct_parity_label_shuffle"
    )
    full = next(row for row in rows if row["model"] == "full_output_mlp")
    if all(checks.values()):
        status = "pass"
        decision = "innovation2_output_parity_prediction_readiness_passed"
        if (
            direct["test_macro_auc"] >= 0.55
            and direct["test_macro_auc"] - shuffled["test_macro_auc"] >= 0.03
        ):
            action = (
                "preregister OP2 multi-key PRESENT r1-r4 calibration with the same "
                "full-output, derived-parity, direct-parity, and shuffled contracts"
            )
            next_adjudication = "op2_multi_key_round_ladder"
        else:
            action = (
                "preregister OP2 one-round mask-geometry calibration comparing contiguous "
                "output nibbles with last-round S-box/P-layer-aligned four-bit masks"
            )
            next_adjudication = "op2_mask_geometry_calibration"
    else:
        status = "fail"
        decision = "innovation2_output_parity_prediction_protocol_invalid"
        action = "repair only the fixed-key output-prediction data or metric contract"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "checks": checks,
        "metrics": {
            "full_bit_accuracy": full["test_accuracy"],
            "full_bit_macro_auc": full["test_macro_auc"],
            "full_derived_parity_accuracy": full["derived_parity_accuracy"],
            "full_derived_parity_macro_auc": full["derived_parity_macro_auc"],
            "direct_parity_accuracy": direct["parity_accuracy"],
            "direct_parity_macro_auc": direct["test_macro_auc"],
            "shuffled_parity_accuracy": shuffled["parity_accuracy"],
            "shuffled_parity_macro_auc": shuffled["test_macro_auc"],
            "direct_minus_derived_parity": direct["parity_accuracy"]
            - full["derived_parity_accuracy"],
            "direct_minus_shuffled_parity": direct["parity_accuracy"]
            - shuffled["parity_accuracy"],
        },
        "claim_scope": (
            "local one-round fixed-key known-plaintext output-prediction readiness; "
            "the binary-valued targets are ciphertext parities, not sample classes, and "
            "this run is not a high-round result, attack-round claim, SOTA comparison, or "
            "evidence that direct parity is better than full-output prediction"
        ),
        "next_action": {
            "action": action,
            "next_adjudication": (
                next_adjudication if status == "pass" else "repair_protocol"
            ),
            "op2_open": status == "pass",
            "remote_scale": False,
            "high_round_training": False,
            "sample_classification": False,
        },
    }


def parity_targets_from_words(
    words: np.ndarray, masks: tuple[int, ...] = MASKS
) -> np.ndarray:
    values = np.asarray(words, dtype=np.uint64)
    return np.asarray(
        [
            [int(int(word & np.uint64(mask)).bit_count() & 1) for mask in masks]
            for word in values
        ],
        dtype=np.float32,
    )


def parity_targets_from_full_bits(
    full_targets: np.ndarray, masks: tuple[int, ...] = MASKS
) -> np.ndarray:
    targets = np.asarray(full_targets, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 64:
        raise ValueError("full targets must have shape [rows, 64]")
    if not masks:
        raise ValueError("masks must be non-empty")
    columns = []
    for mask in masks:
        bit_indices = [bit for bit in range(64) if (mask >> bit) & 1]
        if not bit_indices:
            raise ValueError("each parity mask must select at least one output bit")
        columns.append(np.mod(targets[:, bit_indices].sum(axis=1), 2))
    return np.column_stack(columns).astype(np.float32)


def parity_probabilities_from_bit_probabilities(
    bit_probabilities: np.ndarray,
    masks: tuple[int, ...] = MASKS,
) -> np.ndarray:
    probabilities = np.asarray(bit_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 64:
        raise ValueError("bit probabilities must have shape [rows, 64]")
    if not masks:
        raise ValueError("masks must be non-empty")
    columns = []
    for mask in masks:
        bit_indices = [bit for bit in range(64) if (mask >> bit) & 1]
        if not bit_indices:
            raise ValueError("each parity mask must select at least one output bit")
        signed = 1.0 - 2.0 * probabilities[:, bit_indices]
        columns.append((1.0 - np.prod(signed, axis=1)) / 2.0)
    return np.clip(np.column_stack(columns), 0.0, 1.0).astype(np.float32)


def multilabel_metrics(
    probabilities: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if probabilities.shape != targets.shape:
        raise ValueError("probability and target shapes differ")
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    predictions = probabilities >= 0.5
    labels = targets >= 0.5
    per_column_auc = [
        binary_auc(targets[:, column], probabilities[:, column])
        for column in range(targets.shape[1])
    ]
    prevalence = targets.mean(axis=0)
    return {
        "loss": float(
            np.mean(-(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped)))
        ),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_auc": float(np.mean(per_column_auc)),
        "exact_match": float(np.mean(np.all(predictions == labels, axis=1))),
        "majority_accuracy": float(np.mean(np.maximum(prevalence, 1.0 - prevalence))),
    }


def serializable_config(config: OutputParityPredictionConfig) -> dict[str, Any]:
    return asdict(config)


def train_output_prediction_row(
    config: OutputParityPredictionConfig,
    data: dict[str, Any],
    *,
    row_name: str,
    target_kind: str,
    shuffle_train_labels: bool,
    seed: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    output_dim = (
        64 if target_kind == "full" else data["train"].parity_targets.shape[1]
    )
    model = OutputPredictionMlp(output_dim, config.hidden_dim).to(config.device)
    targets = {
        split_name: (
            split.full_targets if target_kind == "full" else split.parity_targets
        )
        for split_name, split in (
            ("train", data["train"]),
            ("validation", data["validation"]),
            ("test", data["test"]),
        )
    }
    train_targets = targets["train"].copy()
    if shuffle_train_labels:
        rng = np.random.default_rng(930_000 + seed)
        train_targets = train_targets[rng.permutation(len(train_targets))]
    generator = torch.Generator().manual_seed(940_000 + seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data["train"].features),
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
    history: list[dict[str, Any]] = []
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
            model, data["validation"].features, config.batch_size, config.device
        )
        validation_metrics = multilabel_metrics(
            validation_probabilities, targets["validation"]
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
        model, data["test"].features, config.batch_size, config.device
    )
    return {
        "history": history,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "test_probabilities": test_probabilities,
        "test_metrics": multilabel_metrics(test_probabilities, targets["test"]),
        "test_target_identity": (
            "true_full_targets" if target_kind == "full" else "true_parity_targets"
        ),
    }


def _predict(
    model: nn.Module, features: np.ndarray, batch_size: int, device: str
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _unique_uint64_values(count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < count:
        low = int(rng.integers(0, 1 << 32, dtype=np.uint64))
        high = int(rng.integers(0, 1 << 32, dtype=np.uint64))
        value = low | (high << 32)
        if value not in seen:
            seen.add(value)
            values.append(value)
    return np.asarray(values, dtype=np.uint64)


def _words_to_bits(words: np.ndarray) -> np.ndarray:
    values = np.asarray(words, dtype=np.uint64)
    shifts = np.arange(64, dtype=np.uint64)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
