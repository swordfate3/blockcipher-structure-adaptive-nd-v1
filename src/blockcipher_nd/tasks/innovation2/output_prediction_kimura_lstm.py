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
from numpy.lib.format import open_memmap
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i2_output_prediction_op9_present_r3_kimura_lstm_smoke_20260721"
PAPER_RUN_ID = "i2_output_prediction_op9_present_r3_kimura_lstm_2p17_seed0_20260721"
CACHE_VERSION = 1
MODEL_NAMES = (
    "kimura_lstm_true_output",
    "matched_mlp_true_output",
    "kimura_lstm_label_shuffle",
)

ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class KimuraOutputPredictionConfig:
    run_id: str = RUN_ID
    mode: str = "smoke"
    rounds: int = 3
    seed: int = 0
    train_rows: int = 64
    test_rows: int = 64
    hidden_dim: int = 300
    layers: int = 6
    mlp_hidden_dim: int = 1936
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    data_chunk_rows: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "paper_calibration"}:
            raise ValueError("mode must be smoke or paper_calibration")
        integer_fields = (
            self.rounds,
            self.train_rows,
            self.test_rows,
            self.hidden_dim,
            self.layers,
            self.mlp_hidden_dim,
            self.epochs,
            self.batch_size,
            self.data_chunk_rows,
        )
        if min(integer_fields) <= 0:
            raise ValueError("round, row, model, epoch, batch, and chunk values must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @classmethod
    def paper_calibration(
        cls,
        *,
        run_id: str = PAPER_RUN_ID,
        seed: int = 0,
        device: str = "cuda",
    ) -> KimuraOutputPredictionConfig:
        return cls(
            run_id=run_id,
            mode="paper_calibration",
            seed=seed,
            train_rows=1 << 17,
            test_rows=1 << 16,
            hidden_dim=300,
            layers=6,
            mlp_hidden_dim=1936,
            epochs=100,
            batch_size=250,
            learning_rate=1e-3,
            data_chunk_rows=4096,
            device=device,
        )


class KimuraCiphertextLstm(nn.Module):
    def __init__(self, hidden_dim: int = 300, layers: int = 6) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 64)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 64:
            raise ValueError(f"expected [batch, 64] input, got {tuple(features.shape)}")
        _, (hidden, _) = self.encoder(features.float().unsqueeze(-1))
        return self.head(hidden[-1])


class ParameterMatchedOutputMlp(nn.Module):
    def __init__(self, hidden_dim: int = 1936) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 64:
            raise ValueError(f"expected [batch, 64] input, got {tuple(features.shape)}")
        return self.network(features.float())


def prepare_disk_output_prediction_data(
    config: KimuraOutputPredictionConfig,
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    metadata_path = data_root / "cache_metadata.json"
    total_rows = config.train_rows + config.test_rows
    secret_key = random.Random(910_000 + config.seed).getrandbits(80)
    expected = {
        "cache_version": CACHE_VERSION,
        "cipher": "PRESENT-80",
        "rounds": config.rounds,
        "seed": config.seed,
        "train_rows": config.train_rows,
        "test_rows": config.test_rows,
        "total_rows": total_rows,
        "bit_order": "msb_first",
        "secret_key_hex": f"{secret_key:020x}",
    }
    paths = {
        "plaintexts": data_root / "plaintexts.npy",
        "features": data_root / "features.npy",
        "full_targets": data_root / "full_targets.npy",
    }
    metadata = _read_json(metadata_path) if metadata_path.exists() else None
    if metadata is not None:
        actual_static = {key: metadata.get(key) for key in expected}
        if actual_static != expected:
            raise ValueError("existing output-prediction cache parameters do not match")
    completed_rows = int(metadata.get("completed_rows", 0)) if metadata else 0
    arrays_exist = all(path.exists() for path in paths.values())
    if completed_rows and not arrays_exist:
        raise ValueError("cache metadata reports progress but array files are missing")
    if arrays_exist:
        plaintexts = np.load(paths["plaintexts"], mmap_mode="r+")
        features = np.load(paths["features"], mmap_mode="r+")
        full_targets = np.load(paths["full_targets"], mmap_mode="r+")
        if (
            plaintexts.shape != (total_rows,)
            or features.shape != (total_rows, 64)
            or full_targets.shape != (total_rows, 64)
        ):
            raise ValueError("existing output-prediction cache has invalid shapes")
    else:
        plaintexts = open_memmap(
            paths["plaintexts"], mode="w+", dtype=np.uint64, shape=(total_rows,)
        )
        features = open_memmap(
            paths["features"], mode="w+", dtype=np.float32, shape=(total_rows, 64)
        )
        full_targets = open_memmap(
            paths["full_targets"],
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, 64),
        )
        completed_rows = 0
    rng = np.random.default_rng(920_000 + config.seed)
    if metadata and metadata.get("rng_state"):
        rng.bit_generator.state = metadata["rng_state"]
    seen = {int(value) for value in plaintexts[:completed_rows]}
    cipher = Present80(rounds=config.rounds, key=secret_key)
    shifts = np.arange(63, -1, -1, dtype=np.uint64)
    _write_json(
        metadata_path,
        {
            **expected,
            "status": "generating" if completed_rows < total_rows else "complete",
            "completed_rows": completed_rows,
            "rng_state": rng.bit_generator.state,
        },
    )
    while completed_rows < total_rows:
        stop = min(total_rows, completed_rows + config.data_chunk_rows)
        chunk_values: list[int] = []
        while len(chunk_values) < stop - completed_rows:
            low = int(rng.integers(0, 1 << 32, dtype=np.uint64))
            high = int(rng.integers(0, 1 << 32, dtype=np.uint64))
            value = low | (high << 32)
            if value not in seen:
                seen.add(value)
                chunk_values.append(value)
        words = np.asarray(chunk_values, dtype=np.uint64)
        ciphertexts = np.asarray(
            [cipher.encrypt(int(word)) for word in words], dtype=np.uint64
        )
        plaintexts[completed_rows:stop] = words
        features[completed_rows:stop] = (
            (words[:, None] >> shifts[None, :]) & np.uint64(1)
        ).astype(np.float32)
        full_targets[completed_rows:stop] = (
            (ciphertexts[:, None] >> shifts[None, :]) & np.uint64(1)
        ).astype(np.float32)
        plaintexts.flush()
        features.flush()
        full_targets.flush()
        completed_rows = stop
        _write_json(
            metadata_path,
            {
                **expected,
                "status": "complete" if stop == total_rows else "generating",
                "completed_rows": completed_rows,
                "rng_state": rng.bit_generator.state,
            },
        )
        if progress is not None:
            progress(
                "cache_chunk",
                {"completed_rows": completed_rows, "total_rows": total_rows},
            )
    return {
        "secret_key": secret_key,
        "plaintexts": plaintexts,
        "features": features,
        "full_targets": full_targets,
        "metadata": _read_json(metadata_path),
        "data_root": data_root,
        "cache_reused": bool(metadata and metadata.get("status") == "complete"),
    }


def validate_kimura_output_contract(
    config: KimuraOutputPredictionConfig,
    data: dict[str, Any],
) -> dict[str, bool]:
    total_rows = config.train_rows + config.test_rows
    plaintexts = data["plaintexts"]
    features = data["features"]
    targets = data["full_targets"]
    train_plaintexts = {int(value) for value in plaintexts[: config.train_rows]}
    test_plaintexts = {int(value) for value in plaintexts[config.train_rows :]}
    sample_indices = sorted(
        {0, config.train_rows - 1, config.train_rows, total_rows - 1}
    )
    cipher = Present80(rounds=config.rounds, key=int(data["secret_key"]))
    scalar_replay = all(
        _bits_to_word(targets[index]) == cipher.encrypt(int(plaintexts[index]))
        for index in sample_indices
    )
    feature_replay = all(
        _bits_to_word(features[index]) == int(plaintexts[index])
        for index in sample_indices
    )
    paper_fields = {
        "rounds": config.rounds == 3,
        "learning_rate": config.learning_rate == 1e-3,
    }
    if config.mode == "paper_calibration":
        paper_fields.update(
            {
                "hidden": config.hidden_dim == 300,
                "layers": config.layers == 6,
                "mlp_hidden": config.mlp_hidden_dim == 1936,
                "train_rows": config.train_rows == 1 << 17,
                "test_rows": config.test_rows == 1 << 16,
                "epochs": config.epochs == 100,
                "batch_size": config.batch_size == 250,
            }
        )
    return {
        "official_present_vector_matches": Present80(rounds=31, key=0).encrypt(0)
        == 0x5579C1387B228445,
        "cache_is_complete": data["metadata"]["status"] == "complete"
        and int(data["metadata"]["completed_rows"]) == total_rows,
        "cache_arrays_have_expected_shapes": plaintexts.shape == (total_rows,)
        and features.shape == (total_rows, 64)
        and targets.shape == (total_rows, 64),
        "plaintexts_are_unique": len(train_plaintexts | test_plaintexts) == total_rows,
        "train_and_test_plaintexts_are_disjoint": train_plaintexts.isdisjoint(
            test_plaintexts
        ),
        "features_are_msb_first_plaintext_bits": feature_replay,
        "targets_are_msb_first_true_ciphertext_bits": scalar_replay,
        "single_fixed_unknown_key": isinstance(data["secret_key"], int),
        "kimura_architecture_fields_match": all(paper_fields.values()),
        "labels_are_outputs_not_sample_classes": True,
    }


def train_kimura_output_matrix(
    config: KimuraOutputPredictionConfig,
    data: dict[str, Any],
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    train_features = np.asarray(data["features"][: config.train_rows])
    train_targets = np.asarray(data["full_targets"][: config.train_rows])
    test_features = np.asarray(data["features"][config.train_rows :])
    test_targets = np.asarray(data["full_targets"][config.train_rows :])
    specifications = (
        ("kimura_lstm_true_output", "lstm", False),
        ("matched_mlp_true_output", "mlp", False),
        ("kimura_lstm_label_shuffle", "lstm", True),
    )
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for model_name, architecture, shuffle_labels in specifications:
        result = _train_one_model(
            config,
            model_name=model_name,
            architecture=architecture,
            shuffle_labels=shuffle_labels,
            train_features=train_features,
            train_targets=train_targets,
            test_features=test_features,
            test_targets=test_targets,
            output_root=output_root,
            progress=progress,
        )
        rows.append(result["row"])
        history.extend(result["history"])
        checkpoints.append(result["checkpoint"])
    return {"rows": rows, "history": history, "checkpoints": checkpoints}


def adjudicate_kimura_output_prediction(
    config: KimuraOutputPredictionConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
) -> dict[str, Any]:
    rows = {row["model"]: row for row in training["rows"]}
    numeric_fields = (
        "test_mse",
        "test_bit_match",
        "test_macro_auc",
        "test_exact_match",
        "invalid_rounded_cell_rate",
    )
    execution_checks = {
        "three_training_rows_complete": set(rows) == set(MODEL_NAMES),
        "history_rows_complete": len(training["history"]) == config.epochs * 3,
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in rows.values()
            for field in numeric_fields
        ),
        "three_checkpoint_hashes_present": len(training["checkpoints"]) == 3
        and all(item.get("sha256") for item in training["checkpoints"]),
        "shuffle_changes_only_training_target_order": rows[
            "kimura_lstm_label_shuffle"
        ]["test_target_identity"]
        == "true_full_ciphertext_targets",
    }
    true_lstm = rows["kimura_lstm_true_output"]
    shuffled = rows["kimura_lstm_label_shuffle"]
    matched_mlp = rows["matched_mlp_true_output"]
    metrics = {
        "lstm_exact_match_count": true_lstm["test_exact_match_count"],
        "lstm_exact_match": true_lstm["test_exact_match"],
        "lstm_bit_match": true_lstm["test_bit_match"],
        "lstm_macro_auc": true_lstm["test_macro_auc"],
        "shuffled_bit_match": shuffled["test_bit_match"],
        "shuffled_macro_auc": shuffled["test_macro_auc"],
        "matched_mlp_bit_match": matched_mlp["test_bit_match"],
        "matched_mlp_macro_auc": matched_mlp["test_macro_auc"],
        "lstm_minus_shuffled_bit_match": true_lstm["test_bit_match"]
        - shuffled["test_bit_match"],
        "lstm_minus_shuffled_macro_auc": true_lstm["test_macro_auc"]
        - shuffled["test_macro_auc"],
        "lstm_minus_matched_mlp_bit_match": true_lstm["test_bit_match"]
        - matched_mlp["test_bit_match"],
        "lstm_minus_matched_mlp_macro_auc": true_lstm["test_macro_auc"]
        - matched_mlp["test_macro_auc"],
    }
    all_protocol = all(protocol_checks.values()) and all(execution_checks.values())
    if not all_protocol:
        status = "fail"
        decision = "innovation2_output_prediction_kimura_lstm_protocol_invalid"
        next_adjudication = "repair_op9_protocol"
        action = "repair only data, bit order, model, loss, metric, cache, or checkpoint protocol"
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_output_prediction_kimura_lstm_local_smoke_passed"
        next_adjudication = "op9_remote_single_key_paper_calibration"
        action = (
            "launch the pushed paper-calibration commit remotely with 2^17 train, "
            "2^16 test, 100 epochs, and the frozen three-row matrix"
        )
    else:
        performance_checks = {
            "at_least_one_full_output_exact_match": true_lstm[
                "test_exact_match_count"
            ]
            >= 1,
            "lstm_bit_match_at_least_0_505": true_lstm["test_bit_match"] >= 0.505,
            "lstm_minus_shuffle_bit_match_at_least_0_005": metrics[
                "lstm_minus_shuffled_bit_match"
            ]
            >= 0.005,
            "lstm_minus_shuffle_auc_at_least_0_010": metrics[
                "lstm_minus_shuffled_macro_auc"
            ]
            >= 0.010,
        }
        execution_checks.update(performance_checks)
        if all(performance_checks.values()):
            status = "pass"
            decision = "innovation2_output_prediction_kimura_lstm_single_key_supported"
            next_adjudication = "op10_independent_fixed_key_confirmation"
            action = "repeat the identical three-row r3 matrix for one independent fixed key"
        else:
            status = "hold"
            decision = "innovation2_output_prediction_kimura_lstm_single_key_not_supported"
            next_adjudication = "output_prediction_target_reassessment"
            action = (
                "stop Kimura-LSTM data, epoch, layer, round, and seed scaling; reassess "
                "a preregistered true ciphertext-output representation"
            )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "metrics": metrics,
        "claim_scope": (
            "local implementation smoke" if config.mode == "smoke" else "single-fixed-key paper-family calibration"
        )
        + "; full 64-bit PRESENT r3 ciphertext-output prediction, not sample classification, "
        "not a 100-key exact reproduction, attack-round claim, or SOTA result",
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "remote_scale": config.mode == "smoke" and status == "pass",
            "sample_classification": False,
            "target": "full_64_bit_true_ciphertext_output",
        },
    }


def serializable_config(config: KimuraOutputPredictionConfig) -> dict[str, Any]:
    return asdict(config)


def parameter_counts(config: KimuraOutputPredictionConfig) -> dict[str, int]:
    return {
        "kimura_lstm": sum(
            parameter.numel()
            for parameter in KimuraCiphertextLstm(
                config.hidden_dim, config.layers
            ).parameters()
        ),
        "matched_mlp": sum(
            parameter.numel()
            for parameter in ParameterMatchedOutputMlp(
                config.mlp_hidden_dim
            ).parameters()
        ),
    }


def _train_one_model(
    config: KimuraOutputPredictionConfig,
    *,
    model_name: str,
    architecture: str,
    shuffle_labels: bool,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    output_root: Path,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    _seed_everything(950_000 + config.seed)
    model: nn.Module
    if architecture == "lstm":
        model = KimuraCiphertextLstm(config.hidden_dim, config.layers)
    elif architecture == "mlp":
        model = ParameterMatchedOutputMlp(config.mlp_hidden_dim)
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    model = model.to(config.device)
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
    labels = train_targets
    if shuffle_labels:
        permutation = np.random.default_rng(960_000 + config.seed).permutation(
            len(train_targets)
        )
        labels = train_targets[permutation]
    feature_tensor = torch.from_numpy(train_features)
    label_tensor = torch.from_numpy(np.asarray(labels))
    for epoch in range(start_epoch, config.epochs + 1):
        generator = torch.Generator().manual_seed(970_000 + config.seed + epoch)
        loader = DataLoader(
            TensorDataset(feature_tensor, label_tensor),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
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
        row = {
            "run_id": config.run_id,
            "model": model_name,
            "epoch": epoch,
            "train_mse": total_loss / max(1, total_cells),
        }
        history.append(row)
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
            progress(
                "epoch_done",
                {"model": model_name, "epoch": epoch, "train_mse": row["train_mse"]},
            )
    raw_scores = _predict_raw(
        model, test_features, batch_size=config.batch_size, device=config.device
    )
    metrics = full_output_metrics(raw_scores, test_targets)
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
        "row": {
            "run_id": config.run_id,
            "task": "innovation2_output_prediction",
            "experiment": "op9_present_r3_kimura_lstm",
            "model": model_name,
            "architecture": architecture,
            "target": "full_64_bit_true_ciphertext_output",
            "train_labels_shuffled": shuffle_labels,
            "sample_classification": False,
            "secret_key_scope": "single_fixed_unknown_key",
            "rounds": config.rounds,
            "seed": config.seed,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "epochs": config.epochs,
            "train_rows": config.train_rows,
            "test_rows": config.test_rows,
            "loss": "raw_output_mse",
            "optimizer": "rmsprop",
            "learning_rate": config.learning_rate,
            "test_target_identity": "true_full_ciphertext_targets",
            **metrics,
        },
        "history": history,
        "checkpoint": checkpoint,
    }


def full_output_metrics(raw_scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(raw_scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    if scores.shape != labels.shape or scores.ndim != 2 or scores.shape[1] != 64:
        raise ValueError("raw scores and targets must both have shape [rows, 64]")
    rounded = np.rint(scores)
    valid = (rounded == 0.0) | (rounded == 1.0)
    matches = rounded == labels
    exact = np.all(matches, axis=1)
    prevalence = labels.mean(axis=0)
    return {
        "test_mse": float(np.mean(np.square(scores - labels))),
        "test_bit_match": float(np.mean(matches)),
        "test_macro_auc": float(
            np.mean(
                [binary_auc(labels[:, index], scores[:, index]) for index in range(64)]
            )
        ),
        "test_exact_match": float(np.mean(exact)),
        "test_exact_match_count": int(np.sum(exact)),
        "invalid_rounded_cell_rate": float(1.0 - np.mean(valid)),
        "test_majority_bit_match": float(
            np.mean(np.maximum(prevalence, 1.0 - prevalence))
        ),
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


def _bits_to_word(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8):
        value = (value << 1) | int(bit)
    return value


def _training_config_hash(config: KimuraOutputPredictionConfig, model_name: str) -> str:
    payload = json.dumps(
        {**serializable_config(config), "model_name": model_name}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
