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

from blockcipher_nd.tasks.innovation2.output_bit_discovery import (
    per_bit_metric_rows,
)
from blockcipher_nd.tasks.innovation2.output_prediction_kimura_lstm import (
    ParameterMatchedOutputMlp,
    full_output_metrics,
)


MODEL_NAMES = (
    "gift64_full64_mlp_true_output",
    "gift64_full64_mlp_label_shuffle",
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class Gift64DiscoveryTrainingConfig:
    run_id: str = "i2_output_prediction_gx1_gift64_r3_full64_readiness_seed11_20260723"
    mode: str = "readiness"
    seed: int = 11
    hidden_dim: int = 1936
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cpu"
    candidate_limit: int = 8
    minimum_auc: float = 0.510
    minimum_accuracy_margin: float = 0.005
    minimum_shuffle_auc_margin: float = 0.005
    minimum_fresh_confirmed: int = 4

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"readiness", "formal"}:
            raise ValueError("mode must be readiness or formal")
        if min(
            self.hidden_dim,
            self.epochs,
            self.batch_size,
            self.candidate_limit,
            self.minimum_fresh_confirmed,
        ) <= 0:
            raise ValueError("model, epoch, batch, and candidate values must be positive")
        if self.candidate_limit > 64:
            raise ValueError("candidate_limit cannot exceed 64 output bits")
        if self.minimum_fresh_confirmed > self.candidate_limit:
            raise ValueError("minimum_fresh_confirmed cannot exceed candidate_limit")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.mode == "formal" and (
            self.seed != 11
            or self.hidden_dim != 1936
            or self.epochs != 100
            or self.batch_size != 250
            or self.learning_rate != 1e-3
            or self.candidate_limit != 8
            or self.minimum_auc != 0.510
            or self.minimum_accuracy_margin != 0.005
            or self.minimum_shuffle_auc_margin != 0.005
            or self.minimum_fresh_confirmed != 4
        ):
            raise ValueError("formal GX1 training and candidate protocol is frozen")

    @classmethod
    def formal(
        cls,
        *,
        run_id: str = "i2_output_prediction_gx1_gift64_r3_full64_seed11_20260723",
        device: str = "cuda",
    ) -> Gift64DiscoveryTrainingConfig:
        return cls(
            run_id=run_id,
            mode="formal",
            epochs=100,
            batch_size=250,
            device=device,
        )


def train_gift64_discovery_matrix(
    config: Gift64DiscoveryTrainingConfig,
    source: dict[str, Any],
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    source_metadata = source.get("metadata")
    if not isinstance(source_metadata, dict):
        raise ValueError("GX1 source metadata is required")
    train_features = np.asarray(source["train_features"], dtype=np.float32)
    train_targets = np.asarray(source["train_targets"], dtype=np.float32)
    discovery_features = np.asarray(source["discovery_features"], dtype=np.float32)
    discovery_targets = np.asarray(source["discovery_targets"], dtype=np.float32)
    _validate_arrays(train_features, train_targets, split="train")
    _validate_arrays(discovery_features, discovery_targets, split="discovery")
    expected_source = {
        "cipher": "GIFT-64",
        "rounds": 3,
        "seed": config.seed,
        "train_rows": len(train_features),
        "discovery_rows": len(discovery_features),
    }
    if {
        key: source_metadata.get(key) for key in expected_source
    } != expected_source:
        raise ValueError("GX1 source metadata does not match training arrays")
    if not isinstance(source_metadata.get("key_seed"), int):
        raise ValueError("GX1 source metadata must declare an integer key_seed")
    if config.mode == "formal" and (
        source_metadata["key_seed"] != 11
        or len(train_features) != 1 << 17
        or len(discovery_features) != 1 << 16
    ):
        raise ValueError("formal GX1 source data protocol is frozen")
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for model_name, shuffle_labels in (
        (MODEL_NAMES[0], False),
        (MODEL_NAMES[1], True),
    ):
        result = _train_one_model(
            config,
            model_name=model_name,
            shuffle_labels=shuffle_labels,
            train_features=train_features,
            train_targets=train_targets,
            discovery_features=discovery_features,
            discovery_targets=discovery_targets,
            source_metadata=source_metadata,
            output_root=output_root,
            progress=progress,
        )
        rows.append(result["row"])
        history.extend(result["history"])
        checkpoints.append(result["checkpoint"])
    return {"rows": rows, "history": history, "checkpoints": checkpoints}


def evaluate_gift64_output_split(
    config: Gift64DiscoveryTrainingConfig,
    output_root: Path,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    split: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    feature_array = np.asarray(features, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    _validate_arrays(feature_array, target_array, split=split)
    per_bit_rows: list[dict[str, Any]] = []
    full_output_rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        model = _load_final_model(config, output_root, model_name)
        scores = _predict_raw(
            model,
            feature_array,
            batch_size=config.batch_size,
            device=config.device,
        )
        per_bit_rows.extend(
            per_bit_metric_rows(model_name, split, scores, target_array)
        )
        full_output_rows.append(
            {
                "run_id": config.run_id,
                "model": model_name,
                "split": split,
                "target": "full_64_bit_true_gift64_ciphertext_output",
                "sample_classification": False,
                **full_output_metrics(scores, target_array),
            }
        )
        if progress is not None:
            progress("split_model_evaluated", {"split": split, "model": model_name})
    return {"per_bit_rows": per_bit_rows, "full_output_rows": full_output_rows}


def select_gift64_discovery_candidates(
    config: Gift64DiscoveryTrainingConfig,
    discovery_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    indexed = {
        (str(row["model"]), int(row["msb_index"])): row
        for row in discovery_rows
    }
    expected = {(model, bit) for model in MODEL_NAMES for bit in range(64)}
    if set(indexed) != expected:
        raise ValueError("GX1 discovery rows must contain two models x 64 bits exactly")
    ranked: list[dict[str, Any]] = []
    for msb_index in range(64):
        true_row = indexed[(MODEL_NAMES[0], msb_index)]
        shuffle_row = indexed[(MODEL_NAMES[1], msb_index)]
        auc_margin = float(true_row["auc"]) - float(shuffle_row["auc"])
        selection_score = min(
            float(true_row["auc"]) - 0.5,
            float(true_row["accuracy_minus_majority"]),
            auc_margin,
        )
        ranked.append(
            {
                "msb_index": msb_index,
                "integer_bit": 63 - msb_index,
                "nibble_msb_index": msb_index // 4,
                "bit_in_nibble_msb": msb_index % 4,
                "selector_model": MODEL_NAMES[0],
                "shuffle_control_model": MODEL_NAMES[1],
                "shuffle_control_scope": "architecture_matched",
                "eligible": (
                    float(true_row["auc"]) >= config.minimum_auc
                    and float(true_row["accuracy_minus_majority"])
                    >= config.minimum_accuracy_margin
                    and auc_margin >= config.minimum_shuffle_auc_margin
                ),
                "selection_score": selection_score,
                "discovery_auc": float(true_row["auc"]),
                "discovery_accuracy": float(true_row["threshold_accuracy"]),
                "discovery_majority_accuracy": float(
                    true_row["majority_accuracy"]
                ),
                "discovery_accuracy_margin": float(
                    true_row["accuracy_minus_majority"]
                ),
                "discovery_shuffle_auc": float(shuffle_row["auc"]),
                "discovery_auc_minus_shuffle": auc_margin,
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            -float(row["discovery_auc"]),
            int(row["msb_index"]),
        )
    )
    candidates = [row for row in ranked if row["eligible"]][
        : config.candidate_limit
    ]
    return {
        "run_id": config.run_id,
        "cipher": "GIFT-64",
        "rounds": 3,
        "selection_split": "discovery_plaintexts_only",
        "confirmation_split": "fresh_not_generated_or_read_when_frozen",
        "target": "selected_true_ciphertext_output_bits",
        "sample_classification": False,
        "bit_order": "msb_first",
        "candidate_limit": config.candidate_limit,
        "thresholds": {
            "minimum_auc": config.minimum_auc,
            "minimum_accuracy_margin": config.minimum_accuracy_margin,
            "minimum_shuffle_auc_margin": config.minimum_shuffle_auc_margin,
        },
        "candidate_msb_indices": [row["msb_index"] for row in candidates],
        "candidates": candidates,
        "all_64_discovery_ranking": ranked,
    }


def freeze_gift64_candidates(
    candidates: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_path = output_root / "candidates.json"
    candidate_path.write_bytes(_candidate_payload_bytes(candidates))
    digest = gift64_candidate_sha256(candidates)
    (output_root / "candidates.sha256").write_text(
        f"{digest}  candidates.json\n", encoding="ascii"
    )
    return {
        "candidate_path": candidate_path,
        "candidate_sha256": digest,
        "event": "candidates_frozen_before_fresh_generation",
    }


def gift64_candidate_sha256(candidates: dict[str, Any]) -> str:
    return hashlib.sha256(_candidate_payload_bytes(candidates)).hexdigest()


def adjudicate_gift64_discovery(
    config: Gift64DiscoveryTrainingConfig,
    source_checks: dict[str, bool],
    fresh_checks: dict[str, bool],
    training: dict[str, Any],
    discovery_rows: list[dict[str, Any]],
    fresh_rows: list[dict[str, Any]],
    candidates: dict[str, Any],
    *,
    candidate_sha256: str,
) -> dict[str, Any]:
    discovery_index = _index_metric_rows(discovery_rows, split="discovery")
    fresh_index = _index_metric_rows(fresh_rows, split="fresh_confirmation")
    training_rows = {row["model"]: row for row in training["rows"]}
    checkpoint_rows = {row["model"]: row for row in training["checkpoints"]}
    initial_hashes = [
        row.get("initial_state_sha256") for row in checkpoint_rows.values()
    ]
    parameter_counts = [
        int(row.get("parameters", -1)) for row in training_rows.values()
    ]
    training_key_seeds = [row.get("key_seed") for row in training_rows.values()]
    required_metric_fields = (
        "discovery_mse",
        "discovery_bit_match",
        "discovery_macro_auc",
        "discovery_exact_match",
        "discovery_exact_match_count",
        "discovery_invalid_rounded_cell_rate",
        "discovery_majority_bit_match",
    )
    execution_checks = {
        "two_training_rows_complete": set(training_rows) == set(MODEL_NAMES),
        "history_rows_complete": len(training["history"]) == config.epochs * 2,
        "two_checkpoint_hashes_present": set(checkpoint_rows) == set(MODEL_NAMES)
        and all(row.get("sha256") for row in checkpoint_rows.values()),
        "matched_models_share_initialization": len(initial_hashes) == 2
        and all(initial_hashes)
        and len(set(initial_hashes)) == 1,
        "matched_models_have_equal_parameters": len(parameter_counts) == 2
        and min(parameter_counts) > 0
        and len(set(parameter_counts)) == 1,
        "training_rows_declare_same_fixed_gift64_key": len(training_key_seeds)
        == 2
        and all(isinstance(value, int) for value in training_key_seeds)
        and len(set(training_key_seeds)) == 1
        and all(
            row.get("cipher") == "GIFT-64"
            and row.get("secret_key_scope") == "single_fixed_unknown_key"
            for row in training_rows.values()
        ),
        "shuffle_changes_only_training_target_order": training_rows.get(
            MODEL_NAMES[1], {}
        ).get("test_target_identity")
        == "true_full_gift64_ciphertext_targets",
        "discovery_rows_complete": len(discovery_index) == 128,
        "fresh_rows_complete": len(fresh_index) == 128,
        "candidate_count_within_limit": len(candidates.get("candidates", []))
        <= config.candidate_limit,
        "candidate_hash_matches_payload": len(candidate_sha256) == 64
        and all(character in "0123456789abcdef" for character in candidate_sha256)
        and candidate_sha256 == gift64_candidate_sha256(candidates),
        "candidate_freeze_declares_fresh_blindness": candidates.get(
            "confirmation_split"
        )
        == "fresh_not_generated_or_read_when_frozen",
        "all_training_metrics_finite": all(
            field in row and math.isfinite(float(row[field]))
            for row in training_rows.values()
            for field in required_metric_fields
        ),
    }
    confirmations: list[dict[str, Any]] = []
    for candidate in candidates.get("candidates", []):
        msb_index = int(candidate["msb_index"])
        true_row = fresh_index[(MODEL_NAMES[0], msb_index)]
        shuffle_row = fresh_index[(MODEL_NAMES[1], msb_index)]
        auc_margin = float(true_row["auc"]) - float(shuffle_row["auc"])
        passed = (
            float(true_row["auc"]) >= config.minimum_auc
            and float(true_row["accuracy_minus_majority"])
            >= config.minimum_accuracy_margin
            and auc_margin >= config.minimum_shuffle_auc_margin
        )
        confirmations.append(
            {
                "msb_index": msb_index,
                "integer_bit": 63 - msb_index,
                "fresh_auc": float(true_row["auc"]),
                "fresh_accuracy": float(true_row["threshold_accuracy"]),
                "fresh_majority_accuracy": float(true_row["majority_accuracy"]),
                "fresh_accuracy_margin": float(
                    true_row["accuracy_minus_majority"]
                ),
                "fresh_shuffle_auc": float(shuffle_row["auc"]),
                "fresh_auc_minus_shuffle": auc_margin,
                "passed": passed,
            }
        )
    confirmed = [row for row in confirmations if row["passed"]]
    protocol_valid = (
        all(source_checks.values())
        and all(fresh_checks.values())
        and all(execution_checks.values())
    )
    if not protocol_valid:
        status = "fail"
        decision = "innovation2_gift64_r3_output_prediction_protocol_invalid"
        next_adjudication = "repair_gx1_protocol_only"
        action = "repair only GX1 data, source, checkpoint, freeze, metric, or control protocol"
    elif len(confirmed) >= config.minimum_fresh_confirmed:
        status = "pass"
        decision = "innovation2_gift64_r3_true_output_bits_fresh_confirmed"
        next_adjudication = "gx2_selected8_architecture_screen"
        action = (
            "freeze the confirmed GIFT positions and preregister the same-budget "
            "selected-output generic, position-preserving, and GIFT-SPN-aware screen"
        )
    else:
        status = "hold"
        decision = "innovation2_gift64_r3_true_output_bits_not_confirmed"
        next_adjudication = "close_gift_output_position_route"
        action = (
            "stop GIFT position, sample, epoch, seed, and lower-round rescue; "
            "retain the PRESENT boundary and continue the independent SPECK calibration"
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": {
            "source": source_checks,
            "fresh": fresh_checks,
            "execution": execution_checks,
        },
        "metrics": {
            "candidate_count": len(candidates.get("candidates", [])),
            "fresh_confirmed_count": len(confirmed),
            "fresh_confirmed_msb_indices": [row["msb_index"] for row in confirmed],
            "minimum_fresh_confirmed": config.minimum_fresh_confirmed,
        },
        "candidate_confirmation": confirmations,
        "claim_scope": (
            "single-fixed-key GIFT-64 r3 selected true ciphertext output bits on "
            "fresh unseen plaintexts; not full-ciphertext recovery, cross-key evidence, "
            "architecture attribution, higher-round attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "remote_scale": False,
            "target": "selected_true_ciphertext_output_bits",
            "sample_classification": False,
        },
    }


def serializable_training_config(
    config: Gift64DiscoveryTrainingConfig,
) -> dict[str, Any]:
    return asdict(config)


def _train_one_model(
    config: Gift64DiscoveryTrainingConfig,
    *,
    model_name: str,
    shuffle_labels: bool,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    discovery_features: np.ndarray,
    discovery_targets: np.ndarray,
    source_metadata: dict[str, Any],
    output_root: Path,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    _seed_everything(1_140_000 + config.seed)
    model = ParameterMatchedOutputMlp(config.hidden_dim).to(config.device)
    initial_state_sha256 = _state_dict_sha256(model.state_dict())
    parameters = sum(parameter.numel() for parameter in model.parameters())
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
    labels = train_targets
    if shuffle_labels:
        permutation = np.random.default_rng(1_150_000 + config.seed).permutation(
            len(train_targets)
        )
        labels = train_targets[permutation]
    feature_tensor = torch.from_numpy(train_features)
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    for epoch in range(start_epoch, config.epochs + 1):
        generator = torch.Generator().manual_seed(1_160_000 + config.seed + epoch)
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
    raw_scores = _predict_raw(
        model,
        discovery_features,
        batch_size=config.batch_size,
        device=config.device,
    )
    metrics = full_output_metrics(raw_scores, discovery_targets)
    torch.save(
        {
            "config_hash": config_hash,
            "epoch": config.epochs,
            "model_state": model.state_dict(),
            "initial_state_sha256": initial_state_sha256,
        },
        final_path,
    )
    checkpoint_row = {
        "model": model_name,
        "path": str(final_path.relative_to(output_root)),
        "sha256": _sha256(final_path),
        "config_hash": config_hash,
        "initial_state_sha256": initial_state_sha256,
    }
    return {
        "row": {
            "run_id": config.run_id,
            "task": "innovation2_output_prediction",
            "experiment": "gx1_gift64_r3_full64_discovery",
            "cipher": "GIFT-64",
            "model": model_name,
            "architecture": "parameter_matched_output_mlp",
            "target": "full_64_bit_true_gift64_ciphertext_output",
            "train_labels_shuffled": shuffle_labels,
            "sample_classification": False,
            "secret_key_scope": "single_fixed_unknown_key",
            "rounds": 3,
            "seed": config.seed,
            "key_seed": int(source_metadata["key_seed"]),
            "parameters": parameters,
            "epochs": config.epochs,
            "train_rows": len(train_features),
            "discovery_rows": len(discovery_features),
            "loss": "raw_output_mse",
            "optimizer": "rmsprop",
            "learning_rate": config.learning_rate,
            "test_target_identity": "true_full_gift64_ciphertext_targets",
            **{f"discovery_{key.removeprefix('test_')}": value for key, value in metrics.items()},
        },
        "history": history,
        "checkpoint": checkpoint_row,
    }


def _load_final_model(
    config: Gift64DiscoveryTrainingConfig,
    output_root: Path,
    model_name: str,
) -> nn.Module:
    path = output_root / "models" / f"{model_name}_final.pt"
    if not path.exists():
        raise ValueError(f"missing GX1 final checkpoint: {path}")
    checkpoint = torch.load(path, map_location=config.device, weights_only=False)
    if checkpoint.get("config_hash") != _training_config_hash(config, model_name):
        raise ValueError(f"GX1 final checkpoint config mismatch for {model_name}")
    model = ParameterMatchedOutputMlp(config.hidden_dim).to(config.device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def _index_metric_rows(
    rows: list[dict[str, Any]], *, split: str
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {
        (str(row["model"]), int(row["msb_index"])): row
        for row in rows
        if row.get("split") == split
    }
    expected = {(model, bit) for model in MODEL_NAMES for bit in range(64)}
    if set(indexed) != expected:
        raise ValueError(f"GX1 {split} rows must contain two models x 64 bits exactly")
    return indexed


def _validate_arrays(
    features: np.ndarray, targets: np.ndarray, *, split: str
) -> None:
    if (
        features.ndim != 2
        or targets.ndim != 2
        or features.shape != targets.shape
        or features.shape[1] != 64
        or len(features) == 0
    ):
        raise ValueError(f"GX1 {split} features and targets must be non-empty [rows, 64]")


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


def _training_config_hash(
    config: Gift64DiscoveryTrainingConfig, model_name: str
) -> str:
    payload = json.dumps(
        {**serializable_training_config(config), "model_name": model_name},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _candidate_payload_bytes(candidates: dict[str, Any]) -> bytes:
    return (
        json.dumps(candidates, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "MODEL_NAMES",
    "Gift64DiscoveryTrainingConfig",
    "adjudicate_gift64_discovery",
    "evaluate_gift64_output_split",
    "freeze_gift64_candidates",
    "gift64_candidate_sha256",
    "select_gift64_discovery_candidates",
    "serializable_training_config",
    "train_gift64_discovery_matrix",
]
