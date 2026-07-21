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
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
    SelectedOutputBitHeadConfig,
    SelectedOutputMlp,
    prepare_selected_output_data,
    validate_selected_output_contract,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i2_output_prediction_op12_present_r4_structured_xor_smoke_20260721"
REMOTE_RUN_ID = (
    "i2_output_prediction_op12_present_r4_structured_xor_key1_gpu0_20260721"
)

STRUCTURED_MASKS = (
    ("same_sbox_pair_0_32", (0, 32), "same_sbox_pair"),
    ("same_sbox_pair_2_34", (2, 34), "same_sbox_pair"),
    ("same_sbox_pair_8_40", (8, 40), "same_sbox_pair"),
    ("same_sbox_pair_10_42", (10, 42), "same_sbox_pair"),
    ("same_role4_0_2_8_10", (0, 2, 8, 10), "same_role_four"),
    ("same_role4_32_34_40_42", (32, 34, 40, 42), "same_role_four"),
)

GEOMETRY_CONTROL_MASKS = (
    ("output_nibble_pair_0_2", (0, 2), "output_nibble_pair"),
    ("output_nibble_pair_8_10", (8, 10), "output_nibble_pair"),
    ("output_nibble_pair_32_34", (32, 34), "output_nibble_pair"),
    ("output_nibble_pair_40_42", (40, 42), "output_nibble_pair"),
    ("mixed_role4_0_2_32_34", (0, 2, 32, 34), "mixed_role_four"),
    ("mixed_role4_8_10_40_42", (8, 10, 40, 42), "mixed_role_four"),
)

MODEL_SPECS = (
    ("selected8_mlp_true_output", "selected8", False),
    ("structured6_mlp_true_xor", "structured6", False),
    ("geometry6_mlp_true_xor", "geometry6", False),
    ("structured6_mlp_label_shuffle", "structured6", True),
)

ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class StructuredOutputXorConfig:
    run_id: str = RUN_ID
    mode: str = "smoke"
    rounds: int = 4
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
    minimum_control_auc_margin: float = 0.005
    minimum_derived_auc_margin: float = 0.005
    minimum_component_auc_margin: float = 0.002
    minimum_pair_masks: int = 2
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "round_extension"}:
            raise ValueError("invalid structured-XOR mode")
        if self.rounds != 4:
            raise ValueError("OP12 is frozen to PRESENT round four")
        if self.seed != 1:
            raise ValueError("OP12 is frozen to the second fixed-key seed")
        if self.selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("OP12 selected positions must match OP10 and OP11")
        if min(
            self.train_rows,
            self.test_rows,
            self.hidden_dim,
            self.epochs,
            self.batch_size,
            self.data_chunk_rows,
        ) <= 0:
            raise ValueError("row, model, epoch, batch, and chunk values must be positive")

    @classmethod
    def round_extension(
        cls,
        *,
        run_id: str = REMOTE_RUN_ID,
        device: str = "cuda",
    ) -> StructuredOutputXorConfig:
        return cls(
            run_id=run_id,
            mode="round_extension",
            train_rows=1 << 17,
            test_rows=1 << 16,
            epochs=100,
            batch_size=250,
            data_chunk_rows=4096,
            device=device,
        )


def prepare_structured_xor_data(
    config: StructuredOutputXorConfig,
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    selected_config = SelectedOutputBitHeadConfig(
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
    return prepare_selected_output_data(
        selected_config, output_root, progress=progress
    )


def validate_structured_xor_contract(
    config: StructuredOutputXorConfig,
    data: dict[str, Any],
) -> dict[str, bool]:
    base_config = SelectedOutputBitHeadConfig(
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
    checks = validate_selected_output_contract(base_config, data)
    structured_bits = [bits for _, bits, _ in STRUCTURED_MASKS]
    control_bits = [bits for _, bits, _ in GEOMETRY_CONTROL_MASKS]
    pair_sources = [tuple(_inverse_p_source(bit) for bit in bits) for bits in structured_bits[:4]]
    control_sources = [tuple(_inverse_p_source(bit) for bit in bits) for bits in control_bits[:4]]
    role_sources = [tuple(_inverse_p_source(bit) for bit in bits) for bits in structured_bits[4:]]
    selected = set(config.selected_msb_indices)
    checks.update(
        {
            "six_structured_masks_are_unique": len(set(structured_bits)) == 6,
            "six_geometry_masks_are_unique": len(set(control_bits)) == 6,
            "all_masks_use_only_preregistered_bits": all(
                set(bits) <= selected for bits in structured_bits + control_bits
            ),
            "mask_weights_are_pair4_pair4_role4_role4": [
                len(bits) for bits in structured_bits
            ]
            == [2, 2, 2, 2, 4, 4]
            and [len(bits) for bits in control_bits] == [2, 2, 2, 2, 4, 4],
            "primary_pairs_share_inverse_p_sbox": all(
                left // 4 == right // 4 for left, right in pair_sources
            ),
            "pair_controls_do_not_share_inverse_p_sbox": all(
                left // 4 != right // 4 for left, right in control_sources
            ),
            "role4_masks_have_one_inverse_p_role": all(
                len({source % 4 for source in sources}) == 1
                for sources in role_sources
            ),
            "pair_family_bit_frequency_is_matched": _bit_frequencies(
                structured_bits[:4]
            )
            == _bit_frequencies(control_bits[:4]),
            "role4_family_bit_frequency_is_matched": _bit_frequencies(
                structured_bits[4:]
            )
            == _bit_frequencies(control_bits[4:]),
            "labels_are_true_output_xors_not_sample_classes": True,
        }
    )
    return checks


def train_structured_xor_matrix(
    config: StructuredOutputXorConfig,
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
    train_structured = xor_targets(train_full, STRUCTURED_MASKS)
    test_structured = xor_targets(test_full, STRUCTURED_MASKS)
    train_geometry = xor_targets(train_full, GEOMETRY_CONTROL_MASKS)
    test_geometry = xor_targets(test_full, GEOMETRY_CONTROL_MASKS)
    target_sets = {
        "selected8": (train_selected, test_selected),
        "structured6": (train_structured, test_structured),
        "geometry6": (train_geometry, test_geometry),
    }
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    scores: dict[str, np.ndarray] = {}
    for model_name, target_mode, shuffle_labels in MODEL_SPECS:
        train_targets, test_targets = target_sets[target_mode]
        train_targets = np.array(train_targets, copy=True)
        if shuffle_labels:
            permutation = np.random.default_rng(1_220_000 + config.seed).permutation(
                len(train_targets)
            )
            train_targets = train_targets[permutation]
        result = _train_one_model(
            config,
            model_name=model_name,
            target_mode=target_mode,
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
    _attach_structured_comparisons(
        rows,
        selected_scores=scores["selected8_mlp_true_output"],
        selected_targets=test_selected,
        structured_targets=test_structured,
        selected_bits=config.selected_msb_indices,
    )
    return {
        "rows": rows,
        "summaries": summaries,
        "history": history,
        "checkpoints": checkpoints,
    }


def adjudicate_structured_xor(
    config: StructuredOutputXorConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
) -> dict[str, Any]:
    structured = [
        row for row in training["rows"] if row["model"] == "structured6_mlp_true_xor"
    ]
    mask_gates: list[dict[str, Any]] = []
    for row in structured:
        checks = {
            "auc_at_least_0_510": float(row["auc"]) >= config.minimum_auc,
            "accuracy_margin_at_least_0_005": float(
                row["accuracy_minus_majority"]
            )
            >= config.minimum_accuracy_margin,
            "auc_minus_shuffle_at_least_0_005": float(row["auc_minus_shuffle"])
            >= config.minimum_shuffle_auc_margin,
            "auc_minus_geometry_at_least_0_005": float(row["auc_minus_geometry"])
            >= config.minimum_control_auc_margin,
            "auc_minus_derived_at_least_0_005": float(row["auc_minus_derived"])
            >= config.minimum_derived_auc_margin,
            "auc_minus_best_component_at_least_0_002": float(
                row["auc_minus_best_component"]
            )
            >= config.minimum_component_auc_margin,
        }
        mask_gates.append(
            {
                "mask_name": row["mask_name"],
                "mask_bits": row["mask_bits"],
                "family": row["family"],
                "auc": row["auc"],
                "accuracy_minus_majority": row["accuracy_minus_majority"],
                "auc_minus_shuffle": row["auc_minus_shuffle"],
                "auc_minus_geometry": row["auc_minus_geometry"],
                "auc_minus_derived": row["auc_minus_derived"],
                "auc_minus_best_component": row["auc_minus_best_component"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    pair_passed = [
        row
        for row in mask_gates
        if row["family"] == "same_sbox_pair" and row["passed"]
    ]
    role_passed = [
        row
        for row in mask_gates
        if row["family"] == "same_role_four" and row["passed"]
    ]
    pair_family_pass = len(pair_passed) >= config.minimum_pair_masks
    role_family_pass = len(role_passed) == 2
    execution_checks = {
        "four_models_complete": len(training["summaries"]) == 4,
        "twenty_six_result_rows_complete": len(training["rows"]) == 26,
        "history_rows_complete": len(training["history"]) == config.epochs * 4,
        "four_checkpoint_hashes_present": len(training["checkpoints"]) == 4
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
        "structured_rows_have_all_comparisons": len(structured) == 6
        and all(
            field in row
            for row in structured
            for field in (
                "shuffle_auc",
                "geometry_auc",
                "derived_auc",
                "best_component_auc",
            )
        ),
    }
    all_valid = all(protocol_checks.values()) and all(execution_checks.values())
    if not all_valid:
        status = "fail"
        decision = "innovation2_r4_structured_xor_protocol_invalid"
        next_adjudication = "op12_protocol_repair"
        action = "repair only data, mask, target, shuffle, metric, cache, checkpoint, or plotting protocol"
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_r4_structured_xor_local_smoke_passed"
        next_adjudication = "op12_remote_r4_structured_xor"
        action = "launch the frozen four-row matrix at 131072 train, 65536 test, and 100 epochs on A6000"
    elif pair_family_pass or role_family_pass:
        status = "pass"
        decision = "innovation2_r4_structured_xor_supported"
        next_adjudication = "op13_r4_structured_xor_seed0_confirmation"
        action = "repeat the identical masks and four-row matrix under the seed0 fixed secret key before any r5 step"
    else:
        status = "hold"
        decision = "innovation2_r4_structured_xor_not_supported"
        next_adjudication = "innovation2_output_prediction_thesis_boundary"
        action = "stop mask search, data, epoch, model, and round scaling; retain the OP11 r3 selected-bit result"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "mask_gates": mask_gates,
        "metrics": {
            "passed_mask_count": sum(row["passed"] for row in mask_gates),
            "passed_mask_names": [
                row["mask_name"] for row in mask_gates if row["passed"]
            ],
            "pair_family_pass_count": len(pair_passed),
            "pair_family_passed": pair_family_pass,
            "role4_family_pass_count": len(role_passed),
            "role4_family_passed": role_family_pass,
            "mean_structured_auc": _mean_field(structured, "auc"),
            "mean_geometry_auc": _mean_field(structured, "geometry_auc"),
            "mean_shuffle_auc": _mean_field(structured, "shuffle_auc"),
            "mean_derived_auc": _mean_field(structured, "derived_auc"),
            "mean_best_component_auc": _mean_field(
                structured, "best_component_auc"
            ),
        },
        "claim_scope": (
            "local implementation smoke"
            if config.mode == "smoke"
            else "single seed1 fixed-key PRESENT r4 direct prediction of six preregistered structured output-XOR functions"
        )
        + "; not sample classification, integral balance, broad cross-key statistics, r5 evidence, or SOTA",
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "sample_classification": False,
            "target": "preregistered_true_ciphertext_output_xor_values",
        },
    }


def xor_targets(
    full_targets: np.ndarray,
    masks: tuple[tuple[str, tuple[int, ...], str], ...],
) -> np.ndarray:
    return np.stack(
        [
            np.bitwise_xor.reduce(
                np.asarray(full_targets[:, bits], dtype=np.uint8), axis=1
            )
            for _, bits, _ in masks
        ],
        axis=1,
    ).astype(np.float32)


def parameter_counts(config: StructuredOutputXorConfig) -> dict[str, int]:
    return {
        "selected8_mlp": sum(
            parameter.numel()
            for parameter in SelectedOutputMlp(
                config.hidden_dim, output_bits=8
            ).parameters()
        ),
        "xor6_mlp": sum(
            parameter.numel()
            for parameter in SelectedOutputMlp(
                config.hidden_dim, output_bits=6
            ).parameters()
        ),
    }


def serializable_config(config: StructuredOutputXorConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["selected_msb_indices"] = list(config.selected_msb_indices)
    return payload


def _train_one_model(
    config: StructuredOutputXorConfig,
    *,
    model_name: str,
    target_mode: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    output_root: Path,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    _seed_everything(1_210_000 + config.seed)
    output_bits = train_targets.shape[1]
    model: nn.Module = SelectedOutputMlp(config.hidden_dim, output_bits=output_bits)
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
                1_230_000 + config.seed + epoch
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
                "target_kind": "preregistered_true_ciphertext_output_bit",
                "sample_classification": False,
                "msb_index": bit,
                "integer_bit": 63 - bit,
            }
            for index, bit in enumerate(config.selected_msb_indices)
        ]
    else:
        mask_specs = (
            STRUCTURED_MASKS
            if target_mode == "structured6"
            else GEOMETRY_CONTROL_MASKS
        )
        rows = [
            {
                **_binary_metrics(scores[:, index], test_targets[:, index]),
                "model": model_name,
                "target_kind": "preregistered_true_ciphertext_output_xor",
                "sample_classification": False,
                "mask_name": name,
                "mask_bits": list(bits),
                "mask_weight": len(bits),
                "family": family,
                "test_target_identity": "true_ciphertext_output_xor_targets",
            }
            for index, (name, bits, family) in enumerate(mask_specs)
        ]
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
            "target_mode": target_mode,
            "output_bits": output_bits,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train_labels_shuffled": model_name.endswith("label_shuffle"),
            "mean_auc": _mean_field(rows, "auc"),
            "mean_accuracy_margin": _mean_field(
                rows, "accuracy_minus_majority"
            ),
        },
        "history": history,
        "checkpoint": checkpoint,
        "scores": scores,
    }


def _attach_structured_comparisons(
    rows: list[dict[str, Any]],
    *,
    selected_scores: np.ndarray,
    selected_targets: np.ndarray,
    structured_targets: np.ndarray,
    selected_bits: tuple[int, ...],
) -> None:
    bit_to_column = {bit: index for index, bit in enumerate(selected_bits)}
    selected_rows = {
        int(row["msb_index"]): row
        for row in rows
        if row["model"] == "selected8_mlp_true_output"
    }
    geometry_rows = {
        str(row["mask_name"]): row
        for row in rows
        if row["model"] == "geometry6_mlp_true_xor"
    }
    shuffle_rows = {
        str(row["mask_name"]): row
        for row in rows
        if row["model"] == "structured6_mlp_label_shuffle"
    }
    structured_rows = [
        row for row in rows if row["model"] == "structured6_mlp_true_xor"
    ]
    for index, row in enumerate(structured_rows):
        name, bits, _ = STRUCTURED_MASKS[index]
        control_name = GEOMETRY_CONTROL_MASKS[index][0]
        columns = [bit_to_column[bit] for bit in bits]
        component_scores = selected_scores[:, columns]
        clipped = np.clip(component_scores, 0.0, 1.0)
        derived_scores = (
            1.0 - np.prod(1.0 - 2.0 * clipped, axis=1, dtype=np.float64)
        ) / 2.0
        derived = _binary_metrics(derived_scores, structured_targets[:, index])
        geometry = geometry_rows[control_name]
        shuffled = shuffle_rows[name]
        best_component_auc = max(float(selected_rows[bit]["auc"]) for bit in bits)
        row.update(
            {
                "paired_geometry_control": control_name,
                "geometry_auc": float(geometry["auc"]),
                "shuffle_auc": float(shuffled["auc"]),
                "derived_auc": float(derived["auc"]),
                "derived_threshold_accuracy": float(
                    derived["threshold_accuracy"]
                ),
                "derived_clip_rate": float(
                    np.mean((component_scores < 0.0) | (component_scores > 1.0))
                ),
                "best_component_auc": best_component_auc,
                "auc_minus_geometry": float(row["auc"])
                - float(geometry["auc"]),
                "auc_minus_shuffle": float(row["auc"])
                - float(shuffled["auc"]),
                "auc_minus_derived": float(row["auc"])
                - float(derived["auc"]),
                "auc_minus_best_component": float(row["auc"])
                - best_component_auc,
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
        "invalid_numpy_rint_rate": float(
            np.mean(~np.isin(np.rint(bit_scores), (0.0, 1.0)))
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


def _inverse_p_source(msb_index: int) -> int:
    integer_bit = 63 - msb_index
    source_word = Present80.inverse_permutation_layer(1 << integer_bit)
    if source_word.bit_count() != 1:
        raise ValueError("inverse P-layer source must be one bit")
    source = source_word.bit_length() - 1
    if Present80.permutation_layer(1 << source) != 1 << integer_bit:
        raise ValueError("inverse/forward P-layer round trip failed")
    return source


def _bit_frequencies(masks: list[tuple[int, ...]]) -> dict[int, int]:
    frequencies: dict[int, int] = {}
    for bits in masks:
        for bit in bits:
            frequencies[bit] = frequencies.get(bit, 0) + 1
    return frequencies


def _training_config_hash(
    config: StructuredOutputXorConfig, model_name: str
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


def _mean_field(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "GEOMETRY_CONTROL_MASKS",
    "REMOTE_RUN_ID",
    "RUN_ID",
    "STRUCTURED_MASKS",
    "StructuredOutputXorConfig",
    "adjudicate_structured_xor",
    "parameter_counts",
    "prepare_structured_xor_data",
    "serializable_config",
    "train_structured_xor_matrix",
    "validate_structured_xor_contract",
    "xor_targets",
]
