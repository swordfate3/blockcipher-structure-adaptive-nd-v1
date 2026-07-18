from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from blockcipher_nd.models.structure.spn.present_monomial_support_propagation import (
    PresentDegreeSpectrumDistillationNetwork,
    PresentMspnSpec,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    anf_prefix_features,
)
from blockcipher_nd.tasks.innovation2.present_support_identity_collision import (
    load_e48_sources,
    validate_e48_sources,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    possible_active_monomials,
)
from blockcipher_nd.training.metrics import binary_auc


E47_RUN_ID = "i2_present_r4_mspn_neural_attribution_seed0_20260718"
E47_DECISION = "innovation2_present_mspn_candidate_not_ready"
E47_TRUE_AUC = 0.5186727951738006
E47_PARAMETER_COUNT = 17788
E48_RUN_ID = "i2_present_r4_support_identity_collision_20260718"
E48_DECISION = "innovation2_present_support_identity_not_supported"
E48_DEGREE_AUC = 0.6891697787991956
TEACHER_SHUFFLE_SEED = 49001


@dataclass(frozen=True)
class DegreeSpectrumDistillationConfig:
    run_id: str
    mode: str = "smoke"
    epochs: int = 2
    batch_size: int = 32
    hidden_dim: int = 32
    degree_channels: int = 9
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    auxiliary_scale: float = 0.25
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode != "smoke":
            raise ValueError("E49 only defines smoke mode")
        if (
            self.epochs != 2
            or self.batch_size != 32
            or self.hidden_dim != 32
            or self.degree_channels != 9
            or self.dropout != 0.10
            or self.auxiliary_scale != 0.25
            or self.seed != 0
            or self.device != "cpu"
        ):
            raise ValueError("E49 smoke protocol is frozen")


def load_e49_sources(
    atlas_root: Path,
    e45_root: Path,
    e47_root: Path,
    e48_root: Path,
) -> dict[str, Any]:
    sources = load_e48_sources(atlas_root, e45_root, e47_root)
    sources["e48_gate"] = _read_json(e48_root / "gate.json")
    sources["e48_hashes"] = {
        name: _sha256(e48_root / name) for name in ("gate.json", "results.jsonl")
    }
    return sources


def validate_e49_sources(sources: dict[str, Any]) -> dict[str, bool]:
    e48_sources = validate_e48_sources(sources, strict=True)
    e48_gate = sources["e48_gate"]
    return {
        **{f"e48_source_{key}": value for key, value in e48_sources.items()},
        "e48_run_id_matches": e48_gate.get("run_id") == E48_RUN_ID,
        "e48_decision_matches": e48_gate.get("decision") == E48_DECISION,
        "e48_status_hold": e48_gate.get("status") == "hold",
        "e48_degree_auc_matches": math.isclose(
            float(e48_gate["metrics"]["degree_validation_auc"]),
            E48_DEGREE_AUC,
            abs_tol=1e-12,
        ),
        "e48_hashes_present": all(
            len(value) == 64 for value in sources["e48_hashes"].values()
        ),
    }


def build_degree_spectrum_targets(
    data: dict[str, Any], topology_mode: str
) -> np.ndarray:
    if topology_mode not in {"true", "corrupted"}:
        raise ValueError("topology_mode must be true or corrupted")
    player = np.asarray(data["players"], dtype=np.int64)
    if topology_mode == "corrupted":
        player = topology_players(player, "corrupted")
    player = player[0]
    structure_indices = sorted(
        {int(row["structure_index"]) for row in data["rows"]}
    )
    supports = {
        structure_index: {
            rounds: possible_active_monomials(
                tuple(data["structures"][structure_index]["active_bits"]),
                rounds,
                player=player,
            )
            for rounds in (1, 2, 3)
        }
        for structure_index in structure_indices
    }
    targets = np.zeros((len(data["rows"]), 3, 13), dtype=np.float32)
    for row_index, row in enumerate(data["rows"]):
        selected = np.flatnonzero(
            data["output_mask_bits"][int(row["mask_index"])]
        )
        targets[row_index] = anf_prefix_features(
            selected, supports[int(row["structure_index"])]
        ).reshape(3, 13)
    return targets


def build_teacher_bundle(data: dict[str, Any]) -> dict[str, Any]:
    split = np.asarray([row["split"] for row in data["rows"]])
    train = split == "train"
    raw = {
        topology_mode: build_degree_spectrum_targets(data, topology_mode)
        for topology_mode in ("true", "corrupted")
    }
    normalized: dict[str, np.ndarray] = {}
    statistics: dict[str, dict[str, np.ndarray]] = {}
    for topology_mode, targets in raw.items():
        mean = targets[train].mean(axis=0)
        scale = targets[train].std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
        normalized[topology_mode] = ((targets - mean) / scale).astype(np.float32)
        statistics[topology_mode] = {"mean": mean, "scale": scale}
    return {
        "raw": raw,
        "normalized": normalized,
        "statistics": statistics,
        "train_mask": train,
        "validation_mask": split == "validation",
    }


def shuffled_training_targets(
    targets: np.ndarray, train_mask: np.ndarray, *, seed: int = TEACHER_SHUFFLE_SEED
) -> tuple[np.ndarray, np.ndarray]:
    train_indices = np.flatnonzero(train_mask)
    permutation = np.random.default_rng(seed).permutation(len(train_indices))
    output = targets.copy()
    output[train_indices] = targets[train_indices[permutation]]
    return output, permutation


def measure_distillation_contract(
    config: DegreeSpectrumDistillationConfig,
    data: dict[str, Any],
    teachers: dict[str, Any],
) -> dict[str, Any]:
    _seed_everything(config.seed)
    model = _make_model(config, data, "true", dropout=0.0)
    rows = data["rows"][:8]
    arrays = _row_arrays(rows)
    inputs = [torch.from_numpy(array) for array in arrays[:-1]]
    labels = torch.from_numpy(arrays[-1])
    targets = torch.from_numpy(teachers["normalized"]["true"][:8])
    model.train()
    direct_before = model(*inputs)
    with torch.no_grad():
        for parameter in model.spectrum_head.parameters():
            parameter.add_(torch.randn_like(parameter))
    direct_after = model(*inputs)
    logits, spectrum = model.forward_with_auxiliary(*inputs)
    balance_loss = nn.BCEWithLogitsLoss()(logits, labels)
    auxiliary_loss = nn.MSELoss()(spectrum, targets)
    total_loss = balance_loss + config.auxiliary_scale * auxiliary_loss
    total_loss.backward()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    auxiliary_parameters = sum(
        parameter.numel() for parameter in model.spectrum_head.parameters()
    )
    buffer_names = {name for name, _ in model.named_buffers()}
    return {
        "teacher_shape": list(teachers["raw"]["true"].shape),
        "teacher_values_finite": all(
            bool(np.isfinite(values).all())
            for family in (teachers["raw"], teachers["normalized"])
            for values in family.values()
        ),
        "parameter_count": parameter_count,
        "parameter_ratio_to_e47": parameter_count / E47_PARAMETER_COUNT,
        "auxiliary_head_parameter_count": auxiliary_parameters,
        "auxiliary_parameter_ratio_to_e47": auxiliary_parameters
        / E47_PARAMETER_COUNT,
        "auxiliary_prediction_shape": list(spectrum.shape),
        "direct_logit_delta_after_auxiliary_head_change": float(
            torch.max(torch.abs(direct_before - direct_after)).detach()
        ),
        "balance_head_width_unchanged": tuple(model.head[0].normalized_shape)
        == (config.hidden_dim * 4 + config.degree_channels,),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "spectrum_finite": bool(torch.isfinite(spectrum).all()),
        "losses_finite": all(
            bool(torch.isfinite(value))
            for value in (balance_loss, auxiliary_loss, total_loss)
        ),
        "gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ),
        "teacher_buffers_absent": not any(
            token in name
            for name in buffer_names
            for token in ("teacher", "support", "prefix", "certificate", "witness")
        ),
    }


def train_distillation_matrix(
    config: DegreeSpectrumDistillationConfig,
    data: dict[str, Any],
    teachers: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_degree_spectrum_distillation",
            "row_id": "e47_mspn_true_label_only_anchor",
            "model_name": "present_mspn",
            "topology_mode": "true",
            "teacher_mode": "off",
            "seed": 0,
            "best_epoch": 29,
            "train_auc": 0.794375,
            "validation_auc": E47_TRUE_AUC,
            "validation_teacher_normalized_mse": None,
            "parameter_count": E47_PARAMETER_COUNT,
            "training_performed": False,
        }
    ]
    history: list[dict[str, Any]] = []
    for topology_mode, teacher_mode in (
        ("true", "true"),
        ("true", "shuffled"),
        ("corrupted", "corrupted"),
    ):
        output = train_distillation_row(
            config,
            data,
            teachers,
            topology_mode=topology_mode,
            teacher_mode=teacher_mode,
        )
        rows.append(output["result"])
        history.extend(output["history"])
    return {"rows": rows, "history": history}


def train_distillation_row(
    config: DegreeSpectrumDistillationConfig,
    data: dict[str, Any],
    teachers: dict[str, Any],
    *,
    topology_mode: str,
    teacher_mode: str,
) -> dict[str, Any]:
    if (topology_mode, teacher_mode) not in {
        ("true", "true"),
        ("true", "shuffled"),
        ("corrupted", "corrupted"),
    }:
        raise ValueError("unsupported E49 topology/teacher row")
    _seed_everything(config.seed)
    device = torch.device(config.device)
    model = _make_model(config, data, topology_mode).to(device)
    row_indices = np.arange(len(data["rows"]), dtype=np.int64)
    train_indices = row_indices[teachers["train_mask"]]
    validation_indices = row_indices[teachers["validation_mask"]]
    teacher_family = "corrupted" if topology_mode == "corrupted" else "true"
    target_matrix = teachers["normalized"][teacher_family]
    training_targets = target_matrix
    if teacher_mode == "shuffled":
        training_targets, _ = shuffled_training_targets(
            target_matrix, teachers["train_mask"]
        )
    train_arrays = _distillation_arrays(
        data, train_indices, training_targets[train_indices]
    )
    validation_arrays = _distillation_arrays(
        data, validation_indices, target_matrix[validation_indices]
    )
    generator = torch.Generator().manual_seed(49100 + config.seed)
    loader = DataLoader(
        _tensor_dataset(train_arrays),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    balance_criterion = nn.BCEWithLogitsLoss()
    auxiliary_criterion = nn.MSELoss()
    best_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    row_id = (
        "mspn_spectrum_true_seed0"
        if teacher_mode == "true"
        else "mspn_spectrum_target_shuffle_seed0"
        if teacher_mode == "shuffled"
        else "mspn_spectrum_corrupted_seed0"
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        balance_total = 0.0
        auxiliary_total = 0.0
        row_total = 0
        for batch in loader:
            inputs = [tensor.to(device) for tensor in batch[:3]]
            labels = batch[3].to(device)
            teacher = batch[4].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, predictions = model.forward_with_auxiliary(*inputs)
            balance_loss = balance_criterion(logits, labels)
            auxiliary_loss = auxiliary_criterion(predictions, teacher)
            loss = balance_loss + config.auxiliary_scale * auxiliary_loss
            loss.backward()
            optimizer.step()
            balance_total += float(balance_loss.detach()) * len(labels)
            auxiliary_total += float(auxiliary_loss.detach()) * len(labels)
            row_total += len(labels)
        validation = _evaluate_distilled(
            model, validation_arrays, device, config.batch_size
        )
        history.append(
            {
                "row_id": row_id,
                "epoch": epoch,
                "train_balance_loss": balance_total / row_total,
                "train_auxiliary_normalized_mse": auxiliary_total / row_total,
                "validation_balance_loss": validation["balance_loss"],
                "validation_auc": validation["auc"],
                "validation_teacher_normalized_mse": validation[
                    "teacher_normalized_mse"
                ],
            }
        )
        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("E49 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_metrics = _evaluate_distilled(
        model, train_arrays, device, config.batch_size
    )
    validation_metrics = _evaluate_distilled(
        model, validation_arrays, device, config.batch_size
    )
    return {
        "result": {
            "run_id": config.run_id,
            "task": "innovation2_present_degree_spectrum_distillation",
            "row_id": row_id,
            "model_name": "present_mspn_degree_spectrum_distilled",
            "topology_mode": topology_mode,
            "teacher_mode": teacher_mode,
            "seed": config.seed,
            "best_epoch": best_epoch,
            "train_auc": train_metrics["auc"],
            "validation_auc": validation_metrics["auc"],
            "train_balance_loss": train_metrics["balance_loss"],
            "validation_balance_loss": validation_metrics["balance_loss"],
            "train_teacher_normalized_mse": train_metrics[
                "teacher_normalized_mse"
            ],
            "validation_teacher_normalized_mse": validation_metrics[
                "teacher_normalized_mse"
            ],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "training_performed": True,
        },
        "history": history,
    }


def adjudicate_e49(
    config: DegreeSpectrumDistillationConfig,
    source_checks: dict[str, bool],
    contract: dict[str, Any],
    teachers: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    trained = [row for row in matrix["rows"] if row["training_performed"]]
    true = next(row for row in trained if row["teacher_mode"] == "true")
    shuffled = next(row for row in trained if row["teacher_mode"] == "shuffled")
    corrupted = next(row for row in trained if row["teacher_mode"] == "corrupted")
    true_mse = float(true["validation_teacher_normalized_mse"])
    shuffle_mse = float(shuffled["validation_teacher_normalized_mse"])
    true_auc = float(true["validation_auc"])
    shuffled_training, permutation = shuffled_training_targets(
        teachers["normalized"]["true"], teachers["train_mask"]
    )
    train = teachers["train_mask"]
    protocol_checks = {
        **source_checks,
        "expected_four_rows_present": len(matrix["rows"]) == 4,
        "three_distilled_rows_present": len(trained) == 3,
        "all_trained_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in trained
            for key in (
                "train_auc",
                "validation_auc",
                "train_balance_loss",
                "validation_balance_loss",
                "train_teacher_normalized_mse",
                "validation_teacher_normalized_mse",
            )
        ),
        "all_rows_completed_two_epochs": all(
            sum(history["row_id"] == row["row_id"] for history in matrix["history"])
            == config.epochs
            for row in trained
        ),
        "teacher_shape_is_rows_x3x13": contract["teacher_shape"]
        == [len(teachers["train_mask"]), 3, 13],
        "teacher_features_finite": contract["teacher_values_finite"],
        "auxiliary_prediction_shape_is_8x3x13": contract[
            "auxiliary_prediction_shape"
        ]
        == [8, 3, 13],
        "parameter_ratio_at_most_1p15": contract["parameter_ratio_to_e47"]
        <= 1.15,
        "auxiliary_parameter_ratio_at_most_0p15": contract[
            "auxiliary_parameter_ratio_to_e47"
        ]
        <= 0.15,
        "balance_head_width_unchanged": contract["balance_head_width_unchanged"],
        "auxiliary_head_not_direct_balance_input": contract[
            "direct_logit_delta_after_auxiliary_head_change"
        ]
        <= 1e-12,
        "teacher_buffers_absent": contract["teacher_buffers_absent"],
        "forward_loss_gradients_finite": contract["logits_finite"]
        and contract["spectrum_finite"]
        and contract["losses_finite"]
        and contract["gradients_finite"],
        "teacher_shuffle_is_nontrivial_permutation": not np.array_equal(
            permutation, np.arange(len(permutation))
        ),
        "teacher_shuffle_preserves_train_column_distribution": np.allclose(
            np.sort(shuffled_training[train].reshape(len(permutation), -1), axis=0),
            np.sort(
                teachers["normalized"]["true"][train].reshape(
                    len(permutation), -1
                ),
                axis=0,
            ),
        ),
        "true_and_corrupted_teachers_distinct": not np.array_equal(
            teachers["raw"]["true"], teachers["raw"]["corrupted"]
        ),
    }
    teacher_checks = {
        "true_validation_normalized_mse_at_most_0p90": true_mse <= 0.90,
        "true_minus_shuffle_mse_at_most_minus_0p10": true_mse - shuffle_mse
        <= -0.10,
    }
    balance_checks = {"true_validation_auc_at_least_0p48": true_auc >= 0.48}
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_degree_spectrum_protocol_invalid"
        action = "repair source, teacher, leakage, model, control, or training protocol"
    elif not all(teacher_checks.values()):
        status = "hold"
        decision = "innovation2_present_degree_spectrum_not_learned"
        action = "stop degree-spectrum distillation and the certificate propagation neural route"
    elif not all(balance_checks.values()):
        status = "hold"
        decision = "innovation2_present_degree_spectrum_balance_degenerated"
        action = "freeze a separate auxiliary-scale 0.10 sensitivity audit"
    else:
        status = "pass"
        decision = "innovation2_present_degree_spectrum_readiness_passed"
        action = "prepare E50 30-epoch seed0 degree-spectrum attribution plan"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "teacher_learnability_checks": teacher_checks,
        "balance_readiness_checks": balance_checks,
        "metrics": {
            "distilled_true_validation_auc": true_auc,
            "target_shuffle_validation_auc": float(shuffled["validation_auc"]),
            "corrupted_validation_auc": float(corrupted["validation_auc"]),
            "e47_label_only_validation_auc": E47_TRUE_AUC,
            "distilled_true_validation_normalized_mse": true_mse,
            "target_shuffle_validation_normalized_mse": shuffle_mse,
            "corrupted_validation_normalized_mse": float(
                corrupted["validation_teacher_normalized_mse"]
            ),
            "true_minus_shuffle_normalized_mse": true_mse - shuffle_mse,
            "parameter_count": contract["parameter_count"],
            "parameter_ratio_to_e47": contract["parameter_ratio_to_e47"],
            "auxiliary_head_parameter_count": contract[
                "auxiliary_head_parameter_count"
            ],
        },
        "claim_scope": (
            "two-epoch local intermediate degree-spectrum distillation readiness "
            "on the E43 real PRESENT-80 r4 strict benchmark; not an effective "
            "neural result, high-round distinguisher, new attack, remote-scale "
            "result, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "formal_seed0": status == "pass",
            "seed1": False,
            "remote_scale": False,
        },
    }


def teacher_metric_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "row_id": row["row_id"],
            "topology_mode": row["topology_mode"],
            "teacher_mode": row["teacher_mode"],
            "best_epoch": row["best_epoch"],
            "train_teacher_normalized_mse": row[
                "train_teacher_normalized_mse"
            ],
            "validation_teacher_normalized_mse": row[
                "validation_teacher_normalized_mse"
            ],
        }
        for row in matrix["rows"]
        if row["training_performed"]
    ]


def serializable_config(
    config: DegreeSpectrumDistillationConfig,
) -> dict[str, Any]:
    return asdict(config)


def _make_model(
    config: DegreeSpectrumDistillationConfig,
    data: dict[str, Any],
    topology_mode: str,
    *,
    dropout: float | None = None,
) -> PresentDegreeSpectrumDistillationNetwork:
    return PresentDegreeSpectrumDistillationNetwork(
        PresentMspnSpec(
            topology_mode=topology_mode,
            rounds=4,
            degree_channels=config.degree_channels,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout if dropout is None else dropout,
        ),
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    )


def _row_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    return (
        np.zeros(len(rows), dtype=np.int64),
        np.asarray([row["structure_index"] for row in rows], dtype=np.int64),
        np.asarray([row["mask_index"] for row in rows], dtype=np.int64),
        np.asarray([row["label"] for row in rows], dtype=np.float32),
    )


def _distillation_arrays(
    data: dict[str, Any], indices: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, ...]:
    rows = [data["rows"][int(index)] for index in indices]
    return (*_row_arrays(rows), np.asarray(targets, dtype=np.float32))


def _tensor_dataset(arrays: tuple[np.ndarray, ...]) -> TensorDataset:
    return TensorDataset(*(torch.from_numpy(array) for array in arrays))


def _evaluate_distilled(
    model: PresentDegreeSpectrumDistillationNetwork,
    arrays: tuple[np.ndarray, ...],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    balance_losses: list[float] = []
    teacher_losses: list[float] = []
    balance_criterion = nn.BCEWithLogitsLoss(reduction="none")
    teacher_criterion = nn.MSELoss(reduction="none")
    loader = DataLoader(_tensor_dataset(arrays), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            inputs = [tensor.to(device) for tensor in batch[:3]]
            target = batch[3].to(device)
            teacher = batch[4].to(device)
            output, predictions = model.forward_with_auxiliary(*inputs)
            balance_losses.extend(
                balance_criterion(output, target).cpu().numpy().tolist()
            )
            teacher_losses.extend(
                teacher_criterion(predictions, teacher)
                .mean(dim=(1, 2))
                .cpu()
                .numpy()
                .tolist()
            )
            logits.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())
    logit_array = np.concatenate(logits)
    label_array = np.concatenate(labels)
    return {
        "balance_loss": float(np.mean(balance_losses)),
        "teacher_normalized_mse": float(np.mean(teacher_losses)),
        "auc": float(binary_auc(label_array, logit_array)),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
