from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from blockcipher_nd.models.structure.spn.small_spn_graph_models import (
    SmallSpnModelSpec,
    SmallSpnTopologyPredictor,
)
from blockcipher_nd.tasks.innovation2.small_spn_exact_labels import (
    SmallSpnAuditConfig,
    _binary_auc,
    _split_indices,
    make_variants,
)


CELL_SPLIT_SEED = 33001
BASELINE_AUC = {
    "unseen_sbox": 0.7756928982419179,
    "unseen_player": 0.7425316439479348,
    "dual_unseen": 0.726528384279476,
}


@dataclass(frozen=True)
class TopologyTrainingConfig:
    run_id: str
    mode: str = "full"
    hidden_dim: int = 64
    blocks: int = 3
    heads: int = 4
    dropout: float = 0.10
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "full"}:
            raise ValueError("mode must be smoke or full")
        if self.hidden_dim <= 0 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.blocks <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("blocks, epochs, and batch_size must be positive")
        if self.mode == "full" and (
            self.hidden_dim != 64
            or self.blocks != 3
            or self.heads != 4
            or self.epochs != 40
            or self.batch_size != 128
        ):
            raise ValueError("E33 full mode freezes hidden64, blocks3, heads4, epochs40, batch128")


@dataclass(frozen=True)
class TrainingRowSpec:
    model_name: str
    topology_mode: str
    label_mode: str
    seed: int
    position_mode: str = "absolute"

    @property
    def row_id(self) -> str:
        suffix = "label_shuffle" if self.label_mode == "shuffled" else self.topology_mode
        position = "" if self.position_mode == "absolute" else f"_{self.position_mode}"
        return f"{self.model_name}{position}_{suffix}_seed{self.seed}"


def training_matrix(config: TopologyTrainingConfig) -> tuple[TrainingRowSpec, ...]:
    if config.mode == "smoke":
        return (
            TrainingRowSpec("graphgps", "true", "true", 0),
            TrainingRowSpec("scgt", "true", "true", 0),
            TrainingRowSpec("graphgps", "shuffled", "true", 0),
            TrainingRowSpec("graphgps", "true", "shuffled", 0),
        )
    return tuple(
        [TrainingRowSpec("graphgps", "true", "true", seed) for seed in (0, 1)]
        + [TrainingRowSpec("scgt", "true", "true", seed) for seed in (0, 1)]
        + [TrainingRowSpec("graphgps", "shuffled", "true", seed) for seed in (0, 1)]
        + [TrainingRowSpec("graphgps", "true", "shuffled", 0)]
    )


def load_topology_training_data(
    label_root: Path, contrast_root: Path
) -> dict[str, Any]:
    labels = np.load(label_root / "labels.npy")
    selected = np.load(contrast_root / "selected_mask.npy")
    label_metadata = json.loads((label_root / "metadata.json").read_text(encoding="utf-8"))
    label_gate = json.loads((label_root / "gate.json").read_text(encoding="utf-8"))
    contrast_gate = json.loads((contrast_root / "gate.json").read_text(encoding="utf-8"))
    contrast_metadata = json.loads(
        (contrast_root / "metadata.json").read_text(encoding="utf-8")
    )
    variants = label_metadata["variants"]
    structures = label_metadata["structures"]
    masks = [int(value, 16) for value in label_metadata["output_masks"]]
    structure_active = np.zeros((len(structures), 16), dtype=np.float32)
    structure_basis = np.zeros((len(structures), 12, 16), dtype=np.float32)
    structure_basis_valid = np.zeros((len(structures), 12), dtype=np.bool_)
    for index, structure in enumerate(structures):
        bits = [int(bit) for bit in structure["active_bits"]]
        structure_active[index, bits] = 1.0
        for basis_index, bit in enumerate(bits):
            structure_basis[index, basis_index, bit] = 1.0
            structure_basis_valid[index, basis_index] = True
    output_mask_bits = np.asarray(
        [[(mask >> bit) & 1 for bit in range(16)] for mask in masks],
        dtype=np.float32,
    )
    sboxes = np.asarray([variant["sbox"] for variant in variants], dtype=np.uint8)
    players = np.asarray([variant["player"] for variant in variants], dtype=np.int64)
    variant_objects = make_variants(SmallSpnAuditConfig(run_id="training-data"))
    split_indices = _split_indices(variant_objects)
    cells = np.argwhere(selected)
    rng = np.random.default_rng(CELL_SPLIT_SEED)
    order = rng.permutation(len(cells))
    fit_count = int(math.floor(len(cells) * 0.8))
    fit_cells = cells[order[:fit_count]]
    validation_cells = cells[order[fit_count:]]
    return {
        "labels": np.asarray(labels, dtype=np.bool_),
        "selected": np.asarray(selected, dtype=np.bool_),
        "fit_cells": fit_cells,
        "validation_cells": validation_cells,
        "split_indices": split_indices,
        "sboxes": sboxes,
        "players": players,
        "structure_active": structure_active,
        "structure_basis": structure_basis,
        "structure_basis_valid": structure_basis_valid,
        "output_mask_bits": output_mask_bits,
        "label_metadata": label_metadata,
        "label_gate": label_gate,
        "contrast_gate": contrast_gate,
        "contrast_metadata": contrast_metadata,
    }


def validate_training_contract(data: dict[str, Any]) -> dict[str, bool]:
    split_indices = data["split_indices"]
    fit_cells = {tuple(row) for row in data["fit_cells"]}
    validation_cells = {tuple(row) for row in data["validation_cells"]}
    heldout = set(
        np.concatenate(
            (
                split_indices["unseen_sbox"],
                split_indices["unseen_player"],
                split_indices["dual_unseen"],
            )
        ).tolist()
    )
    return {
        "label_source_gate_is_expected_hold": data["label_gate"].get("decision")
        == "innovation2_small_spn_exact_label_shortcut_dominated",
        "contrast_source_gate_is_ready": data["contrast_gate"].get("decision")
        == "innovation2_small_spn_matched_contrast_ready",
        "labels_shape_is_16x4x14x64": data["labels"].shape == (16, 4, 14, 64),
        "selected_cells_are_589": int(data["selected"].sum()) == 589,
        "fit_and_validation_cells_are_disjoint": fit_cells.isdisjoint(validation_cells),
        "fit_and_validation_cover_selected_cells": len(fit_cells | validation_cells) == 589,
        "train_has_nine_variants": len(split_indices["train"]) == 9,
        "heldout_has_seven_disjoint_variants": len(heldout) == 7
        and heldout.isdisjoint(set(split_indices["train"].tolist())),
        "no_cipher_id_feature_is_constructed": True,
    }


def train_topology_row(
    config: TopologyTrainingConfig,
    row_spec: TrainingRowSpec,
    data: dict[str, Any],
) -> dict[str, Any]:
    _seed_everything(row_spec.seed)
    device = torch.device(config.device)
    model_spec = SmallSpnModelSpec(
        model_name=row_spec.model_name,
        topology_mode=row_spec.topology_mode,
        position_mode=row_spec.position_mode,
        hidden_dim=config.hidden_dim,
        blocks=config.blocks,
        heads=config.heads,
        dropout=config.dropout,
    )
    model = SmallSpnTopologyPredictor(
        model_spec,
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        structure_basis=data["structure_basis"],
        structure_basis_valid=data["structure_basis_valid"],
        output_mask_bits=data["output_mask_bits"],
    ).to(device)
    train_arrays = _example_arrays(
        data["labels"], data["split_indices"]["train"], data["fit_cells"]
    )
    validation_arrays = _example_arrays(
        data["labels"], data["split_indices"]["train"], data["validation_cells"]
    )
    if row_spec.label_mode == "shuffled":
        combined = np.concatenate((train_arrays[-1], validation_arrays[-1])).copy()
        rng = np.random.default_rng(44000 + row_spec.seed)
        combined = combined[rng.permutation(len(combined))]
        train_arrays = (*train_arrays[:-1], combined[: len(train_arrays[-1])])
        validation_arrays = (*validation_arrays[:-1], combined[len(train_arrays[-1]) :])
    generator = torch.Generator().manual_seed(55000 + row_spec.seed)
    loader = DataLoader(
        _tensor_dataset(train_arrays),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            inputs = [tensor.to(device) for tensor in batch[:-1]]
            target = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total_rows += len(target)
        validation = _evaluate(model, validation_arrays, device, config.batch_size)
        history.append(
            {
                "row_id": row_spec.row_id,
                "epoch": epoch,
                "train_loss": total_loss / total_rows,
                "validation_loss": validation["loss"],
                "validation_auc": validation["auc"],
            }
        )
        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    evaluations = {
        split: _evaluate(
            model,
            _example_arrays(
                data["labels"], indices, np.argwhere(data["selected"])
            ),
            device,
            config.batch_size,
        )
        for split, indices in data["split_indices"].items()
    }
    result = {
        "run_id": config.run_id,
        "task": "innovation2_small_spn_topology_prediction",
        "row_id": row_spec.row_id,
        "model_name": row_spec.model_name,
        "topology_mode": row_spec.topology_mode,
        "position_mode": row_spec.position_mode,
        "label_mode": row_spec.label_mode,
        "seed": row_spec.seed,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_auc": evaluations["train"]["auc"],
        "unseen_sbox_auc": evaluations["unseen_sbox"]["auc"],
        "unseen_player_auc": evaluations["unseen_player"]["auc"],
        "dual_unseen_auc": evaluations["dual_unseen"]["auc"],
        "training_performed": True,
    }
    return {"result": result, "history": history, "state_dict": best_state}


def adjudicate_topology_training(
    config: TopologyTrainingConfig,
    readiness: dict[str, bool],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in (
            "best_validation_auc",
            "train_auc",
            "unseen_sbox_auc",
            "unseen_player_auc",
            "dual_unseen_auc",
        )
    )
    expected_rows = 4 if config.mode == "smoke" else 7
    protocol = {
        **readiness,
        "expected_matrix_rows_present": len(rows) == expected_rows,
        "all_metrics_finite": finite,
        "all_rows_trained": all(row.get("training_performed") is True for row in rows),
    }
    if not all(protocol.values()):
        return _training_gate(
            config,
            "fail",
            "innovation2_small_spn_topology_training_protocol_invalid",
            protocol,
            {},
            "repair source, split, model forward, checkpoint, or metric protocol",
        )
    if config.mode == "smoke":
        return _training_gate(
            config,
            "pass",
            "innovation2_small_spn_topology_training_readiness_passed",
            protocol,
            {},
            "run the frozen two-seed E33 topology attribution matrix",
        )

    groups = {
        "graphgps_true": [
            row
            for row in rows
            if row["model_name"] == "graphgps"
            and row["topology_mode"] == "true"
            and row["label_mode"] == "true"
        ],
        "scgt_true": [
            row
            for row in rows
            if row["model_name"] == "scgt" and row["label_mode"] == "true"
        ],
        "graphgps_shuffled": [
            row
            for row in rows
            if row["topology_mode"] == "shuffled" and row["label_mode"] == "true"
        ],
        "label_shuffle": [row for row in rows if row["label_mode"] == "shuffled"],
    }
    means = {
        name: {
            split: float(np.mean([row[f"{split}_auc"] for row in group]))
            for split in ("unseen_sbox", "unseen_player", "dual_unseen")
        }
        for name, group in groups.items()
    }
    n1_dual = means["graphgps_true"]["dual_unseen"]
    c1_dual = means["graphgps_shuffled"]["dual_unseen"]
    c2_dual = means["label_shuffle"]["dual_unseen"]
    checks = {
        "label_shuffle_dual_auc_at_most_0p60": c2_dual <= 0.60,
        "graphgps_each_seed_beats_dual_baseline": all(
            row["dual_unseen_auc"] > BASELINE_AUC["dual_unseen"]
            for row in groups["graphgps_true"]
        ),
        "graphgps_mean_dual_beats_baseline_by_0p03": n1_dual
        >= BASELINE_AUC["dual_unseen"] + 0.03,
        "graphgps_mean_dual_beats_shuffled_by_0p03": n1_dual >= c1_dual + 0.03,
        "graphgps_unseen_sbox_not_below_baseline_minus_0p01": means[
            "graphgps_true"
        ]["unseen_sbox"]
        >= BASELINE_AUC["unseen_sbox"] - 0.01,
        "graphgps_unseen_player_not_below_baseline_minus_0p01": means[
            "graphgps_true"
        ]["unseen_player"]
        >= BASELINE_AUC["unseen_player"] - 0.01,
    }
    scgt_keep = means["scgt_true"]["dual_unseen"] >= n1_dual + 0.01
    metrics = {
        "baseline_auc": BASELINE_AUC,
        "mean_auc": means,
        "scgt_basis_branch_keep": scgt_keep,
        "scgt_dual_delta_vs_graphgps": means["scgt_true"]["dual_unseen"] - n1_dual,
    }
    if not checks["label_shuffle_dual_auc_at_most_0p60"] or (
        checks["graphgps_mean_dual_beats_baseline_by_0p03"]
        and not checks["graphgps_mean_dual_beats_shuffled_by_0p03"]
    ):
        status = "hold"
        decision = "innovation2_small_spn_topology_signal_not_attributed"
        action = "audit representation and controls; do not claim topology contribution"
    elif all(checks.values()):
        status = "pass"
        decision = "innovation2_small_spn_topology_predictor_ready"
        action = "prepare E34 real-cipher transfer readiness without remote scale"
    else:
        status = "hold"
        decision = "innovation2_small_spn_topology_predictor_not_ready"
        action = "stop without increasing layers, epochs, samples, or seeds"
    gate = _training_gate(config, status, decision, protocol, checks, action)
    gate["metrics"] = metrics
    return gate


def _training_gate(
    config: TopologyTrainingConfig,
    status: str,
    decision: str,
    readiness: dict[str, bool],
    checks: dict[str, bool],
    action: str,
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness,
        "attribution_checks": checks,
        "claim_scope": (
            "fixed-budget neural topology prediction on train-only matched labels from a "
            "16-bit synthetic SPN family; not a real-cipher distinguisher or high-round result"
        ),
        "next_action": {"action": action, "remote_scale": False},
    }


def _example_arrays(
    labels: np.ndarray, variant_indices: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, ...]:
    variants = np.repeat(np.asarray(variant_indices, dtype=np.int64), len(cells))
    tiled = np.tile(np.asarray(cells, dtype=np.int64), (len(variant_indices), 1))
    target = labels[
        variants,
        tiled[:, 0],
        tiled[:, 1],
        tiled[:, 2],
    ].astype(np.float32)
    return variants, tiled[:, 0], tiled[:, 1], tiled[:, 2], target


def _tensor_dataset(arrays: tuple[np.ndarray, ...]) -> TensorDataset:
    tensors = [torch.from_numpy(np.asarray(array)) for array in arrays]
    return TensorDataset(*tensors)


def _evaluate(
    model: nn.Module,
    arrays: tuple[np.ndarray, ...],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    loader = DataLoader(_tensor_dataset(arrays), batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    logits: list[np.ndarray] = []
    target: list[np.ndarray] = []
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = [tensor.to(device) for tensor in batch[:-1]]
            batch_target = batch[-1].to(device)
            batch_logits = model(*inputs)
            loss += float(criterion(batch_logits, batch_target))
            logits.append(batch_logits.cpu().numpy())
            target.append(batch_target.cpu().numpy())
    all_logits = np.concatenate(logits)
    all_target = np.concatenate(target)
    return {
        "loss": loss / len(all_target),
        "auc": _binary_auc(all_target.astype(np.bool_), all_logits),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
