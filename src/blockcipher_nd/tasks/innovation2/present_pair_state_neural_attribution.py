from __future__ import annotations

import copy
import csv
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

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX
from blockcipher_nd.models.structure.spn.small_spn_pair_relation_models import (
    SmallSpnPairRelationReasoner,
    SmallSpnPairRelationSpec,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.training.metrics import binary_auc


SOURCE_RUN_ID = "i2_present_r4_universal_balance_atlas_20260718"
SOURCE_DECISION = "innovation2_present_universal_balance_atlas_ready"
AUDIT_EPOCHS = 30
AUDIT_BATCH_SIZE = 8
AUDIT_HIDDEN_DIM = 16
AUDIT_PATH_RANK = 2
AUDIT_SEED = 0
UNARY_BASELINE_AUC = 0.5


@dataclass(frozen=True)
class PresentPairStateTrainingConfig:
    run_id: str
    mode: str = "full"
    epochs: int = AUDIT_EPOCHS
    batch_size: int = AUDIT_BATCH_SIZE
    hidden_dim: int = AUDIT_HIDDEN_DIM
    path_rank: int = AUDIT_PATH_RANK
    seed: int = AUDIT_SEED
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "full"}:
            raise ValueError("mode must be smoke or full")
        if min(self.epochs, self.batch_size, self.hidden_dim, self.path_rank) <= 0:
            raise ValueError("training dimensions must be positive")
        if self.mode == "full" and (
            self.epochs != AUDIT_EPOCHS
            or self.batch_size != AUDIT_BATCH_SIZE
            or self.hidden_dim != AUDIT_HIDDEN_DIM
            or self.path_rank != AUDIT_PATH_RANK
            or self.seed != AUDIT_SEED
            or self.dropout != 0.10
            or self.device != "cpu"
        ):
            raise ValueError("E44 full protocol is frozen")


def load_e43_source(source_root: Path) -> dict[str, Any]:
    gate = json.loads((source_root / "gate.json").read_text(encoding="utf-8"))
    structures_payload = json.loads(
        (source_root / "structures.json").read_text(encoding="utf-8")
    )
    masks_payload = json.loads(
        (source_root / "masks.json").read_text(encoding="utf-8")
    )
    with (source_root / "matched_contrast.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    structures = structures_payload["structures"]
    masks = masks_payload["masks"]
    structure_active = np.zeros((len(structures), 64), dtype=np.float32)
    for structure in structures:
        structure_active[int(structure["index"]), structure["active_bits"]] = 1.0
    output_mask_bits = np.zeros((len(masks), 64), dtype=np.float32)
    for mask in masks:
        output_mask_bits[int(mask["index"]), mask["bits"]] = 1.0
    parsed_rows = [
        {
            "split": row["split"],
            "rectangle_index": int(row["rectangle_index"]),
            "structure_index": int(row["structure_index"]),
            "mask_index": int(row["mask_index"]),
            "label": int(row["label"]),
        }
        for row in rows
    ]
    return {
        "gate": gate,
        "structures": structures,
        "masks": masks,
        "rows": parsed_rows,
        "structure_active": structure_active,
        "output_mask_bits": output_mask_bits,
        "sboxes": np.asarray([PRESENT_SBOX], dtype=np.uint8),
        "players": np.asarray(
            [[(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)]],
            dtype=np.int64,
        ),
        "source_hashes": {
            name: _sha256(source_root / name)
            for name in (
                "gate.json",
                "structures.json",
                "masks.json",
                "matched_contrast.csv",
            )
        },
    }


def validate_e43_source(data: dict[str, Any], *, strict: bool) -> dict[str, bool]:
    rows = data["rows"]
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    train_structures = {row["structure_index"] for row in train}
    validation_structures = {row["structure_index"] for row in validation}
    duplicate_edges = len(rows) - len(
        {(row["split"], row["structure_index"], row["mask_index"]) for row in rows}
    )
    structure_balanced = _groups_balanced(rows, "structure_index")
    mask_balanced = _groups_balanced(rows, "mask_index")
    checks = {
        "source_run_id_matches": data["gate"].get("run_id") == SOURCE_RUN_ID,
        "source_decision_ready": data["gate"].get("decision") == SOURCE_DECISION,
        "source_status_pass": data["gate"].get("status") == "pass",
        "only_binary_labels": all(row["label"] in {0, 1} for row in rows),
        "only_train_validation_splits": {row["split"] for row in rows}
        <= {"train", "validation"},
        "train_validation_structures_disjoint": train_structures.isdisjoint(
            validation_structures
        ),
        "matched_edges_unique": duplicate_edges == 0,
        "each_structure_balanced": structure_balanced,
        "each_mask_balanced": mask_balanced,
        "feature_width_is_64": data["structure_active"].shape[1] == 64
        and data["output_mask_bits"].shape[1] == 64,
        "player_is_permutation": np.array_equal(
            np.sort(data["players"][0]), np.arange(64)
        ),
    }
    if strict:
        checks.update(
            {
                "train_rows_are_800": len(train) == 800,
                "validation_rows_are_236": len(validation) == 236,
                "train_is_400_400": sum(row["label"] for row in train) == 400,
                "validation_is_118_118": sum(row["label"] for row in validation)
                == 118,
                "structure_count_is_96": len(data["structures"]) == 96,
                "mask_count_is_300": len(data["masks"]) == 300,
            }
        )
    return checks


def measure_model_contract(
    config: PresentPairStateTrainingConfig, data: dict[str, Any]
) -> dict[str, Any]:
    local = _make_model(config, data, "local", "true", dropout=0.0)
    triangle = _make_model(config, data, "triangle", "true", dropout=0.0)
    corrupted = _make_model(config, data, "local", "corrupted", dropout=0.0)
    _copy_parameters(local, corrupted)
    batch_rows = data["rows"][: min(8, len(data["rows"]))]
    arrays = _rows_to_arrays(batch_rows)
    tensors = [torch.from_numpy(array) for array in arrays[:-1]]
    target = torch.from_numpy(arrays[-1])
    local.train()
    logits = local(*tensors)
    loss = nn.BCEWithLogitsLoss()(logits, target)
    loss.backward()
    local.eval()
    corrupted.eval()
    with torch.no_grad():
        true_logits = local(*tensors)
        corrupted_logits = corrupted(*tensors)
        initial, _ = local.build_initial_relation(*tensors)
    corrupted_players = topology_players(data["players"], "corrupted")
    return {
        "initial_pair_shape": list(initial.shape),
        "pair_count": int(initial.shape[1] * initial.shape[2]),
        "local_parameter_count": sum(parameter.numel() for parameter in local.parameters()),
        "triangle_parameter_count": sum(
            parameter.numel() for parameter in triangle.parameters()
        ),
        "parameter_counts_match": sum(parameter.numel() for parameter in local.parameters())
        == sum(parameter.numel() for parameter in triangle.parameters()),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in local.parameters()
        ),
        "true_corrupted_max_abs_logit_difference": float(
            torch.max(torch.abs(true_logits - corrupted_logits))
        ),
        "corrupted_player_is_permutation": np.array_equal(
            np.sort(corrupted_players[0]), np.arange(64)
        ),
        "corrupted_player_differs": not np.array_equal(
            corrupted_players, data["players"]
        ),
        "step_schedule": local.step_counts(torch.zeros(1, dtype=torch.long)).tolist(),
    }


def train_e44_matrix(
    config: PresentPairStateTrainingConfig, data: dict[str, Any]
) -> dict[str, Any]:
    baseline = {
        "run_id": config.run_id,
        "task": "innovation2_present_pair_state_neural_attribution",
        "row_id": "unary_marginal_baseline",
        "model_name": "unary_marginal_baseline",
        "processor_mode": "none",
        "topology_mode": "none",
        "seed": config.seed,
        "best_epoch": 0,
        "train_auc": UNARY_BASELINE_AUC,
        "validation_auc": UNARY_BASELINE_AUC,
        "parameter_count": 0,
        "training_performed": False,
    }
    trained: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for processor in ("local", "triangle"):
        output = train_pair_state_model(
            config, data, processor_mode=processor, topology_mode="true"
        )
        trained.append(output["result"])
        history.extend(output["history"])
    best_true = max(
        trained,
        key=lambda row: (
            float(row["validation_auc"]),
            row["processor_mode"] == "local",
        ),
    )
    control = train_pair_state_model(
        config,
        data,
        processor_mode=str(best_true["processor_mode"]),
        topology_mode="corrupted",
    )
    history.extend(control["history"])
    return {
        "rows": [baseline, *trained, control["result"]],
        "history": history,
        "selected_processor": best_true["processor_mode"],
    }


def train_pair_state_model(
    config: PresentPairStateTrainingConfig,
    data: dict[str, Any],
    *,
    processor_mode: str,
    topology_mode: str,
) -> dict[str, Any]:
    _seed_everything(config.seed)
    device = torch.device(config.device)
    model = _make_model(config, data, processor_mode, topology_mode).to(device)
    train_rows = [row for row in data["rows"] if row["split"] == "train"]
    validation_rows = [row for row in data["rows"] if row["split"] == "validation"]
    train_arrays = _rows_to_arrays(train_rows)
    validation_arrays = _rows_to_arrays(validation_rows)
    generator = torch.Generator().manual_seed(44000 + config.seed)
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
    row_id = f"pair_{processor_mode}_{topology_mode}_seed{config.seed}"
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
                "row_id": row_id,
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
    train_metrics = _evaluate(model, train_arrays, device, config.batch_size)
    validation_metrics = _evaluate(
        model, validation_arrays, device, config.batch_size
    )
    result = {
        "run_id": config.run_id,
        "task": "innovation2_present_pair_state_neural_attribution",
        "row_id": row_id,
        "model_name": f"present_pair_{processor_mode}",
        "processor_mode": processor_mode,
        "topology_mode": topology_mode,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "train_auc": train_metrics["auc"],
        "validation_auc": validation_metrics["auc"],
        "train_loss": train_metrics["loss"],
        "validation_loss": validation_metrics["loss"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_performed": True,
    }
    return {"result": result, "history": history}


def adjudicate_e44(
    config: PresentPairStateTrainingConfig,
    source_checks: dict[str, bool],
    contract: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    rows = matrix["rows"]
    true_rows = [
        row
        for row in rows
        if row["topology_mode"] == "true" and row["training_performed"]
    ]
    control = next(row for row in rows if row["topology_mode"] == "corrupted")
    best = max(
        true_rows,
        key=lambda row: (
            float(row["validation_auc"]),
            row["processor_mode"] == "local",
        ),
    )
    candidate_delta = float(best["validation_auc"]) - UNARY_BASELINE_AUC
    topology_delta = float(best["validation_auc"]) - float(control["validation_auc"])
    protocol_checks = {
        **source_checks,
        "expected_four_rows_present": len(rows) == 4,
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in ("train_auc", "validation_auc")
        ),
        "selected_control_processor_matches": control["processor_mode"]
        == best["processor_mode"]
        == matrix["selected_processor"],
        "selected_control_seed_matches": control["seed"] == best["seed"] == config.seed,
        "initial_pair_shape_is_8x64x64x16": contract["initial_pair_shape"]
        == [8, 64, 64, config.hidden_dim],
        "pair_count_is_4096": contract["pair_count"] == 4096,
        "local_triangle_parameter_counts_match": contract["parameter_counts_match"],
        "contract_logits_loss_gradients_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "corrupted_player_is_distinct_permutation": contract[
            "corrupted_player_is_permutation"
        ]
        and contract["corrupted_player_differs"],
        "true_corrupted_initial_logits_differ": contract[
            "true_corrupted_max_abs_logit_difference"
        ]
        >= 1e-5,
        "step_schedule_is_4": contract["step_schedule"] == [4],
    }
    candidate_checks = {
        "best_true_validation_auc_at_least_0p60": float(best["validation_auc"])
        >= 0.60,
        "best_true_minus_unary_at_least_0p05": candidate_delta >= 0.05,
    }
    attribution_checks = {
        "best_true_minus_corrupted_at_least_0p03": topology_delta >= 0.03,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_pair_state_attribution_protocol_invalid"
        action = "repair source, model, control, metric, or training protocol"
    elif not all(candidate_checks.values()):
        status = "hold"
        decision = "innovation2_present_pair_state_candidate_not_ready"
        action = "stop r4 pair-state scaling and audit unresolved ANF interaction complexity"
    elif not all(attribution_checks.values()):
        status = "hold"
        decision = "innovation2_present_pair_state_topology_not_attributed"
        action = "audit non-topological active/mask or certificate-complexity shortcuts"
    else:
        status = "pass"
        decision = "innovation2_present_pair_state_topology_attributed"
        action = "run the frozen four-row E44 matrix with seed1 locally"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "candidate_checks": candidate_checks,
        "attribution_checks": attribution_checks,
        "metrics": {
            "selected_processor": best["processor_mode"],
            "best_true_validation_auc": best["validation_auc"],
            "corrupted_validation_auc": control["validation_auc"],
            "unary_baseline_auc": UNARY_BASELINE_AUC,
            "best_true_minus_unary": candidate_delta,
            "best_true_minus_corrupted": topology_delta,
        },
        "claim_scope": (
            "local seed0 neural readiness and topology attribution on the E43 real "
            "PRESENT-80 r4 matched certificate/counterexample benchmark; not a "
            "high-round distinguisher, remote-scale result, or SOTA attack"
        ),
        "next_action": {
            "action": action,
            "seed1": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: PresentPairStateTrainingConfig) -> dict[str, Any]:
    return asdict(config)


def _make_model(
    config: PresentPairStateTrainingConfig,
    data: dict[str, Any],
    processor_mode: str,
    topology_mode: str,
    *,
    dropout: float | None = None,
) -> SmallSpnPairRelationReasoner:
    return SmallSpnPairRelationReasoner(
        SmallSpnPairRelationSpec(
            topology_mode=topology_mode,
            processor_mode=processor_mode,
            state_bits=64,
            round_categories=1,
            round_step_offset=4,
            hidden_dim=config.hidden_dim,
            path_rank=config.path_rank,
            dropout=config.dropout if dropout is None else dropout,
        ),
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    )


def _rows_to_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    return (
        np.zeros(len(rows), dtype=np.int64),
        np.zeros(len(rows), dtype=np.int64),
        np.asarray([row["structure_index"] for row in rows], dtype=np.int64),
        np.asarray([row["mask_index"] for row in rows], dtype=np.int64),
        np.asarray([row["label"] for row in rows], dtype=np.float32),
    )


def _tensor_dataset(arrays: tuple[np.ndarray, ...]) -> TensorDataset:
    return TensorDataset(*(torch.from_numpy(array) for array in arrays))


def _evaluate(
    model: nn.Module,
    arrays: tuple[np.ndarray, ...],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    losses: list[float] = []
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    loader = DataLoader(_tensor_dataset(arrays), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            inputs = [tensor.to(device) for tensor in batch[:-1]]
            target = batch[-1].to(device)
            output = model(*inputs)
            losses.extend(criterion(output, target).cpu().numpy().tolist())
            logits.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())
    logit_array = np.concatenate(logits)
    label_array = np.concatenate(labels)
    return {
        "loss": float(np.mean(losses)),
        "auc": float(binary_auc(label_array, logit_array)),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _copy_parameters(source: nn.Module, target: nn.Module) -> None:
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    if source_parameters.keys() != target_parameters.keys():
        raise ValueError("model parameter names do not match")
    with torch.no_grad():
        for name, parameter in target_parameters.items():
            parameter.copy_(source_parameters[name])


def _groups_balanced(rows: list[dict[str, Any]], field: str) -> bool:
    for split in ("train", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        for value in {row[field] for row in split_rows}:
            labels = [row["label"] for row in split_rows if row[field] == value]
            if sum(labels) * 2 != len(labels):
                return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
