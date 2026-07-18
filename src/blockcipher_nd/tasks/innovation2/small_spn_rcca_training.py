from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from blockcipher_nd.models.structure.spn.small_spn_relation_cross_attention import (
    SmallSpnRelationModelSpec,
    SmallSpnRelationPredictor,
)
from blockcipher_nd.tasks.innovation2.small_spn_multicoordinate_relations import (
    variant_split_indices,
)
from blockcipher_nd.tasks.innovation2.small_spn_topology_training import (
    _evaluate,
    _seed_everything,
    _tensor_dataset,
)


RELATION_SPLIT_SEED = 63001
DUAL_MARGINAL_ANCHOR = 0.6858946784175353


@dataclass(frozen=True)
class RelationTrainingConfig:
    run_id: str
    mode: str = "smoke"
    hidden_dim: int = 32
    layers: int = 2
    heads: int = 4
    epochs: int = 8
    batch_size: int = 128
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "full"}:
            raise ValueError("mode must be smoke or full")
        if min(
            self.hidden_dim,
            self.layers,
            self.heads,
            self.epochs,
            self.batch_size,
        ) <= 0:
            raise ValueError("training dimensions must be positive")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.mode == "full" and (
            self.hidden_dim != 64
            or self.layers != 2
            or self.heads != 4
            or self.epochs != 40
            or self.batch_size != 128
            or self.dropout != 0.10
        ):
            raise ValueError("E63 full protocol is frozen")


@dataclass(frozen=True)
class RelationTrainingRowSpec:
    model_name: str
    topology_mode: str
    label_mode: str
    seed: int

    @property
    def row_id(self) -> str:
        suffix = (
            "label_shuffle"
            if self.label_mode == "shuffled"
            else self.topology_mode
        )
        return f"{self.model_name}_{suffix}_seed{self.seed}"


def readiness_matrix() -> tuple[RelationTrainingRowSpec, ...]:
    return (
        RelationTrainingRowSpec("deepsets", "true", "true", 0),
        RelationTrainingRowSpec("rcca", "true", "true", 0),
        RelationTrainingRowSpec("rcca", "corrupted", "true", 0),
        RelationTrainingRowSpec("rcca", "true", "shuffled", 0),
    )


def phase_a_matrix() -> tuple[RelationTrainingRowSpec, ...]:
    return (
        RelationTrainingRowSpec("deepsets", "true", "true", 0),
        RelationTrainingRowSpec("deepsets", "true", "true", 1),
        RelationTrainingRowSpec("rcca", "true", "true", 0),
        RelationTrainingRowSpec("rcca", "true", "true", 1),
        RelationTrainingRowSpec("rcca", "true", "shuffled", 0),
    )


def load_relation_training_data(e37_root: Path, e62_root: Path) -> dict[str, Any]:
    e37_gate = json.loads((e37_root / "gate.json").read_text(encoding="utf-8"))
    e37_metadata = json.loads(
        (e37_root / "metadata.json").read_text(encoding="utf-8")
    )
    e62_gate = json.loads((e62_root / "gate.json").read_text(encoding="utf-8"))
    e62_metadata = json.loads(
        (e62_root / "metadata.json").read_text(encoding="utf-8")
    )
    variants = e37_metadata["variants"]
    structures = e37_metadata["structures"]
    masks = [int(value, 16) for value in e37_metadata["output_masks"]]
    labels = np.load(e62_root / "relation_labels.npy")
    relation_pairs = np.load(e62_root / "selected_relation_pairs.npy")
    relation_rounds = np.load(e62_root / "selected_round_indices.npy")
    rng = np.random.default_rng(RELATION_SPLIT_SEED)
    permutation = rng.permutation(labels.shape[1])
    validation_count = max(1, round(0.20 * labels.shape[1]))
    validation_relations = np.sort(permutation[:validation_count])
    fit_relations = np.sort(permutation[validation_count:])
    return {
        "e37_gate": e37_gate,
        "e62_gate": e62_gate,
        "e62_metadata": e62_metadata,
        "sboxes": np.asarray([variant["sbox"] for variant in variants], dtype=np.uint8),
        "players": np.asarray(
            [variant["player"] for variant in variants], dtype=np.int64
        ),
        "structure_active": np.asarray(
            [
                [int(bit in structure["active_bits"]) for bit in range(16)]
                for structure in structures
            ],
            dtype=np.float32,
        ),
        "output_mask_bits": np.asarray(
            [[int(mask & (1 << bit) != 0) for bit in range(16)] for mask in masks],
            dtype=np.float32,
        ),
        "relation_pairs": np.asarray(relation_pairs, dtype=np.int64),
        "relation_rounds": np.asarray(relation_rounds, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.bool_),
        "variant_splits": variant_split_indices(),
        "fit_relations": fit_relations,
        "validation_relations": validation_relations,
    }


def train_relation_row(
    config: RelationTrainingConfig,
    row_spec: RelationTrainingRowSpec,
    data: dict[str, Any],
) -> dict[str, Any]:
    _seed_everything(row_spec.seed)
    device = torch.device(config.device)
    model = _make_model(config, row_spec, data).to(device)
    train_variants = data["variant_splits"]["train"]
    train_arrays = _example_arrays(
        data["labels"], train_variants, data["fit_relations"]
    )
    validation_arrays = _example_arrays(
        data["labels"], train_variants, data["validation_relations"]
    )
    if row_spec.label_mode == "shuffled":
        combined = np.concatenate((train_arrays[-1], validation_arrays[-1])).copy()
        rng = np.random.default_rng(63050 + row_spec.seed)
        combined = combined[rng.permutation(len(combined))]
        train_size = len(train_arrays[-1])
        train_arrays = (*train_arrays[:-1], combined[:train_size])
        validation_arrays = (*validation_arrays[:-1], combined[train_size:])
    generator = torch.Generator().manual_seed(63100 + row_spec.seed)
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
        validation = _evaluate(
            model, validation_arrays, device, config.batch_size
        )
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
                data["labels"], variants, np.arange(data["labels"].shape[1])
            ),
            device,
            config.batch_size,
        )
        for split, variants in data["variant_splits"].items()
    }
    result = {
        "run_id": config.run_id,
        "task": "innovation2_small_spn_rcca",
        "row_id": row_spec.row_id,
        "model_name": row_spec.model_name,
        "topology_mode": row_spec.topology_mode,
        "label_mode": row_spec.label_mode,
        "seed": row_spec.seed,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "train_auc": evaluations["train"]["auc"],
        "unseen_sbox_auc": evaluations["unseen_sbox"]["auc"],
        "unseen_player_auc": evaluations["unseen_player"]["auc"],
        "dual_unseen_auc": evaluations["dual_unseen"]["auc"],
        "training_performed": True,
    }
    return {"result": result, "history": history, "state_dict": best_state}


def measure_relation_model_contract(
    config: RelationTrainingConfig, data: dict[str, Any]
) -> dict[str, Any]:
    true_spec = RelationTrainingRowSpec("rcca", "true", "true", 0)
    corrupted_spec = RelationTrainingRowSpec("rcca", "corrupted", "true", 0)
    torch.manual_seed(63200)
    true_model = _make_model(config, true_spec, data).eval()
    corrupted_model = _make_model(config, corrupted_spec, data).eval()
    _copy_parameters(true_model, corrupted_model)
    batch = min(64, data["labels"].shape[1])
    variants = torch.arange(batch, dtype=torch.long) % 64
    relations = torch.arange(batch, dtype=torch.long)
    with torch.no_grad():
        true_logits = true_model(variants, relations)
        corrupted_logits = corrupted_model(variants, relations)
    swapped_data = dict(data)
    swapped_data["relation_pairs"] = data["relation_pairs"][:, ::-1].copy()
    swapped_model = _make_model(config, true_spec, swapped_data).eval()
    _copy_parameters(true_model, swapped_model)
    with torch.no_grad():
        swapped_logits = swapped_model(variants, relations)

    cell_permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    node_permutation = np.asarray(
        [4 * cell_permutation[node // 4] + node % 4 for node in range(16)],
        dtype=np.int64,
    )
    inverse = np.argsort(node_permutation)
    relabeled_data = dict(data)
    relabeled_data["players"] = node_permutation[data["players"][:, inverse]]
    relabeled_data["structure_active"] = data["structure_active"][:, inverse]
    relabeled_data["output_mask_bits"] = data["output_mask_bits"][:, inverse]
    relabeled_model = _make_model(config, true_spec, relabeled_data).eval()
    _copy_parameters(true_model, relabeled_model)
    with torch.no_grad():
        relabeled_logits = relabeled_model(variants, relations)

    parameter_counts = {
        model_name: sum(
            parameter.numel()
            for parameter in _make_model(
                config,
                RelationTrainingRowSpec(model_name, "true", "true", 0),
                data,
            ).parameters()
        )
        for model_name in ("deepsets", "rcca")
    }
    return {
        "relation_swap_max_logit_error": float(
            torch.max(torch.abs(true_logits - swapped_logits))
        ),
        "cell_relabel_max_logit_error": float(
            torch.max(torch.abs(true_logits - relabeled_logits))
        ),
        "true_corrupted_max_logit_delta": float(
            torch.max(torch.abs(true_logits - corrupted_logits))
        ),
        "parameter_counts": parameter_counts,
        "parameter_ratio": max(parameter_counts.values())
        / min(parameter_counts.values()),
        "fit_relations": len(data["fit_relations"]),
        "validation_relations": len(data["validation_relations"]),
    }


def adjudicate_relation_training(
    config: RelationTrainingConfig,
    *,
    data: dict[str, Any],
    contract: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = 4 if config.mode == "smoke" else 5
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
    protocol = {
        "e62_training_gate_passed": data["e62_gate"].get("status") == "pass"
        and data["e62_gate"].get("decision")
        == "innovation2_small_spn_multicoordinate_relation_training_ready",
        "e37_source_gate_passed": data["e37_gate"].get("status") == "pass",
        "label_shape_is_64x2048": data["labels"].shape == (64, 2048),
        "variant_split_sizes_are_36_12_12_4": [
            len(data["variant_splits"][name])
            for name in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
        ]
        == [36, 12, 12, 4],
        "fit_validation_relations_are_disjoint_and_complete": not set(
            data["fit_relations"].tolist()
        ).intersection(data["validation_relations"].tolist())
        and len(data["fit_relations"]) + len(data["validation_relations"]) == 2048,
        "relation_swap_error_at_most_1e_6": contract[
            "relation_swap_max_logit_error"
        ]
        <= 1e-6,
        "cell_relabel_error_at_most_1e_6": contract[
            "cell_relabel_max_logit_error"
        ]
        <= 1e-6,
        "true_corrupted_logit_delta_at_least_1e_5": contract[
            "true_corrupted_max_logit_delta"
        ]
        >= 1e-5,
        "models_each_at_most_300000_parameters": max(
            contract["parameter_counts"].values()
        )
        <= 300000,
        "parameter_ratio_at_most_1p35": contract["parameter_ratio"] <= 1.35,
        "expected_matrix_rows_present": len(rows) == expected,
        "all_rows_trained": all(row.get("training_performed") is True for row in rows),
        "all_metrics_finite": finite,
    }
    if not all(protocol.values()):
        status = "fail"
        decision = "innovation2_small_spn_rcca_protocol_invalid"
        action = "repair model invariance, source ownership, budget, or training protocol"
        metrics: dict[str, Any] = {}
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_small_spn_rcca_readiness_passed"
        action = "run the frozen two-seed DeepSets versus RCCA Phase A matrix"
        metrics = {"rows": rows}
    else:
        deepsets = {
            row["seed"]: row
            for row in rows
            if row["model_name"] == "deepsets" and row["label_mode"] == "true"
        }
        rcca = {
            row["seed"]: row
            for row in rows
            if row["model_name"] == "rcca" and row["label_mode"] == "true"
        }
        shuffled = next(
            row
            for row in rows
            if row["model_name"] == "rcca" and row["label_mode"] == "shuffled"
        )
        paired_deltas = {
            seed: rcca[seed]["dual_unseen_auc"]
            - deepsets[seed]["dual_unseen_auc"]
            for seed in (0, 1)
        }
        rcca_mean = float(
            np.mean([rcca[seed]["dual_unseen_auc"] for seed in (0, 1)])
        )
        phase_checks = {
            "label_shuffle_dual_at_most_0p60": shuffled["dual_unseen_auc"] <= 0.60,
            "deepsets_each_seed_beats_dual_marginal": all(
                deepsets[seed]["dual_unseen_auc"] > DUAL_MARGINAL_ANCHOR
                for seed in (0, 1)
            ),
            "rcca_each_seed_beats_dual_marginal": all(
                rcca[seed]["dual_unseen_auc"] > DUAL_MARGINAL_ANCHOR
                for seed in (0, 1)
            ),
            "rcca_each_seed_beats_paired_deepsets": all(
                paired_deltas[seed] > 0 for seed in (0, 1)
            ),
            "rcca_mean_dual_at_least_0p715895": rcca_mean
            >= DUAL_MARGINAL_ANCHOR + 0.03,
        }
        metrics = {
            "rows": rows,
            "paired_rcca_minus_deepsets_dual": paired_deltas,
            "rcca_mean_dual": rcca_mean,
            "phase_checks": phase_checks,
        }
        if all(phase_checks.values()):
            status = "pass"
            decision = "innovation2_small_spn_rcca_phase_b_ready"
            action = "run paired wrong-P seed0/1 attribution before retaining RCCA"
        else:
            status = "hold"
            decision = "innovation2_small_spn_rcca_not_ready"
            action = "close RCCA without increasing model or training budget"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol,
        "contract": contract,
        "metrics": metrics,
        "thresholds": {
            "dual_marginal_anchor": DUAL_MARGINAL_ANCHOR,
            "required_rcca_mean_dual": DUAL_MARGINAL_ANCHOR + 0.03,
        },
        "claim_scope": (
            "DeepSets/RCCA training on exact all-256-key two-coordinate labels for "
            "16-bit synthetic SPNs; no PRESENT/GIFT result, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "phase_b_wrong_p": decision
            == "innovation2_small_spn_rcca_phase_b_ready",
            "remote_scale": False,
        },
    }


def _make_model(
    config: RelationTrainingConfig,
    row_spec: RelationTrainingRowSpec,
    data: dict[str, Any],
) -> SmallSpnRelationPredictor:
    return SmallSpnRelationPredictor(
        SmallSpnRelationModelSpec(
            model_name=row_spec.model_name,
            topology_mode=row_spec.topology_mode,
            hidden_dim=config.hidden_dim,
            layers=config.layers,
            heads=config.heads,
            dropout=config.dropout,
        ),
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
        relation_pairs=data["relation_pairs"],
        relation_round_indices=data["relation_rounds"],
    )


def _example_arrays(
    labels: np.ndarray,
    variant_indices: np.ndarray,
    relation_indices: np.ndarray,
) -> tuple[np.ndarray, ...]:
    variants = np.repeat(np.asarray(variant_indices, dtype=np.int64), len(relation_indices))
    relations = np.tile(np.asarray(relation_indices, dtype=np.int64), len(variant_indices))
    target = labels[variants, relations].astype(np.float32)
    return variants, relations, target


def _copy_parameters(source: nn.Module, target: nn.Module) -> None:
    target_parameters = dict(target.named_parameters())
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            target_parameters[name].copy_(parameter)
