from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from blockcipher_nd.tasks.innovation2.present_r9_atm_support_component_pu_neural_ranking import (
    PRESENT_P,
    CoordinateDeepSets,
    PoolTensors,
    PuNeuralRankingConfig,
    aggregate_fold_metrics,
    build_neural_folds,
    evaluate_model,
    sha256,
    tensorize_pools,
)


RUN_ID = "i2_present_r9_identity_topology_residual_attribution_seed0_seed1_20260720"
E99_GATE_SHA256 = "a303939a7749452cbf92e4d17b12bec1677b74f20675093877a8907497adab6c"
E99_DECISION = "innovation2_present_r9_pu_generic_neural_signal_only"
MODEL_NAMES = (
    "coordinate_anchor",
    "identity_true_p_residual",
    "identity_wrong_p_residual",
)
SEEDS = (0, 1)
WRONG_P = tuple((PRESENT_P[(bit - 1) % 64] + 1) % 64 for bit in range(64))
E99_ANCHORS = {
    0: {"recall_at_1": 0.9038461538461539, "recall_at_5": 0.9957264957264957, "mrr": 0.9456908831908832},
    1: {"recall_at_1": 0.8888888888888888, "recall_at_5": 0.9935897435897436, "mrr": 0.9355438542938543},
}


@dataclass(frozen=True)
class IdentityTopologyResidualConfig:
    run_id: str = RUN_ID
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    minimum_unlabeled_per_pool: int = 31
    minimum_anchor_recall_at_1_gain: float = 0.020
    minimum_anchor_mrr_gain: float = 0.010
    minimum_wrong_p_recall_at_1_gain: float = 0.010
    minimum_wrong_p_mrr_gain: float = 0.005
    maximum_recall_at_5_drop: float = 0.005
    minimum_fold_recall_at_5: float = 0.95

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError("E100 run_id is frozen")
        if self.epochs != 40 or self.batch_size != 32 or self.minimum_unlabeled_per_pool != 31:
            raise ValueError("E100 training and candidate-width protocol is frozen")


class IdentityTopologyResidual(nn.Module):
    def __init__(self, permutation: tuple[int, ...], hidden: int = 12) -> None:
        super().__init__()
        if sorted(permutation) != list(range(64)):
            raise ValueError("topology residual permutation must be a 64-bit bijection")
        self.input_embedding = nn.Parameter(torch.empty(64, hidden))
        self.output_embedding = nn.Parameter(torch.empty(64, hidden))
        nn.init.normal_(self.input_embedding, std=0.08)
        nn.init.normal_(self.output_embedding, std=0.08)
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
        )
        self.topology_node_mlp = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.topology_project = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, 1),
        )
        inverse = tuple(permutation.index(bit) for bit in range(64))
        self.register_buffer(
            "permutation_inverse",
            torch.tensor(inverse, dtype=torch.long),
            persistent=False,
        )

    def forward(self, coordinates: torch.Tensor, coordinate_mask: torch.Tensor) -> torch.Tensor:
        left = coordinates[..., :64]
        right = coordinates[..., 64:]
        left_embedding = torch.matmul(left, self.input_embedding) / left.sum(-1, keepdim=True).clamp_min(1.0)
        right_embedding = torch.matmul(right, self.output_embedding) / right.sum(-1, keepdim=True).clamp_min(1.0)
        identity = self.coordinate_mlp(torch.cat((left_embedding, right_embedding), dim=-1))

        direct = torch.stack((left, right), dim=-1)
        p_messages = direct.index_select(-2, self.permutation_inverse)
        shape = direct.shape
        cell = direct.reshape(*shape[:-2], 16, 4, 2).mean(dim=-2, keepdim=True)
        cell = cell.expand(*shape[:-2], 16, 4, 2).reshape(shape)
        nodes = self.topology_node_mlp(torch.cat((direct, p_messages, cell), dim=-1))
        topology = self.topology_project(
            torch.cat((nodes.mean(dim=-2), nodes.amax(dim=-2)), dim=-1)
        )
        coordinate = identity + self.residual_scale * topology
        pooled = _masked_mean_max(coordinate, coordinate_mask)
        return self.head(pooled).squeeze(-1)


def make_model(model_name: str) -> nn.Module:
    if model_name == "coordinate_anchor":
        return CoordinateDeepSets()
    if model_name == "identity_true_p_residual":
        return IdentityTopologyResidual(PRESENT_P)
    if model_name == "identity_wrong_p_residual":
        return IdentityTopologyResidual(WRONG_P)
    raise ValueError(f"unknown E100 model: {model_name}")


def train_attribution_matrix(
    config: IdentityTopologyResidualConfig,
    fold_audit: dict[str, Any],
    *,
    device: str,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    fold_metrics: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold_data in fold_audit["folds"]:
            train_tensors = tensorize_pools(fold_data.train_pools)
            test_tensors = tensorize_pools(fold_data.test_pools)
            for model_name in MODEL_NAMES:
                if progress_callback:
                    progress_callback(
                        "model_fold_start",
                        {"model": model_name, "seed": seed, "fold": fold_data.fold},
                    )
                result = train_one_fold(
                    config,
                    model_name=model_name,
                    seed=seed,
                    fold=fold_data.fold,
                    train_tensors=train_tensors,
                    test_tensors=test_tensors,
                    device=device,
                )
                fold_metrics.append(result["metrics"])
                history.extend(result["history"])
                if progress_callback:
                    progress_callback("model_fold_done", result["metrics"])
    return {
        "fold_metrics": fold_metrics,
        "aggregate_rows": aggregate_fold_metrics(fold_metrics),
        "history": history,
    }


def train_one_fold(
    config: IdentityTopologyResidualConfig,
    *,
    model_name: str,
    seed: int,
    fold: int,
    train_tensors: PoolTensors,
    test_tensors: PoolTensors,
    device: str,
) -> dict[str, Any]:
    initialization_seed = seed * 1009 + fold * 37 + 17
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed)
    model = make_model(model_name).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    targets = torch.zeros(len(train_tensors.relation_ids), dtype=torch.long)
    generator = np.random.default_rng(seed * 1009 + fold * 37 + 29)
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    model.train()
    for epoch in range(1, config.epochs + 1):
        permutation = generator.permutation(train_tensors.coordinates.shape[0])
        total_loss = 0.0
        total_pools = 0
        for start in range(0, len(permutation), config.batch_size):
            indices = torch.as_tensor(
                permutation[start : start + config.batch_size], dtype=torch.long
            )
            coordinates = train_tensors.coordinates[indices].to(device)
            coordinate_mask = train_tensors.coordinate_mask[indices].to(device)
            item_mask = train_tensors.item_mask[indices].to(device)
            batch_targets = targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(coordinates, coordinate_mask).masked_fill(~item_mask, -1e9)
            loss = F.cross_entropy(scores, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
            total_pools += len(indices)
        history.append(
            {
                "model": model_name,
                "seed": seed,
                "fold": fold,
                "epoch": epoch,
                "train_listwise_loss": total_loss / total_pools,
                "residual_scale": (
                    float(model.residual_scale.detach().cpu())
                    if isinstance(model, IdentityTopologyResidual)
                    else 0.0
                ),
            }
        )
    training_seconds = time.monotonic() - started
    metrics = evaluate_model(model, test_tensors, device=device, batch_size=config.batch_size)
    return {
        "metrics": {
            "model": model_name,
            "seed": seed,
            "fold": fold,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": training_seconds,
            "final_train_loss": history[-1]["train_listwise_loss"],
            "final_residual_scale": history[-1]["residual_scale"],
            **metrics,
        },
        "history": history,
    }


def adjudicate_identity_topology_residual(
    config: IdentityTopologyResidualConfig,
    *,
    fold_audit: dict[str, Any],
    training: dict[str, Any],
    e99_gate: dict[str, Any],
    e99_gate_hash: str,
) -> dict[str, Any]:
    rows = training["aggregate_rows"]
    metrics = dict(fold_audit["metrics"])
    true_parameters = {
        _row(rows, "identity_true_p_residual", seed)["parameter_count"] for seed in SEEDS
    }
    wrong_parameters = {
        _row(rows, "identity_wrong_p_residual", seed)["parameter_count"] for seed in SEEDS
    }
    protocol_checks = {
        "e99_gate_hash_matches": e99_gate_hash == E99_GATE_SHA256,
        "e99_status_hold": e99_gate.get("status") == "hold",
        "e99_decision_matches": e99_gate.get("decision") == E99_DECISION,
        "exact_468_independent_relations": metrics["independent_relations"] == 468,
        "six_balanced_groups": metrics["groups"] == 6
        and metrics["minimum_group_positives"] == 78
        and metrics["maximum_group_positives"] == 78,
        "train_test_relation_overlap_zero": metrics["maximum_train_test_relation_overlap"] == 0,
        "candidate_positive_support_overlap_zero": metrics["candidate_positive_support_overlap"] == 0,
        "candidate_known_positive_overlap_zero": metrics["candidate_known_positive_overlap"] == 0,
        "candidate_width_met": min(
            metrics["minimum_train_unlabeled"], metrics["minimum_test_unlabeled"]
        )
        >= config.minimum_unlabeled_per_pool,
        "all_model_seed_rows_present": all(
            any(row["model"] == model and row["seed"] == seed for row in rows)
            for model in MODEL_NAMES
            for seed in SEEDS
        ),
        "true_wrong_parameter_counts_equal": true_parameters == wrong_parameters
        and len(true_parameters) == 1,
        "true_wrong_permutations_are_distinct_bijections": sorted(PRESENT_P) == list(range(64))
        and sorted(WRONG_P) == list(range(64))
        and PRESENT_P != WRONG_P,
        "true_wrong_cycle_structures_match": _cycle_histogram(PRESENT_P)
        == _cycle_histogram(WRONG_P),
    }
    anchor_replay_checks: dict[str, bool] = {}
    advance_checks: dict[str, bool] = {}
    capacity_checks: list[bool] = []
    for seed in SEEDS:
        anchor = _row(rows, "coordinate_anchor", seed)
        true_p = _row(rows, "identity_true_p_residual", seed)
        wrong_p = _row(rows, "identity_wrong_p_residual", seed)
        expected = E99_ANCHORS[seed]
        anchor_replay_checks[f"seed{seed}_anchor_recall_at_1_replays"] = abs(
            anchor["recall_at_1"] - expected["recall_at_1"]
        ) <= 0.03
        anchor_replay_checks[f"seed{seed}_anchor_recall_at_5_replays"] = abs(
            anchor["recall_at_5"] - expected["recall_at_5"]
        ) <= 0.01
        anchor_replay_checks[f"seed{seed}_anchor_mrr_replays"] = abs(
            anchor["mean_reciprocal_rank"] - expected["mrr"]
        ) <= 0.03
        advance_checks[f"seed{seed}_true_recall_at_1_beats_anchor"] = true_p[
            "recall_at_1"
        ] >= anchor["recall_at_1"] + config.minimum_anchor_recall_at_1_gain
        advance_checks[f"seed{seed}_true_mrr_beats_anchor"] = true_p[
            "mean_reciprocal_rank"
        ] >= anchor["mean_reciprocal_rank"] + config.minimum_anchor_mrr_gain
        advance_checks[f"seed{seed}_true_recall_at_5_preserved"] = true_p[
            "recall_at_5"
        ] >= anchor["recall_at_5"] - config.maximum_recall_at_5_drop
        advance_checks[f"seed{seed}_true_recall_at_1_beats_wrong_p"] = true_p[
            "recall_at_1"
        ] >= wrong_p["recall_at_1"] + config.minimum_wrong_p_recall_at_1_gain
        advance_checks[f"seed{seed}_true_mrr_beats_wrong_p"] = true_p[
            "mean_reciprocal_rank"
        ] >= wrong_p["mean_reciprocal_rank"] + config.minimum_wrong_p_mrr_gain
        advance_checks[f"seed{seed}_worst_fold_recall_at_5_met"] = true_p[
            "minimum_fold_recall_at_5"
        ] >= config.minimum_fold_recall_at_5
        capacity_checks.append(
            true_p["recall_at_1"]
            >= anchor["recall_at_1"] + config.minimum_anchor_recall_at_1_gain
            and true_p["mean_reciprocal_rank"]
            >= anchor["mean_reciprocal_rank"] + config.minimum_anchor_mrr_gain
        )
    advance_checks["true_p_recall_at_1_gain_direction_consistent"] = all(
        _row(rows, "identity_true_p_residual", seed)["recall_at_1"]
        > _row(rows, "coordinate_anchor", seed)["recall_at_1"]
        for seed in SEEDS
    )
    if not all(protocol_checks.values()) or not all(anchor_replay_checks.values()):
        status = "fail"
        decision = "innovation2_present_r9_identity_topology_residual_protocol_invalid"
        action = "repair source replay, paired model construction, folds, or metrics"
    elif all(advance_checks.values()):
        status = "pass"
        decision = "innovation2_present_r9_identity_true_p_residual_attributed"
        action = "design independent-source or newly generated relation confirmation; keep remote scale closed"
    elif all(capacity_checks):
        status = "hold"
        decision = "innovation2_present_r9_identity_residual_capacity_only"
        action = "retain capacity observation but stop true-P attribution; keep remote scale closed"
    else:
        status = "hold"
        decision = "innovation2_present_r9_coordinate_identity_anchor_remains_best"
        action = "retain E99 generic coordinate result and stop the current PRESENT topology branch"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "anchor_replay_checks": anchor_replay_checks,
        "advance_checks": advance_checks,
        "metrics": metrics,
        "aggregate_metrics": rows,
        "claim_scope": (
            "local paired attribution of a PRESENT-P residual on the public r9 ATM positive-"
            "unlabeled relation-ranking corpus under independent round keys; this is not "
            "strict negative classification, a new relation, PRESENT-80 key-schedule evidence, "
            "an independent-source confirmation, a distinguisher, an attack, remote-scale "
            "evidence, or SOTA"
        ),
        "next_action": {
            "action": action,
            "independent_confirmation_design_open": status == "pass",
            "remote_scale": False,
        },
    }


def result_rows(
    config: IdentityTopologyResidualConfig,
    training: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_identity_topology_residual_attribution",
        "status": gate["status"],
        "decision": gate["decision"],
        "candidate_semantics": "unlabeled_not_negative",
    }
    return [
        {**common, "result_kind": "aggregate", **row}
        for row in training["aggregate_rows"]
    ] + [
        {**common, "result_kind": "fold", **row}
        for row in training["fold_metrics"]
    ]


def serializable_config(config: IdentityTopologyResidualConfig) -> dict[str, Any]:
    return asdict(config)


def build_folds(groups: dict[str, set[Any]]) -> dict[str, Any]:
    return build_neural_folds(groups, PuNeuralRankingConfig())


def _masked_mean_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1)
    total = (values * expanded).sum(dim=-2)
    average = total / expanded.sum(dim=-2).clamp_min(1)
    maximum = values.masked_fill(~expanded, -1e9).amax(dim=-2)
    return torch.cat((average, maximum), dim=-1)


def _cycle_histogram(permutation: tuple[int, ...]) -> dict[int, int]:
    visited: set[int] = set()
    histogram: dict[int, int] = {}
    for start in range(len(permutation)):
        if start in visited:
            continue
        current = start
        length = 0
        while current not in visited:
            visited.add(current)
            current = permutation[current]
            length += 1
        histogram[length] = histogram.get(length, 0) + 1
    return histogram


def _row(rows: list[dict[str, Any]], model: str, seed: int) -> dict[str, Any]:
    return next(row for row in rows if row["model"] == model and row["seed"] == seed)


__all__ = [
    "E99_DECISION",
    "E99_GATE_SHA256",
    "IdentityTopologyResidual",
    "IdentityTopologyResidualConfig",
    "MODEL_NAMES",
    "RUN_ID",
    "WRONG_P",
    "adjudicate_identity_topology_residual",
    "build_folds",
    "make_model",
    "result_rows",
    "serializable_config",
    "sha256",
    "train_attribution_matrix",
]
