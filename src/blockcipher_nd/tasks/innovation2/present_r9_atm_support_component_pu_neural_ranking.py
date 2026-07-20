from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import Property
from blockcipher_nd.tasks.innovation2.present_r9_atm_support_component_pu_readiness import (
    _canonical_independent_basis,
    _filtered_candidates,
    _pack_components,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_support_rotation_orbit_pu_readiness import (
    _support_rotation_components,
)
from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import (
    _absolute_position,
    _canonical_coordinates,
    _relation_id,
    _rotation_candidates,
)


RUN_ID = "i2_present_r9_atm_support_component_pu_neural_ranking_seed0_seed1_20260720"
E98C_GATE_SHA256 = "ebebd137a90c53ea9a45c0f3af8a30b02803d9f1e395f38e4d822bbd31523568"
E98C_DECISION = "innovation2_present_r9_atm_support_orbit_pu_ready"
E98C_ANCHOR_RECALL_AT_5 = 0.1282051282051282
E98C_ANCHOR_MRR = 0.1190011928303896
MODEL_NAMES = (
    "summary_mlp",
    "coordinate_deepsets",
    "present_topology_set",
    "present_topology_set_label_shuffle",
)
SEEDS = (0, 1)
PRESENT_P = tuple((16 * bit) % 63 if bit < 63 else 63 for bit in range(64))
PRESENT_P_INVERSE = tuple(PRESENT_P.index(bit) for bit in range(64))


@dataclass(frozen=True)
class PuNeuralRankingConfig:
    run_id: str = RUN_ID
    group_count: int = 6
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    minimum_unlabeled_per_pool: int = 31
    topology_recall_margin_over_anchor: float = 0.05
    topology_mrr_margin_over_anchor: float = 0.03
    topology_recall_margin_over_summary: float = 0.02
    topology_mrr_margin_over_summary: float = 0.01
    topology_recall_margin_over_coordinate: float = 0.01
    topology_mrr_margin_over_coordinate: float = 0.005
    minimum_fold_recall_at_5: float = 0.10
    maximum_seed_recall_delta: float = 0.10

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError("E99 run_id is frozen")
        if (
            self.group_count != 6
            or self.epochs != 40
            or self.batch_size != 32
            or self.minimum_unlabeled_per_pool != 31
        ):
            raise ValueError("E99 width and training budget are frozen")


@dataclass(frozen=True)
class FoldData:
    fold: int
    train_pools: tuple[dict[str, Any], ...]
    test_pools: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class PoolTensors:
    coordinates: torch.Tensor
    coordinate_mask: torch.Tensor
    item_mask: torch.Tensor
    relation_ids: tuple[tuple[str, ...], ...]


def build_neural_folds(
    groups: dict[str, set[Property]],
    config: PuNeuralRankingConfig,
) -> dict[str, Any]:
    all_known = set().union(*groups.values())
    basis, dropped = _canonical_independent_basis(all_known)
    components, _ = _support_rotation_components(basis)
    packed = _pack_components(components, config.group_count)
    relation_groups = tuple(
        tuple(relation for component in component_group for relation in component)
        for component_group in packed
    )
    folds: list[FoldData] = []
    fold_rows: list[dict[str, Any]] = []
    for heldout_index, heldout_group in enumerate(relation_groups):
        test_positives = set(heldout_group)
        train_positives = set().union(
            *(set(group) for index, group in enumerate(relation_groups) if index != heldout_index)
        )
        train_support = Counter(
            coordinate for relation in train_positives for coordinate in relation
        )
        test_support = Counter(
            coordinate for relation in test_positives for coordinate in relation
        )
        test_pools = tuple(
            _pool(
                relation,
                _filtered_candidates(relation, all_known, train_support),
                group=heldout_index,
                split="test",
            )
            for relation in sorted(test_positives, key=_canonical_coordinates)
        )
        test_unlabeled = {
            candidate
            for pool in test_pools
            for candidate in pool["unlabeled_relations"]
        }
        train_pools = tuple(
            _pool(
                relation,
                tuple(
                    candidate
                    for candidate in _rotation_candidates(relation, all_known)
                    if all(coordinate not in test_support for coordinate in candidate)
                    and candidate not in test_unlabeled
                ),
                group=heldout_index,
                split="train",
            )
            for relation in sorted(train_positives, key=_canonical_coordinates)
        )
        train_relations = {
            relation for pool in train_pools for relation in pool["relations"]
        }
        test_relations = {
            relation for pool in test_pools for relation in pool["relations"]
        }
        train_candidate_test_support_overlap = sum(
            any(coordinate in test_support for coordinate in candidate)
            for pool in train_pools
            for candidate in pool["unlabeled_relations"]
        )
        test_candidate_train_support_overlap = sum(
            any(coordinate in train_support for coordinate in candidate)
            for pool in test_pools
            for candidate in pool["unlabeled_relations"]
        )
        fold_audit = {
            "fold": heldout_index,
            "train_positive_pools": len(train_pools),
            "test_positive_pools": len(test_pools),
            "minimum_train_unlabeled": min(
                pool["unlabeled_count"] for pool in train_pools
            ),
            "minimum_test_unlabeled": min(
                pool["unlabeled_count"] for pool in test_pools
            ),
            "train_test_relation_overlap": len(train_relations & test_relations),
            "train_candidate_test_positive_support_overlap": (
                train_candidate_test_support_overlap
            ),
            "test_candidate_train_positive_support_overlap": (
                test_candidate_train_support_overlap
            ),
            "train_known_positive_candidate_overlap": sum(
                candidate in all_known
                for pool in train_pools
                for candidate in pool["unlabeled_relations"]
            ),
            "test_known_positive_candidate_overlap": sum(
                candidate in all_known
                for pool in test_pools
                for candidate in pool["unlabeled_relations"]
            ),
            "absolute_position_target": mean(
                _absolute_position(relation) for relation in train_positives
            ),
        }
        folds.append(
            FoldData(
                fold=heldout_index,
                train_pools=train_pools,
                test_pools=test_pools,
                audit=fold_audit,
            )
        )
        fold_rows.append(fold_audit)
    return {
        "folds": tuple(folds),
        "fold_rows": fold_rows,
        "metrics": {
            "known_relations": len(all_known),
            "independent_relations": len(basis),
            "dependent_relations_removed": len(dropped),
            "groups": len(relation_groups),
            "minimum_group_positives": min(map(len, relation_groups)),
            "maximum_group_positives": max(map(len, relation_groups)),
            "minimum_train_unlabeled": min(
                row["minimum_train_unlabeled"] for row in fold_rows
            ),
            "minimum_test_unlabeled": min(
                row["minimum_test_unlabeled"] for row in fold_rows
            ),
            "maximum_train_test_relation_overlap": max(
                row["train_test_relation_overlap"] for row in fold_rows
            ),
            "candidate_positive_support_overlap": sum(
                row["train_candidate_test_positive_support_overlap"]
                + row["test_candidate_train_positive_support_overlap"]
                for row in fold_rows
            ),
            "candidate_known_positive_overlap": sum(
                row["train_known_positive_candidate_overlap"]
                + row["test_known_positive_candidate_overlap"]
                for row in fold_rows
            ),
        },
    }


def train_neural_matrix(
    config: PuNeuralRankingConfig,
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
            baseline = evaluate_absolute_position(
                fold_data.test_pools,
                position_target=fold_data.audit["absolute_position_target"],
            )
            fold_metrics.append(
                {
                    "model": "absolute_position",
                    "seed": seed,
                    "fold": fold_data.fold,
                    "parameter_count": 0,
                    "training_seconds": 0.0,
                    **baseline,
                }
            )
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
    aggregate_rows = aggregate_fold_metrics(fold_metrics)
    return {
        "fold_metrics": fold_metrics,
        "aggregate_rows": aggregate_rows,
        "history": history,
    }


def train_one_fold(
    config: PuNeuralRankingConfig,
    *,
    model_name: str,
    seed: int,
    fold: int,
    train_tensors: PoolTensors,
    test_tensors: PoolTensors,
    device: str,
    include_state_dict: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(seed * 1009 + fold * 37 + 17)
    np.random.seed(seed * 1009 + fold * 37 + 17)
    base_name = model_name.replace("_label_shuffle", "")
    model = make_model(base_name).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    shuffle_targets = model_name.endswith("_label_shuffle")
    targets = _training_targets(train_tensors, seed=seed, fold=fold, shuffled=shuffle_targets)
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
            scores = model(coordinates, coordinate_mask)
            scores = scores.masked_fill(~item_mask, -1e9)
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
            }
        )
    training_seconds = time.monotonic() - started
    metrics = evaluate_model(model, test_tensors, device=device, batch_size=config.batch_size)
    result = {
        "metrics": {
            "model": model_name,
            "seed": seed,
            "fold": fold,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": training_seconds,
            "final_train_loss": history[-1]["train_listwise_loss"],
            **metrics,
        },
        "history": history,
    }
    if include_state_dict:
        result["state_dict"] = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    return result


class SummaryMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, coordinates: torch.Tensor, coordinate_mask: torch.Tensor) -> torch.Tensor:
        features = _summary_tensor(coordinates, coordinate_mask)
        return self.network(features).squeeze(-1)


class CoordinateDeepSets(nn.Module):
    def __init__(self, hidden: int = 12) -> None:
        super().__init__()
        self.input_embedding = nn.Parameter(torch.empty(64, hidden))
        self.output_embedding = nn.Parameter(torch.empty(64, hidden))
        nn.init.normal_(self.input_embedding, std=0.08)
        nn.init.normal_(self.output_embedding, std=0.08)
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden * 2), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, 1)
        )

    def forward(self, coordinates: torch.Tensor, coordinate_mask: torch.Tensor) -> torch.Tensor:
        left = coordinates[..., :64]
        right = coordinates[..., 64:]
        left_embedding = torch.matmul(left, self.input_embedding) / left.sum(-1, keepdim=True).clamp_min(1.0)
        right_embedding = torch.matmul(right, self.output_embedding) / right.sum(-1, keepdim=True).clamp_min(1.0)
        encoded = self.coordinate_mlp(torch.cat((left_embedding, right_embedding), dim=-1))
        pooled = _masked_mean_max(encoded, coordinate_mask)
        return self.head(pooled).squeeze(-1)


class PresentTopologySet(nn.Module):
    def __init__(self, hidden: int = 12) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, 1)
        )
        self.register_buffer(
            "p_inverse", torch.tensor(PRESENT_P_INVERSE, dtype=torch.long), persistent=False
        )

    def forward(self, coordinates: torch.Tensor, coordinate_mask: torch.Tensor) -> torch.Tensor:
        left = coordinates[..., :64]
        right = coordinates[..., 64:]
        direct = torch.stack((left, right), dim=-1)
        p_messages = direct.index_select(-2, self.p_inverse)
        shape = direct.shape
        cell = direct.reshape(*shape[:-2], 16, 4, 2).mean(dim=-2, keepdim=True)
        cell = cell.expand(*shape[:-2], 16, 4, 2).reshape(shape)
        node_features = torch.cat((direct, p_messages, cell), dim=-1)
        nodes = self.node_mlp(node_features)
        coordinate = torch.cat((nodes.mean(dim=-2), nodes.amax(dim=-2)), dim=-1)
        coordinate = self.coordinate_mlp(coordinate)
        pooled = _masked_mean_max(coordinate, coordinate_mask)
        return self.head(pooled).squeeze(-1)


def make_model(model_name: str) -> nn.Module:
    if model_name == "summary_mlp":
        return SummaryMlp()
    if model_name == "coordinate_deepsets":
        return CoordinateDeepSets()
    if model_name == "present_topology_set":
        return PresentTopologySet()
    raise ValueError(f"unknown E99 model: {model_name}")


def tensorize_pools(pools: tuple[dict[str, Any], ...]) -> PoolTensors:
    maximum_items = max(len(pool["relations"]) for pool in pools)
    maximum_coordinates = max(len(relation) for pool in pools for relation in pool["relations"])
    coordinates = torch.zeros(
        (len(pools), maximum_items, maximum_coordinates, 128), dtype=torch.float32
    )
    coordinate_mask = torch.zeros(
        (len(pools), maximum_items, maximum_coordinates), dtype=torch.bool
    )
    item_mask = torch.zeros((len(pools), maximum_items), dtype=torch.bool)
    relation_ids: list[tuple[str, ...]] = []
    for pool_index, pool in enumerate(pools):
        ids: list[str] = []
        for item_index, relation in enumerate(pool["relations"]):
            item_mask[pool_index, item_index] = True
            ids.append(_relation_id(relation))
            for coordinate_index, (left, right) in enumerate(sorted(relation)):
                coordinate_mask[pool_index, item_index, coordinate_index] = True
                for bit in range(64):
                    coordinates[pool_index, item_index, coordinate_index, bit] = (
                        left >> bit
                    ) & 1
                    coordinates[pool_index, item_index, coordinate_index, 64 + bit] = (
                        right >> bit
                    ) & 1
        relation_ids.append(tuple(ids))
    return PoolTensors(
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        item_mask=item_mask,
        relation_ids=tuple(relation_ids),
    )


def evaluate_model(
    model: nn.Module,
    tensors: PoolTensors,
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    scores = score_model(model, tensors, device=device, batch_size=batch_size)
    ranks = _ranks(scores, tensors.relation_ids)
    return _ranking_metrics(ranks, [len(ids) for ids in tensors.relation_ids])


def score_model(
    model: nn.Module,
    tensors: PoolTensors,
    *,
    device: str,
    batch_size: int,
) -> list[list[float]]:
    scores: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tensors.coordinates.shape[0], batch_size):
            stop = start + batch_size
            batch_scores = model(
                tensors.coordinates[start:stop].to(device),
                tensors.coordinate_mask[start:stop].to(device),
            )
            batch_mask = tensors.item_mask[start:stop].to(device)
            batch_scores = batch_scores.masked_fill(~batch_mask, -1e9).cpu()
            for row_index, mask in enumerate(tensors.item_mask[start:stop]):
                scores.append(batch_scores[row_index, : int(mask.sum())].tolist())
    return scores


def evaluate_absolute_position(
    pools: tuple[dict[str, Any], ...],
    *,
    position_target: float,
) -> dict[str, Any]:
    scores = [
        [-abs(_absolute_position(relation) - position_target) for relation in pool["relations"]]
        for pool in pools
    ]
    ids = [tuple(_relation_id(relation) for relation in pool["relations"]) for pool in pools]
    ranks = _ranks(scores, ids)
    return _ranking_metrics(ranks, [len(row) for row in ids])


def aggregate_fold_metrics(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted({(row["model"], row["seed"]) for row in fold_rows})
    for model, seed in keys:
        selected = [row for row in fold_rows if row["model"] == model and row["seed"] == seed]
        total = sum(row["ranking_pools"] for row in selected)
        rows.append(
            {
                "model": model,
                "seed": seed,
                "folds": len(selected),
                "ranking_pools": total,
                "parameter_count": max(row["parameter_count"] for row in selected),
                "training_seconds": sum(row["training_seconds"] for row in selected),
                "recall_at_1": sum(row["recall_at_1"] * row["ranking_pools"] for row in selected) / total,
                "recall_at_5": sum(row["recall_at_5"] * row["ranking_pools"] for row in selected) / total,
                "mean_reciprocal_rank": sum(
                    row["mean_reciprocal_rank"] * row["ranking_pools"] for row in selected
                ) / total,
                "top5_enrichment": sum(row["top5_enrichment"] * row["ranking_pools"] for row in selected) / total,
                "minimum_fold_recall_at_5": min(row["recall_at_5"] for row in selected),
                "maximum_fold_recall_at_5": max(row["recall_at_5"] for row in selected),
                "final_train_loss": mean(
                    row.get("final_train_loss", 0.0) for row in selected
                ),
            }
        )
    return rows


def adjudicate_neural_ranking(
    config: PuNeuralRankingConfig,
    *,
    fold_audit: dict[str, Any],
    training: dict[str, Any],
    e98c_gate: dict[str, Any],
    e98c_gate_hash: str,
) -> dict[str, Any]:
    rows = training["aggregate_rows"]
    anchor = _aggregate_row(rows, "absolute_position", 0)
    metrics = {
        **fold_audit["metrics"],
        "anchor_recall_at_5": anchor["recall_at_5"],
        "anchor_mrr": anchor["mean_reciprocal_rank"],
    }
    protocol_checks = {
        "e98c_gate_hash_matches": e98c_gate_hash == E98C_GATE_SHA256,
        "e98c_status_pass": e98c_gate.get("status") == "pass",
        "e98c_decision_matches": e98c_gate.get("decision") == E98C_DECISION,
        "exact_468_independent_relations": metrics["independent_relations"] == 468,
        "six_balanced_groups": metrics["groups"] == 6
        and metrics["minimum_group_positives"] == 78
        and metrics["maximum_group_positives"] == 78,
        "train_test_relation_overlap_zero": metrics[
            "maximum_train_test_relation_overlap"
        ]
        == 0,
        "candidate_positive_support_overlap_zero": metrics[
            "candidate_positive_support_overlap"
        ]
        == 0,
        "candidate_known_positive_overlap_zero": metrics[
            "candidate_known_positive_overlap"
        ]
        == 0,
        "minimum_candidate_width_met": min(
            metrics["minimum_train_unlabeled"], metrics["minimum_test_unlabeled"]
        )
        >= config.minimum_unlabeled_per_pool,
        "absolute_position_anchor_replays": abs(
            anchor["recall_at_5"] - E98C_ANCHOR_RECALL_AT_5
        )
        < 1e-12
        and abs(anchor["mean_reciprocal_rank"] - E98C_ANCHOR_MRR) < 1e-12,
        "all_model_seed_rows_present": all(
            any(row["model"] == model and row["seed"] == seed for row in rows)
            for model in ("absolute_position", *MODEL_NAMES)
            for seed in SEEDS
        ),
    }
    seed_checks: dict[str, bool] = {}
    for seed in SEEDS:
        topology = _aggregate_row(rows, "present_topology_set", seed)
        summary = _aggregate_row(rows, "summary_mlp", seed)
        coordinate = _aggregate_row(rows, "coordinate_deepsets", seed)
        shuffled = _aggregate_row(rows, "present_topology_set_label_shuffle", seed)
        seed_checks[f"seed{seed}_topology_recall_beats_anchor"] = topology[
            "recall_at_5"
        ] >= anchor["recall_at_5"] + config.topology_recall_margin_over_anchor
        seed_checks[f"seed{seed}_topology_mrr_beats_anchor"] = topology[
            "mean_reciprocal_rank"
        ] >= anchor["mean_reciprocal_rank"] + config.topology_mrr_margin_over_anchor
        seed_checks[f"seed{seed}_topology_recall_beats_summary"] = topology[
            "recall_at_5"
        ] >= summary["recall_at_5"] + config.topology_recall_margin_over_summary
        seed_checks[f"seed{seed}_topology_mrr_beats_summary"] = topology[
            "mean_reciprocal_rank"
        ] >= summary["mean_reciprocal_rank"] + config.topology_mrr_margin_over_summary
        seed_checks[f"seed{seed}_topology_recall_beats_coordinate"] = topology[
            "recall_at_5"
        ] >= coordinate["recall_at_5"] + config.topology_recall_margin_over_coordinate
        seed_checks[f"seed{seed}_topology_mrr_beats_coordinate"] = topology[
            "mean_reciprocal_rank"
        ] >= coordinate["mean_reciprocal_rank"] + config.topology_mrr_margin_over_coordinate
        seed_checks[f"seed{seed}_worst_fold_recall_met"] = topology[
            "minimum_fold_recall_at_5"
        ] >= config.minimum_fold_recall_at_5
        seed_checks[f"seed{seed}_shuffle_does_not_pass_anchor_gate"] = not (
            shuffled["recall_at_5"]
            >= anchor["recall_at_5"] + config.topology_recall_margin_over_anchor
            and shuffled["mean_reciprocal_rank"]
            >= anchor["mean_reciprocal_rank"] + config.topology_mrr_margin_over_anchor
        )
    topology_seed_rows = [
        _aggregate_row(rows, "present_topology_set", seed) for seed in SEEDS
    ]
    seed_checks["topology_seed_recall_delta_bounded"] = abs(
        topology_seed_rows[0]["recall_at_5"] - topology_seed_rows[1]["recall_at_5"]
    ) <= config.maximum_seed_recall_delta
    generic_signal = all(
        any(
            _aggregate_row(rows, model, seed)["recall_at_5"]
            >= anchor["recall_at_5"] + config.topology_recall_margin_over_anchor
            and _aggregate_row(rows, model, seed)["mean_reciprocal_rank"]
            >= anchor["mean_reciprocal_rank"] + config.topology_mrr_margin_over_anchor
            for model in ("summary_mlp", "coordinate_deepsets")
        )
        for seed in SEEDS
    )
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_r9_pu_neural_ranking_protocol_invalid"
        action = "repair source replay, fold pools, model matrix, or metric protocol"
    elif all(seed_checks.values()):
        status = "pass"
        decision = "innovation2_present_r9_pu_topology_neural_signal_confirmed"
        action = (
            "design remote scale and independent-source confirmation; do not launch until "
            "the remote dataset cache and claim gates are preregistered"
        )
    elif generic_signal:
        status = "hold"
        decision = "innovation2_present_r9_pu_generic_neural_signal_only"
        action = "redesign the PRESENT topology encoder locally; keep remote scale closed"
    else:
        status = "hold"
        decision = "innovation2_present_r9_pu_public_corpus_neural_route_stopped"
        action = "stop the current public-corpus nine-round neural route; do not scale epochs or width"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "advance_checks": seed_checks,
        "metrics": metrics,
        "aggregate_metrics": rows,
        "claim_scope": (
            "local six-fold positive-unlabeled neural ranking on the corrected public PRESENT "
            "r9 ATM corpus under independent round keys; this is not strict negative "
            "classification, a new relation, PRESENT-80 key-schedule evidence, an independent "
            "publication reproduction, a distinguisher, an attack, remote-scale evidence, or SOTA"
        ),
        "next_action": {
            "action": action,
            "remote_design_open": status == "pass",
            "remote_scale": False,
        },
    }


def result_rows(
    config: PuNeuralRankingConfig,
    training: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_atm_support_component_pu_neural_ranking",
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


def serializable_config(config: PuNeuralRankingConfig) -> dict[str, Any]:
    return asdict(config)


def _pool(
    positive: Property,
    candidates: tuple[Property, ...],
    *,
    group: int,
    split: str,
) -> dict[str, Any]:
    return {
        "group": group,
        "split": split,
        "positive": positive,
        "positive_id": _relation_id(positive),
        "unlabeled_relations": candidates,
        "unlabeled_count": len(candidates),
        "relations": (positive, *candidates),
    }


def _training_targets(
    tensors: PoolTensors,
    *,
    seed: int,
    fold: int,
    shuffled: bool,
) -> torch.Tensor:
    if not shuffled:
        return torch.zeros(len(tensors.relation_ids), dtype=torch.long)
    targets = []
    for relation_ids in tensors.relation_ids:
        digest = hashlib.sha256(
            f"E99-label-shuffle|{seed}|{fold}|{relation_ids[0]}".encode("ascii")
        ).digest()
        targets.append(1 + int.from_bytes(digest[:8], "big") % (len(relation_ids) - 1))
    return torch.tensor(targets, dtype=torch.long)


def _summary_tensor(
    coordinates: torch.Tensor,
    coordinate_mask: torch.Tensor,
) -> torch.Tensor:
    left = coordinates[..., :64]
    right = coordinates[..., 64:]
    mask = coordinate_mask.to(coordinates.dtype)
    count = mask.sum(-1, keepdim=True).clamp_min(1.0)
    left_weight = left.sum(-1)
    right_weight = right.sum(-1)
    xor_weight = (left - right).abs().sum(-1)
    positions = torch.arange(64, device=coordinates.device, dtype=coordinates.dtype)
    left_position = (left * positions).sum(-1) / left_weight.clamp_min(1.0)
    right_position = (right * positions).sum(-1) / right_weight.clamp_min(1.0)

    def masked_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * mask).sum(-1) / count.squeeze(-1)

    def masked_std(value: torch.Tensor) -> torch.Tensor:
        average = masked_mean(value)
        variance = (((value - average.unsqueeze(-1)) ** 2) * mask).sum(-1) / count.squeeze(-1)
        return variance.sqrt()

    features = torch.stack(
        (
            count.squeeze(-1) / 4.0,
            masked_mean(left_weight) / 64.0,
            masked_std(left_weight) / 64.0,
            masked_mean(right_weight) / 64.0,
            masked_std(right_weight) / 64.0,
            masked_mean(xor_weight) / 64.0,
            masked_std(xor_weight) / 64.0,
            masked_mean(left_position) / 63.0,
            masked_std(left_position) / 63.0,
            masked_mean(right_position) / 63.0,
            masked_std(right_position) / 63.0,
            left.amax(dim=(-2, -1)),
            right.amax(dim=(-2, -1)),
            left.mean(dim=(-2, -1)),
            right.mean(dim=(-2, -1)),
            (left * right).mean(dim=(-2, -1)),
        ),
        dim=-1,
    )
    return features


def _masked_mean_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1)
    total = (values * expanded).sum(dim=-2)
    average = total / expanded.sum(dim=-2).clamp_min(1)
    maximum = values.masked_fill(~expanded, -1e9).amax(dim=-2)
    return torch.cat((average, maximum), dim=-1)


def _ranks(
    scores: list[list[float]],
    relation_ids: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
) -> list[int]:
    ranks: list[int] = []
    for pool_scores, ids in zip(scores, relation_ids, strict=True):
        ranked = sorted(zip(pool_scores, ids, strict=True), key=lambda item: (-item[0], item[1]))
        ranks.append(next(index for index, (_, relation_id) in enumerate(ranked, 1) if relation_id == ids[0]))
    return ranks


def _ranking_metrics(ranks: list[int], pool_sizes: list[int]) -> dict[str, Any]:
    random_top5 = mean(min(5 / size, 1.0) for size in pool_sizes)
    recall_at_5 = mean(rank <= 5 for rank in ranks)
    return {
        "ranking_pools": len(ranks),
        "recall_at_1": mean(rank <= 1 for rank in ranks),
        "recall_at_5": recall_at_5,
        "mean_reciprocal_rank": mean(1 / rank for rank in ranks),
        "top5_enrichment": recall_at_5 / random_top5,
        "minimum_rank": min(ranks),
        "maximum_rank": max(ranks),
    }


def _aggregate_row(rows: list[dict[str, Any]], model: str, seed: int) -> dict[str, Any]:
    return next(row for row in rows if row["model"] == model and row["seed"] == seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
