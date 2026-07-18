from __future__ import annotations

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

from blockcipher_nd.ciphers.spn.skinny import cells_to_int, mix_columns, shift_rows
from blockcipher_nd.models.structure.spn.sparse_linear_profile_operator import (
    SparseLinearProfileOperator,
    SparseLinearProfileOperatorSpec,
)
from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_readiness import (
    _batch_tensors,
    _cell_permutation,
    _copy_parameters,
    masked_binary_cross_entropy,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    fit_train_only_ridge,
)
from blockcipher_nd.training.metrics import binary_auc


SOURCE_RUN_ID = "i2_skinny64_r5_unit_balance_profile_transition_20260719"
SOURCE_DECISION = "innovation2_skinny64_r5_unit_balance_profile_transition_ready"
RELATION_MODES = ("independent", "true", "corrupted")
R4_SLICE = slice(39, 52)
RIDGE_LAMBDA = 1e-3
EXPECTED_PARAMETER_COUNT = 4_795


@dataclass(frozen=True)
class Skinny64SparseProfileReadinessConfig:
    run_id: str
    epochs: int = 2
    batch_size: int = 8
    hidden_dim: int = 32
    steps: int = 2
    seed: int = 0
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.epochs != 2
            or self.batch_size != 8
            or self.hidden_dim != 32
            or self.steps != 2
            or self.seed != 0
            or self.dropout != 0.10
            or self.learning_rate != 1e-3
            or self.weight_decay != 1e-4
            or self.device != "cpu"
        ):
            raise ValueError("E83 readiness protocol is frozen")


def load_skinny_profile_sources(root: Path) -> dict[str, Any]:
    gate = json.loads((root / "gate.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    structures = json.loads(
        (root / "structures.json").read_text(encoding="utf-8")
    )["structures"]
    targets = np.load(root / "profile_targets.npy", allow_pickle=False)
    observed = np.load(root / "profile_observed.npy", allow_pickle=False)
    prefix = np.load(root / "prefix_features.npy", allow_pickle=False)
    with (root / "matched_unit_contrast.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        matched_rows = [
            {
                "split": row["split"],
                "structure_index": int(row["structure_index"]),
                "output_bit": int(row["output_bit"]),
                "label": int(row["label"]),
            }
            for row in csv.DictReader(handle)
        ]
    names = (
        "gate.json",
        "metadata.json",
        "structures.json",
        "profile_targets.npy",
        "profile_observed.npy",
        "prefix_features.npy",
        "matched_unit_contrast.csv",
    )
    return {
        "profile_gate": gate,
        "profile_metadata": metadata,
        "structures": structures,
        "profile_targets": np.asarray(targets, dtype=np.int8),
        "profile_observed": np.asarray(observed, dtype=np.bool_),
        "prefix_features": np.asarray(prefix, dtype=np.float64),
        "matched_rows": matched_rows,
        "source_hashes": {name: _sha256(root / name) for name in names},
    }


def validate_skinny_profile_sources(sources: dict[str, Any]) -> dict[str, bool]:
    rows = sources["matched_rows"]
    targets = sources["profile_targets"]
    observed = sources["profile_observed"]
    reconstructed_targets = np.full((96, 64), -1, dtype=np.int8)
    reconstructed_observed = np.zeros((96, 64), dtype=np.bool_)
    for row in rows:
        structure = row["structure_index"]
        output = row["output_bit"]
        if reconstructed_observed[structure, output]:
            return {"matched_edges_unique": False}
        reconstructed_targets[structure, output] = row["label"]
        reconstructed_observed[structure, output] = True
    train_structures = {
        row["structure_index"] for row in rows if row["split"] == "train"
    }
    validation_structures = {
        row["structure_index"] for row in rows if row["split"] == "validation"
    }
    protocol_checks = sources["profile_gate"].get("protocol_checks", {})
    config = sources["profile_metadata"].get("config", {})
    return {
        "profile_run_id_matches": sources["profile_gate"].get("run_id")
        == SOURCE_RUN_ID,
        "profile_decision_matches": sources["profile_gate"].get("decision")
        == SOURCE_DECISION,
        "profile_status_pass": sources["profile_gate"].get("status") == "pass",
        "profile_protocol_checks_pass": bool(protocol_checks)
        and all(protocol_checks.values()),
        "metadata_task_matches": sources["profile_metadata"].get("task")
        == "innovation2_skinny64_r5_unit_balance_profile_transition",
        "metadata_experiment_is_e82": sources["profile_metadata"].get("experiment")
        == "e82",
        "metadata_rounds_are_five": config.get("rounds") == 5,
        "structure_count_is_96": len(sources["structures"]) == 96,
        "target_shape_is_96x64": targets.shape == (96, 64),
        "observed_shape_is_96x64": observed.shape == (96, 64),
        "prefix_shape_is_96x64x52": sources["prefix_features"].shape
        == (96, 64, 52),
        "prefix_features_finite": bool(np.isfinite(sources["prefix_features"]).all()),
        "matched_rows_are_1492": len(rows) == 1_492,
        "observed_edges_are_1492": int(np.sum(observed)) == 1_492,
        "matched_edges_unique": int(np.sum(reconstructed_observed)) == len(rows),
        "targets_replay_matched_csv": np.array_equal(
            reconstructed_targets, targets
        ),
        "observed_replays_matched_csv": np.array_equal(
            reconstructed_observed, observed
        ),
        "observed_targets_binary": bool(np.isin(targets[observed], (0, 1)).all()),
        "unobserved_targets_minus_one": bool(np.all(targets[~observed] == -1)),
        "train_structures_are_47": len(train_structures) == 47,
        "validation_structures_are_17": len(validation_structures) == 17,
        "train_validation_structures_disjoint": train_structures.isdisjoint(
            validation_structures
        ),
        "source_hashes_present": all(
            len(value) == 64 for value in sources["source_hashes"].values()
        ),
    }


def r4_only_sources(sources: dict[str, Any]) -> dict[str, Any]:
    prefix = np.asarray(sources["prefix_features"], dtype=np.float64)
    if prefix.shape != (96, 64, 52):
        raise ValueError("E83 requires the frozen 96x64x52 E82 prefix")
    return {**sources, "prefix_features": prefix[:, :, R4_SLICE].copy()}


def skinny_linear_adjacency() -> np.ndarray:
    adjacency = np.zeros((64, 64), dtype=np.float32)
    for source in range(64):
        cells = [0] * 16
        source_cell = 15 - source // 4
        cells[source_cell] = 1 << (source % 4)
        output = cells_to_int(mix_columns(shift_rows(tuple(cells))))
        for target in range(64):
            adjacency[target, source] = float(bool(output & (1 << target)))
    return adjacency


def corrupted_skinny_linear_adjacency(adjacency: np.ndarray) -> np.ndarray:
    source_permutation = np.asarray(
        [4 * ((bit // 4 + 1) % 16) + bit % 4 for bit in range(64)],
        dtype=np.int64,
    )
    corrupted = np.empty_like(adjacency)
    corrupted[:, source_permutation] = adjacency
    return corrupted


def linear_graph_contract() -> dict[str, Any]:
    true = skinny_linear_adjacency()
    corrupted = corrupted_skinny_linear_adjacency(true)
    true_degree = true.sum(axis=1).astype(np.int64)
    corrupted_degree = corrupted.sum(axis=1).astype(np.int64)
    true_sources = np.argwhere(true > 0)
    corrupted_sources = np.argwhere(corrupted > 0)
    return {
        "true_edge_count": int(true.sum()),
        "corrupted_edge_count": int(corrupted.sum()),
        "true_degree_histogram": {
            str(degree): int(np.sum(true_degree == degree)) for degree in (1, 2, 3)
        },
        "degrees_match": bool(np.array_equal(true_degree, corrupted_degree)),
        "graphs_differ": not np.array_equal(true, corrupted),
        "true_lane_preserved": all(target % 4 == source % 4 for target, source in true_sources),
        "corrupted_lane_preserved": all(
            target % 4 == source % 4 for target, source in corrupted_sources
        ),
    }


def make_skinny_sparse_model(
    config: Skinny64SparseProfileReadinessConfig,
    mode: str,
    *,
    adjacency: np.ndarray | None = None,
    dropout: float | None = None,
) -> SparseLinearProfileOperator:
    if mode not in RELATION_MODES:
        raise ValueError(f"unsupported E83 mode: {mode}")
    true = skinny_linear_adjacency()
    selected = (
        np.asarray(adjacency, dtype=np.float32)
        if adjacency is not None
        else corrupted_skinny_linear_adjacency(true)
        if mode == "corrupted"
        else true
    )
    return SparseLinearProfileOperator(
        SparseLinearProfileOperatorSpec(
            input_dim=13,
            hidden_dim=config.hidden_dim,
            steps=config.steps,
            dropout=config.dropout if dropout is None else dropout,
            relation_mode=mode,
        ),
        torch.from_numpy(selected),
    )


def evaluate_sparse_ridges(sources: dict[str, Any]) -> dict[str, Any]:
    rows = sources["matched_rows"]
    labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
    train = np.asarray([row["split"] == "train" for row in rows])
    validation = ~train
    features = np.asarray(sources["prefix_features"], dtype=np.float64)
    true = skinny_linear_adjacency()
    corrupted = corrupted_skinny_linear_adjacency(true)
    matrices = {
        "local13": _edge_feature_matrix(rows, features, None),
        "true_sparse39": _edge_feature_matrix(rows, features, true),
        "corrupted_sparse39": _edge_feature_matrix(rows, features, corrupted),
    }
    reports: dict[str, Any] = {}
    for name, matrix in matrices.items():
        fitted = fit_train_only_ridge(
            matrix[train], labels[train], matrix[validation], RIDGE_LAMBDA
        )
        reports[name] = {
            "feature_count": int(matrix.shape[1]),
            "train_auc": _safe_auc(labels[train], fitted["train_scores"]),
            "validation_auc": _safe_auc(
                labels[validation], fitted["validation_scores"]
            ),
            "ridge_lambda": RIDGE_LAMBDA,
            "train_standardization_only": True,
        }
    return reports


def _edge_feature_matrix(
    rows: list[dict[str, Any]],
    features: np.ndarray,
    adjacency: np.ndarray | None,
) -> np.ndarray:
    if adjacency is None:
        expanded = features
    else:
        cell = features.reshape(features.shape[0], 16, 4, 13).mean(axis=2)
        cell = np.repeat(cell[:, :, None, :], 4, axis=2).reshape(features.shape)
        normalized = adjacency / adjacency.sum(axis=1, keepdims=True)
        linear = np.einsum("ts,nsh->nth", normalized, features)
        expanded = np.concatenate((features, cell, linear), axis=-1)
    return np.asarray(
        [expanded[row["structure_index"], row["output_bit"]] for row in rows],
        dtype=np.float64,
    )


def measure_sparse_operator_contract(
    config: Skinny64SparseProfileReadinessConfig,
    sources: dict[str, Any],
) -> dict[str, Any]:
    models = {
        mode: make_skinny_sparse_model(config, mode, dropout=0.0)
        for mode in RELATION_MODES
    }
    candidate = models["true"]
    _copy_parameters(candidate, models["independent"])
    _copy_parameters(candidate, models["corrupted"])
    indices = sorted({row["structure_index"] for row in sources["matched_rows"]})[:4]
    features, targets, observed = _batch_tensors(sources, indices, "cpu")
    logits = candidate(features)
    loss = masked_binary_cross_entropy(logits, targets, observed)
    explicit = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[observed], targets[observed]
    )
    loss.backward()
    with torch.no_grad():
        true_logits = candidate(features)
        corrupted_logits = models["corrupted"](features)

    permutation = _cell_permutation()
    relabeled_adjacency = conjugate_adjacency(
        skinny_linear_adjacency(), permutation
    )
    relabeled = make_skinny_sparse_model(
        config, "true", adjacency=relabeled_adjacency, dropout=0.0
    )
    _copy_parameters(candidate, relabeled)
    permuted_features = torch.empty_like(features)
    permuted_features[:, permutation] = features
    with torch.no_grad():
        permuted = relabeled(permuted_features)
    expected = torch.empty_like(true_logits)
    expected[:, permutation] = true_logits
    parameter_counts = {
        mode: sum(parameter.numel() for parameter in model.parameters())
        for mode, model in models.items()
    }
    forbidden = ("certificate", "witness", "parity", "full_cube", "label")
    return {
        "output_shape": list(logits.shape),
        "input_dim": 13,
        "masked_loss_explicit_max_abs_error": float(torch.abs(loss - explicit).detach()),
        "parameter_counts": parameter_counts,
        "parameter_counts_match": len(set(parameter_counts.values())) == 1,
        "true_corrupted_logit_max_abs_difference": float(
            torch.max(torch.abs(true_logits - corrupted_logits))
        ),
        "cell_relabel_max_abs_error": float(torch.max(torch.abs(permuted - expected))),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in candidate.parameters()
        ),
        "forbidden_named_state_absent": not any(
            token in name for name in candidate.state_dict() for token in forbidden
        ),
        "linear_graph": linear_graph_contract(),
    }


def conjugate_adjacency(adjacency: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    relabeled = np.empty_like(adjacency)
    relabeled[np.ix_(permutation, permutation)] = adjacency
    return relabeled


def train_sparse_profile_matrix(
    config: Skinny64SparseProfileReadinessConfig,
    sources: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    rows = sources["matched_rows"]
    train_indices = sorted(
        {row["structure_index"] for row in rows if row["split"] == "train"}
    )
    validation_indices = sorted(
        {row["structure_index"] for row in rows if row["split"] == "validation"}
    )
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    trained_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for mode in RELATION_MODES:
        _seed_everything(config.seed)
        model = make_skinny_sparse_model(config, mode).to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        best: dict[str, Any] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(1, config.epochs + 1):
            model.train()
            generator = torch.Generator().manual_seed(config.seed + epoch)
            order = torch.randperm(len(train_indices), generator=generator).tolist()
            for start in range(0, len(order), config.batch_size):
                selected = order[start : start + config.batch_size]
                batch = [train_indices[index] for index in selected]
                features, targets, observed = _batch_tensors(
                    sources, batch, config.device
                )
                optimizer.zero_grad(set_to_none=True)
                loss = masked_binary_cross_entropy(model(features), targets, observed)
                loss.backward()
                optimizer.step()
            train_metrics = _evaluate(model, sources, train_indices, config)
            validation_metrics = _evaluate(
                model, sources, validation_indices, config
            )
            history = {
                "row_id": f"skinny_r4_sparse_profile_{mode}_seed{config.seed}",
                "relation_mode": mode,
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
            history_rows.append(history)
            if best is None or validation_metrics["auc"] > best["validation_auc"]:
                best = {
                    "run_id": config.run_id,
                    "task": "innovation2_skinny64_r4_only_sparse_profile_readiness",
                    "row_id": history["row_id"],
                    "relation_mode": mode,
                    "seed": config.seed,
                    "best_epoch": epoch,
                    "epochs_completed": epoch,
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{
                        f"validation_{key}": value
                        for key, value in validation_metrics.items()
                    },
                }
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
        if best is None or best_state is None:
            raise RuntimeError("E83 training produced no checkpoint")
        best["epochs_completed"] = config.epochs
        torch.save(best_state, checkpoints / f"{best['row_id']}.pt")
        trained_rows.append(best)
    return {"trained_rows": trained_rows, "history_rows": history_rows}


def _evaluate(
    model: SparseLinearProfileOperator,
    sources: dict[str, Any],
    indices: list[int],
    config: Skinny64SparseProfileReadinessConfig,
) -> dict[str, float]:
    model.eval()
    all_logits = []
    all_targets = []
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), config.batch_size):
            batch = indices[start : start + config.batch_size]
            features, targets, observed = _batch_tensors(
                sources, batch, config.device
            )
            logits = model(features)
            losses.append(float(masked_binary_cross_entropy(logits, targets, observed)))
            all_logits.append(logits[observed].cpu().numpy())
            all_targets.append(targets[observed].cpu().numpy())
    scores = np.concatenate(all_logits)
    labels = np.concatenate(all_targets)
    probabilities = 1.0 / (1.0 + np.exp(-scores))
    return {
        "auc": float(binary_auc(labels.astype(np.float32), scores.astype(np.float64))),
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
        "loss": float(np.mean(losses)),
    }


def adjudicate_sparse_profile_readiness(
    config: Skinny64SparseProfileReadinessConfig,
    source_checks: dict[str, bool],
    contract: dict[str, Any],
    ridges: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    rows = training["trained_rows"]
    by_mode = {row["relation_mode"]: row for row in rows}
    true_auc = float(by_mode.get("true", {}).get("validation_auc", 0.0))
    independent_auc = float(
        by_mode.get("independent", {}).get("validation_auc", 0.0)
    )
    corrupted_auc = float(by_mode.get("corrupted", {}).get("validation_auc", 0.0))
    local_ridge = float(ridges["local13"]["validation_auc"])
    true_ridge = float(ridges["true_sparse39"]["validation_auc"])
    corrupted_ridge = float(ridges["corrupted_sparse39"]["validation_auc"])
    graph = contract["linear_graph"]
    protocol_checks = {
        **source_checks,
        "output_shape_is_4x64": contract["output_shape"] == [4, 64],
        "input_dim_is_13": contract["input_dim"] == 13,
        "masked_loss_matches_explicit": contract[
            "masked_loss_explicit_max_abs_error"
        ]
        <= 1e-7,
        "parameter_counts_match": contract["parameter_counts_match"],
        "parameter_count_is_4795": set(contract["parameter_counts"].values())
        == {EXPECTED_PARAMETER_COUNT},
        "true_corrupted_logits_differ": contract[
            "true_corrupted_logit_max_abs_difference"
        ]
        >= 1e-6,
        "cell_relabel_equivariant": contract["cell_relabel_max_abs_error"] <= 1e-6,
        "forbidden_named_state_absent": contract["forbidden_named_state_absent"],
        "contract_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "true_and_corrupted_have_128_edges": graph["true_edge_count"] == 128
        and graph["corrupted_edge_count"] == 128,
        "degree_profiles_match": graph["degrees_match"],
        "graphs_differ": graph["graphs_differ"],
        "both_graphs_preserve_lane": graph["true_lane_preserved"]
        and graph["corrupted_lane_preserved"],
        "all_three_rows_present": set(by_mode) == set(RELATION_MODES),
        "all_rows_completed_two_epochs": len(rows) == 3
        and all(row["epochs_completed"] == config.epochs for row in rows),
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "train_auc",
                "train_accuracy",
                "train_loss",
                "validation_auc",
                "validation_accuracy",
                "validation_loss",
            )
        ),
        "ridge_reports_train_standardized": all(
            report["train_standardization_only"] for report in ridges.values()
        ),
    }
    deterministic_checks = {
        "true_sparse_ridge_minus_local_at_least_0p03": true_ridge - local_ridge
        >= 0.03,
        "true_sparse_ridge_minus_corrupted_at_least_0p03": true_ridge
        - corrupted_ridge
        >= 0.03,
    }
    readiness_checks = {
        "true_auc_at_least_0p65": true_auc >= 0.65,
        "true_minus_independent_at_least_0p03": true_auc - independent_auc >= 0.03,
        "true_minus_corrupted_at_least_0p03": true_auc - corrupted_auc >= 0.03,
        "true_minus_true_sparse_ridge_at_least_minus_0p03": true_auc - true_ridge
        >= -0.03,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_skinny64_sparse_profile_readiness_protocol_invalid"
        action = "repair E82 source, sparse graph, equivariance, or training protocol"
    elif not all(deterministic_checks.values()):
        status = "hold"
        decision = "innovation2_skinny64_sparse_profile_topology_not_attributed"
        action = "stop SLPO; the fair deterministic graph attribution did not pass"
    elif not all(readiness_checks.values()):
        status = "hold"
        decision = "innovation2_skinny64_sparse_profile_readiness_not_passed"
        action = "retain E82 labels and stop formal SLPO training"
    else:
        status = "pass"
        decision = "innovation2_skinny64_sparse_profile_readiness_passed"
        action = "pre-register the 30-epoch seed0 SLPO attribution matrix"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "deterministic_checks": deterministic_checks,
        "readiness_checks": readiness_checks,
        "metrics": {
            "rows": rows,
            "ridges": ridges,
            "true_sparse_ridge_minus_local": true_ridge - local_ridge,
            "true_sparse_ridge_minus_corrupted": true_ridge - corrupted_ridge,
            "true_minus_independent": true_auc - independent_auc,
            "true_minus_corrupted": true_auc - corrupted_auc,
            "true_minus_true_sparse_ridge": true_auc - true_ridge,
            "contract": contract,
        },
        "claim_scope": (
            "two-epoch local readiness for a SKINNY-64/64 r5 r4-only sparse "
            "linear-layer profile operator on strict 8-bit-cube unit-balance labels; "
            "no formal neural gain, high-round, cross-cipher checkpoint transfer, "
            "attack, remote-scale, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "formal_seed0": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(
    config: Skinny64SparseProfileReadinessConfig,
) -> dict[str, Any]:
    return asdict(config)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(binary_auc(labels.astype(np.float32), scores.astype(np.float64)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "Skinny64SparseProfileReadinessConfig",
    "adjudicate_sparse_profile_readiness",
    "conjugate_adjacency",
    "corrupted_skinny_linear_adjacency",
    "evaluate_sparse_ridges",
    "linear_graph_contract",
    "load_skinny_profile_sources",
    "make_skinny_sparse_model",
    "measure_sparse_operator_contract",
    "r4_only_sources",
    "serializable_config",
    "skinny_linear_adjacency",
    "train_sparse_profile_matrix",
    "validate_skinny_profile_sources",
]
