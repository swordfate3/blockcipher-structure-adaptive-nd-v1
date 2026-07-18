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

from blockcipher_nd.models.structure.spn.ridge_guided_sparse_profile_residual import (
    RidgeGuidedSparseProfileResidual,
    RidgeGuidedSparseResidualSpec,
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
from blockcipher_nd.tasks.innovation2.skinny64_r4_only_sparse_profile_operator_readiness import (
    SOURCE_DECISION,
    SOURCE_RUN_ID,
    _edge_feature_matrix,
    conjugate_adjacency,
    corrupted_skinny_linear_adjacency,
    load_skinny_profile_sources,
    r4_only_sources,
    skinny_linear_adjacency,
    validate_skinny_profile_sources,
)
from blockcipher_nd.training.metrics import binary_auc


E83_RUN_ID = "i2_skinny64_r5_r4_only_sparse_profile_operator_readiness_seed0_20260719"
E83_DECISION = "innovation2_skinny64_sparse_profile_readiness_not_passed"
E83_TRUE_RIDGE_AUC = 0.8620446578265508
RELATION_MODES = ("independent", "true", "corrupted")
RIDGE_LAMBDA = 1e-3
EXPECTED_PARAMETER_COUNT = 4_795


@dataclass(frozen=True)
class Skinny64TrueRidgeResidualConfig:
    run_id: str
    epochs: int = 2
    batch_size: int = 8
    hidden_dim: int = 32
    steps: int = 2
    seed: int = 0
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    residual_bound: float = 0.25
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
            or self.residual_bound != 0.25
            or self.device != "cpu"
        ):
            raise ValueError("E84 readiness protocol is frozen")


def load_e84_sources(profile_root: Path, e83_root: Path) -> dict[str, Any]:
    sources = load_skinny_profile_sources(profile_root)
    gate_path = e83_root / "gate.json"
    results_path = e83_root / "results.jsonl"
    sources["e83_gate"] = json.loads(gate_path.read_text(encoding="utf-8"))
    sources["e83_hashes"] = {
        "gate.json": _sha256(gate_path),
        "results.jsonl": _sha256(results_path),
    }
    return sources


def validate_e84_sources(sources: dict[str, Any]) -> dict[str, bool]:
    profile_checks = validate_skinny_profile_sources(sources)
    e83 = sources["e83_gate"]
    return {
        **{f"e82_{name}": value for name, value in profile_checks.items()},
        "e82_source_run_id_constant_matches": SOURCE_RUN_ID
        == "i2_skinny64_r5_unit_balance_profile_transition_20260719",
        "e82_source_decision_constant_matches": SOURCE_DECISION
        == "innovation2_skinny64_r5_unit_balance_profile_transition_ready",
        "e83_run_id_matches": e83.get("run_id") == E83_RUN_ID,
        "e83_status_is_hold": e83.get("status") == "hold",
        "e83_decision_matches": e83.get("decision") == E83_DECISION,
        "e83_protocol_checks_pass": bool(e83.get("protocol_checks"))
        and all(e83["protocol_checks"].values()),
        "e83_deterministic_checks_pass": bool(e83.get("deterministic_checks"))
        and all(e83["deterministic_checks"].values()),
        "e83_true_ridge_auc_matches": math.isclose(
            float(
                e83.get("metrics", {})
                .get("ridges", {})
                .get("true_sparse39", {})
                .get("validation_auc", float("nan"))
            ),
            E83_TRUE_RIDGE_AUC,
            abs_tol=1e-12,
        ),
        "e83_hashes_present": all(
            len(value) == 64 for value in sources["e83_hashes"].values()
        ),
    }


def build_true_ridge_bundle(sources: dict[str, Any]) -> dict[str, Any]:
    rows = sources["matched_rows"]
    labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
    train = np.asarray([row["split"] == "train" for row in rows])
    validation = ~train
    adjacency = skinny_linear_adjacency()
    features = np.asarray(sources["prefix_features"], dtype=np.float64)
    matrix = _edge_feature_matrix(rows, features, adjacency)
    fitted = fit_train_only_ridge(
        matrix[train], labels[train], matrix[validation], RIDGE_LAMBDA
    )
    return {
        "mean": fitted["mean"],
        "scale": fitted["scale"],
        "weights": fitted["weights"],
        "train_scores": fitted["train_scores"],
        "validation_scores": fitted["validation_scores"],
        "train_auc": _safe_auc(labels[train], fitted["train_scores"]),
        "validation_auc": _safe_auc(
            labels[validation], fitted["validation_scores"]
        ),
        "train_standardization_only": True,
        "ridge_lambda": RIDGE_LAMBDA,
    }


def make_true_ridge_residual_model(
    config: Skinny64TrueRidgeResidualConfig,
    ridge: dict[str, Any],
    mode: str,
    *,
    base_adjacency: np.ndarray | None = None,
    residual_adjacency: np.ndarray | None = None,
    dropout: float | None = None,
) -> RidgeGuidedSparseProfileResidual:
    if mode not in RELATION_MODES:
        raise ValueError(f"unsupported E84 mode: {mode}")
    true = skinny_linear_adjacency()
    residual = (
        np.asarray(residual_adjacency, dtype=np.float32)
        if residual_adjacency is not None
        else corrupted_skinny_linear_adjacency(true)
        if mode == "corrupted"
        else true
    )
    return RidgeGuidedSparseProfileResidual(
        RidgeGuidedSparseResidualSpec(
            input_dim=13,
            hidden_dim=config.hidden_dim,
            steps=config.steps,
            dropout=config.dropout if dropout is None else dropout,
            relation_mode=mode,
            residual_bound=config.residual_bound,
        ),
        base_adjacency=true if base_adjacency is None else base_adjacency,
        residual_adjacency=residual,
        ridge_mean=ridge["mean"],
        ridge_scale=ridge["scale"],
        ridge_weights=ridge["weights"],
    )


def measure_true_ridge_residual_contract(
    config: Skinny64TrueRidgeResidualConfig,
    sources: dict[str, Any],
    ridge: dict[str, Any],
) -> dict[str, Any]:
    models = {
        mode: make_true_ridge_residual_model(config, ridge, mode, dropout=0.0)
        for mode in RELATION_MODES
    }
    candidate = models["true"]
    _copy_parameters(candidate, models["independent"])
    _copy_parameters(candidate, models["corrupted"])
    indices = sorted({row["structure_index"] for row in sources["matched_rows"]})[:4]
    features, targets, observed = _batch_tensors(sources, indices, "cpu")
    base = candidate.base_score(features)
    logits = candidate(features)
    loss = masked_binary_cross_entropy(logits, targets, observed)
    loss.backward()
    candidate.eval()
    models["corrupted"].eval()
    with torch.no_grad():
        zero_errors = {
            mode: float(
                torch.max(torch.abs(model(features) - model.base_score(features)))
            )
            for mode, model in models.items()
        }
        true_embedding = candidate.residual_embedding(features)
        corrupted_embedding = models["corrupted"].residual_embedding(features)

    permutation = _cell_permutation()
    true_adjacency = skinny_linear_adjacency()
    relabeled_adjacency = conjugate_adjacency(true_adjacency, permutation)
    relabeled = make_true_ridge_residual_model(
        config,
        ridge,
        "true",
        base_adjacency=relabeled_adjacency,
        residual_adjacency=relabeled_adjacency,
        dropout=0.0,
    )
    _copy_parameters(candidate, relabeled)
    permuted_features = torch.empty_like(features)
    permuted_features[:, permutation] = features
    with torch.no_grad():
        permuted = relabeled(permuted_features)
    expected = torch.empty_like(logits.detach())
    expected[:, permutation] = logits.detach()
    parameter_counts = {
        mode: sum(parameter.numel() for parameter in model.parameters())
        for mode, model in models.items()
    }
    forbidden = ("certificate", "witness", "parity", "full_cube", "label")
    return {
        "output_shape": list(logits.shape),
        "base_shape": list(base.shape),
        "ridge_validation_auc": ridge["validation_auc"],
        "train_standardization_only": ridge["train_standardization_only"],
        "zero_residual_max_abs_errors": zero_errors,
        "parameter_counts": parameter_counts,
        "parameter_counts_match": len(set(parameter_counts.values())) == 1,
        "true_corrupted_embedding_max_abs_difference": float(
            torch.max(torch.abs(true_embedding - corrupted_embedding))
        ),
        "cell_relabel_max_abs_error": float(
            torch.max(torch.abs(permuted - expected)).detach()
        ),
        "ridge_buffers_require_grad_false": all(
            not buffer.requires_grad
            for model in models.values()
            for buffer in (model.ridge_mean, model.ridge_scale, model.ridge_weights)
        ),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in candidate.parameters()
        ),
        "forbidden_buffers_absent": not any(
            token in name for model in models.values() for name, _ in model.named_buffers() for token in forbidden
        ),
        "residual_bound": config.residual_bound,
    }


def train_true_ridge_residual_matrix(
    config: Skinny64TrueRidgeResidualConfig,
    sources: dict[str, Any],
    ridge: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    rows = sources["matched_rows"]
    train_indices = sorted(
        {row["structure_index"] for row in rows if row["split"] == "train"}
    )
    validation_indices = sorted(
        {row["structure_index"] for row in rows if row["split"] == "validation"}
    )
    results: list[dict[str, Any]] = [
        {
            "run_id": config.run_id,
            "task": "innovation2_skinny64_true_ridge_sparse_residual_readiness",
            "row_id": "skinny_true_sparse39_ridge_anchor",
            "relation_mode": "ridge_only",
            "seed": config.seed,
            "best_epoch": 0,
            "epochs_completed": 0,
            "parameter_count": 0,
            "train_auc": ridge["train_auc"],
            "validation_auc": ridge["validation_auc"],
            "training_performed": False,
            "ridge_weight_max_delta": 0.0,
        }
    ]
    history_rows: list[dict[str, Any]] = []
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for mode in RELATION_MODES:
        _seed_everything(config.seed)
        model = make_true_ridge_residual_model(config, ridge, mode).to(config.device)
        ridge_snapshot = model.ridge_weights.detach().clone()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        initial_train = _evaluate(model, sources, train_indices, config)
        initial_validation = _evaluate(model, sources, validation_indices, config)
        best_auc = initial_validation["auc"]
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
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
            history_rows.append(
                {
                    "row_id": f"skinny_true_ridge_residual_{mode}_seed{config.seed}",
                    "relation_mode": mode,
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{
                        f"validation_{key}": value
                        for key, value in validation_metrics.items()
                    },
                }
            )
            if validation_metrics["auc"] > best_auc:
                best_auc = validation_metrics["auc"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        train_metrics = _evaluate(model, sources, train_indices, config)
        validation_metrics = _evaluate(model, sources, validation_indices, config)
        row_id = f"skinny_true_ridge_residual_{mode}_seed{config.seed}"
        torch.save(best_state, checkpoints / f"{row_id}.pt")
        results.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_skinny64_true_ridge_sparse_residual_readiness",
                "row_id": row_id,
                "relation_mode": mode,
                "seed": config.seed,
                "best_epoch": best_epoch,
                "epochs_completed": config.epochs,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
                "training_performed": True,
                "ridge_weight_max_delta": float(
                    torch.max(torch.abs(model.ridge_weights - ridge_snapshot))
                ),
                "epoch0_train_auc": initial_train["auc"],
                "epoch0_validation_auc": initial_validation["auc"],
            }
        )
    return {"rows": results, "history": history_rows}


def _evaluate(
    model: RidgeGuidedSparseProfileResidual,
    sources: dict[str, Any],
    indices: list[int],
    config: Skinny64TrueRidgeResidualConfig,
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


def adjudicate_true_ridge_residual(
    config: Skinny64TrueRidgeResidualConfig,
    source_checks: dict[str, bool],
    contract: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    anchor = next(row for row in matrix["rows"] if not row["training_performed"])
    trained = {
        row["relation_mode"]: row
        for row in matrix["rows"]
        if row["training_performed"]
    }
    true_auc = float(trained.get("true", {}).get("validation_auc", 0.0))
    independent_auc = float(
        trained.get("independent", {}).get("validation_auc", 0.0)
    )
    corrupted_auc = float(trained.get("corrupted", {}).get("validation_auc", 0.0))
    ridge_auc = float(anchor["validation_auc"])
    protocol_checks = {
        **source_checks,
        "four_rows_present": len(matrix["rows"]) == 4,
        "three_residual_rows_present": set(trained) == set(RELATION_MODES),
        "ridge_validation_auc_reproduced": math.isclose(
            ridge_auc, E83_TRUE_RIDGE_AUC, abs_tol=1e-12
        ),
        "contract_ridge_auc_reproduced": math.isclose(
            float(contract["ridge_validation_auc"]), E83_TRUE_RIDGE_AUC, abs_tol=1e-12
        ),
        "train_standardization_only": contract["train_standardization_only"],
        "zero_residual_matches_ridge": max(
            contract["zero_residual_max_abs_errors"].values()
        )
        <= 1e-7,
        "ridge_buffers_frozen": contract["ridge_buffers_require_grad_false"]
        and all(float(row["ridge_weight_max_delta"]) == 0.0 for row in trained.values()),
        "parameter_count_is_4795": set(contract["parameter_counts"].values())
        == {EXPECTED_PARAMETER_COUNT},
        "parameter_counts_match": contract["parameter_counts_match"],
        "true_corrupted_embeddings_differ": contract[
            "true_corrupted_embedding_max_abs_difference"
        ]
        >= 1e-5,
        "cell_relabel_equivariant": contract["cell_relabel_max_abs_error"] <= 1e-6,
        "residual_bound_is_0p25": contract["residual_bound"] == 0.25,
        "ridge_buffers_have_no_forbidden_state": contract["forbidden_buffers_absent"],
        "contract_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "all_rows_completed_two_epochs": all(
            row["epochs_completed"] == config.epochs for row in trained.values()
        ),
        "all_trained_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in trained.values()
            for key in (
                "train_auc",
                "train_accuracy",
                "train_loss",
                "validation_auc",
                "validation_accuracy",
                "validation_loss",
            )
        ),
    }
    readiness_checks = {
        "true_minus_ridge_at_least_0p02": true_auc - ridge_auc >= 0.02,
        "true_minus_independent_at_least_0p03": true_auc - independent_auc >= 0.03,
        "true_minus_corrupted_at_least_0p03": true_auc - corrupted_auc >= 0.03,
        "true_train_validation_gap_at_most_0p15": float(trained["true"]["train_auc"])
        - true_auc
        <= 0.15,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_skinny64_true_ridge_residual_protocol_invalid"
        action = "repair source, ridge, zero residual, frozen buffers, graph, or training"
    elif not all(readiness_checks.values()):
        status = "hold"
        decision = "innovation2_skinny64_true_ridge_residual_not_ready"
        action = "close SKINNY neural search at the E82 label plus true-ridge baseline"
    else:
        status = "pass"
        decision = "innovation2_skinny64_true_ridge_residual_readiness_passed"
        action = "pre-register the 30-epoch seed0 residual attribution matrix"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "readiness_checks": readiness_checks,
        "metrics": {
            "rows": matrix["rows"],
            "ridge_validation_auc": ridge_auc,
            "true_minus_ridge": true_auc - ridge_auc,
            "true_minus_independent": true_auc - independent_auc,
            "true_minus_corrupted": true_auc - corrupted_auc,
            "contract": contract,
        },
        "claim_scope": (
            "two-epoch local readiness for a frozen true-topology-ridge plus "
            "bounded sparse neural residual on SKINNY-64/64 r5 strict labels; "
            "no formal neural gain, high-round attack, cross-cipher transfer, "
            "remote-scale, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "formal_seed0": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: Skinny64TrueRidgeResidualConfig) -> dict[str, Any]:
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
    "Skinny64TrueRidgeResidualConfig",
    "adjudicate_true_ridge_residual",
    "build_true_ridge_bundle",
    "load_e84_sources",
    "make_true_ridge_residual_model",
    "measure_true_ridge_residual_contract",
    "serializable_config",
    "train_true_ridge_residual_matrix",
    "validate_e84_sources",
]
