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

from blockcipher_nd.models.structure.spn.present_certificate_guided_pair_residual import (
    PresentCertificateGuidedPairResidual,
    PresentCgprSpec,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    RIDGE_LAMBDA,
    fit_train_only_ridge,
    load_sources,
    validate_sources,
)
from blockcipher_nd.tasks.innovation2.present_degree_spectrum_distillation import (
    build_degree_spectrum_targets,
)
from blockcipher_nd.training.metrics import binary_auc


E45_RUN_ID = "i2_present_r4_certificate_complexity_attribution_20260718"
E45_DECISION = "innovation2_present_mspn_route_ready"
E45_PREFIX_AUC = 0.6860815857512209
E49_RUN_ID = "i2_present_r4_degree_spectrum_distillation_readiness_seed0_20260718"
E49_DECISION = "innovation2_present_degree_spectrum_not_learned"


@dataclass(frozen=True)
class CgprReadinessConfig:
    run_id: str
    mode: str = "smoke"
    epochs: int = 2
    batch_size: int = 32
    hidden_dim: int = 16
    path_rank: int = 2
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    residual_bound: float = 0.25
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode != "smoke":
            raise ValueError("E50 only defines smoke mode")
        if (
            self.epochs != 2
            or self.batch_size != 32
            or self.hidden_dim != 16
            or self.path_rank != 2
            or self.dropout != 0.10
            or self.residual_bound != 0.25
            or self.seed != 0
            or self.device != "cpu"
        ):
            raise ValueError("E50 smoke protocol is frozen")


def load_e50_sources(
    atlas_root: Path,
    e44_root: Path,
    e45_root: Path,
    e49_root: Path,
) -> dict[str, Any]:
    sources = load_sources(atlas_root, e44_root)
    sources["e45_gate"] = _read_json(e45_root / "gate.json")
    sources["e49_gate"] = _read_json(e49_root / "gate.json")
    sources["e50_hashes"] = {
        f"e45_{name}": _sha256(e45_root / name)
        for name in ("gate.json", "results.jsonl")
    } | {
        f"e49_{name}": _sha256(e49_root / name)
        for name in ("gate.json", "results.jsonl")
    }
    return sources


def validate_e50_sources(sources: dict[str, Any]) -> dict[str, bool]:
    e45_sources = validate_sources(sources, strict=True)
    e45_gate = sources["e45_gate"]
    e49_gate = sources["e49_gate"]
    return {
        **{f"e45_source_{key}": value for key, value in e45_sources.items()},
        "e45_run_id_matches": e45_gate.get("run_id") == E45_RUN_ID,
        "e45_decision_matches": e45_gate.get("decision") == E45_DECISION,
        "e45_status_pass": e45_gate.get("status") == "pass",
        "e45_prefix_auc_matches": math.isclose(
            float(e45_gate["metrics"]["anf_prefix_validation_auc"]),
            E45_PREFIX_AUC,
            abs_tol=1e-12,
        ),
        "e49_run_id_matches": e49_gate.get("run_id") == E49_RUN_ID,
        "e49_decision_matches": e49_gate.get("decision") == E49_DECISION,
        "e49_status_hold": e49_gate.get("status") == "hold",
        "source_hashes_present": all(
            len(value) == 64 for value in sources["e50_hashes"].values()
        ),
    }


def build_prefix_ridge_bundle(data: dict[str, Any]) -> dict[str, Any]:
    features = build_degree_spectrum_targets(data, "true").reshape(-1, 39)
    features = features.astype(np.float64)
    labels = np.asarray([row["label"] for row in data["rows"]], dtype=np.float64)
    split = np.asarray([row["split"] for row in data["rows"]])
    train = split == "train"
    validation = split == "validation"
    fitted = fit_train_only_ridge(
        features[train], labels[train], features[validation], RIDGE_LAMBDA
    )
    scores = np.zeros(len(features), dtype=np.float64)
    scores[train] = fitted["train_scores"]
    scores[validation] = fitted["validation_scores"]
    return {
        "features": features,
        "labels": labels,
        "train_mask": train,
        "validation_mask": validation,
        "mean": fitted["mean"],
        "scale": fitted["scale"],
        "weights": fitted["weights"],
        "scores": scores,
        "train_auc": _safe_auc(labels[train], scores[train]),
        "validation_auc": _safe_auc(labels[validation], scores[validation]),
        "train_standardization_only": True,
    }


def measure_cgpr_contract(
    config: CgprReadinessConfig,
    data: dict[str, Any],
    ridge: dict[str, Any],
) -> dict[str, Any]:
    _seed_everything(config.seed)
    prefix = _make_model(config, data, ridge, "prefix", "true", dropout=0.0)
    _seed_everything(config.seed)
    true = _make_model(config, data, ridge, "pair", "true", dropout=0.0)
    _seed_everything(config.seed)
    corrupted = _make_model(config, data, ridge, "pair", "corrupted", dropout=0.0)
    _copy_parameters(true, corrupted)
    batch_indices = np.arange(8, dtype=np.int64)
    arrays = _row_arrays(data, batch_indices, ridge["features"][batch_indices])
    tensors = [torch.from_numpy(array) for array in arrays[:-1]]
    labels = torch.from_numpy(arrays[-1])
    prefix.eval()
    true.train()
    corrupted.eval()
    base = true.base_score(tensors[-1])
    true_logits = true(*tensors)
    loss = nn.BCEWithLogitsLoss()(true_logits, labels)
    loss.backward()
    true.eval()
    with torch.no_grad():
        prefix_logits = prefix(*tensors)
        true_logits = true(*tensors)
        corrupted_logits = corrupted(*tensors)
        true_embedding = true.pair_embedding(*tensors[:4])
        corrupted_embedding = corrupted.pair_embedding(*tensors[:4])
    parameter_counts = {
        "prefix": sum(parameter.numel() for parameter in prefix.parameters()),
        "pair_true": sum(parameter.numel() for parameter in true.parameters()),
        "pair_corrupted": sum(
            parameter.numel() for parameter in corrupted.parameters()
        ),
    }
    all_buffer_names = {
        name
        for model in (prefix, true, corrupted)
        for name, _ in model.named_buffers()
    }
    return {
        "prefix_shape": list(ridge["features"].shape),
        "ridge_validation_auc": ridge["validation_auc"],
        "train_standardization_only": ridge["train_standardization_only"],
        "parameter_counts": parameter_counts,
        "parameter_relative_spread": (
            max(parameter_counts.values()) - min(parameter_counts.values())
        )
        / max(parameter_counts.values()),
        "zero_residual_prefix_max_abs_error": float(
            torch.max(torch.abs(prefix_logits - base)).detach()
        ),
        "zero_residual_true_max_abs_error": float(
            torch.max(torch.abs(true_logits - base)).detach()
        ),
        "zero_residual_corrupted_max_abs_error": float(
            torch.max(torch.abs(corrupted_logits - base)).detach()
        ),
        "true_corrupted_pair_embedding_max_abs_difference": float(
            torch.max(torch.abs(true_embedding - corrupted_embedding)).detach()
        ),
        "ridge_buffers_require_grad_false": all(
            not value.requires_grad
            for model in (prefix, true, corrupted)
            for name in ("ridge_mean", "ridge_scale", "ridge_weights")
            for value in (getattr(model, name),)
        ),
        "logits_finite": all(
            bool(torch.isfinite(values).all())
            for values in (prefix_logits, true_logits, corrupted_logits)
        ),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in true.parameters()
        ),
        "forbidden_buffers_absent": not any(
            token in name
            for name in all_buffer_names
            for token in ("full_cube", "certificate", "witness", "parity")
        ),
    }


def train_cgpr_matrix(
    config: CgprReadinessConfig,
    data: dict[str, Any],
    ridge: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_cgpr_readiness",
            "row_id": "e45_anf_prefix_ridge_anchor",
            "model_name": "anf_prefix_ridge",
            "residual_mode": "off",
            "topology_mode": "none",
            "seed": 0,
            "best_epoch": 0,
            "train_auc": ridge["train_auc"],
            "validation_auc": ridge["validation_auc"],
            "parameter_count": 0,
            "ridge_weight_max_delta": 0.0,
            "training_performed": False,
        }
    ]
    history: list[dict[str, Any]] = []
    for residual_mode, topology_mode in (
        ("prefix", "true"),
        ("pair", "true"),
        ("pair", "corrupted"),
    ):
        output = train_cgpr_row(
            config,
            data,
            ridge,
            residual_mode=residual_mode,
            topology_mode=topology_mode,
        )
        rows.append(output["result"])
        history.extend(output["history"])
    return {"rows": rows, "history": history}


def train_cgpr_row(
    config: CgprReadinessConfig,
    data: dict[str, Any],
    ridge: dict[str, Any],
    *,
    residual_mode: str,
    topology_mode: str,
) -> dict[str, Any]:
    if (residual_mode, topology_mode) not in {
        ("prefix", "true"),
        ("pair", "true"),
        ("pair", "corrupted"),
    }:
        raise ValueError("unsupported E50 residual/topology row")
    _seed_everything(config.seed)
    device = torch.device(config.device)
    model = _make_model(config, data, ridge, residual_mode, topology_mode).to(device)
    train_indices = np.flatnonzero(ridge["train_mask"])
    validation_indices = np.flatnonzero(ridge["validation_mask"])
    train_arrays = _row_arrays(data, train_indices, ridge["features"][train_indices])
    validation_arrays = _row_arrays(
        data, validation_indices, ridge["features"][validation_indices]
    )
    generator = torch.Generator().manual_seed(50100 + config.seed)
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
    ridge_snapshot = model.ridge_weights.detach().clone()
    best_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    row_id = (
        "cgpr_prefix_only_seed0"
        if residual_mode == "prefix"
        else f"cgpr_pair_{topology_mode}_seed0"
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            inputs = [tensor.to(device) for tensor in batch[:-1]]
            labels = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total_rows += len(labels)
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
        raise RuntimeError("E50 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_metrics = _evaluate(model, train_arrays, device, config.batch_size)
    validation_metrics = _evaluate(model, validation_arrays, device, config.batch_size)
    return {
        "result": {
            "run_id": config.run_id,
            "task": "innovation2_present_cgpr_readiness",
            "row_id": row_id,
            "model_name": "present_certificate_guided_pair_residual",
            "residual_mode": residual_mode,
            "topology_mode": topology_mode,
            "seed": config.seed,
            "best_epoch": best_epoch,
            "train_auc": train_metrics["auc"],
            "validation_auc": validation_metrics["auc"],
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "ridge_weight_max_delta": float(
                torch.max(torch.abs(model.ridge_weights - ridge_snapshot)).detach()
            ),
            "training_performed": True,
        },
        "history": history,
    }


def adjudicate_e50(
    config: CgprReadinessConfig,
    source_checks: dict[str, bool],
    contract: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    trained = [row for row in matrix["rows"] if row["training_performed"]]
    prefix = next(row for row in trained if row["residual_mode"] == "prefix")
    true = next(
        row
        for row in trained
        if row["residual_mode"] == "pair" and row["topology_mode"] == "true"
    )
    corrupted = next(
        row for row in trained if row["topology_mode"] == "corrupted"
    )
    aucs = [float(row["validation_auc"]) for row in trained]
    protocol_checks = {
        **source_checks,
        "expected_four_rows_present": len(matrix["rows"]) == 4,
        "three_residual_rows_present": len(trained) == 3,
        "ridge_validation_auc_reproduced": math.isclose(
            contract["ridge_validation_auc"], E45_PREFIX_AUC, abs_tol=1e-12
        ),
        "prefix_shape_is_1036x39": contract["prefix_shape"] == [1036, 39],
        "train_standardization_only": contract["train_standardization_only"],
        "zero_residual_matches_ridge": max(
            contract["zero_residual_prefix_max_abs_error"],
            contract["zero_residual_true_max_abs_error"],
            contract["zero_residual_corrupted_max_abs_error"],
        )
        <= 1e-7,
        "ridge_buffers_frozen": contract["ridge_buffers_require_grad_false"]
        and all(float(row["ridge_weight_max_delta"]) == 0.0 for row in trained),
        "true_corrupted_pair_embedding_delta_at_least_1e_5": contract[
            "true_corrupted_pair_embedding_max_abs_difference"
        ]
        >= 1e-5,
        "residual_parameter_relative_spread_at_most_0p01": contract[
            "parameter_relative_spread"
        ]
        <= 0.01,
        "forward_loss_gradients_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "forbidden_buffers_absent": contract["forbidden_buffers_absent"],
        "all_trained_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in trained
            for key in ("train_auc", "validation_auc", "train_loss", "validation_loss")
        ),
        "all_rows_completed_two_epochs": all(
            sum(history["row_id"] == row["row_id"] for history in matrix["history"])
            == config.epochs
            for row in trained
        ),
        "all_validation_auc_in_0p35_0p80": all(0.35 <= auc <= 0.80 for auc in aucs),
    }
    if all(protocol_checks.values()):
        status = "pass"
        decision = "innovation2_present_cgpr_readiness_passed"
        action = "prepare E51 30-epoch seed0 CGPR residual attribution plan"
    else:
        status = "fail"
        decision = "innovation2_present_cgpr_readiness_failed"
        action = "repair source, ridge, zero-equivalence, topology, parameter, or training contract"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "metrics": {
            "ridge_validation_auc": contract["ridge_validation_auc"],
            "prefix_residual_smoke_auc": float(prefix["validation_auc"]),
            "true_pair_residual_smoke_auc": float(true["validation_auc"]),
            "corrupted_pair_residual_smoke_auc": float(
                corrupted["validation_auc"]
            ),
            "parameter_counts": contract["parameter_counts"],
            "parameter_relative_spread": contract["parameter_relative_spread"],
            "zero_residual_max_abs_error": max(
                contract["zero_residual_prefix_max_abs_error"],
                contract["zero_residual_true_max_abs_error"],
                contract["zero_residual_corrupted_max_abs_error"],
            ),
            "true_corrupted_pair_embedding_max_abs_difference": contract[
                "true_corrupted_pair_embedding_max_abs_difference"
            ],
        },
        "claim_scope": (
            "two-epoch local implementation readiness for a certificate-guided "
            "pair-state residual on the E43 real PRESENT-80 r4 strict benchmark; "
            "not an effective neural result, high-round distinguisher, new attack, "
            "remote-scale result, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "formal_seed0": status == "pass",
            "seed1": False,
            "remote_scale": False,
        },
    }


def serializable_config(config: CgprReadinessConfig) -> dict[str, Any]:
    return asdict(config)


def _make_model(
    config: CgprReadinessConfig,
    data: dict[str, Any],
    ridge: dict[str, Any],
    residual_mode: str,
    topology_mode: str,
    *,
    dropout: float | None = None,
) -> PresentCertificateGuidedPairResidual:
    return PresentCertificateGuidedPairResidual(
        PresentCgprSpec(
            residual_mode=residual_mode,
            topology_mode=topology_mode,
            hidden_dim=config.hidden_dim,
            path_rank=config.path_rank,
            dropout=config.dropout if dropout is None else dropout,
            residual_bound=config.residual_bound,
        ),
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
        ridge_mean=ridge["mean"],
        ridge_scale=ridge["scale"],
        ridge_weights=ridge["weights"],
    )


def _row_arrays(
    data: dict[str, Any], indices: np.ndarray, features: np.ndarray
) -> tuple[np.ndarray, ...]:
    rows = [data["rows"][int(index)] for index in indices]
    return (
        np.zeros(len(rows), dtype=np.int64),
        np.zeros(len(rows), dtype=np.int64),
        np.asarray([row["structure_index"] for row in rows], dtype=np.int64),
        np.asarray([row["mask_index"] for row in rows], dtype=np.int64),
        np.asarray(features, dtype=np.float32),
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


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(binary_auc(labels.astype(np.float32), scores.astype(np.float64)))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _copy_parameters(source: nn.Module, target: nn.Module) -> None:
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    if source_parameters.keys() != target_parameters.keys():
        raise ValueError("CGPR parameter names do not match")
    with torch.no_grad():
        for name, parameter in target_parameters.items():
            parameter.copy_(source_parameters[name])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
