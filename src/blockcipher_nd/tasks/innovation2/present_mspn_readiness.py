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
    PresentMonomialSupportPropagationNetwork,
    PresentMspnSpec,
)
from blockcipher_nd.tasks.innovation2.present_pair_state_neural_attribution import (
    load_e43_source,
    validate_e43_source,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    reconstruct_present_sbox_from_anf,
)
from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX
from blockcipher_nd.training.metrics import binary_auc


E44_RUN_ID = "i2_present_r4_pair_state_neural_attribution_seed0_20260718"
E44_DECISION = "innovation2_present_pair_state_candidate_not_ready"
E45_RUN_ID = "i2_present_r4_certificate_complexity_attribution_20260718"
E45_DECISION = "innovation2_present_mspn_route_ready"
E44_TRIANGLE_AUC = 0.5619793162884229
E45_PREFIX_AUC = 0.6860815857512209
E44_PARAMETER_ANCHOR = 10725


@dataclass(frozen=True)
class MspnReadinessConfig:
    run_id: str
    mode: str = "smoke"
    epochs: int = 2
    batch_size: int = 32
    hidden_dim: int = 32
    degree_channels: int = 9
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode != "smoke":
            raise ValueError("E46 only defines smoke mode")
        if min(self.epochs, self.batch_size, self.hidden_dim, self.degree_channels) <= 0:
            raise ValueError("model and training dimensions must be positive")
        if (
            self.epochs != 2
            or self.batch_size != 32
            or self.hidden_dim != 32
            or self.degree_channels != 9
            or self.dropout != 0.10
            or self.seed != 0
            or self.device != "cpu"
        ):
            raise ValueError("E46 smoke protocol is frozen")


def load_readiness_sources(
    atlas_root: Path, e44_root: Path, e45_root: Path
) -> dict[str, Any]:
    atlas = load_e43_source(atlas_root)
    e44_gate = _read_json(e44_root / "gate.json")
    e44_results = _read_jsonl(e44_root / "results.jsonl")
    e45_gate = _read_json(e45_root / "gate.json")
    e45_results = _read_jsonl(e45_root / "results.jsonl")
    return {
        "atlas": atlas,
        "e44_gate": e44_gate,
        "e44_results": e44_results,
        "e45_gate": e45_gate,
        "e45_results": e45_results,
        "source_hashes": {
            "e44_gate.json": _sha256(e44_root / "gate.json"),
            "e44_results.jsonl": _sha256(e44_root / "results.jsonl"),
            "e45_gate.json": _sha256(e45_root / "gate.json"),
            "e45_results.jsonl": _sha256(e45_root / "results.jsonl"),
        },
    }


def validate_readiness_sources(sources: dict[str, Any]) -> dict[str, bool]:
    atlas_checks = validate_e43_source(sources["atlas"], strict=True)
    triangle = next(
        (
            row
            for row in sources["e44_results"]
            if row.get("row_id") == "pair_triangle_true_seed0"
        ),
        None,
    )
    prefix = next(
        (
            row
            for row in sources["e45_results"]
            if row.get("feature_family") == "anf_prefix"
        ),
        None,
    )
    return {
        **{f"atlas_{key}": value for key, value in atlas_checks.items()},
        "e44_run_id_matches": sources["e44_gate"].get("run_id") == E44_RUN_ID,
        "e44_decision_matches": sources["e44_gate"].get("decision") == E44_DECISION,
        "e45_run_id_matches": sources["e45_gate"].get("run_id") == E45_RUN_ID,
        "e45_decision_matches": sources["e45_gate"].get("decision") == E45_DECISION,
        "e45_status_pass": sources["e45_gate"].get("status") == "pass",
        "e44_triangle_anchor_matches": triangle is not None
        and math.isclose(
            float(triangle["validation_auc"]), E44_TRIANGLE_AUC, abs_tol=1e-12
        ),
        "e45_prefix_anchor_matches": prefix is not None
        and math.isclose(
            float(prefix["validation_auc"]), E45_PREFIX_AUC, abs_tol=1e-12
        ),
        "source_hashes_present": all(
            len(value) == 64 for value in sources["source_hashes"].values()
        ),
    }


def measure_mspn_contract(
    config: MspnReadinessConfig, data: dict[str, Any]
) -> dict[str, Any]:
    true = _make_model(config, data, "true", dropout=0.0)
    corrupted = _make_model(config, data, "corrupted", dropout=0.0)
    _copy_parameters(true, corrupted)
    relabeled_data, _ = _cell_relabel_data(data)
    relabeled = _make_model(config, relabeled_data, "true", dropout=0.0)
    _copy_parameters(true, relabeled)
    rows = data["rows"][:8]
    arrays = _rows_to_arrays(rows)
    tensors = [torch.from_numpy(array) for array in arrays[:-1]]
    target = torch.from_numpy(arrays[-1])
    true.train()
    logits = true(*tensors)
    loss = nn.BCEWithLogitsLoss()(logits, target)
    loss.backward()
    true.eval()
    corrupted.eval()
    relabeled.eval()
    with torch.no_grad():
        true_logits = true(*tensors)
        corrupted_logits = corrupted(*tensors)
        relabeled_logits = relabeled(*tensors)
        initial, _, _ = true.build_initial_state(tensors[1], tensors[2])
    parameter_count = sum(parameter.numel() for parameter in true.parameters())
    buffer_names = {name for name, _ in true.named_buffers()}
    return {
        "initial_state_shape": list(initial.shape),
        "shared_step_count": 1,
        "execution_rounds": true.spec.rounds,
        "parameter_count": parameter_count,
        "parameter_ratio_to_e44": parameter_count / E44_PARAMETER_ANCHOR,
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in true.parameters()
        ),
        "true_corrupted_max_abs_logit_difference": float(
            torch.max(torch.abs(true_logits - corrupted_logits))
        ),
        "cell_relabeling_max_abs_logit_error": float(
            torch.max(torch.abs(true_logits - relabeled_logits))
        ),
        "sbox_anf_reconstructs": all(
            reconstruct_present_sbox_from_anf(value) == PRESENT_SBOX[value]
            for value in range(16)
        ),
        "allowed_buffer_names_only": buffer_names
        == {"players", "structure_active_bits", "output_mask_bits"},
        "precomputed_certificate_feature_buffers_absent": not any(
            token in name
            for name in buffer_names
            for token in ("support", "full_cube", "certificate", "prefix", "witness")
        ),
    }


def train_readiness_matrix(
    config: MspnReadinessConfig, data: dict[str, Any]
) -> dict[str, Any]:
    rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_mspn_readiness",
            "row_id": "e45_anf_prefix_ridge_anchor",
            "model_name": "anf_prefix_ridge",
            "topology_mode": "deterministic_prefix",
            "label_mode": "true",
            "seed": 0,
            "best_epoch": 0,
            "train_auc": None,
            "validation_auc": E45_PREFIX_AUC,
            "parameter_count": 0,
            "training_performed": False,
        },
        {
            "run_id": config.run_id,
            "task": "innovation2_present_mspn_readiness",
            "row_id": "e44_triangle_anchor",
            "model_name": "present_pair_triangle",
            "topology_mode": "true",
            "label_mode": "true",
            "seed": 0,
            "best_epoch": 27,
            "train_auc": 0.6113625,
            "validation_auc": E44_TRIANGLE_AUC,
            "parameter_count": E44_PARAMETER_ANCHOR,
            "training_performed": False,
        },
    ]
    history: list[dict[str, Any]] = []
    for topology_mode, label_mode in (
        ("true", "true"),
        ("corrupted", "true"),
        ("true", "shuffled"),
    ):
        output = train_mspn_row(
            config, data, topology_mode=topology_mode, label_mode=label_mode
        )
        rows.append(output["result"])
        history.extend(output["history"])
    return {"rows": rows, "history": history}


def train_mspn_row(
    config: MspnReadinessConfig,
    data: dict[str, Any],
    *,
    topology_mode: str,
    label_mode: str,
) -> dict[str, Any]:
    _seed_everything(config.seed)
    device = torch.device(config.device)
    model = _make_model(config, data, topology_mode).to(device)
    train_rows = [row for row in data["rows"] if row["split"] == "train"]
    validation_rows = [row for row in data["rows"] if row["split"] == "validation"]
    train_arrays = list(_rows_to_arrays(train_rows))
    validation_arrays = _rows_to_arrays(validation_rows)
    if label_mode == "shuffled":
        rng = np.random.default_rng(46000 + config.seed)
        train_arrays[-1] = train_arrays[-1][rng.permutation(len(train_arrays[-1]))]
    train_tuple = tuple(train_arrays)
    generator = torch.Generator().manual_seed(46100 + config.seed)
    loader = DataLoader(
        _tensor_dataset(train_tuple),
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
    row_suffix = "label_shuffle" if label_mode == "shuffled" else topology_mode
    row_id = f"mspn_{row_suffix}_seed{config.seed}"
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
        raise RuntimeError("MSPN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_metrics = _evaluate(model, train_tuple, device, config.batch_size)
    validation_metrics = _evaluate(model, validation_arrays, device, config.batch_size)
    return {
        "result": {
            "run_id": config.run_id,
            "task": "innovation2_present_mspn_readiness",
            "row_id": row_id,
            "model_name": "present_mspn",
            "topology_mode": topology_mode,
            "label_mode": label_mode,
            "seed": config.seed,
            "best_epoch": best_epoch,
            "train_auc": train_metrics["auc"],
            "validation_auc": validation_metrics["auc"],
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "training_performed": True,
        },
        "history": history,
    }


def adjudicate_e46(
    config: MspnReadinessConfig,
    source_checks: dict[str, bool],
    contract: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    trained = [row for row in matrix["rows"] if row["training_performed"]]
    shuffled = next(row for row in trained if row["label_mode"] == "shuffled")
    protocol_checks = {
        **source_checks,
        "expected_five_rows_present": len(matrix["rows"]) == 5,
        "three_mspn_rows_present": len(trained) == 3,
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
        "initial_state_shape_is_8x64x32": contract["initial_state_shape"]
        == [8, 64, config.hidden_dim],
        "one_shared_step_executed_four_times": contract["shared_step_count"] == 1
        and contract["execution_rounds"] == 4,
        "sbox_anf_reconstructs": contract["sbox_anf_reconstructs"],
        "forward_loss_gradients_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "cell_relabel_error_at_most_1e_6": contract[
            "cell_relabeling_max_abs_logit_error"
        ]
        <= 1e-6,
        "true_corrupted_logit_delta_at_least_1e_5": contract[
            "true_corrupted_max_abs_logit_difference"
        ]
        >= 1e-5,
        "parameter_ratio_in_0p5_2p0": 0.5
        <= contract["parameter_ratio_to_e44"]
        <= 2.0,
        "only_allowed_buffers": contract["allowed_buffer_names_only"],
        "precomputed_certificate_features_absent": contract[
            "precomputed_certificate_feature_buffers_absent"
        ],
        "label_shuffle_validation_auc_in_0p35_0p65": 0.35
        <= float(shuffled["validation_auc"])
        <= 0.65,
    }
    if all(protocol_checks.values()):
        status = "pass"
        decision = "innovation2_present_mspn_readiness_passed"
        action = "prepare E47 30-epoch seed0 MSPN attribution plan"
    else:
        status = "fail"
        decision = "innovation2_present_mspn_readiness_failed"
        action = "repair MSPN equivariance, finite training, parameter, source, or leakage contract"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "metrics": {
            "parameter_count": contract["parameter_count"],
            "parameter_ratio_to_e44": contract["parameter_ratio_to_e44"],
            "cell_relabeling_max_abs_logit_error": contract[
                "cell_relabeling_max_abs_logit_error"
            ],
            "true_corrupted_max_abs_logit_difference": contract[
                "true_corrupted_max_abs_logit_difference"
            ],
            "mspn_true_smoke_auc": next(
                row["validation_auc"]
                for row in trained
                if row["topology_mode"] == "true" and row["label_mode"] == "true"
            ),
            "mspn_corrupted_smoke_auc": next(
                row["validation_auc"]
                for row in trained
                if row["topology_mode"] == "corrupted"
            ),
            "mspn_shuffle_smoke_auc": shuffled["validation_auc"],
            "e44_triangle_auc": E44_TRIANGLE_AUC,
            "e45_prefix_ridge_auc": E45_PREFIX_AUC,
        },
        "claim_scope": (
            "two-epoch local implementation and training readiness for a PRESENT-80 "
            "r4 Monomial Support Propagation Network; not an effective neural result, "
            "high-round distinguisher, remote-scale result, or SOTA attack"
        ),
        "next_action": {
            "action": action,
            "formal_seed0": status == "pass",
            "seed1": False,
            "remote_scale": False,
        },
    }


def serializable_config(config: MspnReadinessConfig) -> dict[str, Any]:
    return asdict(config)


def _make_model(
    config: MspnReadinessConfig,
    data: dict[str, Any],
    topology_mode: str,
    *,
    dropout: float | None = None,
) -> PresentMonomialSupportPropagationNetwork:
    return PresentMonomialSupportPropagationNetwork(
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


def _cell_relabel_data(data: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    cell_permutation = np.asarray(
        [5, 12, 1, 9, 0, 14, 3, 10, 6, 15, 8, 2, 13, 4, 11, 7],
        dtype=np.int64,
    )
    node_permutation = np.asarray(
        [4 * cell_permutation[node // 4] + node % 4 for node in range(64)],
        dtype=np.int64,
    )
    inverse = np.argsort(node_permutation)
    relabeled = dict(data)
    relabeled["players"] = node_permutation[data["players"][:, inverse]]
    relabeled["structure_active"] = data["structure_active"][:, inverse]
    relabeled["output_mask_bits"] = data["output_mask_bits"][:, inverse]
    return relabeled, node_permutation


def _rows_to_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    return (
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
        raise ValueError("MSPN parameter names do not match")
    with torch.no_grad():
        for name, parameter in target_parameters.items():
            parameter.copy_(source_parameters[name])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
