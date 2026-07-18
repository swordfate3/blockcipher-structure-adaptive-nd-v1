from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.rectangle_row_typed_shift_operator import (
    RectangleRowTypedShiftOperator,
    RectangleRowTypedShiftOperatorSpec,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_readiness import (
    _batch_tensors,
    _cell_permutation,
    _conjugate_player,
    _copy_parameters,
    _evaluate,
    masked_binary_cross_entropy,
)
from blockcipher_nd.tasks.innovation2.rectangle80_r3_only_profile_operator_readiness import (
    rectangle_model_player,
)


E91_RUN_ID = "i2_rectangle80_row_typed_shift_representation_audit_20260719"
E91_DECISION = "innovation2_rectangle80_row_typed_representation_ready"
MODES = (
    "untyped_true",
    "row_typed_true",
    "row_typed_corrupted",
    "wrong_row_true",
)
EXPECTED_PARAMETER_COUNT = 4_795


@dataclass(frozen=True)
class Rectangle80RowTypedOperatorConfig:
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
            raise ValueError("E92 readiness protocol is frozen")


def load_e91_source(root: Path) -> dict[str, Any]:
    gate_path = root / "gate.json"
    results_path = root / "results.jsonl"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "gate": gate,
        "rows": rows,
        "hashes": {
            "gate": _sha256(gate_path),
            "results": _sha256(results_path),
        },
    }


def validate_e91_source(source: dict[str, Any]) -> dict[str, bool]:
    gate = source["gate"]
    rows = source["rows"]
    return {
        "e91_run_id_matches": gate.get("run_id") == E91_RUN_ID,
        "e91_status_pass": gate.get("status") == "pass",
        "e91_decision_matches": gate.get("decision") == E91_DECISION,
        "e91_protocol_checks_pass": bool(gate.get("protocol_checks"))
        and all(gate["protocol_checks"].values()),
        "e91_mechanism_checks_pass": bool(gate.get("mechanism_checks"))
        and all(gate["mechanism_checks"].values()),
        "e91_no_training": gate.get("next_action", {}).get("training_performed")
        is False,
        "e91_five_rows_present": len(rows) == 5,
        "e91_typed_true_ridge_present": float(
            gate.get("metrics", {})
            .get("ridges", {})
            .get("typed_true", {})
            .get("validation_auc", 0.0)
        )
        >= 0.60,
        "e91_hashes_present": all(
            len(value) == 64 for value in source["hashes"].values()
        ),
    }


def make_row_typed_model(
    config: Rectangle80RowTypedOperatorConfig,
    mode: str,
    *,
    player: np.ndarray | None = None,
    dropout: float | None = None,
) -> RectangleRowTypedShiftOperator:
    if mode not in MODES:
        raise ValueError(f"unsupported E92 mode: {mode}")
    true_player = rectangle_model_player()
    corrupted_player = topology_players(true_player[None, :], "corrupted")[0]
    selected_player = (
        np.asarray(player, dtype=np.int64)
        if player is not None
        else corrupted_player
        if mode == "row_typed_corrupted"
        else true_player
    )
    row_mode = (
        "untyped"
        if mode == "untyped_true"
        else "wrong"
        if mode == "wrong_row_true"
        else "true"
    )
    return RectangleRowTypedShiftOperator(
        RectangleRowTypedShiftOperatorSpec(
            input_dim=13,
            hidden_dim=config.hidden_dim,
            steps=config.steps,
            dropout=config.dropout if dropout is None else dropout,
            row_mode=row_mode,
        ),
        torch.from_numpy(selected_player),
    )


def measure_row_typed_contract(
    config: Rectangle80RowTypedOperatorConfig, sources: dict[str, Any]
) -> dict[str, Any]:
    models = {
        mode: make_row_typed_model(config, mode, dropout=0.0) for mode in MODES
    }
    candidate = models["row_typed_true"]
    for mode, model in models.items():
        if mode != "row_typed_true":
            _copy_parameters(candidate, model)
    indices = sorted(
        {row["structure_index"] for row in sources["matched_rows"]}
    )[:4]
    features, targets, observed = _batch_tensors(sources, indices, "cpu")
    logits = candidate(features)
    loss = masked_binary_cross_entropy(logits, targets, observed)
    explicit = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[observed], targets[observed]
    )
    loss.backward()
    with torch.no_grad():
        candidate_logits = candidate(features)
        untyped_logits = models["untyped_true"](features)
        corrupted_logits = models["row_typed_corrupted"](features)
        wrong_row_logits = models["wrong_row_true"](features)

    permutation = _cell_permutation()
    relabeled_player = _conjugate_player(rectangle_model_player(), permutation)
    relabeled = make_row_typed_model(
        config,
        "row_typed_true",
        player=relabeled_player,
        dropout=0.0,
    )
    _copy_parameters(candidate, relabeled)
    permuted_features = torch.empty_like(features)
    permuted_features[:, permutation] = features
    with torch.no_grad():
        permuted_logits = relabeled(permuted_features)
    expected = torch.empty_like(candidate_logits)
    expected[:, permutation] = candidate_logits

    parameter_counts = {
        mode: sum(parameter.numel() for parameter in model.parameters())
        for mode, model in models.items()
    }
    forbidden = ("certificate", "witness", "parity", "full_cube", "label")
    return {
        "output_shape": list(logits.shape),
        "input_dim": 13,
        "parameter_counts": parameter_counts,
        "parameter_counts_match": len(set(parameter_counts.values())) == 1,
        "masked_loss_explicit_max_abs_error": float(
            torch.abs(loss - explicit).detach()
        ),
        "typed_vs_untyped_logit_max_abs_difference": float(
            torch.max(torch.abs(candidate_logits - untyped_logits))
        ),
        "true_vs_corrupted_logit_max_abs_difference": float(
            torch.max(torch.abs(candidate_logits - corrupted_logits))
        ),
        "true_vs_wrong_row_logit_max_abs_difference": float(
            torch.max(torch.abs(candidate_logits - wrong_row_logits))
        ),
        "cell_relabel_max_abs_error": float(
            torch.max(torch.abs(permuted_logits - expected))
        ),
        "true_and_wrong_channel_maps_differ": not torch.equal(
            candidate.typed_channel_index,
            models["wrong_row_true"].typed_channel_index,
        ),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in candidate.parameters()
        ),
        "forbidden_named_state_absent": not any(
            token in name for name in candidate.state_dict() for token in forbidden
        ),
    }


def train_row_typed_matrix(
    config: Rectangle80RowTypedOperatorConfig,
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
    for mode in MODES:
        _seed_everything(config.seed)
        model = make_row_typed_model(config, mode).to(config.device)
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
                loss = masked_binary_cross_entropy(
                    model(features), targets, observed
                )
                loss.backward()
                optimizer.step()
            train_metrics = _evaluate(model, sources, train_indices, config)
            validation_metrics = _evaluate(
                model, sources, validation_indices, config
            )
            history = {
                "row_id": f"rectangle_row_typed_{mode}_seed{config.seed}",
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
                    "task": "innovation2_rectangle80_row_typed_shift_operator_readiness",
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
            raise RuntimeError("E92 training produced no checkpoint")
        best["epochs_completed"] = config.epochs
        torch.save(best_state, checkpoints / f"{best['row_id']}.pt")
        trained_rows.append(best)
    return {"trained_rows": trained_rows, "history_rows": history_rows}


def adjudicate_row_typed_readiness(
    config: Rectangle80RowTypedOperatorConfig,
    profile_checks: dict[str, bool],
    model_order_checks: dict[str, bool],
    e90_checks: dict[str, bool],
    e91_checks: dict[str, bool],
    contract: dict[str, Any],
    training: dict[str, Any],
    e91_source: dict[str, Any],
) -> dict[str, Any]:
    rows = training["trained_rows"]
    by_mode = {row["relation_mode"]: row for row in rows}
    auc = {
        mode: float(by_mode.get(mode, {}).get("validation_auc", 0.0))
        for mode in MODES
    }
    candidate = auc["row_typed_true"]
    typed_ridge = float(
        e91_source["gate"]["metrics"]["ridges"]["typed_true"]["validation_auc"]
    )
    protocol_checks = {
        **profile_checks,
        **model_order_checks,
        **e90_checks,
        **e91_checks,
        "output_shape_is_4x64": contract["output_shape"] == [4, 64],
        "input_dim_is_13": contract["input_dim"] == 13,
        "masked_loss_matches_explicit": contract[
            "masked_loss_explicit_max_abs_error"
        ]
        <= 1e-7,
        "parameter_counts_match": contract["parameter_counts_match"],
        "all_parameter_counts_are_4795": set(contract["parameter_counts"].values())
        == {EXPECTED_PARAMETER_COUNT},
        "typed_changes_logits": contract[
            "typed_vs_untyped_logit_max_abs_difference"
        ]
        >= 1e-6,
        "topology_changes_logits": contract[
            "true_vs_corrupted_logit_max_abs_difference"
        ]
        >= 1e-6,
        "wrong_row_changes_logits": contract[
            "true_vs_wrong_row_logit_max_abs_difference"
        ]
        >= 1e-6,
        "true_and_wrong_channel_maps_differ": contract[
            "true_and_wrong_channel_maps_differ"
        ],
        "cell_relabel_equivariant": contract["cell_relabel_max_abs_error"] <= 1e-6,
        "forbidden_named_state_absent": contract["forbidden_named_state_absent"],
        "contract_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "all_four_rows_present": set(by_mode) == set(MODES),
        "all_rows_completed_two_epochs": len(rows) == 4
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
    }
    readiness_checks = {
        "row_typed_true_auc_at_least_0p65": candidate >= 0.65,
        "row_typed_true_minus_untyped_at_least_0p01": candidate
        - auc["untyped_true"]
        >= 0.01,
        "row_typed_true_minus_corrupted_at_least_0p03": candidate
        - auc["row_typed_corrupted"]
        >= 0.03,
        "row_typed_true_minus_wrong_row_at_least_0p01": candidate
        - auc["wrong_row_true"]
        >= 0.01,
        "row_typed_true_minus_typed_ridge_at_least_minus_0p03": candidate
        - typed_ridge
        >= -0.03,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_rectangle80_row_typed_shift_operator_protocol_invalid"
        action = "repair E88/E90/E91 replay, typed channel map, or training"
    elif not all(readiness_checks.values()):
        status = "hold"
        decision = "innovation2_rectangle80_row_typed_shift_operator_not_ready"
        action = "close the row-typed neural route without architecture tuning"
    else:
        status = "pass"
        decision = "innovation2_rectangle80_row_typed_shift_operator_readiness_passed"
        action = "run the frozen four-row 30-epoch seed0 attribution"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "readiness_checks": readiness_checks,
        "metrics": {
            "rows": rows,
            "e91_typed_true_ridge_auc": typed_ridge,
            "typed_true_minus_untyped": candidate - auc["untyped_true"],
            "typed_true_minus_corrupted": candidate
            - auc["row_typed_corrupted"],
            "typed_true_minus_wrong_row": candidate - auc["wrong_row_true"],
            "typed_true_minus_typed_ridge": candidate - typed_ridge,
            "contract": contract,
        },
        "claim_scope": (
            "two-epoch local readiness of a parameter-neutral RECTANGLE-80 "
            "Row-Typed Shift Operator on E88 strict labels; no formal gain, "
            "seven-round reproduction, attack, remote-scale, or SOTA"
        ),
        "next_action": {
            "action": action,
            "formal_seed0": status == "pass",
            "remote_scale": False,
        },
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serializable_config(
    config: Rectangle80RowTypedOperatorConfig,
) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "Rectangle80RowTypedOperatorConfig",
    "adjudicate_row_typed_readiness",
    "load_e91_source",
    "make_row_typed_model",
    "measure_row_typed_contract",
    "serializable_config",
    "train_row_typed_matrix",
    "validate_e91_source",
]
