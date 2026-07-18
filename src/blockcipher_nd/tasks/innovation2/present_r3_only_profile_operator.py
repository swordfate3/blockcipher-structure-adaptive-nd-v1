from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.present_balance_profile_operator import (
    PresentBalanceProfileOperator,
    PresentBalanceProfileOperatorSpec,
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
    _present_player,
    _seed_everything,
    masked_binary_cross_entropy,
)


E68_PARAMETER_COUNT = 5_679
R3_SLICE = slice(26, 39)
RELATION_MODES = ("independent", "true", "corrupted")


@dataclass(frozen=True)
class R3OnlyProfileConfig:
    run_id: str
    mode: str = "readiness"
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
        if self.mode not in {"readiness", "formal"}:
            raise ValueError("mode must be readiness or formal")
        if (
            self.batch_size != 8
            or self.hidden_dim != 32
            or self.steps != 2
            or self.dropout != 0.10
            or self.device != "cpu"
        ):
            raise ValueError("E73 shared training protocol is frozen")
        if (self.mode == "readiness" and self.epochs != 2) or (
            self.mode == "formal" and self.epochs != 30
        ):
            raise ValueError("E73 phase epoch budget is frozen")
        if (self.mode == "readiness" and self.seed != 0) or (
            self.mode == "formal" and self.seed not in {0, 1}
        ):
            raise ValueError("E73 phase seed is frozen")


def r3_only_sources(sources: dict[str, Any]) -> dict[str, Any]:
    prefix = np.asarray(sources["prefix_features"])
    if prefix.shape != (96, 64, 39):
        raise ValueError("E73 requires the frozen 96x64x39 E65 prefix")
    return {**sources, "prefix_features": prefix[:, :, R3_SLICE].copy()}


def make_r3_only_model(
    config: R3OnlyProfileConfig,
    mode: str,
    *,
    player: np.ndarray | None = None,
    dropout: float | None = None,
) -> PresentBalanceProfileOperator:
    if mode not in RELATION_MODES:
        raise ValueError(f"unsupported E73 mode: {mode}")
    true_player = _present_player()
    corrupted_player = topology_players(true_player[None, :], "corrupted")[0]
    selected_player = (
        np.asarray(player, dtype=np.int64)
        if player is not None
        else corrupted_player
        if mode == "corrupted"
        else true_player
    )
    return PresentBalanceProfileOperator(
        PresentBalanceProfileOperatorSpec(
            input_dim=13,
            hidden_dim=config.hidden_dim,
            steps=config.steps,
            dropout=config.dropout if dropout is None else dropout,
            relation_mode=mode,
        ),
        torch.from_numpy(selected_player),
    )


def measure_r3_only_contract(
    config: R3OnlyProfileConfig, sources: dict[str, Any]
) -> dict[str, Any]:
    models = {
        mode: make_r3_only_model(config, mode, dropout=0.0)
        for mode in RELATION_MODES
    }
    candidate = models["true"]
    _copy_parameters(candidate, models["independent"])
    _copy_parameters(candidate, models["corrupted"])
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
        true_logits = candidate(features)
        corrupted_logits = models["corrupted"](features)

    player = _present_player()
    permutation = _cell_permutation()
    relabeled_player = _conjugate_player(player, permutation)
    relabeled = make_r3_only_model(
        config, "true", player=relabeled_player, dropout=0.0
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
    candidate_count = parameter_counts["true"]
    forbidden = ("certificate", "witness", "parity", "full_cube", "label")
    return {
        "output_shape": list(logits.shape),
        "masked_loss_explicit_max_abs_error": float(
            torch.abs(loss - explicit).detach()
        ),
        "parameter_counts": parameter_counts,
        "parameter_counts_match": len(set(parameter_counts.values())) == 1,
        "parameter_ratio_to_e68": candidate_count / E68_PARAMETER_COUNT,
        "topology_logit_max_abs_difference": float(
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
        "input_dim": 13,
    }


def train_r3_only_matrix(
    config: R3OnlyProfileConfig,
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
        model = make_r3_only_model(config, mode).to(config.device)
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
                "row_id": f"r3_only_profile_{mode}_seed{config.seed}",
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
                    "task": "innovation2_present_r3_only_profile_readiness",
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
            raise RuntimeError("E73 training produced no checkpoint")
        best["epochs_completed"] = config.epochs
        torch.save(best_state, checkpoints / f"{best['row_id']}.pt")
        trained_rows.append(best)
    return {"trained_rows": trained_rows, "history_rows": history_rows}


def adjudicate_r3_only_readiness(
    config: R3OnlyProfileConfig,
    source_checks: dict[str, bool | float],
    contract: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    rows = training["trained_rows"]
    by_mode = {row["relation_mode"]: row for row in rows}
    true_auc = float(by_mode.get("true", {}).get("validation_auc", 0.0))
    independent_auc = float(
        by_mode.get("independent", {}).get("validation_auc", 0.0)
    )
    corrupted_auc = float(by_mode.get("corrupted", {}).get("validation_auc", 0.0))
    protocol_checks = {
        **{
            key: value
            for key, value in source_checks.items()
            if isinstance(value, bool)
        },
        "output_shape_is_4x64": contract["output_shape"] == [4, 64],
        "input_dim_is_13": contract["input_dim"] == 13,
        "masked_loss_matches_explicit": contract[
            "masked_loss_explicit_max_abs_error"
        ]
        <= 1e-7,
        "parameter_counts_match": contract["parameter_counts_match"],
        "parameters_reduced_at_least_10_percent": contract["parameter_ratio_to_e68"]
        <= 0.90,
        "topology_changes_logits": contract["topology_logit_max_abs_difference"]
        >= 1e-6,
        "cell_relabel_equivariant": contract["cell_relabel_max_abs_error"] <= 1e-6,
        "forbidden_named_state_absent": contract["forbidden_named_state_absent"],
        "contract_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
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
    }
    readiness_checks = {
        "true_auc_at_least_0p75": true_auc >= 0.75,
        "true_minus_independent_at_least_0p03": true_auc - independent_auc >= 0.03,
        "true_minus_corrupted_at_least_0p03": true_auc - corrupted_auc >= 0.03,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_r3_only_profile_protocol_invalid"
        action = "repair source, r3 slice, fairness, or training protocol"
    elif not all(readiness_checks.values()):
        status = "hold"
        decision = "innovation2_present_r3_only_profile_readiness_not_passed"
        action = "retain full 39-d E68 and stop r3-only compression"
    else:
        status = "pass"
        decision = "innovation2_present_r3_only_profile_readiness_passed"
        action = "run the frozen 30-epoch seed0 r3-only attribution"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "readiness_checks": readiness_checks,
        "metrics": {
            "rows": rows,
            "true_minus_independent": true_auc - independent_auc,
            "true_minus_corrupted": true_auc - corrupted_auc,
            "contract": contract,
        },
        "claim_scope": (
            "two-epoch local readiness for an r3-only prefix-guided profile "
            "operator on PRESENT-80 r4 strict 8-bit-cube unit-balance labels; "
            "no high-round, cross-dimension, attack, remote-scale, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "formal_seed0": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: R3OnlyProfileConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "R3OnlyProfileConfig",
    "adjudicate_r3_only_readiness",
    "make_r3_only_model",
    "measure_r3_only_contract",
    "r3_only_sources",
    "serializable_config",
    "train_r3_only_matrix",
]
