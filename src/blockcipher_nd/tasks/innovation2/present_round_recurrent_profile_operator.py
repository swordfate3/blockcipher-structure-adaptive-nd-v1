from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.present_round_recurrent_profile_operator import (
    PresentRoundRecurrentProfileOperator,
    PresentRoundRecurrentProfileOperatorSpec,
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
MODE_SPECS = {
    "true_order_true_P": ((0, 1, 2), "true"),
    "wrong_order_true_P": ((2, 1, 0), "true"),
    "true_order_corrupted_P": ((0, 1, 2), "corrupted"),
}


@dataclass(frozen=True)
class RoundRecurrentProfileConfig:
    run_id: str
    epochs: int = 2
    batch_size: int = 8
    hidden_dim: int = 22
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
            or self.hidden_dim != 22
            or self.seed != 0
            or self.dropout != 0.10
            or self.device != "cpu"
        ):
            raise ValueError("E71 Phase A readiness protocol is frozen")


def make_round_recurrent_model(
    config: RoundRecurrentProfileConfig,
    mode: str,
    *,
    player: np.ndarray | None = None,
    dropout: float | None = None,
) -> PresentRoundRecurrentProfileOperator:
    if mode not in MODE_SPECS:
        raise ValueError(f"unsupported E71 mode: {mode}")
    round_order, relation_mode = MODE_SPECS[mode]
    true_player = _present_player()
    corrupted_player = topology_players(true_player[None, :], "corrupted")[0]
    selected_player = (
        np.asarray(player, dtype=np.int64)
        if player is not None
        else corrupted_player
        if relation_mode == "corrupted"
        else true_player
    )
    return PresentRoundRecurrentProfileOperator(
        PresentRoundRecurrentProfileOperatorSpec(
            round_input_dim=13,
            hidden_dim=config.hidden_dim,
            prefix_rounds=3,
            dropout=config.dropout if dropout is None else dropout,
            relation_mode=relation_mode,
            round_order=round_order,
        ),
        torch.from_numpy(selected_player),
    )


def measure_round_recurrent_contract(
    config: RoundRecurrentProfileConfig,
    sources: dict[str, Any],
) -> dict[str, Any]:
    models = {
        mode: make_round_recurrent_model(config, mode, dropout=0.0)
        for mode in MODE_SPECS
    }
    candidate = models["true_order_true_P"]
    for mode, model in models.items():
        if mode != "true_order_true_P":
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
        wrong_order_logits = models["wrong_order_true_P"](features)
        corrupted_logits = models["true_order_corrupted_P"](features)

    player = _present_player()
    permutation = _cell_permutation()
    relabeled_player = _conjugate_player(player, permutation)
    relabeled = make_round_recurrent_model(
        config,
        "true_order_true_P",
        player=relabeled_player,
        dropout=0.0,
    )
    _copy_parameters(candidate, relabeled)
    permuted_features = torch.empty_like(features)
    permuted_features[:, permutation] = features
    with torch.no_grad():
        permuted = relabeled(permuted_features)
    expected = torch.empty_like(candidate_logits)
    expected[:, permutation] = candidate_logits

    parameter_counts = {
        mode: sum(parameter.numel() for parameter in model.parameters())
        for mode, model in models.items()
    }
    candidate_count = parameter_counts["true_order_true_P"]
    forbidden = ("certificate", "witness", "parity", "full_cube", "label")
    return {
        "output_shape": list(logits.shape),
        "masked_loss_explicit_max_abs_error": float(
            torch.abs(loss - explicit).detach()
        ),
        "parameter_counts": parameter_counts,
        "parameter_counts_match": len(set(parameter_counts.values())) == 1,
        "parameter_ratio_to_e68": candidate_count / E68_PARAMETER_COUNT,
        "round_order_logit_max_abs_difference": float(
            torch.max(torch.abs(candidate_logits - wrong_order_logits))
        ),
        "topology_logit_max_abs_difference": float(
            torch.max(torch.abs(candidate_logits - corrupted_logits))
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
    }


def train_round_recurrent_matrix(
    config: RoundRecurrentProfileConfig,
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
    for mode in MODE_SPECS:
        _seed_everything(config.seed)
        model = make_round_recurrent_model(config, mode).to(config.device)
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
                "row_id": f"round_recurrent_{mode}_seed{config.seed}",
                "mode": mode,
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
                    "task": "innovation2_present_round_recurrent_profile_readiness",
                    "row_id": history["row_id"],
                    "mode": mode,
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
            raise RuntimeError("E71 training produced no checkpoint")
        best["epochs_completed"] = config.epochs
        torch.save(best_state, checkpoints / f"{best['row_id']}.pt")
        trained_rows.append(best)
    return {"trained_rows": trained_rows, "history_rows": history_rows}


def adjudicate_round_recurrent_readiness(
    config: RoundRecurrentProfileConfig,
    source_checks: dict[str, bool | float],
    contract: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    rows = training["trained_rows"]
    by_mode = {row["mode"]: row for row in rows}
    candidate_auc = float(
        by_mode.get("true_order_true_P", {}).get("validation_auc", 0.0)
    )
    wrong_order_auc = float(
        by_mode.get("wrong_order_true_P", {}).get("validation_auc", 0.0)
    )
    corrupted_auc = float(
        by_mode.get("true_order_corrupted_P", {}).get("validation_auc", 0.0)
    )
    protocol_checks = {
        **{
            key: value
            for key, value in source_checks.items()
            if isinstance(value, bool)
        },
        "output_shape_is_4x64": contract["output_shape"] == [4, 64],
        "masked_loss_matches_explicit": contract[
            "masked_loss_explicit_max_abs_error"
        ]
        <= 1e-7,
        "parameter_counts_match": contract["parameter_counts_match"],
        "parameter_budget_within_10_percent": 0.90
        <= contract["parameter_ratio_to_e68"]
        <= 1.10,
        "round_order_changes_logits": contract[
            "round_order_logit_max_abs_difference"
        ]
        >= 1e-6,
        "topology_changes_logits": contract["topology_logit_max_abs_difference"]
        >= 1e-6,
        "cell_relabel_equivariant": contract["cell_relabel_max_abs_error"] <= 1e-6,
        "forbidden_named_state_absent": contract["forbidden_named_state_absent"],
        "contract_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "all_three_rows_present": set(by_mode) == set(MODE_SPECS),
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
        "candidate_auc_at_least_0p70": candidate_auc >= 0.70,
        "candidate_minus_wrong_order_at_least_0p02": candidate_auc
        - wrong_order_auc
        >= 0.02,
        "candidate_minus_corrupted_at_least_0p02": candidate_auc - corrupted_auc
        >= 0.02,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_round_recurrent_protocol_invalid"
        action = "repair source, invariance, fairness, or training protocol"
    elif not all(readiness_checks.values()):
        status = "hold"
        decision = "innovation2_present_round_recurrent_readiness_not_passed"
        action = "stop RR-PGPO without adding capacity or epochs"
    else:
        status = "pass"
        decision = "innovation2_present_round_recurrent_readiness_passed"
        action = "run the frozen 30-epoch seed0 attribution matrix"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "readiness_checks": readiness_checks,
        "metrics": {
            "rows": rows,
            "candidate_minus_wrong_order": candidate_auc - wrong_order_auc,
            "candidate_minus_corrupted": candidate_auc - corrupted_auc,
            "contract": contract,
        },
        "claim_scope": (
            "two-epoch local readiness for a round-recurrent profile operator on "
            "PRESENT-80 r4 strict 8-bit-cube unit-balance labels; no high-round, "
            "cross-dimension, new-attack, remote-scale, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "formal_seed0": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: RoundRecurrentProfileConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "MODE_SPECS",
    "RoundRecurrentProfileConfig",
    "adjudicate_round_recurrent_readiness",
    "make_round_recurrent_model",
    "measure_round_recurrent_contract",
    "serializable_config",
    "train_round_recurrent_matrix",
]
