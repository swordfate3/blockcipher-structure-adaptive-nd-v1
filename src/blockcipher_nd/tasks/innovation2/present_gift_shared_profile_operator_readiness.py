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

from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.models.structure.spn.topology_parameterized_profile_operator import (
    TopologyParameterizedProfileOperator,
    TopologyParameterizedProfileOperatorSpec,
)
from blockcipher_nd.tasks.innovation2.gift64_r3_only_profile_operator_readiness import (
    gift_player,
    load_gift_profile_sources,
    r3_only_sources as gift_r3_only_sources,
    validate_gift_profile_sources,
)
from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_readiness import (
    _batch_tensors,
    _cell_permutation,
    _conjugate_player,
    _present_player,
    load_profile_operator_sources,
    masked_binary_cross_entropy,
    validate_profile_operator_sources,
)
from blockcipher_nd.tasks.innovation2.present_r3_only_profile_operator import (
    r3_only_sources as present_r3_only_sources,
)
from blockcipher_nd.training.metrics import binary_auc


RELATION_MODES = ("independent", "true", "corrupted")
EXPECTED_PARAMETER_COUNT = 4_795
PRESENT_READINESS_RUN_ID = (
    "i2_present_r4_r3_only_profile_operator_readiness_seed0_20260718"
)
PRESENT_READINESS_DECISION = "innovation2_present_r3_only_profile_readiness_passed"
PRESENT_READINESS_TRUE_AUC = 0.8341666666666666
GIFT_READINESS_RUN_ID = (
    "i2_gift64_r4_r3_only_profile_operator_readiness_seed0_20260719"
)
GIFT_READINESS_DECISION = "innovation2_gift64_r3_only_prefix_not_sufficient"
GIFT_READINESS_TRUE_AUC = 0.7606659729448492
E80_RUN_ID = "i2_cross_spn_r3_profile_operator_method_synthesis_20260719"
E80_DECISION = "innovation2_cross_spn_r3_profile_method_confirmed_skinny_labels_not_ready"
E84_RUN_ID = (
    "i2_skinny64_r5_true_ridge_sparse_residual_readiness_seed0_20260719"
)
E84_DECISION = "innovation2_skinny64_true_ridge_residual_not_ready"


@dataclass(frozen=True)
class SharedProfileReadinessConfig:
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
            raise ValueError("E85 readiness protocol is frozen")

    @property
    def task_name(self) -> str:
        return "innovation2_present_gift_shared_profile_operator_readiness"


def load_e85_sources(
    present_profile_root: Path,
    present_atlas_root: Path,
    gift_profile_root: Path,
    present_readiness_root: Path,
    gift_readiness_root: Path,
    e80_root: Path,
    e84_root: Path,
) -> dict[str, Any]:
    tracked = {
        "present_readiness": present_readiness_root,
        "gift_readiness": gift_readiness_root,
        "e80": e80_root,
        "e84": e84_root,
    }
    gates = {
        name: json.loads((root / "gate.json").read_text(encoding="utf-8"))
        for name, root in tracked.items()
    }
    hashes = {
        name: {
            filename: _sha256(root / filename)
            for filename in ("gate.json", "results.jsonl")
        }
        for name, root in tracked.items()
    }
    return {
        "present_full": load_profile_operator_sources(
            present_profile_root, present_atlas_root
        ),
        "gift_full": load_gift_profile_sources(gift_profile_root),
        "gates": gates,
        "route_hashes": hashes,
    }


def validate_e85_sources(sources: dict[str, Any]) -> dict[str, bool]:
    present_checks = validate_profile_operator_sources(
        sources["present_full"], strict=True
    )
    gift_checks = validate_gift_profile_sources(sources["gift_full"])
    gates = sources["gates"]
    present_rows = gates["present_readiness"].get("metrics", {}).get("rows", [])
    gift_rows = gates["gift_readiness"].get("metrics", {}).get("rows", [])
    present_true = next(
        (row for row in present_rows if row.get("relation_mode") == "true"), {}
    )
    gift_true = next(
        (row for row in gift_rows if row.get("relation_mode") == "true"), {}
    )
    return {
        **{
            f"present_{key}": value
            for key, value in present_checks.items()
            if isinstance(value, bool)
        },
        **{
            f"gift_{key}": value
            for key, value in gift_checks.items()
            if isinstance(value, bool)
        },
        "present_readiness_run_id_matches": gates["present_readiness"].get(
            "run_id"
        )
        == PRESENT_READINESS_RUN_ID,
        "present_readiness_status_pass": gates["present_readiness"].get("status")
        == "pass",
        "present_readiness_decision_matches": gates["present_readiness"].get(
            "decision"
        )
        == PRESENT_READINESS_DECISION,
        "present_readiness_true_auc_matches": math.isclose(
            float(present_true.get("validation_auc", float("nan"))),
            PRESENT_READINESS_TRUE_AUC,
            abs_tol=1e-12,
        ),
        "gift_readiness_run_id_matches": gates["gift_readiness"].get("run_id")
        == GIFT_READINESS_RUN_ID,
        "gift_readiness_status_hold": gates["gift_readiness"].get("status")
        == "hold",
        "gift_readiness_decision_matches": gates["gift_readiness"].get("decision")
        == GIFT_READINESS_DECISION,
        "gift_readiness_true_auc_matches": math.isclose(
            float(gift_true.get("validation_auc", float("nan"))),
            GIFT_READINESS_TRUE_AUC,
            abs_tol=1e-12,
        ),
        "e80_run_id_matches": gates["e80"].get("run_id") == E80_RUN_ID,
        "e80_status_pass": gates["e80"].get("status") == "pass",
        "e80_decision_matches": gates["e80"].get("decision") == E80_DECISION,
        "e84_run_id_matches": gates["e84"].get("run_id") == E84_RUN_ID,
        "e84_status_hold": gates["e84"].get("status") == "hold",
        "e84_decision_matches": gates["e84"].get("decision") == E84_DECISION,
        "route_hashes_present": all(
            len(value) == 64
            for group in sources["route_hashes"].values()
            for value in group.values()
        ),
    }


def prepare_e85_sources(sources: dict[str, Any]) -> dict[str, dict[str, Any]]:
    present = present_r3_only_sources(sources["present_full"])
    gift = gift_r3_only_sources(sources["gift_full"])
    return {"present": present, "gift": gift}


def make_shared_profile_model(
    config: SharedProfileReadinessConfig,
    mode: str,
    *,
    dropout: float | None = None,
) -> TopologyParameterizedProfileOperator:
    if mode not in RELATION_MODES:
        raise ValueError(f"unsupported E85 relation mode: {mode}")
    return TopologyParameterizedProfileOperator(
        TopologyParameterizedProfileOperatorSpec(
            input_dim=13,
            hidden_dim=config.hidden_dim,
            steps=config.steps,
            dropout=config.dropout if dropout is None else dropout,
            relation_mode=mode,
        )
    )


def relation_players(mode: str) -> dict[str, np.ndarray]:
    players = {"present": _present_player(), "gift": gift_player()}
    if mode == "corrupted":
        return {
            name: topology_players(player[None, :], "corrupted")[0]
            for name, player in players.items()
        }
    return players


def inverse_player(player: np.ndarray, device: str = "cpu") -> torch.Tensor:
    values = np.asarray(player, dtype=np.int64)
    if values.shape != (64,) or not np.array_equal(np.sort(values), np.arange(64)):
        raise ValueError("player must be a 64-bit permutation")
    inverse = np.empty_like(values)
    inverse[values] = np.arange(64)
    return torch.from_numpy(inverse).to(device=device, dtype=torch.long)


def measure_shared_profile_contract(
    config: SharedProfileReadinessConfig,
    data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    models: dict[str, TopologyParameterizedProfileOperator] = {}
    for mode in RELATION_MODES:
        _seed_everything(config.seed)
        models[mode] = make_shared_profile_model(config, mode, dropout=0.0)
    candidate = models["true"]
    candidate_state = dict(candidate.named_parameters())
    initial_parameter_max_abs_difference = 0.0
    for mode in ("independent", "corrupted"):
        other = dict(models[mode].named_parameters())
        initial_parameter_max_abs_difference = max(
            initial_parameter_max_abs_difference,
            max(
                float(
                    torch.max(torch.abs(candidate_state[name] - other[name])).detach()
                )
                for name in candidate_state
            ),
        )

    parameter_counts = {
        mode: sum(parameter.numel() for parameter in model.parameters())
        for mode, model in models.items()
    }
    true_players = relation_players("true")
    corrupted_players = relation_players("corrupted")
    topology_differences: dict[str, float] = {}
    relabel_errors: dict[str, float] = {}
    output_shapes: dict[str, list[int]] = {}
    loss_errors: dict[str, float] = {}
    logits_finite = True
    loss_finite = True
    gradient_finite = True
    permutation = _cell_permutation()
    for cipher in ("present", "gift"):
        indices = sorted(
            {row["structure_index"] for row in data[cipher]["matched_rows"]}
        )[:4]
        features, targets, observed = _batch_tensors(
            data[cipher], indices, config.device
        )
        true_inverse = inverse_player(true_players[cipher])
        corrupted_inverse = inverse_player(corrupted_players[cipher])
        logits = candidate(features, true_inverse)
        loss = masked_binary_cross_entropy(logits, targets, observed)
        explicit = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[observed], targets[observed]
        )
        candidate.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            corrupted_logits = candidate(features, corrupted_inverse)
        topology_differences[cipher] = float(
            torch.max(torch.abs(logits.detach() - corrupted_logits))
        )
        output_shapes[cipher] = list(logits.shape)
        loss_errors[cipher] = float(torch.abs(loss.detach() - explicit.detach()))
        logits_finite = logits_finite and bool(torch.isfinite(logits).all())
        loss_finite = loss_finite and bool(torch.isfinite(loss))
        gradient_finite = gradient_finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in candidate.parameters()
        )

        relabeled_player = _conjugate_player(true_players[cipher], permutation)
        permuted_features = torch.empty_like(features)
        permuted_features[:, permutation] = features
        with torch.no_grad():
            relabeled = candidate(
                permuted_features, inverse_player(relabeled_player)
            )
        expected = torch.empty_like(logits)
        expected[:, permutation] = logits.detach()
        relabel_errors[cipher] = float(torch.max(torch.abs(relabeled - expected)))

    forbidden = ("cipher", "adapter", "film", "certificate", "witness", "parity", "label")
    named_state = tuple(candidate.state_dict())
    return {
        "output_shapes": output_shapes,
        "masked_loss_explicit_max_abs_errors": loss_errors,
        "parameter_counts": parameter_counts,
        "parameter_counts_match": len(set(parameter_counts.values())) == 1,
        "initial_parameter_max_abs_difference": initial_parameter_max_abs_difference,
        "runtime_topology_state_absent": not any(
            token in name for name in named_state for token in ("player", "topology")
        ),
        "cipher_specific_named_state_absent": not any(
            token in name for name in named_state for token in forbidden
        ),
        "topology_logit_max_abs_differences": topology_differences,
        "cell_relabel_max_abs_errors": relabel_errors,
        "true_players_are_distinct": not np.array_equal(
            true_players["present"], true_players["gift"]
        ),
        "corrupted_players_are_distinct_from_true": all(
            not np.array_equal(true_players[name], corrupted_players[name])
            for name in true_players
        ),
        "all_players_are_permutations": all(
            np.array_equal(np.sort(player), np.arange(64))
            for player in (*true_players.values(), *corrupted_players.values())
        ),
        "logits_finite": logits_finite,
        "loss_finite": loss_finite,
        "gradients_finite": gradient_finite,
    }


def train_shared_profile_matrix(
    config: SharedProfileReadinessConfig,
    data: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    split_indices = {
        cipher: {
            split: sorted(
                {
                    row["structure_index"]
                    for row in sources["matched_rows"]
                    if row["split"] == split
                }
            )
            for split in ("train", "validation")
        }
        for cipher, sources in data.items()
    }
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    schedule_audits: dict[str, Any] = {}
    for mode in RELATION_MODES:
        _seed_everything(config.seed)
        model = make_shared_profile_model(config, mode).to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        players = relation_players(mode)
        inverses = {
            cipher: inverse_player(player, config.device)
            for cipher, player in players.items()
        }
        best: dict[str, Any] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        mode_audit = {"epoch_batch_counts": [], "total_updates": 0}
        for epoch in range(1, config.epochs + 1):
            schedule = _epoch_schedule(
                split_indices, config.batch_size, config.seed + epoch
            )
            counts = {
                cipher: sum(item[0] == cipher for item in schedule)
                for cipher in ("present", "gift")
            }
            mode_audit["epoch_batch_counts"].append(counts)
            mode_audit["total_updates"] += len(schedule)
            loss_weights = {
                cipher: len(schedule) / (2.0 * counts[cipher])
                for cipher in counts
            }
            model.train()
            for cipher, batch in schedule:
                features, targets, observed = _batch_tensors(
                    data[cipher], batch, config.device
                )
                optimizer.zero_grad(set_to_none=True)
                loss = masked_binary_cross_entropy(
                    model(features, inverses[cipher]), targets, observed
                )
                (loss * loss_weights[cipher]).backward()
                optimizer.step()

            metrics = {
                cipher: {
                    split: _evaluate_shared(
                        model,
                        data[cipher],
                        split_indices[cipher][split],
                        inverses[cipher],
                        config,
                    )
                    for split in ("train", "validation")
                }
                for cipher in ("present", "gift")
            }
            macro_validation_auc = float(
                np.mean(
                    [metrics[cipher]["validation"]["auc"] for cipher in metrics]
                )
            )
            history_row = {
                "row_id": f"shared_profile_{mode}_seed{config.seed}",
                "relation_mode": mode,
                "epoch": epoch,
                "macro_validation_auc": macro_validation_auc,
            }
            for cipher in ("present", "gift"):
                for split in ("train", "validation"):
                    history_row.update(
                        {
                            f"{cipher}_{split}_{key}": value
                            for key, value in metrics[cipher][split].items()
                        }
                    )
            history.append(history_row)
            if best is None or macro_validation_auc > best["macro_validation_auc"]:
                best = {
                    "run_id": config.run_id,
                    "task": config.task_name,
                    "row_id": history_row["row_id"],
                    "relation_mode": mode,
                    "seed": config.seed,
                    "best_epoch": epoch,
                    "epochs_completed": epoch,
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "macro_validation_auc": macro_validation_auc,
                    **{
                        key: value
                        for key, value in history_row.items()
                        if key.startswith("present_") or key.startswith("gift_")
                    },
                }
                best_state = copy.deepcopy(model.state_dict())
        if best is None or best_state is None:
            raise RuntimeError("E85 training produced no checkpoint")
        best["epochs_completed"] = config.epochs
        torch.save(best_state, checkpoints / f"{best['row_id']}.pt")
        rows.append(best)
        schedule_audits[mode] = mode_audit
    return {"rows": rows, "history": history, "schedule_audits": schedule_audits}


def adjudicate_shared_profile_readiness(
    config: SharedProfileReadinessConfig,
    source_checks: dict[str, bool],
    contract: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    rows = matrix["rows"]
    by_mode = {row["relation_mode"]: row for row in rows}
    protocol_checks = {
        **source_checks,
        "r3_shapes_match": source_checks.get("present_r3_shape_is_96x64x13", False)
        and source_checks.get("gift_r3_shape_is_192x64x13", False),
        "output_shapes_are_4x64": set(
            tuple(value) for value in contract["output_shapes"].values()
        )
        == {(4, 64)},
        "masked_losses_match_explicit": max(
            contract["masked_loss_explicit_max_abs_errors"].values()
        )
        <= 1e-7,
        "parameter_counts_are_4795": set(contract["parameter_counts"].values())
        == {EXPECTED_PARAMETER_COUNT},
        "parameter_counts_match": contract["parameter_counts_match"],
        "initial_parameters_match": contract[
            "initial_parameter_max_abs_difference"
        ]
        == 0.0,
        "runtime_topology_state_absent": contract[
            "runtime_topology_state_absent"
        ],
        "cipher_specific_named_state_absent": contract[
            "cipher_specific_named_state_absent"
        ],
        "runtime_topology_changes_logits": min(
            contract["topology_logit_max_abs_differences"].values()
        )
        >= 1e-6,
        "cell_relabel_equivariant": max(
            contract["cell_relabel_max_abs_errors"].values()
        )
        <= 1e-6,
        "true_players_are_distinct": contract["true_players_are_distinct"],
        "corrupted_players_are_distinct_from_true": contract[
            "corrupted_players_are_distinct_from_true"
        ],
        "all_players_are_permutations": contract["all_players_are_permutations"],
        "contract_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "three_rows_present": set(by_mode) == set(RELATION_MODES),
        "all_rows_completed_two_epochs": len(rows) == 3
        and all(row["epochs_completed"] == config.epochs for row in rows),
        "schedule_is_7_plus_14_each_epoch": all(
            audit["epoch_batch_counts"]
            == [{"present": 7, "gift": 14}, {"present": 7, "gift": 14}]
            for audit in matrix["schedule_audits"].values()
        ),
        "each_row_has_42_updates": all(
            audit["total_updates"] == 42
            for audit in matrix["schedule_audits"].values()
        ),
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if key.endswith(("_auc", "_accuracy", "_loss"))
        ),
    }
    readiness_checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {"rows": rows, "contract": contract}
    for cipher in ("present", "gift"):
        true_auc = float(by_mode.get("true", {}).get(f"{cipher}_validation_auc", 0.0))
        independent_auc = float(
            by_mode.get("independent", {}).get(f"{cipher}_validation_auc", 0.0)
        )
        corrupted_auc = float(
            by_mode.get("corrupted", {}).get(f"{cipher}_validation_auc", 0.0)
        )
        anchor = (
            PRESENT_READINESS_TRUE_AUC if cipher == "present" else GIFT_READINESS_TRUE_AUC
        )
        readiness_checks[f"{cipher}_true_within_0p05_of_anchor"] = (
            true_auc >= anchor - 0.05
        )
        readiness_checks[f"{cipher}_true_minus_independent_at_least_0p03"] = (
            true_auc - independent_auc >= 0.03
        )
        readiness_checks[f"{cipher}_true_minus_corrupted_at_least_0p03"] = (
            true_auc - corrupted_auc >= 0.03
        )
        metrics[f"{cipher}_true_minus_independent"] = true_auc - independent_auc
        metrics[f"{cipher}_true_minus_corrupted"] = true_auc - corrupted_auc
        metrics[f"{cipher}_true_minus_anchor"] = true_auc - anchor
    macro_true = float(by_mode.get("true", {}).get("macro_validation_auc", 0.0))
    readiness_checks["macro_true_auc_at_least_0p75"] = macro_true >= 0.75
    metrics["macro_true_auc"] = macro_true

    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_shared_profile_operator_protocol_invalid"
        action = "repair sources, runtime topology, parameter sharing, budget, or equivariance"
    elif not all(readiness_checks.values()):
        status = "hold"
        decision = "innovation2_shared_profile_operator_readiness_not_passed"
        action = "retain separate E73/E79 models and close current shared-parameter branch"
    else:
        status = "pass"
        decision = "innovation2_shared_profile_operator_readiness_passed"
        action = "run E86 frozen 30-epoch seed0 shared-operator attribution"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "readiness_checks": readiness_checks,
        "metrics": metrics,
        "claim_scope": (
            "two-epoch local readiness for one 4,795-parameter topology-"
            "parameterized profile operator jointly trained on PRESENT-80 r4 and "
            "GIFT-64 r4 strict unit-balance profiles; no formal gain, zero-shot "
            "transfer, high-round distinguisher, attack, remote-scale, novelty, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "formal_seed0": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: SharedProfileReadinessConfig) -> dict[str, Any]:
    return asdict(config)


def _epoch_schedule(
    split_indices: dict[str, dict[str, list[int]]],
    batch_size: int,
    seed: int,
) -> list[tuple[str, list[int]]]:
    schedule: list[tuple[str, list[int]]] = []
    for offset, cipher in enumerate(("present", "gift")):
        indices = split_indices[cipher]["train"]
        generator = torch.Generator().manual_seed(seed + 100 * offset)
        order = torch.randperm(len(indices), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            schedule.append(
                (cipher, [indices[index] for index in order[start : start + batch_size]])
            )
    random.Random(seed + 1_000).shuffle(schedule)
    return schedule


def _evaluate_shared(
    model: TopologyParameterizedProfileOperator,
    sources: dict[str, Any],
    indices: list[int],
    inverse: torch.Tensor,
    config: SharedProfileReadinessConfig,
) -> dict[str, float]:
    model.eval()
    logits_all = []
    targets_all = []
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), config.batch_size):
            batch = indices[start : start + config.batch_size]
            features, targets, observed = _batch_tensors(
                sources, batch, config.device
            )
            logits = model(features, inverse)
            losses.append(float(masked_binary_cross_entropy(logits, targets, observed)))
            logits_all.append(logits[observed].cpu().numpy())
            targets_all.append(targets[observed].cpu().numpy())
    scores = np.concatenate(logits_all)
    labels = np.concatenate(targets_all)
    probabilities = 1.0 / (1.0 + np.exp(-scores))
    return {
        "auc": float(binary_auc(labels.astype(np.float32), scores.astype(np.float64))),
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
        "loss": float(np.mean(losses)),
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


__all__ = [
    "SharedProfileReadinessConfig",
    "adjudicate_shared_profile_readiness",
    "inverse_player",
    "load_e85_sources",
    "make_shared_profile_model",
    "measure_shared_profile_contract",
    "prepare_e85_sources",
    "relation_players",
    "serializable_config",
    "train_shared_profile_matrix",
    "validate_e85_sources",
]
