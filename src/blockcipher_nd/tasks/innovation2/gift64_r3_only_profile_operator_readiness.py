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

from blockcipher_nd.ciphers.spn.gift import _GIFT64_PERM
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
    masked_binary_cross_entropy,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    fit_train_only_ridge,
)
from blockcipher_nd.training.metrics import binary_auc


SOURCE_RUN_ID = "i2_gift64_r4_unit_balance_profile_192_structures_20260719"
SOURCE_DECISION = "innovation2_gift64_unit_balance_profile_expansion_ready"
RELATION_MODES = ("independent", "true", "corrupted")
R3_SLICE = slice(26, 39)
RIDGE_LAMBDA = 1e-3
EXPECTED_PARAMETER_COUNT = 4_795


@dataclass(frozen=True)
class Gift64R3ProfileReadinessConfig:
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
            raise ValueError("E76 readiness protocol is frozen")


def load_gift_profile_sources(root: Path) -> dict[str, Any]:
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


def validate_gift_profile_sources(sources: dict[str, Any]) -> dict[str, bool]:
    rows = sources["matched_rows"]
    targets = sources["profile_targets"]
    observed = sources["profile_observed"]
    reconstructed_targets = np.full((192, 64), -1, dtype=np.int8)
    reconstructed_observed = np.zeros((192, 64), dtype=np.bool_)
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
    return {
        "profile_run_id_matches": sources["profile_gate"].get("run_id")
        == SOURCE_RUN_ID,
        "profile_decision_matches": sources["profile_gate"].get("decision")
        == SOURCE_DECISION,
        "profile_status_pass": sources["profile_gate"].get("status") == "pass",
        "profile_protocol_checks_pass": bool(protocol_checks)
        and all(protocol_checks.values()),
        "metadata_task_matches": sources["profile_metadata"].get("task")
        == "innovation2_gift64_unit_balance_profile_expansion",
        "structure_count_is_192": len(sources["structures"]) == 192,
        "target_shape_is_192x64": targets.shape == (192, 64),
        "observed_shape_is_192x64": observed.shape == (192, 64),
        "prefix_shape_is_192x64x39": sources["prefix_features"].shape
        == (192, 64, 39),
        "prefix_features_finite": bool(np.isfinite(sources["prefix_features"]).all()),
        "matched_rows_are_620": len(rows) == 620,
        "observed_edges_are_620": int(np.sum(observed)) == 620,
        "matched_edges_unique": int(np.sum(reconstructed_observed)) == len(rows),
        "targets_replay_matched_csv": np.array_equal(
            reconstructed_targets, targets
        ),
        "observed_replays_matched_csv": np.array_equal(
            reconstructed_observed, observed
        ),
        "observed_targets_binary": bool(np.isin(targets[observed], (0, 1)).all()),
        "unobserved_targets_minus_one": bool(np.all(targets[~observed] == -1)),
        "train_structures_are_110": len(train_structures) == 110,
        "validation_structures_are_33": len(validation_structures) == 33,
        "train_validation_structures_disjoint": train_structures.isdisjoint(
            validation_structures
        ),
        "source_hashes_present": all(
            len(value) == 64 for value in sources["source_hashes"].values()
        ),
    }


def r3_only_sources(sources: dict[str, Any]) -> dict[str, Any]:
    prefix = np.asarray(sources["prefix_features"], dtype=np.float64)
    if prefix.shape != (192, 64, 39):
        raise ValueError("E76 requires the frozen 192x64x39 E75 prefix")
    return {**sources, "prefix_features": prefix[:, :, R3_SLICE].copy()}


def evaluate_deterministic_ridges(sources: dict[str, Any]) -> dict[str, Any]:
    rows = sources["matched_rows"]
    labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
    train = np.asarray([row["split"] == "train" for row in rows])
    validation = ~train
    full = np.asarray(
        [
            sources["prefix_features"][row["structure_index"], row["output_bit"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    reports: dict[str, Any] = {}
    for name, matrix in (("full39", full), ("r3_only", full[:, R3_SLICE])):
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


def gift_player() -> np.ndarray:
    return np.asarray(_GIFT64_PERM, dtype=np.int64)


def make_gift_r3_model(
    config: Gift64R3ProfileReadinessConfig,
    mode: str,
    *,
    player: np.ndarray | None = None,
    dropout: float | None = None,
) -> PresentBalanceProfileOperator:
    if mode not in RELATION_MODES:
        raise ValueError(f"unsupported E76 mode: {mode}")
    true_player = gift_player()
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


def measure_gift_r3_contract(
    config: Gift64R3ProfileReadinessConfig, sources: dict[str, Any]
) -> dict[str, Any]:
    models = {
        mode: make_gift_r3_model(config, mode, dropout=0.0)
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

    permutation = _cell_permutation()
    relabeled_player = _conjugate_player(gift_player(), permutation)
    relabeled = make_gift_r3_model(
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
    forbidden = ("certificate", "witness", "parity", "full_cube", "label")
    return {
        "output_shape": list(logits.shape),
        "input_dim": 13,
        "masked_loss_explicit_max_abs_error": float(
            torch.abs(loss - explicit).detach()
        ),
        "parameter_counts": parameter_counts,
        "parameter_counts_match": len(set(parameter_counts.values())) == 1,
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
    }


def train_gift_r3_matrix(
    config: Gift64R3ProfileReadinessConfig,
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
        model = make_gift_r3_model(config, mode).to(config.device)
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
                "row_id": f"gift_r3_profile_{mode}_seed{config.seed}",
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
                    "task": getattr(
                        config,
                        "task_name",
                        "innovation2_gift64_r3_only_profile_readiness",
                    ),
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
            raise RuntimeError("E76 training produced no checkpoint")
        best["epochs_completed"] = config.epochs
        torch.save(best_state, checkpoints / f"{best['row_id']}.pt")
        trained_rows.append(best)
    return {"trained_rows": trained_rows, "history_rows": history_rows}


def adjudicate_gift_r3_readiness(
    config: Gift64R3ProfileReadinessConfig,
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
    r3_auc = float(ridges["r3_only"]["validation_auc"])
    full_auc = float(ridges["full39"]["validation_auc"])
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
        "ridge_reports_train_standardized": all(
            report["train_standardization_only"] for report in ridges.values()
        ),
    }
    deterministic_checks = {
        "r3_ridge_auc_at_least_0p60": r3_auc >= 0.60,
        "r3_minus_full_ridge_at_least_minus_0p03": r3_auc - full_auc >= -0.03,
    }
    readiness_checks = {
        "true_auc_at_least_0p65": true_auc >= 0.65,
        "true_minus_independent_at_least_0p03": true_auc - independent_auc >= 0.03,
        "true_minus_corrupted_at_least_0p03": true_auc - corrupted_auc >= 0.03,
        "true_minus_r3_ridge_at_least_minus_0p03": true_auc - r3_auc >= -0.03,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_gift64_r3_only_profile_protocol_invalid"
        action = "repair E75 source, GIFT topology, fairness, or training protocol"
    elif not all(deterministic_checks.values()):
        status = "hold"
        decision = "innovation2_gift64_r3_only_prefix_not_sufficient"
        action = "test the full 39-d GIFT profile operator only if full-prefix ridge is strong"
    elif not all(readiness_checks.values()):
        status = "hold"
        decision = "innovation2_gift64_r3_only_profile_readiness_not_passed"
        action = "stop GIFT r3-only formal training and retain E75 label evidence"
    else:
        status = "pass"
        decision = "innovation2_gift64_r3_only_profile_readiness_passed"
        action = "run the frozen 30-epoch seed0 GIFT r3-only attribution matrix"
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
            "r3_minus_full_ridge": r3_auc - full_auc,
            "true_minus_independent": true_auc - independent_auc,
            "true_minus_corrupted": true_auc - corrupted_auc,
            "true_minus_r3_ridge": true_auc - r3_auc,
            "contract": contract,
        },
        "claim_scope": (
            "two-epoch local readiness for a GIFT-64 r4 r3-only prefix-guided "
            "profile operator on strict 8-bit-cube unit-balance labels; no formal "
            "neural gain, high-round, cross-cipher generalization, attack, or SOTA claim"
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


def serializable_config(config: Gift64R3ProfileReadinessConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "Gift64R3ProfileReadinessConfig",
    "adjudicate_gift_r3_readiness",
    "evaluate_deterministic_ridges",
    "gift_player",
    "load_gift_profile_sources",
    "make_gift_r3_model",
    "measure_gift_r3_contract",
    "r3_only_sources",
    "serializable_config",
    "train_gift_r3_matrix",
    "validate_gift_profile_sources",
]
