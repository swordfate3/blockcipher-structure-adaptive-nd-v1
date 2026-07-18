from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.tasks.innovation2.gift64_r3_only_profile_operator_readiness import (
    Gift64R3ProfileReadinessConfig,
    gift_player,
    load_gift_profile_sources,
    make_gift_r3_model,
    r3_only_sources,
    validate_gift_profile_sources,
)
from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_readiness import (
    _copy_parameters,
    _evaluate,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    fit_train_only_ridge,
)
from blockcipher_nd.training.metrics import binary_auc


E76_RUN_ID = "i2_gift64_r4_r3_only_profile_operator_readiness_seed0_20260719"
E76_DECISION = "innovation2_gift64_r3_only_prefix_not_sufficient"
RIDGE_LAMBDA = 1e-3


@dataclass(frozen=True)
class Gift64TopologyInteractionConfig:
    run_id: str
    ridge_lambda: float = RIDGE_LAMBDA
    corruption_shifts: tuple[int, ...] = (1, 2, 3)
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.ridge_lambda != RIDGE_LAMBDA
            or self.corruption_shifts != (1, 2, 3)
            or self.device != "cpu"
        ):
            raise ValueError("E77 protocol is frozen")


def load_e76_source(root: Path) -> dict[str, Any]:
    gate_path = root / "gate.json"
    results_path = root / "results.jsonl"
    checkpoint_path = root / "checkpoints" / "gift_r3_profile_true_seed0.pt"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "gate": gate,
        "rows": rows,
        "checkpoint_path": checkpoint_path,
        "hashes": {
            "gate": _sha256(gate_path),
            "results": _sha256(results_path),
            "checkpoint": _sha256(checkpoint_path),
        },
    }


def validate_e76_source(source: dict[str, Any]) -> dict[str, bool]:
    gate = source["gate"]
    rows = source["rows"]
    deterministic = gate.get("deterministic_checks", {})
    readiness = gate.get("readiness_checks", {})
    return {
        "e76_run_id_matches": gate.get("run_id") == E76_RUN_ID,
        "e76_status_is_hold": gate.get("status") == "hold",
        "e76_decision_matches": gate.get("decision") == E76_DECISION,
        "e76_protocol_checks_pass": bool(gate.get("protocol_checks"))
        and all(gate["protocol_checks"].values()),
        "e76_neural_readiness_checks_pass": bool(readiness)
        and all(readiness.values()),
        "e76_only_r3_absolute_ridge_gate_failed": deterministic
        == {
            "r3_ridge_auc_at_least_0p60": False,
            "r3_minus_full_ridge_at_least_minus_0p03": True,
        },
        "e76_three_rows_present": {row.get("relation_mode") for row in rows}
        == {"independent", "true", "corrupted"},
        "e76_true_checkpoint_exists": source["checkpoint_path"].is_file(),
        "e76_hashes_present": all(
            len(value) == 64 for value in source["hashes"].values()
        ),
    }


def gift_player_variants() -> dict[str, np.ndarray]:
    true = gift_player()
    variants = {"true": true}
    for shift in (1, 2, 3):
        destination_rotation = np.asarray(
            [
                4 * ((node // 4 + shift) % 16) + node % 4
                for node in range(64)
            ],
            dtype=np.int64,
        )
        variants[f"corrupted_shift{shift}"] = destination_rotation[true]
    return variants


def topology_expanded_ridges(
    config: Gift64TopologyInteractionConfig,
    sources: dict[str, Any],
) -> dict[str, Any]:
    rows = sources["matched_rows"]
    prefix = sources["prefix_features"][:, :, 26:39]
    labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
    train = np.asarray([row["split"] == "train" for row in rows])
    validation = ~train
    local = np.asarray(
        [prefix[row["structure_index"], row["output_bit"]] for row in rows],
        dtype=np.float64,
    )
    cell = np.asarray(
        [
            prefix[
                row["structure_index"],
                4 * (row["output_bit"] // 4) : 4 * (row["output_bit"] // 4 + 1),
            ].mean(axis=0)
            for row in rows
        ],
        dtype=np.float64,
    )
    matrices = {"local": local}
    for name, player in gift_player_variants().items():
        inverse = np.empty(64, dtype=np.int64)
        inverse[player] = np.arange(64)
        predecessor = np.asarray(
            [
                prefix[row["structure_index"], inverse[row["output_bit"]]]
                for row in rows
            ],
            dtype=np.float64,
        )
        matrices[name] = np.concatenate((local, cell, predecessor), axis=1)
    reports: dict[str, Any] = {}
    for name, matrix in matrices.items():
        fitted = fit_train_only_ridge(
            matrix[train],
            labels[train],
            matrix[validation],
            config.ridge_lambda,
        )
        reports[name] = {
            "feature_count": int(matrix.shape[1]),
            "train_auc": _safe_auc(labels[train], fitted["train_scores"]),
            "validation_auc": _safe_auc(
                labels[validation], fitted["validation_scores"]
            ),
            "ridge_lambda": config.ridge_lambda,
            "train_standardization_only": True,
        }
    return reports


def checkpoint_topology_counterfactuals(
    config: Gift64TopologyInteractionConfig,
    sources: dict[str, Any],
    e76_source: dict[str, Any],
) -> dict[str, Any]:
    training_config = Gift64R3ProfileReadinessConfig(run_id=E76_RUN_ID)
    source_model = make_gift_r3_model(training_config, "true", dropout=0.0)
    state = torch.load(
        e76_source["checkpoint_path"], map_location="cpu", weights_only=True
    )
    source_model.load_state_dict(state)
    validation_indices = sorted(
        {
            row["structure_index"]
            for row in sources["matched_rows"]
            if row["split"] == "validation"
        }
    )
    reports: dict[str, Any] = {}
    for name, player in gift_player_variants().items():
        model = make_gift_r3_model(
            training_config, "true", player=player, dropout=0.0
        )
        _copy_parameters(source_model, model)
        reports[name] = _evaluate(model, sources, validation_indices, training_config)
    return reports


def adjudicate_topology_interaction(
    config: Gift64TopologyInteractionConfig,
    profile_checks: dict[str, bool],
    e76_checks: dict[str, bool],
    ridges: dict[str, Any],
    counterfactuals: dict[str, Any],
    e76_source: dict[str, Any],
) -> dict[str, Any]:
    players = gift_player_variants()
    true_ridge = float(ridges["true"]["validation_auc"])
    local_ridge = float(ridges["local"]["validation_auc"])
    corrupted_ridge = max(
        float(ridges[f"corrupted_shift{shift}"]["validation_auc"])
        for shift in config.corruption_shifts
    )
    true_inference = float(counterfactuals["true"]["auc"])
    corrupted_inference = max(
        float(counterfactuals[f"corrupted_shift{shift}"]["auc"])
        for shift in config.corruption_shifts
    )
    recorded_true = next(
        float(row["validation_auc"])
        for row in e76_source["rows"]
        if row["relation_mode"] == "true"
    )
    protocol_checks = {
        **profile_checks,
        **e76_checks,
        "four_players_present": set(players)
        == {"true", "corrupted_shift1", "corrupted_shift2", "corrupted_shift3"},
        "all_players_are_permutations": all(
            np.array_equal(np.sort(player), np.arange(64))
            for player in players.values()
        ),
        "all_corrupted_players_distinct": len(
            {player.tobytes() for player in players.values()}
        )
        == 4,
        "all_ridges_train_standardized": all(
            report["train_standardization_only"] for report in ridges.values()
        ),
        "all_counterfactual_metrics_finite": all(
            np.isfinite(float(report[key]))
            for report in counterfactuals.values()
            for key in ("auc", "accuracy", "loss")
        ),
        "correct_checkpoint_auc_reproduces_e76": abs(true_inference - recorded_true)
        <= 1e-12,
    }
    deterministic_checks = {
        "true_topology_ridge_auc_at_least_0p60": true_ridge >= 0.60,
        "true_minus_local_ridge_at_least_0p03": true_ridge - local_ridge >= 0.03,
        "true_minus_max_corrupted_ridge_at_least_0p03": true_ridge
        - corrupted_ridge
        >= 0.03,
    }
    checkpoint_checks = {
        "true_minus_max_corrupted_inference_at_least_0p03": true_inference
        - corrupted_inference
        >= 0.03,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_gift64_topology_interaction_protocol_invalid"
        action = "repair E75/E76 replay, player variants, ridge, or checkpoint loading"
    elif not all(deterministic_checks.values()) or not all(checkpoint_checks.values()):
        status = "hold"
        decision = "innovation2_gift64_topology_interaction_not_confirmed"
        action = "close GIFT r3-only and stop architecture or corruption search"
    else:
        status = "pass"
        decision = "innovation2_gift64_topology_interaction_gate_repaired"
        action = "prepare a separate frozen 30-epoch seed0 attribution plan"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "deterministic_checks": deterministic_checks,
        "checkpoint_checks": checkpoint_checks,
        "metrics": {
            "ridges": ridges,
            "checkpoint_counterfactuals": counterfactuals,
            "e76_recorded_true_auc": recorded_true,
            "true_minus_local_ridge": true_ridge - local_ridge,
            "true_minus_max_corrupted_ridge": true_ridge - corrupted_ridge,
            "true_minus_max_corrupted_inference": true_inference
            - corrupted_inference,
        },
        "claim_scope": (
            "no-training GIFT-64 r4 attribution audit aligning deterministic "
            "local/cell/P-layer information and frozen-checkpoint topology "
            "counterfactuals; no formal neural gain, high-round, cross-cipher, "
            "attack, remote-scale, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "formal_plan_allowed": status == "pass",
            "training_performed": False,
            "remote_scale": False,
        },
    }


def result_rows(
    config: Gift64TopologyInteractionConfig,
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "run_id": config.run_id,
        "task": "innovation2_gift64_topology_interaction_readjudication",
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": False,
    }
    rows = [
        {
            **common,
            "family": "deterministic_ridge",
            "variant": name,
            **report,
        }
        for name, report in gate["metrics"]["ridges"].items()
    ]
    rows.extend(
        {
            **common,
            "family": "frozen_checkpoint_inference",
            "variant": name,
            **report,
        }
        for name, report in gate["metrics"]["checkpoint_counterfactuals"].items()
    )
    return rows


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


def serializable_config(config: Gift64TopologyInteractionConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "Gift64TopologyInteractionConfig",
    "adjudicate_topology_interaction",
    "checkpoint_topology_counterfactuals",
    "gift_player_variants",
    "load_e76_source",
    "load_gift_profile_sources",
    "result_rows",
    "serializable_config",
    "topology_expanded_ridges",
    "validate_e76_source",
    "validate_gift_profile_sources",
]
