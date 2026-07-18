from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    fit_train_only_ridge,
)
from blockcipher_nd.tasks.innovation2.rectangle80_r3_only_profile_operator_readiness import (
    rectangle_model_player,
)
from blockcipher_nd.training.metrics import binary_auc


E90_RUN_ID = "i2_rectangle80_r4_r3_only_profile_operator_attribution_seed0_20260719"
E90_DECISION = "innovation2_rectangle80_r3_only_topology_not_attributed"
E89_TRUE_RIDGE_AUC = 0.8246824848549261
RIDGE_LAMBDA = 1e-3


@dataclass(frozen=True)
class Rectangle80RowTypedAuditConfig:
    run_id: str
    ridge_lambda: float = RIDGE_LAMBDA

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.ridge_lambda != RIDGE_LAMBDA:
            raise ValueError("E91 audit protocol is frozen")


def load_e90_source(root: Path) -> dict[str, Any]:
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


def validate_e90_source(source: dict[str, Any]) -> dict[str, bool]:
    gate = source["gate"]
    rows = source["rows"]
    failed_relation = {
        name for name, passed in gate.get("relation_checks", {}).items() if not passed
    }
    return {
        "e90_run_id_matches": gate.get("run_id") == E90_RUN_ID,
        "e90_status_hold": gate.get("status") == "hold",
        "e90_decision_matches": gate.get("decision") == E90_DECISION,
        "e90_protocol_checks_pass": bool(gate.get("protocol_checks"))
        and all(gate["protocol_checks"].values()),
        "e90_candidate_checks_pass": bool(gate.get("candidate_checks"))
        and all(gate["candidate_checks"].values()),
        "e90_only_corrupted_relation_failed": failed_relation
        == {"true_minus_corrupted_at_least_0p03"},
        "e90_three_rows_completed_30_epochs": len(rows) == 3
        and {row.get("relation_mode") for row in rows}
        == {"independent", "true", "corrupted"}
        and all(row.get("epochs_completed") == 30 for row in rows),
        "e90_hashes_present": all(
            len(value) == 64 for value in source["hashes"].values()
        ),
    }


def build_row_typed_matrices(sources: dict[str, Any]) -> dict[str, np.ndarray]:
    rows = sources["matched_rows"]
    prefix = np.asarray(sources["prefix_features"], dtype=np.float64)[:, :, 26:39]
    true_player = rectangle_model_player()
    corrupted_player = topology_players(true_player[None, :], "corrupted")[0]
    local_rows: list[np.ndarray] = []
    cell_rows: list[np.ndarray] = []
    true_predecessors: list[np.ndarray] = []
    corrupted_predecessors: list[np.ndarray] = []
    true_types: list[int] = []
    wrong_types: list[int] = []
    true_inverse = np.empty(64, dtype=np.int64)
    true_inverse[true_player] = np.arange(64)
    corrupted_inverse = np.empty(64, dtype=np.int64)
    corrupted_inverse[corrupted_player] = np.arange(64)
    for row in rows:
        structure = row["structure_index"]
        node = row["output_bit"]
        cell = node // 4
        lane = node % 4
        local_rows.append(prefix[structure, node])
        cell_rows.append(prefix[structure, 4 * cell : 4 * (cell + 1)].mean(axis=0))
        true_predecessors.append(prefix[structure, true_inverse[node]])
        corrupted_predecessors.append(prefix[structure, corrupted_inverse[node]])
        true_types.append(lane)
        wrong_types.append((lane + 1 + structure % 3) % 4)
    local = np.asarray(local_rows)
    cell_context = np.asarray(cell_rows)
    true_predecessor = np.asarray(true_predecessors)
    corrupted_predecessor = np.asarray(corrupted_predecessors)
    return {
        "untyped_true": np.concatenate(
            (local, cell_context, true_predecessor), axis=1
        ),
        "untyped_corrupted": np.concatenate(
            (local, cell_context, corrupted_predecessor), axis=1
        ),
        "typed_true": _typed_matrix(
            cell_context, local, true_predecessor, np.asarray(true_types)
        ),
        "typed_corrupted": _typed_matrix(
            cell_context, local, corrupted_predecessor, np.asarray(true_types)
        ),
        "wrong_row_typed_true": _typed_matrix(
            cell_context, local, true_predecessor, np.asarray(wrong_types)
        ),
    }


def _typed_matrix(
    cell_context: np.ndarray,
    local: np.ndarray,
    predecessor: np.ndarray,
    row_types: np.ndarray,
) -> np.ndarray:
    typed_local = np.zeros((len(local), 4, local.shape[1]), dtype=np.float64)
    typed_predecessor = np.zeros_like(typed_local)
    indices = np.arange(len(local))
    typed_local[indices, row_types] = local
    typed_predecessor[indices, row_types] = predecessor
    return np.concatenate(
        (
            cell_context,
            typed_local.reshape(len(local), -1),
            typed_predecessor.reshape(len(local), -1),
        ),
        axis=1,
    )


def evaluate_row_typed_ridges(
    config: Rectangle80RowTypedAuditConfig,
    sources: dict[str, Any],
) -> dict[str, Any]:
    rows = sources["matched_rows"]
    matrices = build_row_typed_matrices(sources)
    labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
    train = np.asarray([row["split"] == "train" for row in rows])
    validation = ~train
    reports: dict[str, Any] = {}
    for name, matrix in matrices.items():
        fitted = fit_train_only_ridge(
            matrix[train], labels[train], matrix[validation], config.ridge_lambda
        )
        reports[name] = {
            "feature_count": int(matrix.shape[1]),
            "train_auc": _safe_auc(labels[train], fitted["train_scores"]),
            "validation_auc": _safe_auc(
                labels[validation], fitted["validation_scores"]
            ),
            "ridge_lambda": config.ridge_lambda,
            "train_standardization_only": True,
            "finite": bool(np.isfinite(matrix).all()),
        }
    return reports


def adjudicate_row_typed_audit(
    config: Rectangle80RowTypedAuditConfig,
    profile_checks: dict[str, bool],
    model_order_checks: dict[str, bool],
    e90_checks: dict[str, bool],
    reports: dict[str, Any],
) -> dict[str, Any]:
    untyped_true = float(reports["untyped_true"]["validation_auc"])
    typed_true = float(reports["typed_true"]["validation_auc"])
    typed_corrupted = float(reports["typed_corrupted"]["validation_auc"])
    wrong_typed_true = float(reports["wrong_row_typed_true"]["validation_auc"])
    protocol_checks = {
        **profile_checks,
        **model_order_checks,
        **e90_checks,
        "five_frozen_reports_present": set(reports)
        == {
            "untyped_true",
            "untyped_corrupted",
            "typed_true",
            "typed_corrupted",
            "wrong_row_typed_true",
        },
        "untyped_features_are_39": all(
            reports[name]["feature_count"] == 39
            for name in ("untyped_true", "untyped_corrupted")
        ),
        "typed_features_are_117": all(
            reports[name]["feature_count"] == 117
            for name in (
                "typed_true",
                "typed_corrupted",
                "wrong_row_typed_true",
            )
        ),
        "all_ridges_train_standardized": all(
            report["train_standardization_only"] for report in reports.values()
        ),
        "all_matrices_finite": all(report["finite"] for report in reports.values()),
        "all_auc_finite": all(
            math.isfinite(float(report[key]))
            for report in reports.values()
            for key in ("train_auc", "validation_auc")
        ),
        "e89_untyped_true_ridge_reproduced": abs(
            untyped_true - E89_TRUE_RIDGE_AUC
        )
        <= 1e-12,
    }
    mechanism_checks = {
        "typed_true_minus_untyped_true_at_least_0p01": typed_true
        - untyped_true
        >= 0.01,
        "typed_true_minus_typed_corrupted_at_least_0p03": typed_true
        - typed_corrupted
        >= 0.03,
        "typed_true_minus_wrong_row_typed_at_least_0p01": typed_true
        - wrong_typed_true
        >= 0.01,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_rectangle80_row_typed_representation_protocol_invalid"
        action = "repair E88/E90 replay or row-typed feature construction"
    elif not all(mechanism_checks.values()):
        status = "hold"
        decision = "innovation2_rectangle80_row_typed_representation_not_ready"
        action = "stop RECTANGLE architecture enumeration and retain E88-E90 evidence"
    else:
        status = "pass"
        decision = "innovation2_rectangle80_row_typed_representation_ready"
        action = "design a capacity-matched Row-Typed Shift Operator readiness"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "mechanism_checks": mechanism_checks,
        "metrics": {
            "ridges": reports,
            "typed_true_minus_untyped_true": typed_true - untyped_true,
            "typed_true_minus_typed_corrupted": typed_true - typed_corrupted,
            "typed_true_minus_wrong_row_typed": typed_true - wrong_typed_true,
        },
        "claim_scope": (
            "no-training RECTANGLE-80 row-typed r3 representation audit on E88 "
            "strict labels; no neural gain, seven-round reproduction, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "neural_readiness_allowed": status == "pass",
            "training_performed": False,
            "remote_scale": False,
        },
    }


def result_rows(
    config: Rectangle80RowTypedAuditConfig, gate: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": config.run_id,
            "task": "innovation2_rectangle80_row_typed_shift_representation_audit",
            "family": "deterministic_ridge",
            "variant": name,
            "status": gate["status"],
            "decision": gate["decision"],
            "training_performed": False,
            **report,
        }
        for name, report in gate["metrics"]["ridges"].items()
    ]


def serializable_config(config: Rectangle80RowTypedAuditConfig) -> dict[str, Any]:
    return asdict(config)


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
    "Rectangle80RowTypedAuditConfig",
    "adjudicate_row_typed_audit",
    "build_row_typed_matrices",
    "evaluate_row_typed_ridges",
    "load_e90_source",
    "result_rows",
    "serializable_config",
    "validate_e90_source",
]
