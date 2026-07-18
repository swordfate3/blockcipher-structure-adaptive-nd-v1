from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


E89_RUN_ID = "i2_rectangle80_r4_r3_only_profile_operator_readiness_seed0_20260719"
E89_DECISION = "innovation2_rectangle80_r3_only_profile_readiness_passed"
RIDGE_LAMBDA = 1e-3


@dataclass(frozen=True)
class Rectangle80R3AttributionConfig:
    run_id: str
    epochs: int = 30
    batch_size: int = 8
    hidden_dim: int = 32
    steps: int = 2
    seed: int = 0
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ridge_lambda: float = RIDGE_LAMBDA
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.epochs != 30
            or self.batch_size != 8
            or self.hidden_dim != 32
            or self.steps != 2
            or self.seed != 0
            or self.dropout != 0.10
            or self.learning_rate != 1e-3
            or self.weight_decay != 1e-4
            or self.ridge_lambda != RIDGE_LAMBDA
            or self.device != "cpu"
        ):
            raise ValueError("E90 formal seed0 protocol is frozen")

    @property
    def task_name(self) -> str:
        return "innovation2_rectangle80_r3_only_profile_attribution"

    @property
    def row_prefix(self) -> str:
        return "rectangle_r3_profile_attribution"


def load_e89_source(root: Path) -> dict[str, Any]:
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


def validate_e89_source(source: dict[str, Any]) -> dict[str, bool]:
    gate = source["gate"]
    rows = source["rows"]
    return {
        "e89_run_id_matches": gate.get("run_id") == E89_RUN_ID,
        "e89_status_pass": gate.get("status") == "pass",
        "e89_decision_matches": gate.get("decision") == E89_DECISION,
        "e89_protocol_checks_pass": bool(gate.get("protocol_checks"))
        and all(gate["protocol_checks"].values()),
        "e89_deterministic_checks_pass": bool(gate.get("deterministic_checks"))
        and all(gate["deterministic_checks"].values()),
        "e89_readiness_checks_pass": bool(gate.get("readiness_checks"))
        and all(gate["readiness_checks"].values()),
        "e89_three_rows_present": {row.get("relation_mode") for row in rows}
        == {"independent", "true", "corrupted"},
        "e89_rows_completed_two_epochs": len(rows) == 3
        and all(row.get("epochs_completed") == 2 for row in rows),
        "e89_true_fair_ridge_present": float(
            gate.get("metrics", {})
            .get("ridges", {})
            .get("true", {})
            .get("validation_auc", 0.0)
        )
        >= 0.60,
        "e89_hashes_present": all(
            len(value) == 64 for value in source["hashes"].values()
        ),
    }


def adjudicate_rectangle_r3_attribution(
    config: Rectangle80R3AttributionConfig,
    profile_checks: dict[str, bool],
    model_order_checks: dict[str, bool],
    e89_checks: dict[str, bool],
    contract: dict[str, Any],
    training: dict[str, Any],
    e89_source: dict[str, Any],
) -> dict[str, Any]:
    rows = training["trained_rows"]
    by_mode = {row["relation_mode"]: row for row in rows}
    true_auc = float(by_mode.get("true", {}).get("validation_auc", 0.0))
    independent_auc = float(
        by_mode.get("independent", {}).get("validation_auc", 0.0)
    )
    corrupted_auc = float(by_mode.get("corrupted", {}).get("validation_auc", 0.0))
    true_train_auc = float(by_mode.get("true", {}).get("train_auc", 0.0))
    fair_ridge_auc = float(
        e89_source["gate"]["metrics"]["ridges"]["true"]["validation_auc"]
    )
    protocol_checks = {
        **profile_checks,
        **model_order_checks,
        **e89_checks,
        "output_shape_is_4x64": contract["output_shape"] == [4, 64],
        "input_dim_is_13": contract["input_dim"] == 13,
        "masked_loss_matches_explicit": contract[
            "masked_loss_explicit_max_abs_error"
        ]
        <= 1e-7,
        "parameter_counts_match": contract["parameter_counts_match"],
        "parameter_count_is_4795": set(contract["parameter_counts"].values())
        == {4_795},
        "topology_changes_logits": contract["topology_logit_max_abs_difference"]
        >= 1e-6,
        "cell_relabel_equivariant": contract["cell_relabel_max_abs_error"] <= 1e-6,
        "forbidden_named_state_absent": contract["forbidden_named_state_absent"],
        "contract_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "all_three_rows_present": set(by_mode)
        == {"independent", "true", "corrupted"},
        "all_rows_completed_30_epochs": len(rows) == 3
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
    candidate_checks = {
        "true_auc_at_least_0p80": true_auc >= 0.80,
        "true_train_validation_gap_at_most_0p15": true_train_auc - true_auc
        <= 0.15,
        "true_minus_e89_fair_ridge_at_least_0p03": true_auc - fair_ridge_auc
        >= 0.03,
    }
    relation_checks = {
        "true_minus_independent_at_least_0p03": true_auc - independent_auc >= 0.03,
        "true_minus_corrupted_at_least_0p03": true_auc - corrupted_auc >= 0.03,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_rectangle80_r3_only_attribution_protocol_invalid"
        action = "repair E88/E89 replay, cell-major topology, or 30-epoch protocol"
    elif not all(candidate_checks.values()):
        status = "hold"
        decision = "innovation2_rectangle80_r3_only_quality_not_confirmed"
        action = "retain E89 evidence and close the formal RECTANGLE neural route"
    elif not all(relation_checks.values()):
        status = "hold"
        decision = "innovation2_rectangle80_r3_only_topology_not_attributed"
        action = "retain E89 evidence and close the formal RECTANGLE neural route"
    else:
        status = "pass"
        decision = "innovation2_rectangle80_r3_only_neural_gain_attributed"
        action = "run the identical 30-epoch matrix with seed1"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "candidate_checks": candidate_checks,
        "relation_checks": relation_checks,
        "metrics": {
            "rows": rows,
            "e89_true_topology_ridge_auc": fair_ridge_auc,
            "true_minus_e89_fair_ridge": true_auc - fair_ridge_auc,
            "true_minus_independent": true_auc - independent_auc,
            "true_minus_corrupted": true_auc - corrupted_auc,
            "true_train_validation_gap": true_train_auc - true_auc,
            "contract": contract,
        },
        "claim_scope": (
            "30-epoch local seed0 attribution of a RECTANGLE-80 r4 r3-only "
            "profile operator on strict unit-balance labels; no two-seed, "
            "seven-round reproduction, transfer, attack, remote-scale, or SOTA"
        ),
        "next_action": {
            "action": action,
            "seed1": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: Rectangle80R3AttributionConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "Rectangle80R3AttributionConfig",
    "adjudicate_rectangle_r3_attribution",
    "load_e89_source",
    "serializable_config",
    "validate_e89_source",
]
