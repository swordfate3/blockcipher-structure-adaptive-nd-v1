from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_attribution import (
    E65_PREFIX_RIDGE_AUC,
)
from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_readiness import (
    ProfileOperatorReadinessConfig,
)


SOURCE_RUN_ID = (
    "i2_present_r4_prefix_guided_profile_operator_attribution_seed0_20260718"
)
SOURCE_DECISION = "innovation2_present_profile_operator_neural_gain_attributed"


@dataclass(frozen=True)
class ProfileOperatorReplicationConfig:
    run_id: str
    epochs: int = 30
    batch_size: int = 8
    hidden_dim: int = 32
    steps: int = 2
    seed: int = 1
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.epochs != 30
            or self.batch_size != 8
            or self.hidden_dim != 32
            or self.steps != 2
            or self.seed != 1
            or self.dropout != 0.10
            or self.learning_rate != 1e-3
            or self.weight_decay != 1e-4
            or self.device != "cpu"
        ):
            raise ValueError("E68 replication protocol is frozen")

    def as_training_config(self) -> ProfileOperatorReadinessConfig:
        return ProfileOperatorReadinessConfig(
            run_id=self.run_id,
            mode="smoke",
            epochs=self.epochs,
            batch_size=self.batch_size,
            hidden_dim=self.hidden_dim,
            steps=self.steps,
            seed=self.seed,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            device=self.device,
        )


def load_seed0_source(source_root: Path) -> dict[str, Any]:
    gate = json.loads((source_root / "gate.json").read_text(encoding="utf-8"))
    results = [
        json.loads(line)
        for line in (source_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "gate": gate,
        "results": results,
        "hashes": {
            name: _sha256(source_root / name)
            for name in ("gate.json", "results.jsonl", "metadata.json")
        },
    }


def validate_seed0_source(source: dict[str, Any]) -> dict[str, bool]:
    rows = source["results"]
    return {
        "seed0_run_id_matches": source["gate"].get("run_id") == SOURCE_RUN_ID,
        "seed0_decision_matches": source["gate"].get("decision") == SOURCE_DECISION,
        "seed0_status_pass": source["gate"].get("status") == "pass",
        "seed0_candidate_checks_pass": all(
            source["gate"].get("candidate_checks", {}).values()
        ),
        "seed0_relation_checks_pass": all(
            source["gate"].get("relation_checks", {}).values()
        ),
        "seed0_method_gain_checks_pass": all(
            source["gate"].get("method_gain_checks", {}).values()
        ),
        "seed0_three_modes_present": {row.get("relation_mode") for row in rows}
        == {"independent", "true", "corrupted"},
        "seed0_rows_completed_30_epochs": len(rows) == 3
        and all(row.get("epochs_completed") == 30 for row in rows),
        "seed0_hashes_present": all(len(value) == 64 for value in source["hashes"].values()),
    }


def adjudicate_profile_operator_replication(
    config: ProfileOperatorReplicationConfig,
    seed0_checks: dict[str, bool],
    profile_checks: dict[str, bool | float],
    contract: dict[str, Any],
    seed0_source: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    seed0 = {row["relation_mode"]: row for row in seed0_source["results"]}
    seed1_rows = training["trained_rows"]
    seed1 = {row["relation_mode"]: row for row in seed1_rows}
    boolean_profile_checks = {
        key: value for key, value in profile_checks.items() if isinstance(value, bool)
    }
    protocol_checks = {
        **seed0_checks,
        **boolean_profile_checks,
        "contract_output_shape_is_4x64": contract.get("output_shape") == [4, 64],
        "contract_masked_loss_matches": contract.get(
            "masked_loss_explicit_max_abs_error", math.inf
        )
        <= 1e-7,
        "contract_parameter_counts_match": bool(contract.get("parameter_counts_match")),
        "contract_cell_relabel_equivariant": contract.get(
            "cell_relabel_max_abs_error", math.inf
        )
        <= 1e-6,
        "contract_true_corrupted_differ": contract.get(
            "true_corrupted_logit_max_abs_difference", 0.0
        )
        >= 1e-6,
        "seed1_three_modes_present": set(seed1)
        == {"independent", "true", "corrupted"},
        "seed1_rows_completed_30_epochs": len(seed1_rows) == 3
        and all(row["epochs_completed"] == config.epochs for row in seed1_rows),
        "all_seed1_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in seed1_rows
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
    true1 = float(seed1.get("true", {}).get("validation_auc", 0.0))
    independent1 = float(seed1.get("independent", {}).get("validation_auc", 0.0))
    corrupted1 = float(seed1.get("corrupted", {}).get("validation_auc", 0.0))
    train1 = float(seed1.get("true", {}).get("train_auc", 0.0))
    seed1_checks = {
        "seed1_true_auc_at_least_0p78": true1 >= 0.78,
        "seed1_train_validation_gap_at_most_0p15": train1 - true1 <= 0.15,
        "seed1_true_minus_independent_at_least_0p03": true1 - independent1 >= 0.03,
        "seed1_true_minus_corrupted_at_least_0p03": true1 - corrupted1 >= 0.03,
        "seed1_true_minus_ridge_at_least_0p02": true1 - E65_PREFIX_RIDGE_AUC >= 0.02,
    }
    seed_metrics = {
        "seed0": _seed_metrics(seed0),
        "seed1": _seed_metrics(seed1),
    }
    mean_metrics = {
        key: float((seed_metrics["seed0"][key] + seed_metrics["seed1"][key]) / 2.0)
        for key in seed_metrics["seed0"]
    }
    joint_checks = {
        "both_seeds_true_auc_at_least_0p78": all(
            seed_metrics[seed]["true_auc"] >= 0.78 for seed in ("seed0", "seed1")
        ),
        "mean_true_minus_independent_at_least_0p03": mean_metrics[
            "true_minus_independent"
        ]
        >= 0.03,
        "mean_true_minus_corrupted_at_least_0p03": mean_metrics[
            "true_minus_corrupted"
        ]
        >= 0.03,
        "mean_true_minus_ridge_at_least_0p02": mean_metrics["true_minus_ridge"]
        >= 0.02,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_profile_operator_replication_protocol_invalid"
        action = "repair seed0 source, profile source, contract, or seed1 protocol"
    elif not all(seed1_checks.values()) or not all(joint_checks.values()):
        status = "hold"
        decision = "innovation2_present_profile_operator_seed_not_replicated"
        action = "retain seed0-only evidence and stop this operator"
    else:
        status = "pass"
        decision = "innovation2_present_profile_operator_two_seed_confirmed"
        action = "audit multi-bit linear-mask profile width before adding a mask-query decoder"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "seed1_checks": seed1_checks,
        "joint_checks": joint_checks,
        "metrics": {
            "e65_prefix_ridge_validation_auc": E65_PREFIX_RIDGE_AUC,
            "seed_metrics": seed_metrics,
            "mean_metrics": mean_metrics,
            "seed1_rows": seed1_rows,
            "contract": contract,
        },
        "claim_scope": (
            "two-seed local confirmation of a prefix-guided 64-node profile operator "
            "on PRESENT-80 r4 strict unit-balance labels; no high-round, new-attack, "
            "remote-scale, or SOTA claim"
        ),
        "next_action": {"action": action, "remote_scale": False},
    }


def _seed_metrics(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    true_auc = float(rows["true"]["validation_auc"])
    return {
        "true_auc": true_auc,
        "independent_auc": float(rows["independent"]["validation_auc"]),
        "corrupted_auc": float(rows["corrupted"]["validation_auc"]),
        "true_minus_independent": true_auc
        - float(rows["independent"]["validation_auc"]),
        "true_minus_corrupted": true_auc
        - float(rows["corrupted"]["validation_auc"]),
        "true_minus_ridge": true_auc - E65_PREFIX_RIDGE_AUC,
        "train_validation_gap": float(rows["true"]["train_auc"]) - true_auc,
    }


def serializable_config(config: ProfileOperatorReplicationConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ProfileOperatorReplicationConfig",
    "adjudicate_profile_operator_replication",
    "load_seed0_source",
    "serializable_config",
    "validate_seed0_source",
]
