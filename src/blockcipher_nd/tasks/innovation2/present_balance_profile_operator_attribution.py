from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_readiness import (
    ProfileOperatorReadinessConfig,
)


SOURCE_RUN_ID = (
    "i2_present_r4_prefix_guided_profile_operator_readiness_seed0_20260718"
)
SOURCE_DECISION = "innovation2_present_profile_operator_readiness_passed"
FORMAL_EPOCHS = 30
FORMAL_BATCH_SIZE = 8
FORMAL_HIDDEN_DIM = 32
FORMAL_STEPS = 2
FORMAL_SEED = 0
E65_PREFIX_RIDGE_AUC = 0.7936111111111112


@dataclass(frozen=True)
class ProfileOperatorAttributionConfig:
    run_id: str
    epochs: int = FORMAL_EPOCHS
    batch_size: int = FORMAL_BATCH_SIZE
    hidden_dim: int = FORMAL_HIDDEN_DIM
    steps: int = FORMAL_STEPS
    seed: int = FORMAL_SEED
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.epochs != FORMAL_EPOCHS
            or self.batch_size != FORMAL_BATCH_SIZE
            or self.hidden_dim != FORMAL_HIDDEN_DIM
            or self.steps != FORMAL_STEPS
            or self.seed != FORMAL_SEED
            or self.dropout != 0.10
            or self.learning_rate != 1e-3
            or self.weight_decay != 1e-4
            or self.device != "cpu"
        ):
            raise ValueError("E67 formal protocol is frozen")

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


def load_e66_source(source_root: Path) -> dict[str, Any]:
    gate = json.loads((source_root / "gate.json").read_text(encoding="utf-8"))
    metadata = json.loads((source_root / "metadata.json").read_text(encoding="utf-8"))
    results = [
        json.loads(line)
        for line in (source_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "gate": gate,
        "metadata": metadata,
        "results": results,
        "hashes": {
            name: _sha256(source_root / name)
            for name in ("gate.json", "metadata.json", "results.jsonl")
        },
    }


def validate_e66_source(source: dict[str, Any]) -> dict[str, bool]:
    rows = source["results"]
    by_mode = {row.get("relation_mode"): row for row in rows}
    gate_rows = {
        row.get("relation_mode"): row
        for row in source["gate"].get("metrics", {}).get("rows", [])
    }
    return {
        "e66_run_id_matches": source["gate"].get("run_id") == SOURCE_RUN_ID,
        "e66_decision_matches": source["gate"].get("decision") == SOURCE_DECISION,
        "e66_status_pass": source["gate"].get("status") == "pass",
        "e66_protocol_checks_pass": all(
            source["gate"].get("protocol_checks", {}).values()
        ),
        "e66_optimization_checks_pass": all(
            source["gate"].get("optimization_checks", {}).values()
        ),
        "e66_three_modes_present": set(by_mode)
        == {"independent", "true", "corrupted"},
        "e66_results_match_gate": set(by_mode) == set(gate_rows)
        and all(
            math.isclose(
                float(by_mode[mode]["validation_auc"]),
                float(gate_rows[mode]["validation_auc"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for mode in by_mode
        ),
        "e66_hashes_present": all(len(value) == 64 for value in source["hashes"].values()),
    }


def adjudicate_profile_operator_attribution(
    config: ProfileOperatorAttributionConfig,
    source_checks: dict[str, bool],
    profile_source_checks: dict[str, bool | float],
    contract: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    rows = training["trained_rows"]
    by_mode = {row["relation_mode"]: row for row in rows}
    boolean_profile_checks = {
        key: value
        for key, value in profile_source_checks.items()
        if isinstance(value, bool)
    }
    true_auc = float(by_mode.get("true", {}).get("validation_auc", 0.0))
    independent_auc = float(
        by_mode.get("independent", {}).get("validation_auc", 0.0)
    )
    corrupted_auc = float(by_mode.get("corrupted", {}).get("validation_auc", 0.0))
    true_train_auc = float(by_mode.get("true", {}).get("train_auc", 0.0))
    protocol_checks = {
        **source_checks,
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
        "contract_finite": bool(contract.get("logits_finite"))
        and bool(contract.get("loss_finite"))
        and bool(contract.get("gradients_finite")),
        "all_three_modes_present": set(by_mode)
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
        "true_validation_auc_at_least_0p78": true_auc >= 0.78,
        "true_train_validation_gap_at_most_0p15": true_train_auc - true_auc <= 0.15,
    }
    relation_checks = {
        "true_minus_independent_at_least_0p03": true_auc - independent_auc >= 0.03,
        "true_minus_corrupted_at_least_0p03": true_auc - corrupted_auc >= 0.03,
    }
    method_gain_checks = {
        "true_minus_e65_prefix_ridge_at_least_0p02": true_auc
        - E65_PREFIX_RIDGE_AUC
        >= 0.02,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_profile_operator_attribution_protocol_invalid"
        action = "repair source, contract, metric, or formal training protocol"
    elif not all(candidate_checks.values()):
        status = "hold"
        decision = "innovation2_present_profile_operator_candidate_not_ready"
        action = "stop the profile operator without adding capacity or epochs"
    elif not all(relation_checks.values()):
        status = "hold"
        decision = "innovation2_present_profile_operator_relation_not_attributed"
        action = "retain prediction only as a control; stop topology attribution"
    elif not all(method_gain_checks.values()):
        status = "hold"
        decision = "innovation2_present_profile_operator_no_ridge_gain"
        action = "retain topology-attributed method evidence but do not run seed1"
    else:
        status = "pass"
        decision = "innovation2_present_profile_operator_neural_gain_attributed"
        action = "run the same 30-epoch matrix with seed1 for replication"
    metrics = {
        "e65_prefix_ridge_validation_auc": E65_PREFIX_RIDGE_AUC,
        "rows": rows,
        "true_minus_independent": true_auc - independent_auc,
        "true_minus_corrupted": true_auc - corrupted_auc,
        "true_minus_e65_prefix_ridge": true_auc - E65_PREFIX_RIDGE_AUC,
        "true_train_validation_gap": true_train_auc - true_auc,
        "contract": contract,
    }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "candidate_checks": candidate_checks,
        "relation_checks": relation_checks,
        "method_gain_checks": method_gain_checks,
        "metrics": metrics,
        "claim_scope": (
            "30-epoch local seed0 attribution of a prefix-guided 64-node profile "
            "operator on PRESENT-80 r4 strict unit-balance labels; no high-round, "
            "new-attack, remote-scale, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "seed1": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: ProfileOperatorAttributionConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ProfileOperatorAttributionConfig",
    "adjudicate_profile_operator_attribution",
    "load_e66_source",
    "serializable_config",
    "validate_e66_source",
]
