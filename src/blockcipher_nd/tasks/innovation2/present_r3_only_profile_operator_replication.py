from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SEED0_RUN_ID = "i2_present_r4_r3_only_profile_operator_attribution_seed0_20260718"
SEED0_DECISION = "innovation2_present_r3_only_neural_gain_attributed"
E68_SEED1_TRUE_AUC = 0.9613888888888888
E68_MEAN_TRUE_AUC = 0.9572222222222222


@dataclass(frozen=True)
class R3OnlyReplicationConfig:
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
            or self.device != "cpu"
        ):
            raise ValueError("E73 seed1 protocol is frozen")

    def as_training_config(self):
        from blockcipher_nd.tasks.innovation2.present_r3_only_profile_operator import (
            R3OnlyProfileConfig,
        )

        return R3OnlyProfileConfig(
            run_id=self.run_id,
            mode="formal",
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


def load_r3_seed0_source(root: Path) -> dict[str, Any]:
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


def validate_r3_seed0_source(source: dict[str, Any]) -> dict[str, bool]:
    return {
        "seed0_run_id_matches": source["gate"].get("run_id") == SEED0_RUN_ID,
        "seed0_decision_matches": source["gate"].get("decision")
        == SEED0_DECISION,
        "seed0_status_pass": source["gate"].get("status") == "pass",
        "seed0_candidate_checks_pass": all(
            source["gate"].get("candidate_checks", {}).values()
        ),
        "seed0_relation_checks_pass": all(
            source["gate"].get("relation_checks", {}).values()
        ),
        "seed0_three_modes_present": {
            row.get("relation_mode") for row in source["rows"]
        }
        == {"independent", "true", "corrupted"},
        "seed0_hashes_present": all(
            len(value) == 64 for value in source["hashes"].values()
        ),
    }


def adjudicate_r3_only_replication(
    config: R3OnlyReplicationConfig,
    seed0_checks: dict[str, bool],
    source_checks: dict[str, bool | float],
    contract: dict[str, Any],
    seed0_source: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    rows = training["trained_rows"]
    by_mode = {row["relation_mode"]: row for row in rows}
    seed0_rows = {row["relation_mode"]: row for row in seed0_source["rows"]}
    seed1_true = float(by_mode.get("true", {}).get("validation_auc", 0.0))
    seed1_independent = float(
        by_mode.get("independent", {}).get("validation_auc", 0.0)
    )
    seed1_corrupted = float(
        by_mode.get("corrupted", {}).get("validation_auc", 0.0)
    )
    seed1_train = float(by_mode.get("true", {}).get("train_auc", 0.0))
    seed0_true = float(seed0_rows["true"]["validation_auc"])
    mean_true = (seed0_true + seed1_true) / 2.0
    protocol_checks = {
        **seed0_checks,
        **{
            key: value
            for key, value in source_checks.items()
            if isinstance(value, bool)
        },
        "output_shape_is_4x64": contract["output_shape"] == [4, 64],
        "input_dim_is_13": contract["input_dim"] == 13,
        "parameter_counts_match": contract["parameter_counts_match"],
        "parameters_reduced_at_least_10_percent": contract["parameter_ratio_to_e68"]
        <= 0.90,
        "cell_relabel_equivariant": contract["cell_relabel_max_abs_error"] <= 1e-6,
        "all_three_seed1_rows_present": set(by_mode)
        == {"independent", "true", "corrupted"},
        "all_seed1_rows_completed_30_epochs": len(rows) == 3
        and all(row["epochs_completed"] == config.epochs for row in rows),
        "all_seed1_metrics_finite": all(
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
    seed1_checks = {
        "seed1_true_auc_at_least_0p93": seed1_true >= 0.93,
        "seed1_train_validation_gap_at_most_0p15": seed1_train - seed1_true <= 0.15,
        "seed1_minus_e68_full_prefix_at_least_minus_0p02": seed1_true
        - E68_SEED1_TRUE_AUC
        >= -0.02,
        "seed1_true_minus_independent_at_least_0p03": seed1_true
        - seed1_independent
        >= 0.03,
        "seed1_true_minus_corrupted_at_least_0p03": seed1_true - seed1_corrupted
        >= 0.03,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_r3_only_replication_protocol_invalid"
        action = "repair seed0/source/contract or seed1 protocol"
    elif not all(seed1_checks.values()):
        status = "hold"
        decision = "innovation2_present_r3_only_seed_not_replicated"
        action = "retain E68 full-prefix method and seed0 r3-only evidence only"
    else:
        status = "pass"
        decision = "innovation2_present_r3_only_two_seed_confirmed"
        action = "promote r3-only as the simpler confirmed PRESENT-r4 profile method"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "seed1_checks": seed1_checks,
        "metrics": {
            "seed0_rows": list(seed0_rows.values()),
            "seed1_rows": rows,
            "seed0_true_auc": seed0_true,
            "seed1_true_auc": seed1_true,
            "mean_true_auc": mean_true,
            "mean_minus_e68_full_prefix": mean_true - E68_MEAN_TRUE_AUC,
            "seed1_minus_e68_full_prefix": seed1_true - E68_SEED1_TRUE_AUC,
            "seed1_true_minus_independent": seed1_true - seed1_independent,
            "seed1_true_minus_corrupted": seed1_true - seed1_corrupted,
            "seed1_train_validation_gap": seed1_train - seed1_true,
            "contract": contract,
        },
        "claim_scope": (
            "two-seed local confirmation of a parameter-reduced r3-only profile "
            "operator on PRESENT-80 r4 strict 8-bit-cube unit-balance labels; no "
            "high-round, cross-dimension, attack, remote-scale, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "method_promoted": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: R3OnlyReplicationConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "R3OnlyReplicationConfig",
    "adjudicate_r3_only_replication",
    "load_r3_seed0_source",
    "serializable_config",
    "validate_r3_seed0_source",
]
