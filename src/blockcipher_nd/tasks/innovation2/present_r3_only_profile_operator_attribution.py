from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


READINESS_RUN_ID = "i2_present_r4_r3_only_profile_operator_readiness_seed0_20260718"
READINESS_DECISION = "innovation2_present_r3_only_profile_readiness_passed"
E67_TRUE_AUC = 0.9530555555555555


@dataclass(frozen=True)
class R3OnlyAttributionConfig:
    run_id: str
    epochs: int = 30
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
            self.epochs != 30
            or self.batch_size != 8
            or self.hidden_dim != 32
            or self.steps != 2
            or self.seed != 0
            or self.dropout != 0.10
            or self.device != "cpu"
        ):
            raise ValueError("E73 Phase B protocol is frozen")

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


def load_r3_readiness_source(root: Path) -> dict[str, Any]:
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


def validate_r3_readiness_source(source: dict[str, Any]) -> dict[str, bool]:
    rows = source["rows"]
    return {
        "readiness_run_id_matches": source["gate"].get("run_id")
        == READINESS_RUN_ID,
        "readiness_decision_matches": source["gate"].get("decision")
        == READINESS_DECISION,
        "readiness_status_pass": source["gate"].get("status") == "pass",
        "readiness_protocol_pass": all(
            source["gate"].get("protocol_checks", {}).values()
        ),
        "readiness_signal_pass": all(
            source["gate"].get("readiness_checks", {}).values()
        ),
        "readiness_three_modes_present": {
            row.get("relation_mode") for row in rows
        }
        == {"independent", "true", "corrupted"},
        "readiness_hashes_present": all(
            len(value) == 64 for value in source["hashes"].values()
        ),
    }


def adjudicate_r3_only_attribution(
    config: R3OnlyAttributionConfig,
    readiness_checks: dict[str, bool],
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
    true_train_auc = float(by_mode.get("true", {}).get("train_auc", 0.0))
    protocol_checks = {
        **readiness_checks,
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
        "contract_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "all_three_rows_present": set(by_mode) == {"independent", "true", "corrupted"},
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
        "true_auc_at_least_0p93": true_auc >= 0.93,
        "true_train_validation_gap_at_most_0p15": true_train_auc - true_auc <= 0.15,
        "true_minus_e67_full_prefix_at_least_minus_0p02": true_auc - E67_TRUE_AUC
        >= -0.02,
    }
    relation_checks = {
        "true_minus_independent_at_least_0p03": true_auc - independent_auc >= 0.03,
        "true_minus_corrupted_at_least_0p03": true_auc - corrupted_auc >= 0.03,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_r3_only_attribution_protocol_invalid"
        action = "repair readiness, source, contract, or 30-epoch protocol"
    elif not all(candidate_checks.values()):
        status = "hold"
        decision = "innovation2_present_r3_only_quality_not_retained"
        action = "retain full 39-d E68 and stop r3-only compression"
    elif not all(relation_checks.values()):
        status = "hold"
        decision = "innovation2_present_r3_only_topology_not_attributed"
        action = "retain r3 sufficiency only as deterministic evidence"
    else:
        status = "pass"
        decision = "innovation2_present_r3_only_neural_gain_attributed"
        action = "run the same 30-epoch r3-only matrix with seed1"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "candidate_checks": candidate_checks,
        "relation_checks": relation_checks,
        "metrics": {
            "rows": rows,
            "e67_full_prefix_true_auc": E67_TRUE_AUC,
            "true_minus_e67_full_prefix": true_auc - E67_TRUE_AUC,
            "true_minus_independent": true_auc - independent_auc,
            "true_minus_corrupted": true_auc - corrupted_auc,
            "true_train_validation_gap": true_train_auc - true_auc,
            "contract": contract,
        },
        "claim_scope": (
            "30-epoch local seed0 attribution of an r3-only prefix-guided "
            "profile operator on PRESENT-80 r4 strict 8-bit-cube unit-balance "
            "labels; no high-round, cross-dimension, attack, remote-scale, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "seed1": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: R3OnlyAttributionConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "R3OnlyAttributionConfig",
    "adjudicate_r3_only_attribution",
    "load_r3_readiness_source",
    "serializable_config",
    "validate_r3_readiness_source",
]
