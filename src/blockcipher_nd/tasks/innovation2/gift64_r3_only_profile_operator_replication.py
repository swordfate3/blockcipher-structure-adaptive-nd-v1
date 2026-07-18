from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


E78_RUN_ID = "i2_gift64_r4_r3_only_profile_operator_attribution_seed0_20260719"
E78_DECISION = "innovation2_gift64_r3_only_neural_gain_attributed"


@dataclass(frozen=True)
class Gift64R3ReplicationConfig:
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
            raise ValueError("E79 seed1 protocol is frozen")

    @property
    def task_name(self) -> str:
        return "innovation2_gift64_r3_only_profile_replication"


def load_e78_source(root: Path) -> dict[str, Any]:
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


def validate_e78_source(source: dict[str, Any]) -> dict[str, bool]:
    gate = source["gate"]
    rows = source["rows"]
    return {
        "e78_run_id_matches": gate.get("run_id") == E78_RUN_ID,
        "e78_status_pass": gate.get("status") == "pass",
        "e78_decision_matches": gate.get("decision") == E78_DECISION,
        "e78_protocol_checks_pass": bool(gate.get("protocol_checks"))
        and all(gate["protocol_checks"].values()),
        "e78_candidate_checks_pass": bool(gate.get("candidate_checks"))
        and all(gate["candidate_checks"].values()),
        "e78_relation_checks_pass": bool(gate.get("relation_checks"))
        and all(gate["relation_checks"].values()),
        "e78_three_rows_present": {row.get("relation_mode") for row in rows}
        == {"independent", "true", "corrupted"},
        "e78_rows_seed0": all(row.get("seed") == 0 for row in rows),
        "e78_hashes_present": all(
            len(value) == 64 for value in source["hashes"].values()
        ),
    }


def adjudicate_gift_r3_replication(
    config: Gift64R3ReplicationConfig,
    profile_checks: dict[str, bool],
    e78_checks: dict[str, bool],
    contract: dict[str, Any],
    training: dict[str, Any],
    e78_source: dict[str, Any],
) -> dict[str, Any]:
    seed1_rows = training["trained_rows"]
    seed0_rows = e78_source["rows"]
    seed1 = {row["relation_mode"]: row for row in seed1_rows}
    seed0 = {row["relation_mode"]: row for row in seed0_rows}
    e77_ridge_auc = float(e78_source["gate"]["metrics"]["e77_true_topology_ridge_auc"])
    true_auc = float(seed1.get("true", {}).get("validation_auc", 0.0))
    independent_auc = float(
        seed1.get("independent", {}).get("validation_auc", 0.0)
    )
    corrupted_auc = float(seed1.get("corrupted", {}).get("validation_auc", 0.0))
    true_train_auc = float(seed1.get("true", {}).get("train_auc", 0.0))
    protocol_checks = {
        **profile_checks,
        **e78_checks,
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
        "all_three_seed1_rows_present": set(seed1) == {"independent", "true", "corrupted"},
        "all_seed1_rows_completed_30_epochs": len(seed1_rows) == 3
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
    seed1_candidate_checks = {
        "seed1_true_auc_at_least_0p80": true_auc >= 0.80,
        "seed1_true_train_validation_gap_at_most_0p15": true_train_auc - true_auc
        <= 0.15,
        "seed1_true_minus_e77_ridge_at_least_0p03": true_auc - e77_ridge_auc
        >= 0.03,
    }
    seed1_relation_checks = {
        "seed1_true_minus_independent_at_least_0p03": true_auc - independent_auc
        >= 0.03,
        "seed1_true_minus_corrupted_at_least_0p03": true_auc - corrupted_auc >= 0.03,
    }
    joint_checks = {
        "seed0_source_all_gates_pass": all(e78_checks.values()),
        "seed1_candidate_checks_pass": all(seed1_candidate_checks.values()),
        "seed1_relation_checks_pass": all(seed1_relation_checks.values()),
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_gift64_r3_only_replication_protocol_invalid"
        action = "repair E75/E78 source, contract, or seed1 protocol"
    elif not all(joint_checks.values()):
        status = "hold"
        decision = "innovation2_gift64_r3_only_seed_not_replicated"
        action = "retain seed0 and deterministic evidence; stop extra seeds and tuning"
    else:
        status = "pass"
        decision = "innovation2_gift64_r3_only_two_seed_confirmed"
        action = "freeze the GIFT result and move to cross-SPN method synthesis"
    per_seed = {}
    for seed, by_mode in ((0, seed0), (1, seed1)):
        per_seed[f"seed{seed}"] = {
            "true_auc": float(by_mode["true"]["validation_auc"]),
            "independent_auc": float(by_mode["independent"]["validation_auc"]),
            "corrupted_auc": float(by_mode["corrupted"]["validation_auc"]),
            "true_minus_independent": float(by_mode["true"]["validation_auc"])
            - float(by_mode["independent"]["validation_auc"]),
            "true_minus_corrupted": float(by_mode["true"]["validation_auc"])
            - float(by_mode["corrupted"]["validation_auc"]),
            "true_minus_e77_ridge": float(by_mode["true"]["validation_auc"])
            - e77_ridge_auc,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "seed1_candidate_checks": seed1_candidate_checks,
        "seed1_relation_checks": seed1_relation_checks,
        "joint_checks": joint_checks,
        "metrics": {
            "seed0_rows": seed0_rows,
            "seed1_rows": seed1_rows,
            "e77_true_topology_ridge_auc": e77_ridge_auc,
            "per_seed": per_seed,
            "mean_true_auc": sum(item["true_auc"] for item in per_seed.values()) / 2,
            "mean_true_minus_independent": sum(
                item["true_minus_independent"] for item in per_seed.values()
            )
            / 2,
            "mean_true_minus_corrupted": sum(
                item["true_minus_corrupted"] for item in per_seed.values()
            )
            / 2,
            "mean_true_minus_e77_ridge": sum(
                item["true_minus_e77_ridge"] for item in per_seed.values()
            )
            / 2,
            "contract": contract,
        },
        "claim_scope": (
            "two-seed 30-epoch attribution of a GIFT-64 r4 r3-only "
            "prefix-guided profile operator on strict 8-bit-cube unit-balance "
            "labels; no high-round, zero-shot cross-cipher, attack, remote-scale, "
            "or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "freeze_gift_result": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: Gift64R3ReplicationConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "Gift64R3ReplicationConfig",
    "adjudicate_gift_r3_replication",
    "load_e78_source",
    "serializable_config",
    "validate_e78_source",
]
