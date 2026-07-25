from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_experiment import (
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
    adjudicate_joint_experiment,
    load_and_validate_joint_config,
)


def load_and_validate_gated_config(path: Path) -> dict[str, Any]:
    payload = load_and_validate_joint_config(path)
    if payload.get("experiment") != (
        "innovation1_runtime_spn_primitive_gated_modulation_five_cipher_joint"
    ):
        raise ValueError("primitive gated-modulation experiment name drifted")
    if payload["model"].get("primitive_adapter_effect") != "multiplicative_gate":
        raise ValueError("primitive gated-modulation effect drifted")
    if not str(payload["training"].get("cache_source_root", "")).endswith("/cache"):
        raise ValueError("primitive gated-modulation cache source drifted")
    gate = payload["gate"]
    expected = {
        "source_margin": 0.005,
        "source_per_cipher_floor": -0.005,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ValueError(f"primitive gated-modulation gate field {key} drifted")
    if not str(gate.get("source_anchor_results", "")).endswith("/results.jsonl"):
        raise ValueError("primitive gated-modulation source anchor path drifted")
    if len(str(gate.get("source_anchor_config_sha256", ""))) != 64:
        raise ValueError("primitive gated-modulation source hash drifted")
    return payload


def adjudicate_gated_modulation_experiment(
    payload: dict[str, Any],
    *,
    project_root: Path,
    source_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = payload["config"]
    base = adjudicate_joint_experiment(payload)
    gate_config = config["gate"]
    if source_rows is None:
        source_path = project_root / gate_config["source_anchor_results"]
        source_rows = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    source_correct = [row for row in source_rows if row.get("role") == "correct"]
    source_valid = bool(
        len(source_correct) == len(EXPECTED_SEEDS) * len(EXPECTED_CIPHERS)
        and all(
            row.get("config_sha256") == gate_config["source_anchor_config_sha256"]
            and row.get("samples_per_class") == config["training"]["samples_per_class"]
            and row.get("validation_samples_per_class")
            == config["training"]["validation_samples_per_class"]
            and row.get("pairs_per_sample") == config["training"]["pairs_per_sample"]
            and row.get("epochs") == config["training"]["epochs"]
            and row.get("negative_mode") == config["training"]["negative_mode"]
            for row in source_correct
        )
    )
    source_auc = {
        (int(row["seed"]), row["cipher"]): float(row["metrics"]["validation"]["auc"])
        for row in source_correct
    }
    protocol_by_name = {item["name"]: item for item in config["protocols"]}
    per_seed: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        candidate = {
            cipher: payload["per_cipher_metrics"][seed_key]["correct"][cipher][
                "validation"
            ]["auc"]
            for cipher in EXPECTED_CIPHERS
        }
        deltas = {
            cipher: candidate[cipher] - source_auc.get((seed, cipher), float("nan"))
            for cipher in EXPECTED_CIPHERS
        }
        core = [
            deltas[cipher]
            for cipher in EXPECTED_CIPHERS
            if protocol_by_name[cipher]["group"] == "core"
        ]
        stress = [
            deltas[cipher]
            for cipher in EXPECTED_CIPHERS
            if protocol_by_name[cipher]["group"] == "stress"
        ]
        source_checks = {
            "core_macro_at_least_margin": float(np.mean(core))
            >= gate_config["source_margin"],
            "stress_macro_at_least_margin": float(np.mean(stress))
            >= gate_config["source_margin"],
            "each_core_cipher_above_floor": min(core)
            >= gate_config["source_per_cipher_floor"],
            "each_stress_cipher_above_floor": min(stress)
            >= gate_config["source_per_cipher_floor"],
        }
        base_seed = base["per_seed"][seed_key]
        core_pass = bool(
            base_seed["core_pass"]
            and source_checks["core_macro_at_least_margin"]
            and source_checks["each_core_cipher_above_floor"]
        )
        stress_pass = bool(
            base_seed["stress_pass"]
            and source_checks["stress_macro_at_least_margin"]
            and source_checks["each_stress_cipher_above_floor"]
        )
        per_seed[seed_key] = {
            **base_seed,
            "gated_minus_additive_by_cipher": deltas,
            "gated_minus_additive_core_macro": float(np.mean(core)),
            "gated_minus_additive_stress_macro": float(np.mean(stress)),
            "source_anchor_checks": source_checks,
            "core_pass": core_pass,
            "stress_pass": stress_pass,
            "full_pass": core_pass and stress_pass,
        }
    protocol_valid = bool(base["protocol_valid"] and source_valid)
    full_pass = protocol_valid and all(
        per_seed[str(seed)]["full_pass"] for seed in EXPECTED_SEEDS
    )
    core_pass = protocol_valid and all(
        per_seed[str(seed)]["core_pass"] for seed in EXPECTED_SEEDS
    )
    if full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_primitive_gated_modulation_supported"
        next_action = (
            "keep gated modulation and preregister whole-cipher holdouts for "
            "RECTANGLE, Dialga, and uKNIT; learned routing remains closed"
        )
    elif core_pass:
        status = "hold"
        decision = "innovation1_runtime_spn_primitive_gated_core_supported_stress_hold"
        next_action = (
            "preserve only the core result and refine one local heterogeneous "
            "transition descriptor; do not claim five-cipher support"
        )
    elif protocol_valid:
        status = "hold"
        decision = "innovation1_runtime_spn_primitive_gated_modulation_not_supported"
        next_action = (
            "discard gated modulation; rank one parameter-matched dense conditional "
            "basis/FiLM candidate against stopping the differentiated route"
        )
    else:
        status = "invalid"
        decision = "innovation1_runtime_spn_primitive_gated_modulation_protocol_invalid"
        next_action = (
            "repair readiness or source-anchor alignment before interpretation"
        )
    return {
        **base,
        "status": status,
        "decision": decision,
        "protocol_valid": protocol_valid,
        "source_anchor_valid": source_valid,
        "core_pass": core_pass,
        "full_pass": full_pass,
        "per_seed": per_seed,
        "claim_scope": (
            "local 2048/class/cipher two-seed gated-modulation diagnostic only; "
            "not formal scale, unseen-cipher transfer, or a universal claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "increase scale, rank, experts, samples, epochs, or remote compute after hold",
            "use a high Dialga AUC to hide uKNIT or core failures",
            "start learned routing without a full joint and holdout pass",
        ],
    }


__all__ = [
    "adjudicate_gated_modulation_experiment",
    "load_and_validate_gated_config",
]
