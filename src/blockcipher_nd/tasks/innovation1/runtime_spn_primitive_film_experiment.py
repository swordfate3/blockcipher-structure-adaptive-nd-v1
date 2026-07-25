from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_experiment import (
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
    load_and_validate_joint_config,
)


def load_and_validate_true_film_config(path: Path) -> dict[str, Any]:
    payload = load_and_validate_joint_config(path)
    if payload.get("experiment") != (
        "innovation1_runtime_spn_primitive_true_film_five_cipher_joint"
    ):
        raise ValueError("primitive True FiLM experiment name drifted")
    model = payload["model"]
    expected_model = {
        "primitive_conditioning": "true_film",
        "primitive_film_descriptor_dim": 128,
        "primitive_film_rank": 10,
        "primitive_film_scale": 0.1,
        "primitive_film_position": "pre_mixer_transition",
    }
    for key, value in expected_model.items():
        if model.get(key) != value:
            raise ValueError(f"primitive True FiLM model field {key} drifted")
    training = payload["training"]
    if not str(training.get("cache_source_root", "")).endswith(
        "i1_runtime_spn_primitive_adapter_five_cipher_joint_2048_seed0_seed1_20260725/cache"
    ):
        raise ValueError("primitive True FiLM cache source drifted")
    gate = payload["gate"]
    expected_gate = {
        "source_margin": 0.005,
        "source_per_cipher_floor": -0.005,
        "expected_parameter_count": 446_562,
    }
    for key, value in expected_gate.items():
        if gate.get(key) != value:
            raise ValueError(f"primitive True FiLM gate field {key} drifted")
    if not str(gate.get("source_anchor_results", "")).endswith("/results.jsonl"):
        raise ValueError("primitive True FiLM source anchor path drifted")
    if len(str(gate.get("source_anchor_config_sha256", ""))) != 64:
        raise ValueError("primitive True FiLM source hash drifted")
    return payload


def adjudicate_true_film_experiment(
    payload: dict[str, Any],
    *,
    project_root: Path,
    source_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = payload["config"]
    gate_config = config["gate"]
    if source_rows is None:
        source_path = project_root / gate_config["source_anchor_results"]
        source_rows = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    source_correct = [row for row in source_rows if row.get("role") == "correct"]
    source_valid = _source_anchor_valid(source_correct, config)
    source_auc = {
        (int(row["seed"]), row["cipher"]): float(row["metrics"]["validation"]["auc"])
        for row in source_correct
    }

    protocol_by_name = {item["name"]: item for item in config["protocols"]}
    per_seed: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        candidate = {
            cipher: float(
                payload["per_cipher_metrics"][seed_key]["correct"][cipher][
                    "validation"
                ]["auc"]
            )
            for cipher in EXPECTED_CIPHERS
        }
        control_deltas: dict[str, dict[str, Any]] = {}
        for control in ("dense", "uniform", "shuffled"):
            by_cipher = {
                cipher: candidate[cipher]
                - float(
                    payload["per_cipher_metrics"][seed_key][control][cipher][
                        "validation"
                    ]["auc"]
                )
                for cipher in EXPECTED_CIPHERS
            }
            control_deltas[control] = _group_deltas(by_cipher, protocol_by_name)

        source_by_cipher = {
            cipher: candidate[cipher] - source_auc.get((seed, cipher), float("nan"))
            for cipher in EXPECTED_CIPHERS
        }
        source_delta = _group_deltas(source_by_cipher, protocol_by_name)
        control_checks = {
            control: _delta_checks(
                values,
                margin=float(gate_config["margin"]),
                floor=float(gate_config["per_cipher_floor"]),
            )
            for control, values in control_deltas.items()
        }
        source_checks = _delta_checks(
            source_delta,
            margin=float(gate_config["source_margin"]),
            floor=float(gate_config["source_per_cipher_floor"]),
        )
        traffic = sum(
            float(
                payload["router_utilization"][seed_key]["correct"][cipher].get(
                    "film", 0.0
                )
            )
            for cipher in EXPECTED_CIPHERS
        )
        gradient_payload = payload["gradient_diagnostics"][seed_key]["correct"]
        film_gradients = {
            name: float(value)
            for name, value in gradient_payload["adapter_gradient_mean_abs_sum"].items()
            if name.startswith("primitive_film_conditioner.")
        }
        attribution_checks = {
            "film_has_traffic": traffic > 0.0,
            "film_down_and_affine_have_gradients": (
                film_gradients.get("primitive_film_conditioner.down", 0.0) > 0.0
                and film_gradients.get("primitive_film_conditioner.affine", 0.0) > 0.0
            ),
            "all_gradients_finite": bool(gradient_payload["all_gradients_finite"]),
        }
        controls_core_pass = all(
            checks["core_macro_at_least_margin"]
            and checks["each_core_cipher_above_floor"]
            for checks in control_checks.values()
        )
        controls_stress_pass = all(
            checks["stress_macro_at_least_margin"]
            and checks["each_stress_cipher_above_floor"]
            for checks in control_checks.values()
        )
        core_pass = bool(
            controls_core_pass
            and source_checks["core_macro_at_least_margin"]
            and source_checks["each_core_cipher_above_floor"]
            and all(attribution_checks.values())
        )
        stress_pass = bool(
            controls_stress_pass
            and source_checks["stress_macro_at_least_margin"]
            and source_checks["each_stress_cipher_above_floor"]
        )
        per_seed[seed_key] = {
            "control_deltas": control_deltas,
            "film_minus_additive": source_delta,
            "control_checks": control_checks,
            "source_anchor_checks": source_checks,
            "film_traffic": traffic,
            "film_gradients": film_gradients,
            "attribution_checks": attribution_checks,
            "core_pass": core_pass,
            "stress_pass": stress_pass,
            "full_pass": core_pass and stress_pass,
        }

    validation = payload["validation"]
    protocol_valid = bool(
        validation.get("status") == "pass"
        and validation.get("result_rows") == 40
        and validation.get("all_checkpoints_exist") is True
        and validation.get("parameter_counts")
        == [gate_config["expected_parameter_count"]]
        and validation.get("parameter_matched") is True
        and validation.get("task_specific_trainable_state") is False
        and validation.get("strict_negative_mode") == "encrypted_random_plaintexts"
        and validation.get("cache_source_root")
        == config["training"]["cache_source_root"]
        and source_valid
    )
    full_pass = protocol_valid and all(
        per_seed[str(seed)]["full_pass"] for seed in EXPECTED_SEEDS
    )
    core_pass = protocol_valid and all(
        per_seed[str(seed)]["core_pass"] for seed in EXPECTED_SEEDS
    )
    if full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_primitive_true_film_supported"
        next_action = (
            "keep True FiLM and preregister whole-cipher holdouts with one entire "
            "cipher absent from scratch training; learned MoE remains closed"
        )
    elif core_pass:
        status = "hold"
        decision = (
            "innovation1_runtime_spn_primitive_true_film_core_supported_stress_hold"
        )
        next_action = (
            "preserve the core-only evidence but do not enter holdouts or remote scale; "
            "compare typed GNN-FiLM against method consolidation"
        )
    elif protocol_valid:
        status = "hold"
        decision = "innovation1_runtime_spn_primitive_true_film_not_supported"
        next_action = (
            "discard True FiLM and stop deterministic Adapter/FiLM/MoE scaling; "
            "rank typed GNN-FiLM against consolidating the supported Runtime-E4 route"
        )
    else:
        status = "invalid"
        decision = "innovation1_runtime_spn_primitive_true_film_protocol_invalid"
        next_action = (
            "repair readiness, archive alignment, or source-anchor mismatch only"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": protocol_valid,
        "source_anchor_valid": source_valid,
        "full_pass": full_pass,
        "core_pass": core_pass,
        "margin": gate_config["margin"],
        "per_cipher_floor": gate_config["per_cipher_floor"],
        "per_seed": per_seed,
        "claim_scope": (
            "local 2048/class/cipher two-seed True-FiLM diagnostic only; not formal "
            "scale, unseen-cipher transfer, or a universal claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "increase rank, scale, experts, samples, epochs, or remote compute after hold",
            "hide a failed cipher or seed behind the five-cipher macro average",
            "start learned routing before a full joint and whole-cipher holdout pass",
        ],
    }


def _source_anchor_valid(rows: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    gate = config["gate"]
    training = config["training"]
    return bool(
        len(rows) == len(EXPECTED_SEEDS) * len(EXPECTED_CIPHERS)
        and {(int(row["seed"]), row["cipher"]) for row in rows}
        == {(seed, cipher) for seed in EXPECTED_SEEDS for cipher in EXPECTED_CIPHERS}
        and all(
            row.get("config_sha256") == gate["source_anchor_config_sha256"]
            and row.get("samples_per_class") == training["samples_per_class"]
            and row.get("validation_samples_per_class")
            == training["validation_samples_per_class"]
            and row.get("pairs_per_sample") == training["pairs_per_sample"]
            and row.get("epochs") == training["epochs"]
            and row.get("negative_mode") == training["negative_mode"]
            for row in rows
        )
    )


def _group_deltas(
    by_cipher: dict[str, float],
    protocol_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    core = [
        by_cipher[cipher]
        for cipher in EXPECTED_CIPHERS
        if protocol_by_name[cipher]["group"] == "core"
    ]
    stress = [
        by_cipher[cipher]
        for cipher in EXPECTED_CIPHERS
        if protocol_by_name[cipher]["group"] == "stress"
    ]
    return {
        "by_cipher": by_cipher,
        "core_macro": float(np.mean(core)),
        "stress_macro": float(np.mean(stress)),
        "five_macro": float(np.mean(list(by_cipher.values()))),
        "core_values": core,
        "stress_values": stress,
    }


def _delta_checks(
    values: dict[str, Any], *, margin: float, floor: float
) -> dict[str, bool]:
    return {
        "core_macro_at_least_margin": values["core_macro"] >= margin,
        "stress_macro_at_least_margin": values["stress_macro"] >= margin,
        "each_core_cipher_above_floor": min(values["core_values"]) >= floor,
        "each_stress_cipher_above_floor": min(values["stress_values"]) >= floor,
    }


__all__ = [
    "adjudicate_true_film_experiment",
    "load_and_validate_true_film_config",
]
