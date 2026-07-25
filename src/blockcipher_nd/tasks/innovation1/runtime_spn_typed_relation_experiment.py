from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_experiment import (
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
    load_and_validate_joint_config,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_film_experiment import (
    _delta_checks,
    _group_deltas,
)


def load_and_validate_typed_relation_config(path: Path) -> dict[str, Any]:
    payload = load_and_validate_joint_config(path)
    if payload.get("experiment") != (
        "innovation1_runtime_spn_typed_relation_gnn_film_five_cipher_joint"
    ):
        raise ValueError("typed-relation experiment name drifted")
    expected_model = {
        "primitive_conditioning": "typed_relation_gnn_film",
        "typed_relation_types": 16,
        "typed_relation_parameter_count": 4096,
        "typed_relation_scale": 0.1,
        "typed_relation_position": "pre_mixer_exact_gf2_residual",
    }
    for key, value in expected_model.items():
        if payload["model"].get(key) != value:
            raise ValueError(f"typed-relation model field {key} drifted")
    expected_modes = {
        "dense": "dense",
        "correct": "correct",
        "uniform": "agnostic",
        "shuffled": "shuffled",
    }
    if payload["training"].get("roles") != expected_modes:
        raise ValueError("typed-relation role semantics drifted")
    if not str(payload["training"].get("cache_source_root", "")).endswith(
        "i1_runtime_spn_primitive_adapter_five_cipher_joint_2048_seed0_seed1_20260725/cache"
    ):
        raise ValueError("typed-relation cache source drifted")
    gate = payload["gate"]
    expected_gate = {
        "margin": 0.005,
        "per_cipher_floor": -0.005,
        "source_min_macro_delta": 0.0,
        "source_per_cipher_floor": -0.005,
        "expected_parameter_count": 446_562,
    }
    for key, value in expected_gate.items():
        if gate.get(key) != value:
            raise ValueError(f"typed-relation gate field {key} drifted")
    anchors = gate.get("source_anchors", {})
    if set(anchors) != {"additive", "true_film"}:
        raise ValueError("typed-relation source anchors drifted")
    for anchor in anchors.values():
        if not str(anchor.get("results", "")).endswith("/results.jsonl"):
            raise ValueError("typed-relation source result path drifted")
        if len(str(anchor.get("config_sha256", ""))) != 64:
            raise ValueError("typed-relation source hash drifted")
    return payload


def adjudicate_typed_relation_experiment(
    payload: dict[str, Any],
    *,
    project_root: Path,
    source_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    config = payload["config"]
    gate_config = config["gate"]
    if source_rows is None:
        source_rows = {
            name: [
                json.loads(line)
                for line in (project_root / anchor["results"])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            for name, anchor in gate_config["source_anchors"].items()
        }
    source_correct = {
        name: [row for row in rows if row.get("role") == "correct"]
        for name, rows in source_rows.items()
    }
    source_valid = {
        name: _source_anchor_valid(
            rows,
            config=config,
            expected_sha256=gate_config["source_anchors"][name]["config_sha256"],
        )
        for name, rows in source_correct.items()
    }
    source_auc = {
        name: {
            (int(row["seed"]), row["cipher"]): float(
                row["metrics"]["validation"]["auc"]
            )
            for row in rows
        }
        for name, rows in source_correct.items()
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
        control_deltas = {
            control: _group_deltas(
                {
                    cipher: candidate[cipher]
                    - float(
                        payload["per_cipher_metrics"][seed_key][control][cipher][
                            "validation"
                        ]["auc"]
                    )
                    for cipher in EXPECTED_CIPHERS
                },
                protocol_by_name,
            )
            for control in ("dense", "uniform", "shuffled")
        }
        control_checks = {
            control: _delta_checks(
                values,
                margin=float(gate_config["margin"]),
                floor=float(gate_config["per_cipher_floor"]),
            )
            for control, values in control_deltas.items()
        }
        source_deltas = {
            name: _group_deltas(
                {
                    cipher: candidate[cipher]
                    - values.get((seed, cipher), float("nan"))
                    for cipher in EXPECTED_CIPHERS
                },
                protocol_by_name,
            )
            for name, values in source_auc.items()
        }
        source_checks = {
            name: _delta_checks(
                values,
                margin=float(gate_config["source_min_macro_delta"]),
                floor=float(gate_config["source_per_cipher_floor"]),
            )
            for name, values in source_deltas.items()
        }
        traffic = sum(
            float(
                payload["router_utilization"][seed_key]["correct"][cipher].get(
                    "typed_relation", 0.0
                )
            )
            for cipher in EXPECTED_CIPHERS
        )
        gradient_payload = payload["gradient_diagnostics"][seed_key]["correct"]
        gradients = gradient_payload["adapter_gradient_mean_abs_sum"]
        attribution_checks = {
            "typed_relation_has_traffic": traffic > 0.0,
            "gamma_has_gradient": (
                gradients.get("typed_relation_message.gamma", 0.0) > 0.0
            ),
            "beta_has_gradient": (
                gradients.get("typed_relation_message.beta", 0.0) > 0.0
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
        sources_core_pass = all(
            checks["core_macro_at_least_margin"]
            and checks["each_core_cipher_above_floor"]
            for checks in source_checks.values()
        )
        sources_stress_pass = all(
            checks["stress_macro_at_least_margin"]
            and checks["each_stress_cipher_above_floor"]
            for checks in source_checks.values()
        )
        core_pass = bool(
            controls_core_pass
            and sources_core_pass
            and all(attribution_checks.values())
        )
        stress_pass = bool(controls_stress_pass and sources_stress_pass)
        per_seed[seed_key] = {
            "control_deltas": control_deltas,
            "control_checks": control_checks,
            "typed_minus_sources": source_deltas,
            "source_anchor_checks": source_checks,
            "typed_relation_traffic": traffic,
            "typed_relation_gradients": {
                name: value
                for name, value in gradients.items()
                if name.startswith("typed_relation_message.")
            },
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
        and all(source_valid.values())
    )
    full_pass = protocol_valid and all(
        per_seed[str(seed)]["full_pass"] for seed in EXPECTED_SEEDS
    )
    core_pass = protocol_valid and all(
        per_seed[str(seed)]["core_pass"] for seed in EXPECTED_SEEDS
    )
    if full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_typed_relation_supported"
        next_action = (
            "keep the typed relation path and preregister one entire-cipher holdout; "
            "do not increase sample scale before the holdout protocol is frozen"
        )
    elif core_pass:
        status = "hold"
        decision = "innovation1_runtime_spn_typed_relation_core_only_stress_hold"
        next_action = (
            "retain the core-only diagnostic but do not enter holdouts or remote scale; "
            "consolidate Runtime-E4 unless one stress-protocol mismatch is identified"
        )
    elif protocol_valid:
        status = "hold"
        decision = "innovation1_runtime_spn_typed_relation_not_supported"
        next_action = (
            "close the typed Adapter/FiLM/GNN residual branch and consolidate the "
            "supported Runtime-E4 method for the thesis"
        )
    else:
        status = "invalid"
        decision = "innovation1_runtime_spn_typed_relation_protocol_invalid"
        next_action = "repair only readiness, cache, checkpoint, or source-anchor alignment"
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
            "local 2048/class/cipher two-seed typed-relation diagnostic only; not "
            "formal scale, unseen-cipher transfer, universality, or breakthrough"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "increase relation layers, scale, samples, epochs, or remote compute after hold",
            "hide a failed cipher or seed behind the five-cipher macro average",
            "start learned MoE or cipher-ID routing",
            "claim unseen-cipher structure learning before an entire-cipher holdout",
        ],
    }


def _source_anchor_valid(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    expected_sha256: str,
) -> bool:
    training = config["training"]
    return bool(
        len(rows) == len(EXPECTED_SEEDS) * len(EXPECTED_CIPHERS)
        and {(int(row["seed"]), row["cipher"]) for row in rows}
        == {(seed, cipher) for seed in EXPECTED_SEEDS for cipher in EXPECTED_CIPHERS}
        and all(
            row.get("config_sha256") == expected_sha256
            and row.get("samples_per_class") == training["samples_per_class"]
            and row.get("validation_samples_per_class")
            == training["validation_samples_per_class"]
            and row.get("pairs_per_sample") == training["pairs_per_sample"]
            and row.get("epochs") == training["epochs"]
            and row.get("negative_mode") == training["negative_mode"]
            for row in rows
        )
    )


__all__ = [
    "adjudicate_typed_relation_experiment",
    "load_and_validate_typed_relation_config",
]
