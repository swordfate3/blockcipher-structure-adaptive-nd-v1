from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_experiment import (
    EXPECTED_CIPHERS,
    EXPECTED_ROLES,
    EXPECTED_SEEDS,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_film_experiment import (
    adjudicate_true_film_experiment,
    load_and_validate_true_film_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_primitive_true_film_five_cipher_joint_2048_seed0_seed1.json"
)


def test_frozen_true_film_joint_config_is_valid() -> None:
    config = load_and_validate_true_film_config(CONFIG)

    assert config["model"]["primitive_conditioning"] == "true_film"
    assert config["model"]["primitive_film_descriptor_dim"] == 128
    assert config["model"]["primitive_film_rank"] == 10
    assert config["training"]["samples_per_class"] == 2048
    assert config["training"]["cache_source_root"].endswith("/cache")
    assert config["gate"]["expected_parameter_count"] == 446_562


def test_true_film_gate_requires_all_controls_and_additive_source_gain() -> None:
    config = load_and_validate_true_film_config(CONFIG)
    payload, source_rows = _passing_payload(config)

    gate = adjudicate_true_film_experiment(
        payload,
        project_root=ROOT,
        source_rows=source_rows,
    )

    assert gate["status"] == "pass"
    assert gate["source_anchor_valid"] is True
    assert gate["full_pass"] is True
    assert gate["decision"].endswith("true_film_supported")


def test_true_film_gate_cannot_hide_one_cipher_control_failure() -> None:
    config = load_and_validate_true_film_config(CONFIG)
    payload, source_rows = _passing_payload(config)
    payload = deepcopy(payload)
    payload["per_cipher_metrics"]["1"]["shuffled"]["uknit64"]["validation"]["auc"] = (
        0.56
    )

    gate = adjudicate_true_film_experiment(
        payload,
        project_root=ROOT,
        source_rows=source_rows,
    )

    assert gate["status"] == "hold"
    assert gate["full_pass"] is False
    assert (
        gate["per_seed"]["1"]["control_checks"]["shuffled"][
            "each_stress_cipher_above_floor"
        ]
        is False
    )


def _passing_payload(
    config: dict,
) -> tuple[dict, list[dict]]:
    per_cipher = {}
    router = {}
    gradients = {}
    source_rows = []
    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        per_cipher[seed_key] = {}
        router[seed_key] = {}
        gradients[seed_key] = {}
        for role in EXPECTED_ROLES:
            auc = 0.54 if role == "correct" else 0.52
            per_cipher[seed_key][role] = {
                cipher: {
                    "train": {"auc": auc},
                    "validation": {"auc": auc},
                }
                for cipher in EXPECTED_CIPHERS
            }
            router[seed_key][role] = {
                cipher: {"film": 1.0} for cipher in EXPECTED_CIPHERS
            }
            gradients[seed_key][role] = {
                "all_gradients_finite": True,
                "adapter_gradient_mean_abs_sum": {
                    "primitive_film_conditioner.down": 1.0,
                    "primitive_film_conditioner.affine": 1.0,
                },
            }
        for cipher in EXPECTED_CIPHERS:
            source_rows.append(
                {
                    "seed": seed,
                    "cipher": cipher,
                    "role": "correct",
                    "config_sha256": config["gate"]["source_anchor_config_sha256"],
                    "samples_per_class": 2048,
                    "validation_samples_per_class": 1024,
                    "pairs_per_sample": 4,
                    "epochs": 10,
                    "negative_mode": "encrypted_random_plaintexts",
                    "metrics": {"validation": {"auc": 0.53}},
                }
            )
    payload = {
        "config": config,
        "per_cipher_metrics": per_cipher,
        "router_utilization": router,
        "gradient_diagnostics": gradients,
        "validation": {
            "status": "pass",
            "result_rows": 40,
            "all_checkpoints_exist": True,
            "parameter_counts": [446_562],
            "parameter_matched": True,
            "task_specific_trainable_state": False,
            "strict_negative_mode": "encrypted_random_plaintexts",
            "cache_source_root": config["training"]["cache_source_root"],
        },
    }
    return payload, source_rows
