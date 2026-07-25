from __future__ import annotations

from pathlib import Path

from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_experiment import (
    EXPECTED_CIPHERS,
    EXPECTED_ROLES,
    EXPECTED_SEEDS,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_gated_experiment import (
    adjudicate_gated_modulation_experiment,
    load_and_validate_gated_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_primitive_gated_modulation_five_cipher_joint_2048_seed0_seed1.json"
)


def test_frozen_gated_joint_config_is_valid() -> None:
    config = load_and_validate_gated_config(CONFIG)

    assert config["model"]["primitive_adapter_effect"] == "multiplicative_gate"
    assert config["training"]["samples_per_class"] == 2048
    assert config["training"]["cache_source_root"].endswith("/cache")
    assert config["gate"]["source_margin"] == 0.005


def test_gated_gate_requires_controls_and_additive_source_gain() -> None:
    config = load_and_validate_gated_config(CONFIG)
    per_cipher = {}
    aggregates = {}
    router = {}
    gradients = {}
    source_rows = []
    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        per_cipher[seed_key] = {}
        aggregates[seed_key] = {}
        router[seed_key] = {}
        gradients[seed_key] = {}
        for role in EXPECTED_ROLES:
            auc = 0.52 if role != "correct" else 0.54
            per_cipher[seed_key][role] = {
                cipher: {
                    "train": {"auc": auc},
                    "validation": {"auc": auc},
                }
                for cipher in EXPECTED_CIPHERS
            }
            aggregates[seed_key][role] = {
                "core_macro_auc": auc,
                "stress_macro_auc": auc,
                "five_macro_auc": auc,
            }
            router[seed_key][role] = {
                cipher: {"fan_in_1": 1.0, "multi_source": 1.0}
                for cipher in EXPECTED_CIPHERS
            }
            gradients[seed_key][role] = {
                "all_gradients_finite": True,
                "adapter_gradient_mean_abs_sum": {
                    "primitive_adapters.fan_in_1": 1.0,
                    "primitive_adapters.multi_source": 1.0,
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
        "aggregates": aggregates,
        "router_utilization": router,
        "gradient_diagnostics": gradients,
        "validation": {"status": "pass"},
    }

    gate = adjudicate_gated_modulation_experiment(
        payload,
        project_root=ROOT,
        source_rows=source_rows,
    )

    assert gate["status"] == "pass"
    assert gate["source_anchor_valid"] is True
    assert gate["full_pass"] is True
    assert gate["decision"].endswith("gated_modulation_supported")
