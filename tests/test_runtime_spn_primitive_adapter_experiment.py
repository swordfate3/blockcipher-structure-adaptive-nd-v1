from __future__ import annotations

from pathlib import Path

import pytest

from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_experiment import (
    EXPECTED_CIPHERS,
    EXPECTED_ROLES,
    EXPECTED_SEEDS,
    adjudicate_joint_experiment,
    load_and_validate_joint_config,
    render_joint_margin_svg,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_primitive_adapter_five_cipher_joint_2048_seed0_seed1.json"
)


def _payload(*, uknit_correct: float = 0.52, core_correct: float = 0.52):
    config = load_and_validate_joint_config(CONFIG)
    per_cipher_metrics = {}
    aggregates = {}
    router = {}
    gradients = {}
    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        per_cipher_metrics[seed_key] = {}
        aggregates[seed_key] = {}
        router[seed_key] = {}
        gradients[seed_key] = {}
        for role in EXPECTED_ROLES:
            per_cipher_metrics[seed_key][role] = {}
            for cipher in EXPECTED_CIPHERS:
                auc = 0.50
                if role == "correct":
                    auc = (
                        uknit_correct
                        if cipher == "uknit64"
                        else 0.53
                        if cipher == "dialga128"
                        else core_correct
                    )
                per_cipher_metrics[seed_key][role][cipher] = {
                    "train": {"auc": auc},
                    "validation": {"auc": auc},
                }
            core = [
                per_cipher_metrics[seed_key][role][cipher]["validation"]["auc"]
                for cipher in ("gift64", "skinny64", "rectangle80")
            ]
            stress = [
                per_cipher_metrics[seed_key][role][cipher]["validation"]["auc"]
                for cipher in ("uknit64", "dialga128")
            ]
            five = core + stress
            aggregates[seed_key][role] = {
                "core_macro_auc": sum(core) / len(core),
                "stress_macro_auc": sum(stress) / len(stress),
                "five_macro_auc": sum(five) / len(five),
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
    return {
        "config": config,
        "per_cipher_metrics": per_cipher_metrics,
        "aggregates": aggregates,
        "router_utilization": router,
        "gradient_diagnostics": gradients,
        "validation": {"status": "pass"},
    }


def test_frozen_joint_config_is_valid() -> None:
    config = load_and_validate_joint_config(CONFIG)

    assert config["training"]["samples_per_class"] == 2048
    assert config["training"]["validation_samples_per_class"] == 1024
    assert tuple(config["training"]["roles"]) == EXPECTED_ROLES
    assert tuple(item["name"] for item in config["protocols"]) == EXPECTED_CIPHERS


def test_joint_config_rejects_a_protocol_drift(tmp_path: Path) -> None:
    config = load_and_validate_joint_config(CONFIG)
    config["protocols"][3]["rounds"] = 6
    changed = tmp_path / "changed.json"
    changed.write_text(__import__("json").dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="protocol uknit64 field rounds drifted"):
        load_and_validate_joint_config(changed)


def test_joint_gate_requires_both_core_and_stress_panels() -> None:
    full = adjudicate_joint_experiment(_payload())
    stress_hold = adjudicate_joint_experiment(_payload(uknit_correct=0.49))
    core_hold = adjudicate_joint_experiment(_payload(core_correct=0.501))

    assert full["status"] == "pass"
    assert full["full_pass"] is True
    assert stress_hold["status"] == "hold"
    assert stress_hold["core_pass"] is True
    assert stress_hold["full_pass"] is False
    assert "core_supported_new_cipher_hold" in stress_hold["decision"]
    assert core_hold["status"] == "hold"
    assert core_hold["core_pass"] is False
    assert core_hold["decision"].endswith("joint_not_supported")


def test_joint_margin_svg_renders_both_seeds(tmp_path: Path) -> None:
    payload = _payload()
    output = tmp_path / "curves.svg"

    render_joint_margin_svg(payload, output)

    text = output.read_text(encoding="utf-8")
    assert output.stat().st_size > 10_000
    assert "seed0" in text
    assert "seed1" in text
    assert "正确路由" in text
