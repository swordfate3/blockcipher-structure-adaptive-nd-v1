from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_experiment import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_identifiability import (
    EXPECTED_PROBES,
    adapter_rank_profile,
    adjudicate_adapter_identifiability,
    counterfactual_metrics,
    load_and_validate_identifiability_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_primitive_adapter_identifiability_audit_seed0_seed1.json"
)


def test_frozen_identifiability_config_is_valid() -> None:
    config = load_and_validate_identifiability_config(CONFIG, project_root=ROOT)

    assert tuple(config["audit"]["probes"]) == EXPECTED_PROBES
    assert config["audit"]["rows_per_cipher"] == 4096
    assert config["audit"]["probes"]["amplified"]["scale"] == 0.5


def test_counterfactual_metrics_report_relative_logit_effect() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    reference = np.asarray([-1.0, -0.5, 0.5, 1.0])
    changed = reference * 2.0

    metrics = counterfactual_metrics(
        labels=labels,
        logits=changed,
        reference_logits=reference,
    )

    assert metrics["auc"] == 1.0
    assert metrics["mean_abs_logit_delta_vs_reference"] == 0.75
    assert metrics["relative_rms_logit_delta_vs_reference"] == 1.0
    assert metrics["threshold_flip_fraction_vs_reference"] == 0.0


def test_rank_profile_detects_full_low_rank_proxy() -> None:
    state = {}
    for adapter in ("fan_in_1", "multi_source"):
        state[f"primitive_adapters.{adapter}.down.weight"] = torch.eye(4)
        state[f"primitive_adapters.{adapter}.up.weight"] = torch.eye(4)

    profile = adapter_rank_profile(state)

    assert profile["fan_in_1"]["linearized_numerical_rank"] == 4
    assert profile["fan_in_1"]["linearized_effective_rank"] == 4.0


def test_gate_replaces_functionally_weak_adapter() -> None:
    config = load_and_validate_identifiability_config(CONFIG, project_root=ROOT)
    rows = []
    for seed in (0, 1):
        for task in EXPECTED_CIPHERS:
            for probe in EXPECTED_PROBES:
                effect = 0.01 if probe == "source" else 0.2
                rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "probe": probe,
                        "auc": 0.6,
                        "relative_rms_logit_delta_vs_reference": effect,
                    }
                )
    rank = {
        str(seed): {
            adapter: {"linearized_effective_rank": 6.0}
            for adapter in ("fan_in_1", "multi_source")
        }
        for seed in (0, 1)
    }
    payload = {
        "config": config,
        "rows": rows,
        "adapter_rank": rank,
        "validation": {"status": "pass"},
    }

    gate = adjudicate_adapter_identifiability(payload)

    assert gate["status"] == "pass"
    assert gate["functionally_active_both_seeds"] is False
    assert gate["decision"].endswith("additive_adapter_functionally_weak")
