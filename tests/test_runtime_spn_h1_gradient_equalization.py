from __future__ import annotations

from pathlib import Path

from blockcipher_nd.tasks.innovation1.runtime_spn_h1_gradient_equalization import (
    adjudicate_h1_gradient_equalization,
    load_and_validate_h1_gradient_equalization_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_h1_representation_gradient_equalization_a2_2048_seed0_seed1.json"
)


def test_frozen_h1_a2_config_is_valid() -> None:
    config = load_and_validate_h1_gradient_equalization_config(
        CONFIG,
        project_root=ROOT,
    )

    assert config["candidate"]["gradient_combination"] == (
        "representation_l2_equalized"
    )
    assert config["candidate"]["expected_parameter_count"] == 442466
    assert config["gate"]["target_topology_margin"] == 0.005


def test_gate_passes_dual_seed_attributed_candidate() -> None:
    payload = _payload(seed1_correct=0.61, seed1_corrupted=0.59, seed1_none=0.58)

    gate = adjudicate_h1_gradient_equalization(payload)

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("gradient_equalization_supported")
    assert gate["full_pass"] is True


def test_gate_retains_partial_margin_improvement_without_full_pass() -> None:
    payload = _payload(seed1_correct=0.59, seed1_corrupted=0.594, seed1_none=0.59)

    gate = adjudicate_h1_gradient_equalization(payload)

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("gradient_equalization_partial")
    assert gate["seed1_partial_margin_improvement"] is True


def test_gate_fails_closed_on_protocol_error() -> None:
    payload = _payload(seed1_correct=0.61, seed1_corrupted=0.59, seed1_none=0.58)
    payload["validation"] = {"status": "fail"}

    gate = adjudicate_h1_gradient_equalization(payload)

    assert gate["status"] == "invalid"
    assert gate["decision"].endswith("protocol_invalid")


def _payload(
    *,
    seed1_correct: float,
    seed1_corrupted: float,
    seed1_none: float,
) -> dict[str, object]:
    config = load_and_validate_h1_gradient_equalization_config(
        CONFIG,
        project_root=ROOT,
    )
    return {
        "config": config,
        "candidate_target_auc": {
            "0": {
                "candidate_correct": 0.67,
                "candidate_corrupted_target": 0.64,
                "candidate_no_topology_target": 0.63,
            },
            "1": {
                "candidate_correct": seed1_correct,
                "candidate_corrupted_target": seed1_corrupted,
                "candidate_no_topology_target": seed1_none,
            },
        },
        "anchor_target_auc": {
            "0": {
                "candidate_correct": 0.674,
                "candidate_corrupted_target": 0.635,
                "candidate_no_topology_target": 0.625,
            },
            "1": {
                "candidate_correct": 0.588,
                "candidate_corrupted_target": 0.5885,
                "candidate_no_topology_target": 0.606,
            },
        },
        "candidate_source_macro_auc": {"0": 0.63, "1": 0.61},
        "anchor_source_macro_auc": {"0": 0.631, "1": 0.610},
        "validation": {"status": "pass"},
    }
