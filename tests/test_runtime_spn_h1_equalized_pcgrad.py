from __future__ import annotations

from pathlib import Path

from blockcipher_nd.tasks.innovation1.runtime_spn_h1_equalized_pcgrad import (
    adjudicate_h1_equalized_pcgrad,
    load_and_validate_h1_equalized_pcgrad_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_h1_equalized_pcgrad_a3_2048_seed0_seed1.json"
)


def test_frozen_h1_a3_config_is_valid() -> None:
    config = load_and_validate_h1_equalized_pcgrad_config(CONFIG, project_root=ROOT)

    assert config["candidate"]["gradient_combination"].endswith(
        "pcgrad_fixed_order"
    )
    assert config["gate"]["skinny_auc_improvement_over_a2"] == 0.01


def test_gate_passes_when_target_source_and_skinny_all_recover() -> None:
    payload = _payload(skinny=(0.52, 0.50), target=(0.67, 0.65))

    gate = adjudicate_h1_equalized_pcgrad(payload)

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("equalized_pcgrad_supported")


def test_gate_keeps_only_partial_skinny_improvement() -> None:
    payload = _payload(skinny=(0.496, 0.48), target=(0.62, 0.60))

    gate = adjudicate_h1_equalized_pcgrad(payload)

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("equalized_pcgrad_partial")


def test_gate_fails_closed_when_projection_not_observed() -> None:
    payload = _payload(skinny=(0.52, 0.50), target=(0.67, 0.65))
    payload["validation"] = {"status": "fail"}

    gate = adjudicate_h1_equalized_pcgrad(payload)

    assert gate["status"] == "invalid"
    assert gate["decision"].endswith("protocol_invalid")


def _payload(
    *,
    skinny: tuple[float, float],
    target: tuple[float, float],
) -> dict[str, object]:
    config = load_and_validate_h1_equalized_pcgrad_config(CONFIG, project_root=ROOT)
    candidate_source = {
        str(seed): {
            "gift64": 0.54 if seed == 0 else 0.51,
            "skinny64": skinny[seed],
            "uknit64": 0.52,
            "dialga128": 0.94,
        }
        for seed in (0, 1)
    }
    return {
        "config": config,
        "candidate_target_auc": {
            str(seed): {
                "candidate_correct": target[seed],
                "candidate_corrupted_target": target[seed] - 0.03,
                "candidate_no_topology_target": target[seed] - 0.04,
            }
            for seed in (0, 1)
        },
        "a2_target_auc": {
            "0": {"candidate_correct": 0.684},
            "1": {"candidate_correct": 0.660},
        },
        "candidate_source_auc": candidate_source,
        "anchor_source_auc": {
            "0": {
                "gift64": 0.539,
                "skinny64": 0.536,
                "uknit64": 0.509,
                "dialga128": 0.941,
            },
            "1": {
                "gift64": 0.478,
                "skinny64": 0.535,
                "uknit64": 0.484,
                "dialga128": 0.942,
            },
        },
        "a2_source_auc": {
            "0": {
                "gift64": 0.537,
                "skinny64": 0.490,
                "uknit64": 0.509,
                "dialga128": 0.931,
            },
            "1": {
                "gift64": 0.503,
                "skinny64": 0.474,
                "uknit64": 0.520,
                "dialga128": 0.934,
            },
        },
        "conflict_projections_by_seed": {"0": 10, "1": 12},
        "validation": {"status": "pass"},
    }
