from __future__ import annotations

from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_topology_only import (
    SOURCE_CIPHERS,
    adjudicate_topology_only,
    load_a8_references,
    load_and_validate_topology_only_config,
    run_topology_only_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/"
    "innovation1_runtime_spn_topology_only_c1_2048_seed0_seed1_20260726.json"
)
A8_RESULTS = (
    ROOT
    / "outputs/local_diagnostic/"
    "i1_runtime_spn_dialga_holdout_a8_2048_seed0_seed1_20260726/results.jsonl"
)


def test_real_c1_config_and_readiness_pass() -> None:
    config = load_and_validate_topology_only_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    readiness = run_topology_only_readiness(
        config=config,
        project_root=ROOT,
    )

    assert tuple(config["source_ciphers"]) == SOURCE_CIPHERS
    assert readiness["status"] == "pass"
    assert all(readiness["checks"].values())
    assert set(readiness["parameter_counts"].values()) == {442466}
    assert all(value == 0.0 for value in readiness["sbox_logit_deltas"].values())
    assert all(
        value > 0.0
        for by_cipher in readiness["topology_logit_deltas"].values()
        for value in by_cipher.values()
    )
    assert readiness["target_training_rows"] == 0
    assert readiness["target_optimizer_steps"] == 0


def test_c1_a8_reference_panel_is_exact() -> None:
    references = load_a8_references(A8_RESULTS)

    assert len(references["source_rows"]) == 8
    assert len(references["target_rows"]) == 4
    for seed in ("0", "1"):
        assert set(references["source_auc"][seed]) == set(SOURCE_CIPHERS)
        assert set(references["target_auc"][seed]) == {
            "candidate_correct",
            "no_topology_trained_anchor",
        }


def test_c1_gate_distinguishes_supported_hold_and_invalid() -> None:
    payload = _gate_payload()
    passed = adjudicate_topology_only(payload)
    assert passed["status"] == "pass"
    assert passed["decision"].endswith("topology_only_dialga_supported")

    held_payload = _gate_payload()
    held_payload["target_auc"]["1"]["candidate_corrupted_target"] = 0.70
    held = adjudicate_topology_only(held_payload)
    assert held["status"] == "hold"
    assert held["decision"].endswith("topology_only_dialga_not_supported")

    invalid_payload = _gate_payload()
    invalid_payload["validation"] = {"status": "fail", "checks": {"ok": False}}
    invalid = adjudicate_topology_only(invalid_payload)
    assert invalid["status"] == "invalid"
    assert invalid["decision"].endswith("topology_only_c1_protocol_invalid")


def _gate_payload() -> dict[str, Any]:
    config = load_and_validate_topology_only_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    target = {
        "candidate_correct": 0.70,
        "candidate_corrupted_target": 0.65,
        "candidate_no_topology_target": 0.52,
        "candidate_wrong_sbox_target": 0.70,
        "a8_correct": 0.695,
        "a8_trained_no_topology": 0.51,
    }
    source = {"candidate": 0.60, "a8_correct": 0.599}
    return {
        "config": config,
        "validation": {"status": "pass", "checks": {"ok": True}},
        "source_macro_auc": {"0": dict(source), "1": dict(source)},
        "target_auc": {"0": dict(target), "1": dict(target)},
        "sbox_probability_delta": {"0": 0.0, "1": 0.0},
        "conflict_projections_by_seed": {"0": 2, "1": 2},
    }
