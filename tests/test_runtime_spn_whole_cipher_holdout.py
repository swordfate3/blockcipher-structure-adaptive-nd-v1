from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    rectangle80_runtime_structure,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    EXPECTED_SOURCES,
    RelationModeRuntimeE4,
    _target_control_probe,
    adjudicate_holdout_experiment,
    load_and_validate_holdout_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_rectangle_whole_cipher_holdout_h1_2048_seed0_seed1.json"
)


def test_h1_config_excludes_rectangle_from_all_source_tasks() -> None:
    config = load_and_validate_holdout_config(CONFIG)

    assert config["holdout_cipher"] == "rectangle80"
    assert tuple(config["source_ciphers"]) == EXPECTED_SOURCES
    assert "rectangle80" not in config["source_ciphers"]
    assert config["model"]["conditioner"] == "none"
    assert config["model"]["expected_parameter_count"] == 442_466


def test_plain_shared_state_handles_target_counterfactuals() -> None:
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=64,
        pair_embedding_dim=128,
        processor_steps=2,
        dropout=0.0,
        sbox_context_mode="edge_gate",
        cell_input_mode="state_triplet",
        round_window_mode="recurrent_window",
    )
    model = RelationModeRuntimeE4(spec, "true")

    assert sum(parameter.numel() for parameter in model.parameters()) == 442_466
    assert all(_target_control_probe(model, rectangle80_runtime_structure(2)).values())


def test_h1_gate_passes_only_with_source_and_target_structure_margins() -> None:
    config = load_and_validate_holdout_config(CONFIG)
    payload = _passing_payload(config)

    gate = adjudicate_holdout_experiment(payload)

    assert gate["status"] == "pass"
    assert gate["full_pass"] is True
    assert all(seed["pass"] for seed in gate["per_seed"].values())


def test_h1_gate_fails_closed_on_leakage_contract_failure() -> None:
    config = load_and_validate_holdout_config(CONFIG)
    payload = _passing_payload(config)
    payload = deepcopy(payload)
    payload["validation"]["status"] = "fail"

    gate = adjudicate_holdout_experiment(payload)

    assert gate["status"] == "fail"
    assert gate["protocol_valid"] is False
    assert gate["decision"].endswith("protocol_invalid")


def test_h1_gate_holds_when_same_checkpoint_target_topology_is_not_used() -> None:
    config = load_and_validate_holdout_config(CONFIG)
    payload = _passing_payload(config)
    payload = deepcopy(payload)
    payload["target_metrics"]["1"]["candidate_corrupted_target"] = 0.58

    gate = adjudicate_holdout_experiment(payload)

    assert gate["status"] == "hold"
    assert gate["per_seed"]["1"]["checks"]["target_controls"] is False


def _passing_payload(config: dict) -> dict:
    target = {
        str(seed): {
            "candidate_correct": 0.58,
            "candidate_corrupted_target": 0.56,
            "candidate_no_topology_target": 0.54,
            "corrupted_source_control": 0.55,
            "no_topology_source_control": 0.53,
        }
        for seed in (0, 1)
    }
    source = {
        str(seed): {"correct": 0.62, "corrupted": 0.60, "no_topology": 0.55}
        for seed in (0, 1)
    }
    return {
        "config": config,
        "validation": {"status": "pass"},
        "target_metrics": target,
        "source_macro_auc": source,
    }
