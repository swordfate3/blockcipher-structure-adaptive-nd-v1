from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from blockcipher_nd.cli.plot_uknit_family_component_separated_structure_gate_k1ay import (
    render_k1ay_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_component_separated_structure_gate_k1ay import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_STATE_ENTRIES,
    _component_gradient_metrics,
    adjudicate,
    audit_candidate_geometry,
    build_candidate,
    load_and_validate_config,
    load_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1av import (
    build_candidate as build_k1aw_candidate,
)


def test_k1ay_config_source_and_geometry_are_exact() -> None:
    config = load_and_validate_config()
    (
        readiness,
        k1av,
        _dataset_rows,
        _datasets,
        _structures,
        _controls,
        _summary_rows,
        checkpoints,
        source_checks,
    ) = load_authority(config)
    assert all(source_checks.values()), source_checks
    assert set(checkpoints) == {0, 1}

    geometry_checks, report = audit_candidate_geometry(
        readiness_config=readiness,
        k1av_config=k1av,
        config=config,
    )
    assert all(geometry_checks.values()), geometry_checks
    assert set(report["parameter_counts"].values()) == {EXPECTED_PARAMETER_COUNT}
    assert set(report["state_entries"].values()) == {EXPECTED_STATE_ENTRIES}


def test_k1ay_strict_load_compatibility_and_component_isolation() -> None:
    config = load_and_validate_config()
    (
        readiness,
        k1av,
        _dataset_rows,
        datasets,
        structures,
        controls,
        _summary_rows,
        checkpoints,
        source_checks,
    ) = load_authority(config)
    assert all(source_checks.values()), source_checks
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness["ciphers"]
    }
    source = build_k1aw_candidate(
        cipher_configs["uknit64"], readiness["model"], k1av["model"]
    )
    candidate = build_candidate(
        cipher_configs["uknit64"], readiness["model"], config["model"]
    )
    source.load_state_dict(checkpoints[0]["state_dict"], strict=True)
    candidate.load_state_dict(checkpoints[0]["state_dict"], strict=True)
    assert tuple(source.state_dict()) == tuple(candidate.state_dict())
    assert all(
        torch.equal(source.state_dict()[name], candidate.state_dict()[name])
        for name in source.state_dict()
    )

    structure = structures["uknit64"]
    correct = controls["uknit64"]["correct_descriptor"]
    sbox_mismatch = controls["uknit64"]["sbox_only_mismatch"]
    linear_mismatch = controls["uknit64"]["linear_only_mismatch"]
    assert correct is not None
    assert sbox_mismatch is not None
    assert linear_mismatch is not None
    gradients = _component_gradient_metrics(candidate, structure, correct)
    assert gradients["edge_linear_summary_jacobian_l2"] >= 1e-6
    assert gradients["edge_sbox_summary_jacobian_l2"] == 0.0
    assert gradients["transition_sbox_summary_jacobian_l2"] >= 1e-6
    assert gradients["transition_linear_summary_jacobian_l2"] == 0.0

    correct_gates = candidate.effective_path_gates(
        structure,
        summary=correct,
        component_separation_enabled=True,
    )
    sbox_gates = candidate.effective_path_gates(
        structure,
        summary=sbox_mismatch,
        component_separation_enabled=True,
    )
    linear_gates = candidate.effective_path_gates(
        structure,
        summary=linear_mismatch,
        component_separation_enabled=True,
    )
    assert torch.equal(correct_gates[0], sbox_gates[0])
    assert not torch.equal(correct_gates[1], sbox_gates[1])
    assert not torch.equal(correct_gates[0], linear_gates[0])
    assert torch.equal(correct_gates[1], linear_gates[1])

    dataset = datasets[("uknit64", 3, "same_key_fresh")]
    features = torch.as_tensor(
        np.array(dataset.features[:4], copy=True), dtype=torch.float32
    )
    source.eval()
    candidate.eval()
    with torch.inference_mode():
        expected = source.logits_with_runtime(
            features,
            structure,
            apply_sboxes=True,
            transition_branch_enabled=True,
            gate_summary=correct,
            dual_path_enabled=True,
        )
        actual = candidate.logits_with_runtime(
            features,
            structure,
            apply_sboxes=True,
            transition_branch_enabled=True,
            gate_summary=correct,
            dual_path_enabled=True,
            component_separation_enabled=False,
        )
        separated = candidate.logits_with_runtime(
            features,
            structure,
            apply_sboxes=True,
            transition_branch_enabled=True,
            gate_summary=correct,
            dual_path_enabled=True,
            component_separation_enabled=True,
        )
    assert torch.equal(actual, expected)
    assert not torch.equal(separated, actual)


def test_k1ay_gate_separates_pass_hold_and_invalid() -> None:
    config = load_and_validate_config()
    results = _synthetic_results()
    controls = _synthetic_controls()
    loads = _synthetic_loads()
    passed = adjudicate(
        config=config,
        source_checks={"source": True},
        geometry_checks={"geometry": True},
        results=results,
        controls=controls,
        loads=loads,
    )
    assert passed["status"] == "pass"
    assert passed["next_training_authorized"] is True
    assert "K1-AZ" in passed["next_action"]

    leaked = deepcopy(controls)
    for row in leaked:
        if row["condition"] == "sbox_only_mismatch":
            row["edge_gate_delta"] = 1e-4
    held = adjudicate(
        config=config,
        source_checks={"source": True},
        geometry_checks={"geometry": True},
        results=results,
        controls=leaked,
        loads=loads,
    )
    assert held["status"] == "hold"
    assert held["next_training_authorized"] is False
    assert "sbox_mismatch_isolated_to_transition_gate" in held[
        "failed_research_checks"
    ]

    replay_drift = deepcopy(results)
    replay_drift[0]["disabled_k1aw_max_abs_logit_delta"] = 1e-7
    invalid = adjudicate(
        config=config,
        source_checks={"source": True},
        geometry_checks={"geometry": True},
        results=replay_drift,
        controls=controls,
        loads=loads,
    )
    assert invalid["status"] == "invalid"
    assert "disabled_mode_exactly_replays_k1aw" in invalid[
        "failed_protocol_checks"
    ]


def test_k1ay_plot_writes_four_panel_chinese_svg(tmp_path) -> None:
    gate = adjudicate(
        config=load_and_validate_config(),
        source_checks={"source": True},
        geometry_checks={"geometry": True},
        results=_synthetic_results(),
        controls=_synthetic_controls(),
        loads=_synthetic_loads(),
    )
    output = tmp_path / "curves.svg"

    report = render_k1ay_svg(
        gate,
        _synthetic_results(),
        _synthetic_controls(),
        output,
    )

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["result_rows"] == 12
    assert "S盒与线性扩散信息已被隔离到各自门控" in text
    assert "不增加16 pairs" in text


def _synthetic_results() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "split": split,
            "disabled_k1aw_max_abs_logit_delta": 0.0,
            "enabled_max_abs_logit_delta": 0.01,
            "edge_linear_summary_jacobian_l2": 0.02,
            "edge_sbox_summary_jacobian_l2": 0.0,
            "transition_sbox_summary_jacobian_l2": 0.03,
            "transition_linear_summary_jacobian_l2": 0.0,
            "all_gate_values_finite_bounded": True,
            "state_immutable": True,
            "training_performed": False,
            "optimizer_steps": 0,
        }
        for replica in (0, 1)
        for cipher in ("uknit64", "midori64", "dialga128")
        for split in ("same_key_fresh", "cross_key_validation")
    ]


def _synthetic_controls() -> list[dict[str, object]]:
    deltas = {
        "full_mismatch": (0.01, 0.02),
        "sbox_only_mismatch": (0.0, 0.02),
        "linear_only_mismatch": (0.01, 0.0),
    }
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "split": split,
            "condition": condition,
            "edge_gate_delta": deltas[condition][0],
            "transition_gate_delta": deltas[condition][1],
            "max_abs_logit_delta": 0.01,
            "state_immutable": True,
            "training_performed": False,
            "optimizer_steps": 0,
        }
        for replica in (0, 1)
        for cipher in ("uknit64", "midori64", "dialga128")
        for split in ("same_key_fresh", "cross_key_validation")
        for condition in deltas
    ]


def _synthetic_loads() -> list[dict[str, object]]:
    return [
        {"replica": replica, "strict_load_exact": True} for replica in (0, 1)
    ]
