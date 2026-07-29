from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from blockcipher_nd.cli.plot_uknit_family_dual_path_structure_modulation_k1av import (
    render_k1av_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1as import (
    build_candidate as build_k1as_candidate,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
    MISMATCH_CONDITIONS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_gate_identifiability_k1au import (
    EXPECTED_REPLICAS,
    REPLICA_DATASET_SEEDS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1av import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_STATE_ENTRIES,
    OUTPUT_WEIGHT_KEY,
    _gradient_metrics,
    adjudicate,
    audit_candidate_geometry,
    build_candidate,
    build_structure_report,
    load_and_validate_config,
    load_authority,
    migrate_k1at_state,
)


def test_k1av_config_geometry_and_source_authority_are_exact() -> None:
    config = load_and_validate_config()
    (
        readiness,
        _k1as,
        _datasets,
        structures,
        structure_controls,
        checkpoints,
        _checkpoint_rows,
        source_checks,
    ) = load_authority(config)
    assert all(source_checks.values())
    assert set(checkpoints) == set(EXPECTED_REPLICAS)

    geometry_checks, report = audit_candidate_geometry(
        readiness_config=readiness,
        config=config,
    )
    assert all(geometry_checks.values())
    assert set(report["parameter_counts"].values()) == {EXPECTED_PARAMETER_COUNT}
    assert set(report["state_entries"].values()) == {EXPECTED_STATE_ENTRIES}

    structure_rows, structure_checks = build_structure_report(
        structures=structures,
        structure_controls=structure_controls,
        tolerance=0.0,
    )
    assert len(structure_rows) == 3
    assert all(structure_checks.values())


def test_k1av_migration_and_compatibility_mode_exactly_replay_k1at() -> None:
    config = load_and_validate_config()
    (
        readiness,
        k1as,
        datasets,
        structures,
        structure_controls,
        checkpoints,
        _checkpoint_rows,
        source_checks,
    ) = load_authority(config)
    assert all(source_checks.values())
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness["ciphers"]
    }
    source = build_k1as_candidate(
        cipher_configs["uknit64"], readiness["model"], k1as["model"]
    )
    source.load_state_dict(checkpoints[0]["state_dict"], strict=True)
    with torch.random.fork_rng():
        torch.manual_seed(config["model"]["initialization_seed"])
        candidate = build_candidate(
            cipher_configs["uknit64"], readiness["model"], config["model"]
        )
    migration = migrate_k1at_state(candidate, checkpoints[0]["state_dict"])
    assert migration["only_final_projection_expanded"] is True
    assert migration["transition_row_exact"] is True
    assert migration["new_edge_row_finite_nonzero"] is True
    assert tuple(candidate.state_dict()[OUTPUT_WEIGHT_KEY].shape) == (2, 12)

    structure = structures["uknit64"]
    summary = structure_controls["uknit64"]["correct_descriptor"]
    assert summary is not None
    dataset = datasets[
        (
            "uknit64",
            REPLICA_DATASET_SEEDS[0]["uknit64"],
            "same_key_fresh",
        )
    ]
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
            gate_summary=summary,
            structure_gate_enabled=True,
        )
        actual = candidate.logits_with_runtime(
            features,
            structure,
            apply_sboxes=True,
            transition_branch_enabled=True,
            gate_summary=summary,
            dual_path_enabled=False,
        )
    assert torch.equal(actual, expected)


def test_k1av_channels_have_separate_output_rows_and_component_sensitivity() -> None:
    config = load_and_validate_config()
    (
        readiness,
        _k1as,
        _datasets,
        structures,
        structure_controls,
        checkpoints,
        _checkpoint_rows,
        source_checks,
    ) = load_authority(config)
    assert all(source_checks.values())
    cipher = readiness["ciphers"][0]
    with torch.random.fork_rng():
        torch.manual_seed(config["model"]["initialization_seed"])
        candidate = build_candidate(cipher, readiness["model"], config["model"])
    migrate_k1at_state(candidate, checkpoints[0]["state_dict"])
    summary = structure_controls["uknit64"]["correct_descriptor"]
    assert summary is not None
    metrics = _gradient_metrics(candidate, structures["uknit64"], summary)
    assert metrics["all_channel_gradients_finite"] is True
    assert metrics["edge_linear_summary_jacobian_l2"] >= 1e-6
    assert metrics["transition_sbox_summary_jacobian_l2"] >= 1e-6
    assert metrics["edge_own_row_parameter_jacobian_l2"] >= 1e-6
    assert metrics["transition_own_row_parameter_jacobian_l2"] >= 1e-6
    assert metrics["edge_cross_row_parameter_jacobian_l2"] == 0.0
    assert metrics["transition_cross_row_parameter_jacobian_l2"] == 0.0


def _synthetic_results() -> list[dict]:
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "split": split,
            "disabled_k1at_max_abs_logit_delta": 0.0,
            "enabled_max_abs_logit_delta": 0.01,
            "edge_linear_summary_jacobian_l2": 0.02,
            "transition_sbox_summary_jacobian_l2": 0.03,
            "edge_own_row_parameter_jacobian_l2": 0.4,
            "edge_cross_row_parameter_jacobian_l2": 0.0,
            "transition_own_row_parameter_jacobian_l2": 0.5,
            "transition_cross_row_parameter_jacobian_l2": 0.0,
            "all_gate_values_finite_bounded": True,
            "state_immutable": True,
            "training_performed": False,
            "optimizer_steps": 0,
        }
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    ]


def _synthetic_controls() -> list[dict]:
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "split": split,
            "condition": condition,
            "edge_gate_delta": 0.01,
            "transition_gate_delta": 0.01,
            "max_abs_logit_delta": 0.01,
            "state_immutable": True,
            "training_performed": False,
            "optimizer_steps": 0,
        }
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
        for condition in MISMATCH_CONDITIONS
    ]


def _synthetic_migrations() -> list[dict]:
    return [
        {
            "replica": replica,
            "only_final_projection_expanded": True,
            "transition_row_exact": True,
            "new_edge_row_finite_nonzero": True,
        }
        for replica in EXPECTED_REPLICAS
    ]


def test_k1av_gate_separates_pass_hold_and_protocol_invalid() -> None:
    config = load_and_validate_config()
    results = _synthetic_results()
    controls = _synthetic_controls()
    migrations = _synthetic_migrations()
    passed = adjudicate(
        config=config,
        source_checks={"source": True},
        geometry_checks={"geometry": True},
        structure_checks={"structure": True},
        results=results,
        controls=controls,
        migrations=migrations,
    )
    assert passed["status"] == "pass"
    assert passed["next_training_authorized"] is True
    assert "K1-AW" in passed["next_action"]

    weak_controls = deepcopy(controls)
    for row in weak_controls:
        if row["condition"] == "linear_only_mismatch":
            row["edge_gate_delta"] = 0.0
    held = adjudicate(
        config=config,
        source_checks={"source": True},
        geometry_checks={"geometry": True},
        structure_checks={"structure": True},
        results=results,
        controls=weak_controls,
        migrations=migrations,
    )
    assert held["status"] == "hold"
    assert held["next_training_authorized"] is False
    assert "linear_mismatch_changes_edge_gate" in held["failed_research_checks"]

    replay_drift = deepcopy(results)
    replay_drift[0]["disabled_k1at_max_abs_logit_delta"] = 1e-7
    invalid = adjudicate(
        config=config,
        source_checks={"source": True},
        geometry_checks={"geometry": True},
        structure_checks={"structure": True},
        results=replay_drift,
        controls=controls,
        migrations=migrations,
    )
    assert invalid["status"] == "invalid"
    assert "disabled_mode_exactly_replays_k1at" in invalid[
        "failed_protocol_checks"
    ]


def test_k1av_plot_writes_four_panel_chinese_svg(tmp_path) -> None:
    gate = {
        "status": "pass",
        "maximum_disabled_k1at_logit_replay_delta": 0.0,
        "minimum_edge_linear_summary_jacobian_l2": 0.016,
        "minimum_transition_sbox_summary_jacobian_l2": 0.035,
    }
    output = tmp_path / "curves.svg"
    report = render_k1av_svg(
        gate,
        _synthetic_results(),
        _synthetic_controls(),
        output,
    )
    text = output.read_text(encoding="utf-8")
    assert report["status"] == "rendered_pending_visual_qa"
    assert report["panels"] == 4
    assert "双通道候选" in text
    assert "同一结构摘要" in text
    assert "跨通道严格为0" in text
