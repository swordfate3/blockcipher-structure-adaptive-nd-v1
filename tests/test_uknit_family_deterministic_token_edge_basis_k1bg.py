from __future__ import annotations

from copy import deepcopy
import math

import torch

from blockcipher_nd.cli.plot_uknit_family_deterministic_token_edge_basis_k1bg import (
    render_k1bg_svg,
)
from blockcipher_nd.models.structure.spn.deterministic_token_edge_basis import (
    DeterministicTokenEdgeBasisOperatorEncoder,
)
from blockcipher_nd.models.structure.spn.position_preserving_operator import (
    OPERATOR_TOKEN_DIM,
    PositionPreservingOperatorSpec,
    trainable_parameter_geometry,
)
from blockcipher_nd.tasks.innovation1.uknit_family_deterministic_token_edge_basis_k1bg import (
    BASIS_GRAM_TOLERANCE,
    EXPECTED_TRAINABLE_PARAMETERS,
    RELABEL_TOLERANCE,
    SOURCE_REPLAY_TOLERANCE,
    WRONG_CONDITIONS,
    adjudicate_readiness,
    build_candidate_probe,
    load_and_validate_config,
    load_authority,
    measure_gradient_coverage,
    measure_panel,
)
from blockcipher_nd.tasks.innovation1.uknit_family_mandatory_token_gate_k1be import (
    build_candidate_probe as build_k1be_probe,
    load_and_validate_config as load_k1be_config,
)
from blockcipher_nd.tasks.innovation1.uknit_family_position_preserving_operator_k1bc import (
    build_probe as build_k1bc_probe,
    load_and_validate_config as load_k1bc_config,
)


def test_k1bg_fixed_basis_is_full_rank_deterministic_and_non_trainable() -> None:
    encoder = DeterministicTokenEdgeBasisOperatorEncoder(
        PositionPreservingOperatorSpec()
    )
    projection = encoder.basis_projection
    gram_error = float(
        torch.max(torch.abs(projection @ projection.T - torch.eye(OPERATOR_TOKEN_DIM)))
    )
    tokens = torch.zeros(2, OPERATOR_TOKEN_DIM)
    tokens[:, -1] = 1.0
    tokens[1, 0] = 1.0
    first = encoder.fixed_edge_basis(tokens)
    second = encoder.fixed_edge_basis(tokens.clone())

    assert int(torch.linalg.matrix_rank(projection)) == OPERATOR_TOKEN_DIM
    assert gram_error <= BASIS_GRAM_TOLERANCE
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    assert "basis_projection" in dict(encoder.named_buffers())
    assert "basis_projection" not in dict(encoder.named_parameters())
    assert not hasattr(encoder, "token_encoder")


def test_k1bg_authority_geometry_and_real_panel_are_exact() -> None:
    config = load_and_validate_config()
    (
        runtime_config,
        dataset_rows,
        datasets,
        structures,
        summaries,
        source_checkpoints,
        corrupted,
        cross,
        source_panels,
        checks,
    ) = load_authority(config)
    common = {
        "runtime_config": runtime_config,
        "structures": structures,
        "checkpoint": source_checkpoints[0],
        "initialization_seed": 40,
        "device": "cpu",
    }
    candidate = build_candidate_probe(**common, model_config=config["model"])
    k1be = build_k1be_probe(
        **common,
        model_config=load_k1be_config()["model"],
    )
    k1bc = build_k1bc_probe(
        **common,
        model_config=load_k1bc_config()["model"],
    )
    structure = structures["uknit64"]
    summary = summaries["uknit64"]["correct_descriptor"]
    assert summary is not None

    gradient = measure_gradient_coverage(
        probe=candidate,
        dataset=datasets[("uknit64", 3, "train_seen")],
        runtime_structure=structure,
        operator_structure=structure,
        summary=summary,
        replica=0,
        cipher="uknit64",
        device="cpu",
    )
    panel = measure_panel(
        candidate=candidate,
        k1be=k1be,
        k1bc=k1bc,
        dataset=datasets[("uknit64", 3, "same_key_fresh")],
        structure=structure,
        corrupted=corrupted["uknit64"],
        cross_operator=cross["uknit64"],
        summary=summary,
        source_panel=source_panels[(0, "uknit64", "same_key_fresh")],
        replica=0,
        cipher="uknit64",
        seed=3,
        split="same_key_fresh",
        device="cpu",
    )

    assert all(checks.values()), checks
    assert len(dataset_rows) == len(datasets) == 18
    assert set(corrupted) == set(cross) == set(structures)
    assert len(source_panels) == 12
    assert sum(
        parameter.numel()
        for parameter in candidate.parameters()
        if parameter.requires_grad
    ) == EXPECTED_TRAINABLE_PARAMETERS
    geometry = trainable_parameter_geometry(candidate.operator_encoder)
    assert all("token_encoder" not in name for name in geometry)
    assert all("structure_projection" not in name for name in geometry)
    assert gradient["graph_connected_tensor_count"] == gradient["parameter_tensor_count"]
    assert gradient["persistent_grads_none"] is True
    assert panel["matched_k1be_source_replay_max_abs_delta"] <= SOURCE_REPLAY_TOLERANCE
    assert panel["disabled_k1az_logit_replay_delta"] == 0.0
    assert panel["joint_relabel_modulation_delta"] <= RELABEL_TOLERANCE
    assert panel["joint_relabel_logit_delta"] <= RELABEL_TOLERANCE
    assert all(
        math.isfinite(float(value))
        for key, value in panel.items()
        if key.endswith(("_rms", "_share", "_delta"))
    )


def test_k1bg_gate_separates_pass_hold_and_invalid() -> None:
    config = load_and_validate_config()
    panels = _synthetic_panels()
    gradients = _synthetic_gradients()
    geometry = _synthetic_geometry()

    passed = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        panels=panels,
        gradients=gradients,
        geometry=geometry,
    )
    assert passed["status"] == "pass"
    assert "readiness_authorized" in passed["decision"]

    weak = deepcopy(panels)
    weak[0]["candidate_whole_path_probability_rms"] = 0.004
    held_path = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        panels=weak,
        gradients=gradients,
        geometry=geometry,
    )
    assert held_path["status"] == "hold"
    assert "path_too_weak" in held_path["decision"]

    weak_topology = deepcopy(panels)
    for row in weak_topology:
        for condition in WRONG_CONDITIONS:
            row[f"candidate_{condition}_topology_share"] = 0.03
    held_topology = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        panels=weak_topology,
        gradients=gradients,
        geometry=geometry,
    )
    assert held_topology["status"] == "hold"
    assert "topology_lift_not_supported" in held_topology["decision"]

    replay_drift = deepcopy(panels)
    replay_drift[0]["matched_k1be_source_replay_max_abs_delta"] = 1e-5
    held_compatibility = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        panels=replay_drift,
        gradients=gradients,
        geometry=geometry,
    )
    assert held_compatibility["status"] == "hold"
    assert "compatibility_incomplete" in held_compatibility["decision"]

    updated = deepcopy(panels)
    updated[0]["optimizer_steps"] = 1
    invalid = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        panels=updated,
        gradients=gradients,
        geometry=geometry,
    )
    assert invalid["status"] == "invalid"
    assert "zero_updates_and_immutable_states" in invalid["failed_protocol_checks"]


def test_k1bg_plot_writes_clear_chinese_svg(tmp_path) -> None:
    panels = _synthetic_panels()
    gradients = _synthetic_gradients()
    geometry = _synthetic_geometry()
    for row in panels:
        for condition in WRONG_CONDITIONS:
            row[f"candidate_{condition}_topology_share"] = 0.005
    gate = adjudicate_readiness(
        config=load_and_validate_config(),
        source_checks={"source": True},
        panels=panels,
        gradients=gradients,
        geometry=geometry,
    )
    output = tmp_path / "curves.svg"

    report = render_k1bg_svg(gate, panels, gradients, geometry, output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["result_panels"] == 12
    assert "固定正交边基没有救回拓扑依赖" in text
    assert "停止学习边消息再池化的路线" in text
    assert "不是准确率提升" in text


def _synthetic_panels() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for replica in (0, 1):
        for cipher in ("uknit64", "midori64", "dialga128"):
            for split in ("same_key_fresh", "cross_key_validation"):
                row: dict[str, object] = {
                    "replica": replica,
                    "cipher_key": cipher,
                    "split": split,
                    "candidate_whole_path_probability_rms": 0.01,
                    "k1be_whole_path_probability_rms": 0.01,
                    "k1bc_whole_path_probability_rms": 0.01,
                    "matched_k1be_source_replay_max_abs_delta": 0.0,
                    "disabled_k1az_logit_replay_delta": 0.0,
                    "joint_relabel_modulation_delta": 1e-6,
                    "joint_relabel_logit_delta": 1e-6,
                    "training_performed": False,
                    "optimizer_steps": 0,
                    "candidate_state_immutable": True,
                    "k1be_state_immutable": True,
                    "k1bc_state_immutable": True,
                }
                for condition in WRONG_CONDITIONS:
                    for model, share in (
                        ("candidate", 0.05),
                        ("k1be", 0.02),
                        ("k1bc", 0.01),
                    ):
                        row.update(
                            {
                                f"{model}_{condition}_modulation_rms": 0.01,
                                f"{model}_{condition}_logit_rms": 0.001,
                                f"{model}_{condition}_probability_rms": 0.0005,
                                f"{model}_{condition}_topology_share": share,
                            }
                        )
                rows.append(row)
    return rows


def _synthetic_gradients() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "parameter_tensor_count": 18,
            "graph_connected_tensor_count": 18,
            "persistent_grads_none": True,
            "loss": 0.25,
            "training_performed": False,
            "optimizer_steps": 0,
            "candidate_state_immutable": True,
            "k1be_state_immutable": True,
            "k1bc_state_immutable": True,
        }
        for replica in (0, 1)
        for cipher in ("uknit64", "midori64", "dialga128")
    ]


def _synthetic_geometry() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
            "trainable_parameter_geometry": {"shared.weight": [32, 32]},
            "basis_projection_rank": 18,
            "basis_projection_gram_max_abs_error": 1e-8,
            "basis_projection_sha256": "a" * 64,
            "basis_projection_trainable": False,
            "token_encoder_present": False,
            "sample_only_bypass": False,
            "readiness_only_projection_present": False,
            "uses_cipher_identity": False,
            "uses_per_cipher_parameters": False,
        }
        for replica in (0, 1)
        for cipher in ("uknit64", "midori64", "dialga128")
    ]
