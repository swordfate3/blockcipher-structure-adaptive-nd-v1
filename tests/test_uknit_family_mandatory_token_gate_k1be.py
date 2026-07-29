from __future__ import annotations

from copy import deepcopy
import math

from blockcipher_nd.cli.plot_uknit_family_mandatory_token_gate_k1be import (
    render_k1be_svg,
)
from blockcipher_nd.models.structure.spn.position_preserving_operator import (
    trainable_parameter_geometry,
)
from blockcipher_nd.tasks.innovation1.uknit_family_mandatory_token_gate_k1be import (
    EXPECTED_TRAINABLE_PARAMETERS,
    RELABEL_TOLERANCE,
    WRONG_CONDITIONS,
    adjudicate_readiness,
    build_candidate_probe,
    load_and_validate_config,
    load_authority,
    measure_gradient_coverage,
    measure_panel,
)
from blockcipher_nd.tasks.innovation1.uknit_family_position_preserving_operator_k1bc import (
    build_probe as build_anchor_probe,
    load_and_validate_config as load_k1bc_config,
)


def test_k1be_authority_and_non_bypass_geometry_are_exact() -> None:
    config = load_and_validate_config()
    (
        runtime_config,
        dataset_rows,
        datasets,
        structures,
        _summaries,
        source_checkpoints,
        corrupted,
        cross,
        checks,
    ) = load_authority(config)
    candidate = build_candidate_probe(
        runtime_config=runtime_config,
        structures=structures,
        checkpoint=source_checkpoints[0],
        initialization_seed=40,
        model_config=config["model"],
        device="cpu",
    )

    assert all(checks.values()), checks
    assert len(dataset_rows) == len(datasets) == 18
    assert set(corrupted) == set(cross) == set(structures)
    assert sum(
        parameter.numel()
        for parameter in candidate.parameters()
        if parameter.requires_grad
    ) == EXPECTED_TRAINABLE_PARAMETERS
    assert not hasattr(candidate.operator_encoder, "structure_projection")
    assert candidate.sample_only_bypass is False
    assert candidate.readiness_only_projection_present is False
    assert candidate.uses_cipher_identity is False
    assert candidate.uses_per_cipher_parameters is False

    geometry = trainable_parameter_geometry(candidate.operator_encoder)
    assert "sample_message.0.weight" in geometry
    assert all("structure_projection" not in name for name in geometry)
    assert structures["uknit64"].block_bits == 64
    assert structures["dialga128"].block_bits == 128


def test_k1be_real_probe_connects_all_tensors_and_preserves_compatibility() -> None:
    config = load_and_validate_config()
    (
        runtime_config,
        _dataset_rows,
        datasets,
        structures,
        summaries,
        source_checkpoints,
        corrupted,
        cross,
        checks,
    ) = load_authority(config)
    assert all(checks.values()), checks
    candidate = build_candidate_probe(
        runtime_config=runtime_config,
        structures=structures,
        checkpoint=source_checkpoints[0],
        initialization_seed=40,
        model_config=config["model"],
        device="cpu",
    )
    source_config = load_k1bc_config()
    anchor = build_anchor_probe(
        runtime_config=runtime_config,
        structures=structures,
        checkpoint=source_checkpoints[0],
        initialization_seed=40,
        model_config=source_config["model"],
        device="cpu",
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
        anchor=anchor,
        dataset=datasets[("uknit64", 3, "same_key_fresh")],
        structure=structure,
        corrupted=corrupted["uknit64"],
        cross_operator=cross["uknit64"],
        summary=summary,
        replica=0,
        cipher="uknit64",
        seed=3,
        split="same_key_fresh",
        device="cpu",
    )

    assert gradient["parameter_tensor_count"] > 0
    assert (
        gradient["graph_connected_tensor_count"]
        == gradient["parameter_tensor_count"]
    )
    assert gradient["persistent_grads_none"] is True
    assert panel["disabled_k1az_logit_replay_delta"] == 0.0
    assert panel["joint_relabel_modulation_delta"] <= RELABEL_TOLERANCE
    assert panel["joint_relabel_logit_delta"] <= RELABEL_TOLERANCE
    assert all(
        math.isfinite(float(value))
        for key, value in panel.items()
        if key.endswith(("_rms", "_share", "_delta"))
    )


def test_k1be_gate_separates_pass_hold_and_invalid() -> None:
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
    held = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        panels=weak,
        gradients=gradients,
        geometry=geometry,
    )
    assert held["status"] == "hold"
    assert "path_too_weak" in held["decision"]

    disconnected = deepcopy(gradients)
    disconnected[0]["graph_connected_tensor_count"] -= 1
    compatibility_hold = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        panels=panels,
        gradients=disconnected,
        geometry=geometry,
    )
    assert compatibility_hold["status"] == "hold"
    assert "compatibility_incomplete" in compatibility_hold["decision"]

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


def test_k1be_plot_writes_clear_chinese_svg(tmp_path) -> None:
    config = load_and_validate_config()
    panels = _synthetic_panels()
    gradients = _synthetic_gradients()
    for row in panels:
        for condition in WRONG_CONDITIONS:
            row[f"candidate_{condition}_topology_share"] = 0.009
    gate = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        panels=panels,
        gradients=gradients,
        geometry=_synthetic_geometry(),
    )
    output = tmp_path / "curves.svg"

    report = render_k1be_svg(gate, panels, gradients, output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["result_panels"] == 12
    assert "必经乘法门保住了路径，但没有增加拓扑依赖" in text
    assert "停止K1-BF训练" in text
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
                    "anchor_whole_path_probability_rms": 0.01,
                    "disabled_k1az_logit_replay_delta": 0.0,
                    "joint_relabel_modulation_delta": 1e-6,
                    "joint_relabel_logit_delta": 1e-6,
                    "training_performed": False,
                    "optimizer_steps": 0,
                    "candidate_state_immutable": True,
                    "anchor_state_immutable": True,
                }
                for condition in WRONG_CONDITIONS:
                    row.update(
                        {
                            f"candidate_{condition}_modulation_rms": 0.01,
                            f"candidate_{condition}_logit_rms": 0.001,
                            f"candidate_{condition}_probability_rms": 0.001,
                            f"candidate_{condition}_topology_share": 0.1,
                            f"anchor_{condition}_modulation_rms": 0.01,
                            f"anchor_{condition}_logit_rms": 0.001,
                            f"anchor_{condition}_probability_rms": 0.0001,
                            f"anchor_{condition}_topology_share": 0.01,
                        }
                    )
                rows.append(row)
    return rows


def _synthetic_gradients() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "parameter_tensor_count": 20,
            "graph_connected_tensor_count": 20,
            "persistent_grads_none": True,
            "loss": 0.25,
            "training_performed": False,
            "optimizer_steps": 0,
            "candidate_state_immutable": True,
            "anchor_state_immutable": True,
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
            "sample_only_bypass": False,
            "readiness_only_projection_present": False,
            "uses_cipher_identity": False,
            "uses_per_cipher_parameters": False,
        }
        for replica in (0, 1)
        for cipher in ("uknit64", "midori64", "dialga128")
    ]
