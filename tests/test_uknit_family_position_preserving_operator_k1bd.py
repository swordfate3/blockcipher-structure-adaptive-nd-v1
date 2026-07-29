from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from blockcipher_nd.cli.plot_uknit_family_position_preserving_operator_k1bd import (
    render_k1bd_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_position_preserving_operator_k1bc import (
    build_probe,
    load_and_validate_config as load_source_config,
)
from blockcipher_nd.tasks.innovation1.uknit_family_position_preserving_operator_k1bd import (
    ENCODER_STATES,
    EXPECTED_CIPHERS,
    OPERATOR_CONDITIONS,
    PARAMETER_GROUPS,
    REPLICAS,
    WRONG_OPERATOR_CONDITIONS,
    adjudicate,
    aggregate_results,
    load_and_validate_config,
    load_authority,
    measure_gradient_groups,
)


def test_k1bd_authority_reconstructs_initial_and_selected_encoders() -> None:
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
        encoder_checkpoints,
        checks,
    ) = load_authority(config)

    assert all(checks.values()), checks
    assert len(dataset_rows) == len(datasets) == 18
    assert set(corrupted) == set(cross) == set(structures)
    assert set(encoder_checkpoints) == {0, 1}
    probe = build_probe(
        runtime_config=runtime_config,
        structures=structures,
        checkpoint=source_checkpoints[0],
        initialization_seed=40,
        model_config=load_source_config()["model"],
        device="cpu",
    )
    assert (
        tensor_mapping_sha256(probe.operator_encoder.state_dict())
        == encoder_checkpoints[0]["initial_encoder_state_sha256"]
    )


def test_k1bd_gradient_groups_expose_disconnected_readiness_projection() -> None:
    config = load_and_validate_config()
    (
        runtime_config,
        _dataset_rows,
        datasets,
        structures,
        summaries,
        source_checkpoints,
        _corrupted,
        _cross,
        _encoder_checkpoints,
        checks,
    ) = load_authority(config)
    assert all(checks.values()), checks
    probe = build_probe(
        runtime_config=runtime_config,
        structures=structures,
        checkpoint=source_checkpoints[0],
        initialization_seed=40,
        model_config=load_source_config()["model"],
        device="cpu",
    )
    dataset = datasets[("uknit64", 3, "train_seen")]
    positive = np.flatnonzero(np.asarray(dataset.labels).reshape(-1) == 1)[:32]
    negative = np.flatnonzero(np.asarray(dataset.labels).reshape(-1) == 0)[:32]
    indices = np.concatenate((positive, negative))
    features = torch.as_tensor(
        np.array(dataset.features[indices], copy=True), dtype=torch.float32
    )
    labels = torch.as_tensor(
        np.array(dataset.labels[indices], copy=True), dtype=torch.float32
    ).reshape(-1, 1)

    metrics, connected, loss, _output_hash = measure_gradient_groups(
        probe=probe,
        features=features,
        labels=labels,
        runtime_structure=structures["uknit64"],
        operator_structure=structures["uknit64"],
        summary=summaries["uknit64"]["correct_descriptor"],
    )

    assert loss > 0.0
    assert float(torch.linalg.vector_norm(connected)) > 0.0
    assert metrics["structure_projection"]["gradient_norm"] == 0.0
    assert metrics["structure_projection"]["nonzero_elements"] == 0
    assert all(
        float(metrics[group]["gradient_norm"]) > 0.0
        for group in PARAMETER_GROUPS
        if group != "structure_projection"
    )
    assert all(parameter.grad is None for parameter in probe.parameters())


def test_k1bd_gate_separates_topology_indistinguishable_hold_and_invalid() -> None:
    norm_rows, topology_rows, cross_rows, interventions = _synthetic_rows()
    results = aggregate_results(
        norm_rows, topology_rows, cross_rows, interventions
    )
    passed = adjudicate(
        source_checks={"source": True},
        gradient_state_checks=_gradient_state_checks(),
        intervention_state_checks=_intervention_state_checks(),
        norm_rows=norm_rows,
        topology_rows=topology_rows,
        cross_rows=cross_rows,
        intervention_rows=interventions,
        results=results,
    )
    assert passed["status"] == "pass"
    assert passed["topology_gradient_indistinguishable"] is True
    assert "topology_indistinguishable_supported" in passed["decision"]
    assert passed["disconnected_structure_projection_supported"] is True

    held_topology = deepcopy(topology_rows)
    for row in held_topology:
        if (
            row["encoder_state"] == "selected_encoder"
            and row["replica"] == 0
            and row["cipher_key"] == "uknit64"
            and row["wrong_condition"]
            == "same_summary_corrupted_operator"
        ):
            row["cosine"] = 0.5
    held_results = aggregate_results(
        norm_rows, held_topology, cross_rows, interventions
    )
    held = adjudicate(
        source_checks={"source": True},
        gradient_state_checks=_gradient_state_checks(),
        intervention_state_checks=_intervention_state_checks(),
        norm_rows=norm_rows,
        topology_rows=held_topology,
        cross_rows=cross_rows,
        intervention_rows=interventions,
        results=held_results,
    )
    assert held["status"] == "hold"
    assert "coupling_redesign_required" in held["decision"]

    invalid_norms = deepcopy(norm_rows)
    invalid_norms[0]["optimizer_steps"] = 1
    invalid = adjudicate(
        source_checks={"source": True},
        gradient_state_checks=_gradient_state_checks(),
        intervention_state_checks=_intervention_state_checks(),
        norm_rows=invalid_norms,
        topology_rows=topology_rows,
        cross_rows=cross_rows,
        intervention_rows=interventions,
        results=results,
    )
    assert invalid["status"] == "invalid"
    assert "all_rows_zero_optimizer_steps" in invalid["failed_protocol_checks"]


def test_k1bd_plot_writes_clear_chinese_svg(tmp_path) -> None:
    norm_rows, topology_rows, cross_rows, interventions = _synthetic_rows()
    results = aggregate_results(
        norm_rows, topology_rows, cross_rows, interventions
    )
    gate = adjudicate(
        source_checks={"source": True},
        gradient_state_checks=_gradient_state_checks(),
        intervention_state_checks=_intervention_state_checks(),
        norm_rows=norm_rows,
        topology_rows=topology_rows,
        cross_rows=cross_rows,
        intervention_rows=interventions,
        results=results,
    )
    for row in results:
        if row.get("metric_type") == "fresh_intervention":
            row.update(
                {
                    "correct_vs_disabled_probability_rms": 0.02,
                    "same_summary_corrupted_operator_probability_rms": 2e-5,
                    "cross_cipher_operator_probability_rms": 3e-5,
                }
            )
    output = tmp_path / "curves.svg"

    report = render_k1bd_svg(gate, results, output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["topology_summary_rows"] == 12
    assert "学强了通用调制，却没有学强拓扑特定调制" in text
    assert "拓扑token梯度远弱于后端样本对投影" in text
    assert "不是新训练结果" in text


def _synthetic_rows() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    norm_rows = []
    topology_rows = []
    cross_rows = []
    interventions = []
    for encoder_state in ENCODER_STATES:
        for replica in REPLICAS:
            for batch_index in range(64):
                for cipher in EXPECTED_CIPHERS:
                    for condition in OPERATOR_CONDITIONS:
                        for group in PARAMETER_GROUPS:
                            disconnected = group == "structure_projection"
                            norm_rows.append(
                                {
                                    "encoder_state": encoder_state,
                                    "replica": replica,
                                    "batch_index": batch_index,
                                    "cipher_key": cipher,
                                    "condition": condition,
                                    "parameter_group": group,
                                    "gradient_norm": 0.0 if disconnected else 1.0,
                                    "nonzero_elements": 0 if disconnected else 10,
                                    "parameter_elements": 10,
                                    "nonzero_fraction": 0.0 if disconnected else 1.0,
                                    "optimizer_steps": 0,
                                }
                            )
                    for condition in WRONG_OPERATOR_CONDITIONS:
                        topology_rows.append(
                            {
                                "encoder_state": encoder_state,
                                "replica": replica,
                                "batch_index": batch_index,
                                "cipher_key": cipher,
                                "wrong_condition": condition,
                                "cosine": 0.999,
                                "relative_norm_difference": 0.001,
                                "optimizer_steps": 0,
                            }
                        )
                for left_index, left in enumerate(EXPECTED_CIPHERS):
                    for right in EXPECTED_CIPHERS[left_index + 1 :]:
                        cross_rows.append(
                            {
                                "encoder_state": encoder_state,
                                "replica": replica,
                                "batch_index": batch_index,
                                "cipher_pair": f"{left}__{right}",
                                "cosine": 0.2,
                                "optimizer_steps": 0,
                            }
                        )
            for cipher in EXPECTED_CIPHERS:
                for split in ("same_key_fresh", "cross_key_validation"):
                    interventions.append(
                        {
                            "encoder_state": encoder_state,
                            "replica": replica,
                            "cipher_key": cipher,
                            "split": split,
                            "correct_contribution_rms": 0.01,
                            "optimizer_steps": 0,
                        }
                    )
    return norm_rows, topology_rows, cross_rows, interventions


def _gradient_state_checks() -> dict[str, bool]:
    return {
        f"replica{replica}_{state}_{suffix}": True
        for replica in REPLICAS
        for state in ENCODER_STATES
        for suffix in ("state_immutable", "grads_none")
    }


def _intervention_state_checks() -> dict[str, bool]:
    return {
        f"replica{replica}_{state}_intervention_state_immutable": True
        for replica in REPLICAS
        for state in ENCODER_STATES
    }
