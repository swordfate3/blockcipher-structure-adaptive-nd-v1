from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from blockcipher_nd.cli.plot_uknit_family_position_preserving_operator_k1bc import (
    render_k1bc_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_position_preserving_operator_k1bc import (
    EXPECTED_TRAINABLE_PARAMETERS,
    adjudicate_training,
    build_probe,
    load_and_validate_config,
    load_authority,
)


def test_k1bc_authority_and_cross_operator_controls_are_exact() -> None:
    config = load_and_validate_config()
    (
        _runtime,
        dataset_rows,
        datasets,
        structures,
        _summaries,
        checkpoints,
        anchors,
        corrupted,
        cross,
        checks,
    ) = load_authority(config)

    assert all(checks.values()), checks
    assert len(dataset_rows) == len(datasets) == 18
    assert len(checkpoints) == 2
    assert len(anchors) == 12
    assert set(corrupted) == set(cross) == set(structures)
    assert cross["uknit64"].block_bits == 64
    assert cross["midori64"].block_bits == 64
    assert cross["dialga128"].block_bits == 128
    assert torch.equal(
        cross["dialga128"].linear_matrices[:, :64, :64],
        structures["uknit64"].linear_matrices,
    )


def test_k1bc_one_step_updates_only_shared_operator_encoder() -> None:
    config = load_and_validate_config()
    (
        runtime,
        _dataset_rows,
        datasets,
        structures,
        summaries,
        checkpoints,
        _anchors,
        _corrupted,
        _cross,
        checks,
    ) = load_authority(config)
    assert all(checks.values()), checks
    replica = config["replicas"][0]
    probe = build_probe(
        runtime_config=runtime,
        structures=structures,
        checkpoint=checkpoints[0],
        initialization_seed=replica["encoder_initialization_seed"],
        model_config=config["model"],
        device="cpu",
    )
    anchor_before = tensor_mapping_sha256(probe.anchor.state_dict())
    encoder_before = tensor_mapping_sha256(probe.operator_encoder.state_dict())
    dataset = datasets[("uknit64", 3, "train_seen")]
    features = torch.as_tensor(
        np.array(dataset.features[:64], copy=True),
        dtype=torch.float32,
    )
    labels = torch.as_tensor(
        np.array(dataset.labels[:64], copy=True),
        dtype=torch.float32,
    ).reshape(-1, 1)
    optimizer = torch.optim.Adam(probe.operator_encoder.parameters(), lr=1e-4)

    logits = probe.logits_with_operator(
        features,
        structures["uknit64"],
        structures["uknit64"],
        gate_summary=summaries["uknit64"]["correct_descriptor"],
    )
    loss = nn.functional.mse_loss(torch.sigmoid(logits), labels)
    loss.backward()
    optimizer.step()

    assert sum(
        parameter.numel()
        for parameter in probe.parameters()
        if parameter.requires_grad
    ) == EXPECTED_TRAINABLE_PARAMETERS
    assert tensor_mapping_sha256(probe.anchor.state_dict()) == anchor_before
    assert tensor_mapping_sha256(probe.operator_encoder.state_dict()) != encoder_before


def test_k1bc_gate_separates_pass_hold_and_invalid() -> None:
    config = load_and_validate_config()
    training_rows = _synthetic_training_rows()
    controls = _synthetic_controls()
    checkpoints = _synthetic_checkpoints()

    passed = adjudicate_training(
        config=config,
        source_checks={"source": True},
        training_rows=training_rows,
        evaluation_rows=controls,
        checkpoints=checkpoints,
    )
    assert passed["status"] == "pass"
    assert "training_supported" in passed["decision"]

    held_controls = deepcopy(controls)
    for row in held_controls:
        if row["condition"] == "same_summary_corrupted_operator":
            row["auc"] = next(
                float(candidate["auc"]) + 0.01
                for candidate in held_controls
                if candidate["replica"] == row["replica"]
                and candidate["cipher_key"] == row["cipher_key"]
                and candidate["split"] == row["split"]
                and candidate["condition"] == "correct_operator"
            )
    held = adjudicate_training(
        config=config,
        source_checks={"source": True},
        training_rows=training_rows,
        evaluation_rows=held_controls,
        checkpoints=checkpoints,
    )
    assert held["status"] == "hold"
    assert held["correct_topology_attribution_all"] is False

    invalid_rows = deepcopy(controls)
    invalid_rows[0]["state_immutable_across_controls"] = False
    invalid = adjudicate_training(
        config=config,
        source_checks={"source": True},
        training_rows=training_rows,
        evaluation_rows=invalid_rows,
        checkpoints=checkpoints,
    )
    assert invalid["status"] == "invalid"
    assert "evaluation_zero_step_same_checkpoint_immutable" in invalid[
        "failed_protocol_checks"
    ]


def test_k1bc_plot_writes_clear_chinese_svg(tmp_path: Path) -> None:
    config = load_and_validate_config()
    training_rows = _synthetic_training_rows()
    controls = _synthetic_controls()
    gate = adjudicate_training(
        config=config,
        source_checks={"source": True},
        training_rows=training_rows,
        evaluation_rows=controls,
        checkpoints=_synthetic_checkpoints(),
    )
    history = [
        {
            "replica": replica,
            "epoch": epoch,
            "cross_key_macro_auc": 0.70 + 0.01 * replica + 0.001 * epoch,
            "cross_key_minimum_auc": 0.60 + 0.001 * epoch,
        }
        for replica in (0, 1)
        for epoch in range(1, 11)
    ]
    output = tmp_path / "curves.svg"

    report = render_k1bc_svg(gate, history, controls, output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["evaluation_panels"] == 12
    assert "位置保持算子能表示，但训练后没有使用正确拓扑" in text
    assert "正确拓扑归因" in text
    assert "不是正式训练" in text


def _synthetic_training_rows() -> list[dict[str, object]]:
    geometry = {"weight": [10, 10]}
    return [
        {
            "replica": replica,
            "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
            "trainable_parameter_geometry": geometry,
            "anchor_all_parameters_frozen": True,
            "anchor_state_immutable": True,
            "uses_cipher_identity": False,
            "uses_per_cipher_parameters": False,
            "training": {
                "epochs": 10,
                "optimizer_steps": 1920,
                "optimizer_state_step_min": 1920,
                "optimizer_state_step_max": 1920,
                "one_shared_optimizer": True,
                "equal_batches_per_cipher": True,
                "anchor_frozen": True,
                "operator_encoder_only": True,
            },
        }
        for replica in (0, 1)
    ]


def _synthetic_controls() -> list[dict[str, object]]:
    rows = []
    for replica in (0, 1):
        for cipher in ("uknit64", "midori64", "dialga128"):
            for split in ("same_key_fresh", "cross_key_validation"):
                correct = 0.70 + 0.01 * replica
                for condition, auc in (
                    ("correct_operator", correct),
                    ("same_summary_corrupted_operator", correct - 0.01),
                    ("cross_cipher_operator", correct - 0.01),
                    ("disabled_k1az", correct - 0.002),
                ):
                    rows.append(
                        {
                            "replica": replica,
                            "cipher_key": cipher,
                            "split": split,
                            "condition": condition,
                            "auc": auc,
                            "k1az_anchor_auc": correct - 0.002,
                            "correct_minus_k1az_auc": 0.002,
                            "correct_minus_condition_auc": correct - auc,
                            "state_immutable_across_controls": True,
                            "runtime_structure_held_correct": True,
                            "runtime_structure_cipher_key": cipher,
                            "disabled_probability_replay_exact": True,
                            "disabled_auc_replay_delta": 0.0,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows


def _synthetic_checkpoints() -> dict[int, dict[str, object]]:
    root = Path("/tmp")
    checkpoints = {}
    for replica in (0, 1):
        path = root / f"k1bc-test-replica{replica}.pt"
        path.write_bytes(f"replica-{replica}".encode("ascii"))
        from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256

        checkpoints[replica] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "best_epoch": 5,
        }
    return checkpoints
