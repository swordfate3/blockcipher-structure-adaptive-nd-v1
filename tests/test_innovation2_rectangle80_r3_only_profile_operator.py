from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.plot_innovation2_rectangle80_r3_only_profile_operator import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.rectangle80_r3_only_profile_operator_readiness import (
    Rectangle80R3ProfileReadinessConfig,
    adjudicate_rectangle_r3_readiness,
    make_rectangle_r3_model,
    rectangle_cell_major_order,
    rectangle_model_player,
    to_rectangle_model_order,
    validate_rectangle_model_order,
    validate_rectangle_profile_sources,
)


def test_rectangle_r3_models_are_same_size_and_output_64_logits() -> None:
    config = Rectangle80R3ProfileReadinessConfig(run_id="e89-test")
    models = {
        mode: make_rectangle_r3_model(config, mode)
        for mode in ("independent", "true", "corrupted")
    }
    features = torch.randn(3, 64, 13)
    counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    }

    assert models["true"](features).shape == (3, 64)
    assert counts == {4_795}


def test_rectangle_cell_major_order_and_players_are_valid() -> None:
    config = Rectangle80R3ProfileReadinessConfig(run_id="e89-test")
    order = rectangle_cell_major_order()
    true_model = make_rectangle_r3_model(config, "true")
    corrupted_model = make_rectangle_r3_model(config, "corrupted")

    np.testing.assert_array_equal(order[:4], (0, 16, 32, 48))
    np.testing.assert_array_equal(np.sort(order), np.arange(64))
    np.testing.assert_array_equal(np.sort(rectangle_model_player()), np.arange(64))
    assert not torch.equal(true_model.player, corrupted_model.player)
    assert torch.equal(
        torch.sort(corrupted_model.player).values, torch.arange(64)
    )


def _source_fixture() -> dict[str, object]:
    train_structures = list(range(144))
    validation_structures = list(range(144, 192))
    train_edges = [
        (structure, output)
        for output in range(64)
        for structure in train_structures
    ][:2416]
    validation_edges = [
        (structure, output)
        for output in range(64)
        for structure in validation_structures
    ][:776]
    rows = [
        {
            "split": split,
            "structure_index": structure,
            "output_bit": output,
            "label": index % 2,
        }
        for split, edges in (
            ("train", train_edges),
            ("validation", validation_edges),
        )
        for index, (structure, output) in enumerate(edges)
    ]
    targets = np.full((192, 64), -1, dtype=np.int8)
    observed = np.zeros((192, 64), dtype=np.bool_)
    for row in rows:
        targets[row["structure_index"], row["output_bit"]] = row["label"]
        observed[row["structure_index"], row["output_bit"]] = True
    return {
        "profile_gate": {
            "run_id": "i2_rectangle80_r4_unit_balance_profile_192_structures_20260719",
            "decision": "innovation2_rectangle80_unit_profile_expansion_ready",
            "status": "pass",
            "protocol_checks": {"valid": True},
        },
        "profile_metadata": {
            "task": "innovation2_rectangle80_unit_balance_profile_expansion"
        },
        "structures": [{} for _ in range(192)],
        "profile_targets": targets,
        "profile_observed": observed,
        "prefix_features": np.arange(192 * 64 * 39, dtype=np.float64).reshape(
            192, 64, 39
        ),
        "matched_rows": rows,
        "source_hashes": {"gate.json": "a" * 64},
    }


def test_rectangle_profile_source_and_model_order_replay() -> None:
    physical = _source_fixture()
    model = to_rectangle_model_order(physical)

    assert all(validate_rectangle_profile_sources(physical).values())
    assert all(validate_rectangle_model_order(physical, model).values())
    np.testing.assert_array_equal(
        model["prefix_features"][:, 1], physical["prefix_features"][:, 16]
    )


def _rows(true_auc: float = 0.78) -> list[dict[str, object]]:
    aucs = {
        "independent": true_auc - 0.09,
        "corrupted": true_auc - 0.05,
        "true": true_auc,
    }
    return [
        {
            "relation_mode": mode,
            "epochs_completed": 2,
            "train_auc": auc + 0.02,
            "train_accuracy": 0.70,
            "train_loss": 0.60,
            "validation_auc": auc,
            "validation_accuracy": 0.68,
            "validation_loss": 0.62,
        }
        for mode, auc in aucs.items()
    ]


def _contract() -> dict[str, object]:
    return {
        "output_shape": [4, 64],
        "input_dim": 13,
        "masked_loss_explicit_max_abs_error": 0.0,
        "parameter_counts": {
            mode: 4_795 for mode in ("independent", "true", "corrupted")
        },
        "parameter_counts_match": True,
        "topology_logit_max_abs_difference": 0.03,
        "cell_relabel_max_abs_error": 1e-7,
        "forbidden_named_state_absent": True,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
    }


def _ridges() -> dict[str, object]:
    return {
        "local": {
            "validation_auc": 0.60,
            "train_standardization_only": True,
        },
        "corrupted": {
            "validation_auc": 0.64,
            "train_standardization_only": True,
        },
        "true": {
            "validation_auc": 0.70,
            "train_standardization_only": True,
        },
    }


def test_rectangle_r3_gate_opens_formal_only_after_fair_controls() -> None:
    gate = adjudicate_rectangle_r3_readiness(
        Rectangle80R3ProfileReadinessConfig(run_id="e89-test"),
        {"source_valid": True},
        {"model_order_valid": True},
        _contract(),
        _ridges(),
        {"trained_rows": _rows()},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_rectangle80_r3_only_profile_readiness_passed"
    )


def test_rectangle_r3_gate_holds_when_fair_ridge_control_fails() -> None:
    ridges = _ridges()
    ridges["corrupted"]["validation_auc"] = 0.69
    gate = adjudicate_rectangle_r3_readiness(
        Rectangle80R3ProfileReadinessConfig(run_id="e89-test"),
        {"source_valid": True},
        {"model_order_valid": True},
        _contract(),
        ridges,
        {"trained_rows": _rows()},
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_rectangle80_r3_only_topology_baseline_not_ready"
    )


def test_plot_writes_clear_chinese_e89_svg(tmp_path: Path) -> None:
    rows = _rows()
    gate = adjudicate_rectangle_r3_readiness(
        Rectangle80R3ProfileReadinessConfig(run_id="e89-test"),
        {"source_valid": True},
        {"model_order_valid": True},
        _contract(),
        _ridges(),
        {"trained_rows": rows},
    )
    summary = {
        "trained_rows": rows,
        "deterministic_ridges": _ridges(),
        "contract": _contract(),
        "gate": gate,
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E89" in svg
    assert "真实RECTANGLE P层" in svg
    assert "信息范围对齐的确定性基线" in svg


def test_e89_training_protocol_is_frozen() -> None:
    try:
        Rectangle80R3ProfileReadinessConfig(run_id="e89-test", epochs=3)
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("E89 readiness budget must remain frozen")
