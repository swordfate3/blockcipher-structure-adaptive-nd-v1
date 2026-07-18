from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.plot_innovation2_gift64_r3_only_profile_operator import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.gift64_r3_only_profile_operator_readiness import (
    Gift64R3ProfileReadinessConfig,
    adjudicate_gift_r3_readiness,
    gift_player,
    make_gift_r3_model,
    validate_gift_profile_sources,
)


def test_gift_r3_models_are_same_size_and_output_64_logits() -> None:
    config = Gift64R3ProfileReadinessConfig(run_id="e76-test")
    models = {
        mode: make_gift_r3_model(config, mode)
        for mode in ("independent", "true", "corrupted")
    }
    features = torch.randn(3, 64, 13)
    counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    }

    assert models["true"](features).shape == (3, 64)
    assert counts == {4_795}


def test_gift_true_and_corrupted_players_are_distinct_permutations() -> None:
    config = Gift64R3ProfileReadinessConfig(run_id="e76-test")
    true_model = make_gift_r3_model(config, "true")
    corrupted_model = make_gift_r3_model(config, "corrupted")

    assert np.array_equal(np.sort(gift_player()), np.arange(64))
    assert not torch.equal(true_model.player, corrupted_model.player)
    assert torch.equal(
        torch.sort(corrupted_model.player).values, torch.arange(64)
    )


def _source_fixture() -> dict[str, object]:
    train_structures = list(range(110))
    validation_structures = list(range(110, 143))
    train_edges = [
        (structure, output)
        for output in range(64)
        for structure in train_structures
    ][:496]
    validation_edges = [
        (structure, output)
        for output in range(64)
        for structure in validation_structures
    ][:124]
    rows = [
        {
            "split": split,
            "structure_index": structure,
            "output_bit": output,
            "label": index % 2,
        }
        for split, edges in (("train", train_edges), ("validation", validation_edges))
        for index, (structure, output) in enumerate(edges)
    ]
    targets = np.full((192, 64), -1, dtype=np.int8)
    observed = np.zeros((192, 64), dtype=np.bool_)
    for row in rows:
        targets[row["structure_index"], row["output_bit"]] = row["label"]
        observed[row["structure_index"], row["output_bit"]] = True
    return {
        "profile_gate": {
            "run_id": "i2_gift64_r4_unit_balance_profile_192_structures_20260719",
            "decision": "innovation2_gift64_unit_balance_profile_expansion_ready",
            "status": "pass",
            "protocol_checks": {"valid": True},
        },
        "profile_metadata": {
            "task": "innovation2_gift64_unit_balance_profile_expansion"
        },
        "structures": [{} for _ in range(192)],
        "profile_targets": targets,
        "profile_observed": observed,
        "prefix_features": np.zeros((192, 64, 39), dtype=np.float64),
        "matched_rows": rows,
        "source_hashes": {"gate.json": "a" * 64},
    }


def test_gift_profile_source_validation_replays_all_620_edges() -> None:
    checks = validate_gift_profile_sources(_source_fixture())

    assert all(checks.values())


def _rows(true_auc: float = 0.72) -> list[dict[str, object]]:
    aucs = {
        "independent": true_auc - 0.07,
        "corrupted": true_auc - 0.06,
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
        "parameter_counts": {mode: 4_795 for mode in ("independent", "true", "corrupted")},
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
        "full39": {
            "validation_auc": 0.72,
            "train_standardization_only": True,
        },
        "r3_only": {
            "validation_auc": 0.70,
            "train_standardization_only": True,
        },
    }


def test_gift_r3_gate_opens_formal_only_after_all_margins() -> None:
    gate = adjudicate_gift_r3_readiness(
        Gift64R3ProfileReadinessConfig(run_id="e76-test"),
        {"source_valid": True},
        _contract(),
        _ridges(),
        {"trained_rows": _rows()},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_gift64_r3_only_profile_readiness_passed"


def test_gift_r3_gate_holds_when_true_p_does_not_beat_corrupted() -> None:
    rows = _rows()
    rows[1]["validation_auc"] = rows[2]["validation_auc"] - 0.01
    gate = adjudicate_gift_r3_readiness(
        Gift64R3ProfileReadinessConfig(run_id="e76-test"),
        {"source_valid": True},
        _contract(),
        _ridges(),
        {"trained_rows": rows},
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_gift64_r3_only_profile_readiness_not_passed"
    )


def test_plot_writes_clear_chinese_e76_svg(tmp_path: Path) -> None:
    rows = _rows()
    gate = adjudicate_gift_r3_readiness(
        Gift64R3ProfileReadinessConfig(run_id="e76-test"),
        {"source_valid": True},
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
    assert "创新2 E76" in svg
    assert "真实P-layer" in svg
    assert "确定性安全基线" in svg


def test_e76_training_protocol_is_frozen() -> None:
    try:
        Gift64R3ProfileReadinessConfig(run_id="e76-test", epochs=3)
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("E76 readiness budget must remain frozen")
