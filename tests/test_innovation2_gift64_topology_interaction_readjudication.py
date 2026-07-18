from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_gift64_topology_interaction_readjudication import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.gift64_topology_interaction_readjudication import (
    Gift64TopologyInteractionConfig,
    adjudicate_topology_interaction,
    gift_player_variants,
    validate_e76_source,
)


def test_e77_player_variants_are_distinct_same_size_permutations() -> None:
    variants = gift_player_variants()

    assert set(variants) == {
        "true",
        "corrupted_shift1",
        "corrupted_shift2",
        "corrupted_shift3",
    }
    assert len({player.tobytes() for player in variants.values()}) == 4
    assert all(np.array_equal(np.sort(player), np.arange(64)) for player in variants.values())
    assert all(
        np.array_equal(player % 4, variants["true"] % 4)
        for player in variants.values()
    )


def test_e76_source_validation_requires_the_exact_hold_pattern(tmp_path: Path) -> None:
    checkpoint = tmp_path / "true.pt"
    checkpoint.write_bytes(b"checkpoint")
    source = {
        "gate": {
            "run_id": "i2_gift64_r4_r3_only_profile_operator_readiness_seed0_20260719",
            "status": "hold",
            "decision": "innovation2_gift64_r3_only_prefix_not_sufficient",
            "protocol_checks": {"valid": True},
            "readiness_checks": {"true_margin": True},
            "deterministic_checks": {
                "r3_ridge_auc_at_least_0p60": False,
                "r3_minus_full_ridge_at_least_minus_0p03": True,
            },
        },
        "rows": [
            {"relation_mode": mode}
            for mode in ("independent", "true", "corrupted")
        ],
        "checkpoint_path": checkpoint,
        "hashes": {"gate": "a" * 64, "results": "b" * 64, "checkpoint": "c" * 64},
    }

    assert all(validate_e76_source(source).values())


def _ridges() -> dict[str, dict[str, float | bool]]:
    return {
        "local": {"validation_auc": 0.55, "train_standardization_only": True},
        "corrupted_shift1": {
            "validation_auc": 0.58,
            "train_standardization_only": True,
        },
        "corrupted_shift2": {
            "validation_auc": 0.57,
            "train_standardization_only": True,
        },
        "corrupted_shift3": {
            "validation_auc": 0.56,
            "train_standardization_only": True,
        },
        "true": {"validation_auc": 0.65, "train_standardization_only": True},
    }


def _counterfactuals() -> dict[str, dict[str, float]]:
    return {
        "corrupted_shift1": {"auc": 0.70, "accuracy": 0.65, "loss": 0.62},
        "corrupted_shift2": {"auc": 0.69, "accuracy": 0.64, "loss": 0.63},
        "corrupted_shift3": {"auc": 0.68, "accuracy": 0.63, "loss": 0.64},
        "true": {"auc": 0.76, "accuracy": 0.70, "loss": 0.58},
    }


def _e76_source() -> dict[str, object]:
    return {
        "rows": [
            {"relation_mode": "independent", "validation_auc": 0.56},
            {"relation_mode": "true", "validation_auc": 0.76},
            {"relation_mode": "corrupted", "validation_auc": 0.70},
        ]
    }


def test_e77_gate_repairs_information_misaligned_ridge_gate() -> None:
    gate = adjudicate_topology_interaction(
        Gift64TopologyInteractionConfig(run_id="e77-test"),
        {"profile_valid": True},
        {"e76_valid": True},
        _ridges(),
        _counterfactuals(),
        _e76_source(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_gift64_topology_interaction_gate_repaired"


def test_e77_gate_holds_when_true_ridge_does_not_beat_corruptions() -> None:
    ridges = _ridges()
    ridges["corrupted_shift1"]["validation_auc"] = 0.64
    gate = adjudicate_topology_interaction(
        Gift64TopologyInteractionConfig(run_id="e77-test"),
        {"profile_valid": True},
        {"e76_valid": True},
        ridges,
        _counterfactuals(),
        _e76_source(),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_gift64_topology_interaction_not_confirmed"


def test_plot_writes_clear_chinese_e77_svg(tmp_path: Path) -> None:
    ridges = _ridges()
    for report in ridges.values():
        report["train_auc"] = 0.70
        report["feature_count"] = 39
        report["ridge_lambda"] = 1e-3
    gate = adjudicate_topology_interaction(
        Gift64TopologyInteractionConfig(run_id="e77-test"),
        {"profile_valid": True},
        {"e76_valid": True},
        ridges,
        _counterfactuals(),
        _e76_source(),
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps({"gate": gate}), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E77" in svg
    assert "同权重反事实" in svg
    assert "不训练新模型" in svg


def test_e77_protocol_is_frozen() -> None:
    try:
        Gift64TopologyInteractionConfig(
            run_id="e77-test", corruption_shifts=(1, 2)
        )
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("E77 corruption family must remain frozen")
