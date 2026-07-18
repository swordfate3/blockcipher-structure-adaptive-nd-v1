from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_rectangle80_r3_only_profile_operator_attribution import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.rectangle80_r3_only_profile_operator_attribution import (
    Rectangle80R3AttributionConfig,
    adjudicate_rectangle_r3_attribution,
    validate_e89_source,
)


def _e89_source() -> dict[str, object]:
    rows = [
        {"relation_mode": mode, "epochs_completed": 2}
        for mode in ("independent", "true", "corrupted")
    ]
    return {
        "gate": {
            "run_id": "i2_rectangle80_r4_r3_only_profile_operator_readiness_seed0_20260719",
            "status": "pass",
            "decision": "innovation2_rectangle80_r3_only_profile_readiness_passed",
            "protocol_checks": {"valid": True},
            "deterministic_checks": {"valid": True},
            "readiness_checks": {"valid": True},
            "metrics": {
                "ridges": {"true": {"validation_auc": 0.8246824848549261}}
            },
        },
        "rows": rows,
        "hashes": {"gate": "a" * 64, "results": "b" * 64},
    }


def test_e90_validates_e89_pass_source() -> None:
    assert all(validate_e89_source(_e89_source()).values())


def _rows(true_auc: float = 0.93) -> list[dict[str, object]]:
    aucs = {
        "independent": true_auc - 0.18,
        "corrupted": true_auc - 0.08,
        "true": true_auc,
    }
    return [
        {
            "relation_mode": mode,
            "epochs_completed": 30,
            "best_epoch": 24,
            "train_auc": auc + 0.01,
            "train_accuracy": 0.80,
            "train_loss": 0.40,
            "validation_auc": auc,
            "validation_accuracy": 0.78,
            "validation_loss": 0.44,
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


def test_e90_gate_opens_seed1_only_after_quality_and_topology_pass() -> None:
    gate = adjudicate_rectangle_r3_attribution(
        Rectangle80R3AttributionConfig(run_id="e90-test"),
        {"profile": True},
        {"model_order": True},
        {"e89": True},
        _contract(),
        {"trained_rows": _rows()},
        _e89_source(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_rectangle80_r3_only_neural_gain_attributed"
    )
    assert gate["next_action"]["seed1"] is True


def test_e90_gate_holds_when_true_does_not_beat_fair_ridge() -> None:
    rows = _rows(true_auc=0.84)
    gate = adjudicate_rectangle_r3_attribution(
        Rectangle80R3AttributionConfig(run_id="e90-test"),
        {"profile": True},
        {"model_order": True},
        {"e89": True},
        _contract(),
        {"trained_rows": rows},
        _e89_source(),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_rectangle80_r3_only_quality_not_confirmed"
    )
    assert gate["next_action"]["seed1"] is False


def test_e90_plot_writes_clear_chinese_svg(tmp_path: Path) -> None:
    rows = _rows()
    gate = adjudicate_rectangle_r3_attribution(
        Rectangle80R3AttributionConfig(run_id="e90-test"),
        {"profile": True},
        {"model_order": True},
        {"e89": True},
        _contract(),
        {"trained_rows": rows},
        _e89_source(),
    )
    summary = {"trained_rows": rows, "gate": gate}
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E90" in svg
    assert "信息范围对齐的确定性安全锚点" in svg
    assert "不是双seed" in svg


def test_e90_training_protocol_is_frozen() -> None:
    try:
        Rectangle80R3AttributionConfig(run_id="e90-test", epochs=29)
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("E90 attribution budget must remain frozen")
