from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.cli.plot_innovation2_rectangle80_row_typed_shift_operator import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.rectangle80_row_typed_shift_operator_readiness import (
    MODES,
    Rectangle80RowTypedOperatorConfig,
    adjudicate_row_typed_readiness,
    make_row_typed_model,
    validate_e91_source,
)


def test_e92_models_are_parameter_neutral_and_change_row_context() -> None:
    config = Rectangle80RowTypedOperatorConfig(run_id="e92-test")
    models = {mode: make_row_typed_model(config, mode) for mode in MODES}
    counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    }
    features = torch.randn(3, 64, 13)

    assert counts == {4_795}
    assert all(model(features).shape == (3, 64) for model in models.values())
    assert not torch.equal(
        models["row_typed_true"].typed_channel_index,
        models["untyped_true"].typed_channel_index,
    )
    assert not torch.equal(
        models["row_typed_true"].typed_channel_index,
        models["wrong_row_true"].typed_channel_index,
    )
    assert not torch.equal(
        models["row_typed_true"].player,
        models["row_typed_corrupted"].player,
    )


def _e91_source() -> dict[str, object]:
    return {
        "gate": {
            "run_id": "i2_rectangle80_row_typed_shift_representation_audit_20260719",
            "status": "pass",
            "decision": "innovation2_rectangle80_row_typed_representation_ready",
            "protocol_checks": {"valid": True},
            "mechanism_checks": {"valid": True},
            "metrics": {
                "ridges": {"typed_true": {"validation_auc": 0.8389640238069933}}
            },
            "next_action": {"training_performed": False},
        },
        "rows": [{"variant": str(index)} for index in range(5)],
        "hashes": {"gate": "a" * 64, "results": "b" * 64},
    }


def test_e92_validates_e91_mechanism_source() -> None:
    assert all(validate_e91_source(_e91_source()).values())


def _rows(candidate: float = 0.88) -> list[dict[str, object]]:
    aucs = {
        "untyped_true": candidate - 0.03,
        "row_typed_true": candidate,
        "row_typed_corrupted": candidate - 0.05,
        "wrong_row_true": candidate - 0.02,
    }
    return [
        {
            "relation_mode": mode,
            "epochs_completed": 2,
            "train_auc": auc + 0.01,
            "train_accuracy": 0.75,
            "train_loss": 0.50,
            "validation_auc": auc,
            "validation_accuracy": 0.73,
            "validation_loss": 0.53,
        }
        for mode, auc in aucs.items()
    ]


def _contract() -> dict[str, object]:
    return {
        "output_shape": [4, 64],
        "input_dim": 13,
        "parameter_counts": {mode: 4_795 for mode in MODES},
        "parameter_counts_match": True,
        "masked_loss_explicit_max_abs_error": 0.0,
        "typed_vs_untyped_logit_max_abs_difference": 0.1,
        "true_vs_corrupted_logit_max_abs_difference": 0.1,
        "true_vs_wrong_row_logit_max_abs_difference": 0.1,
        "cell_relabel_max_abs_error": 1e-7,
        "true_and_wrong_channel_maps_differ": True,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
        "forbidden_named_state_absent": True,
    }


def test_e92_gate_opens_formal_only_after_all_typed_controls() -> None:
    gate = adjudicate_row_typed_readiness(
        Rectangle80RowTypedOperatorConfig(run_id="e92-test"),
        {"profile": True},
        {"model_order": True},
        {"e90": True},
        {"e91": True},
        _contract(),
        {"trained_rows": _rows()},
        _e91_source(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_rectangle80_row_typed_shift_operator_readiness_passed"
    )
    assert gate["next_action"]["formal_seed0"] is True


def test_e92_gate_holds_when_wrong_row_is_too_close() -> None:
    rows = _rows()
    by_mode = {row["relation_mode"]: row for row in rows}
    by_mode["wrong_row_true"]["validation_auc"] = (
        by_mode["row_typed_true"]["validation_auc"] - 0.005
    )
    gate = adjudicate_row_typed_readiness(
        Rectangle80RowTypedOperatorConfig(run_id="e92-test"),
        {"profile": True},
        {"model_order": True},
        {"e90": True},
        {"e91": True},
        _contract(),
        {"trained_rows": rows},
        _e91_source(),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_rectangle80_row_typed_shift_operator_not_ready"
    )


def test_e92_plot_writes_clear_chinese_svg(tmp_path: Path) -> None:
    rows = _rows()
    gate = adjudicate_row_typed_readiness(
        Rectangle80RowTypedOperatorConfig(run_id="e92-test"),
        {"profile": True},
        {"model_order": True},
        {"e90": True},
        {"e91": True},
        _contract(),
        {"trained_rows": rows},
        _e91_source(),
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(
        json.dumps({"gate": gate, "trained_rows": rows, "contract": _contract()}),
        encoding="utf-8",
    )

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E92" in svg
    assert "参数零增量" in svg
    assert "没有row embedding参数" in svg


def test_e92_protocol_is_frozen() -> None:
    try:
        Rectangle80RowTypedOperatorConfig(run_id="e92-test", hidden_dim=36)
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("E92 readiness budget must remain frozen")
