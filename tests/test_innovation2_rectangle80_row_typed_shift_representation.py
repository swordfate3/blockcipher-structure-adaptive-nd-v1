from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_rectangle80_row_typed_shift_representation import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.rectangle80_row_typed_shift_representation_audit import (
    E89_TRUE_RIDGE_AUC,
    Rectangle80RowTypedAuditConfig,
    adjudicate_row_typed_audit,
    build_row_typed_matrices,
    validate_e90_source,
)


def _e90_source() -> dict[str, object]:
    return {
        "gate": {
            "run_id": "i2_rectangle80_r4_r3_only_profile_operator_attribution_seed0_20260719",
            "status": "hold",
            "decision": "innovation2_rectangle80_r3_only_topology_not_attributed",
            "protocol_checks": {"valid": True},
            "candidate_checks": {"valid": True},
            "relation_checks": {
                "true_minus_independent_at_least_0p03": True,
                "true_minus_corrupted_at_least_0p03": False,
            },
        },
        "rows": [
            {"relation_mode": mode, "epochs_completed": 30}
            for mode in ("independent", "true", "corrupted")
        ],
        "hashes": {"gate": "a" * 64, "results": "b" * 64},
    }


def test_e91_validates_exact_e90_hold_boundary() -> None:
    assert all(validate_e90_source(_e90_source()).values())


def test_row_typed_matrices_have_frozen_dimensions_and_wrong_control() -> None:
    prefix = np.arange(3 * 64 * 39, dtype=np.float64).reshape(3, 64, 39)
    sources = {
        "prefix_features": prefix,
        "matched_rows": [
            {"structure_index": 0, "output_bit": 0, "label": 0, "split": "train"},
            {"structure_index": 1, "output_bit": 5, "label": 1, "split": "train"},
            {
                "structure_index": 2,
                "output_bit": 10,
                "label": 0,
                "split": "validation",
            },
        ],
    }

    matrices = build_row_typed_matrices(sources)

    assert matrices["untyped_true"].shape == (3, 39)
    assert matrices["untyped_corrupted"].shape == (3, 39)
    assert matrices["typed_true"].shape == (3, 117)
    assert matrices["typed_corrupted"].shape == (3, 117)
    assert matrices["wrong_row_typed_true"].shape == (3, 117)
    assert not np.array_equal(
        matrices["typed_true"], matrices["wrong_row_typed_true"]
    )


def _reports(
    typed_true: float = 0.87,
    typed_corrupted: float = 0.82,
    wrong_typed: float = 0.84,
) -> dict[str, object]:
    values = {
        "untyped_true": E89_TRUE_RIDGE_AUC,
        "untyped_corrupted": 0.774,
        "typed_true": typed_true,
        "typed_corrupted": typed_corrupted,
        "wrong_row_typed_true": wrong_typed,
    }
    return {
        name: {
            "feature_count": 39 if name.startswith("untyped") else 117,
            "train_auc": value + 0.01,
            "validation_auc": value,
            "ridge_lambda": 1e-3,
            "train_standardization_only": True,
            "finite": True,
        }
        for name, value in values.items()
    }


def test_e91_gate_opens_row_typed_network_only_after_three_margins() -> None:
    gate = adjudicate_row_typed_audit(
        Rectangle80RowTypedAuditConfig(run_id="e91-test"),
        {"profile": True},
        {"model_order": True},
        {"e90": True},
        _reports(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_rectangle80_row_typed_representation_ready"
    )
    assert gate["next_action"]["neural_readiness_allowed"] is True


def test_e91_gate_holds_when_wrong_row_explains_typed_gain() -> None:
    gate = adjudicate_row_typed_audit(
        Rectangle80RowTypedAuditConfig(run_id="e91-test"),
        {"profile": True},
        {"model_order": True},
        {"e90": True},
        _reports(wrong_typed=0.865),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_rectangle80_row_typed_representation_not_ready"
    )
    assert gate["next_action"]["neural_readiness_allowed"] is False


def test_e91_plot_writes_clear_chinese_svg(tmp_path: Path) -> None:
    gate = adjudicate_row_typed_audit(
        Rectangle80RowTypedAuditConfig(run_id="e91-test"),
        {"profile": True},
        {"model_order": True},
        {"e90": True},
        _reports(),
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps({"gate": gate}), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E91" in svg
    assert "错误row控制" in svg
    assert "不训练神经网络" in svg
