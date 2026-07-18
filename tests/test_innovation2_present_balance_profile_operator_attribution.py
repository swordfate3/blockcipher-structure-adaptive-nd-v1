from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_balance_profile_operator_attribution import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_attribution import (
    ProfileOperatorAttributionConfig,
    adjudicate_profile_operator_attribution,
)


def trained_rows(true_auc: float = 0.84) -> list[dict[str, float | int | str]]:
    return [
        {
            "relation_mode": mode,
            "validation_auc": validation_auc,
            "train_auc": train_auc,
            "validation_accuracy": 0.70,
            "train_accuracy": 0.75,
            "validation_loss": 0.60,
            "train_loss": 0.50,
            "epochs_completed": 30,
            "best_epoch": 18,
            "parameter_count": 5679,
        }
        for mode, validation_auc, train_auc in (
            ("independent", 0.76, 0.80),
            ("true", true_auc, true_auc + 0.05),
            ("corrupted", 0.74, 0.79),
        )
    ]


def valid_contract() -> dict[str, object]:
    return {
        "output_shape": [4, 64],
        "masked_loss_explicit_max_abs_error": 0.0,
        "parameter_counts_match": True,
        "cell_relabel_max_abs_error": 1e-7,
        "true_corrupted_logit_max_abs_difference": 0.1,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
    }


def test_formal_gate_attributes_neural_gain_only_after_all_controls() -> None:
    gate = adjudicate_profile_operator_attribution(
        ProfileOperatorAttributionConfig(run_id="e67-test"),
        {"e66_valid": True},
        {"profile_valid": True, "replay_error": 0.0},
        valid_contract(),
        {"trained_rows": trained_rows()},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_present_profile_operator_neural_gain_attributed"
    )
    assert gate["next_action"]["seed1"] is True


def test_formal_gate_holds_when_true_does_not_beat_ridge() -> None:
    gate = adjudicate_profile_operator_attribution(
        ProfileOperatorAttributionConfig(run_id="e67-test"),
        {"e66_valid": True},
        {"profile_valid": True},
        valid_contract(),
        {"trained_rows": trained_rows(true_auc=0.81)},
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_profile_operator_no_ridge_gain"
    assert gate["next_action"]["seed1"] is False


def test_plot_writes_chinese_e67_svg(tmp_path: Path) -> None:
    rows = trained_rows()
    summary = {
        "trained_rows": rows,
        "gate": {
            "decision": "innovation2_present_profile_operator_neural_gain_attributed",
            "metrics": {
                "e65_prefix_ridge_validation_auc": 0.793611,
                "true_minus_independent": 0.08,
                "true_minus_corrupted": 0.10,
                "true_minus_e65_prefix_ridge": 0.046,
                "true_train_validation_gap": 0.05,
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert (
        plot_main(["--summary", str(summary_path), "--output", str(output_path)])
        == 0
    )
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E67" in svg
    assert "PRESENT平衡谱算子正式归因" in svg
