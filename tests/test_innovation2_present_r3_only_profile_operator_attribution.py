from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_r3_only_profile_operator_attribution import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_r3_only_profile_operator_attribution import (
    R3OnlyAttributionConfig,
    adjudicate_r3_only_attribution,
)


def rows(true_auc: float = 0.95) -> list[dict[str, object]]:
    aucs = {"independent": 0.76, "corrupted": 0.80, "true": true_auc}
    return [
        {
            "relation_mode": mode,
            "epochs_completed": 30,
            "train_auc": auc + 0.02,
            "train_accuracy": 0.80,
            "train_loss": 0.30,
            "validation_auc": auc,
            "validation_accuracy": 0.78,
            "validation_loss": 0.35,
        }
        for mode, auc in aucs.items()
    ]


def contract() -> dict[str, object]:
    return {
        "output_shape": [4, 64],
        "input_dim": 13,
        "masked_loss_explicit_max_abs_error": 0.0,
        "parameter_counts_match": True,
        "parameter_ratio_to_e68": 0.844,
        "topology_logit_max_abs_difference": 0.03,
        "cell_relabel_max_abs_error": 1e-7,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
    }


def test_r3_only_formal_gate_opens_seed1_when_quality_is_retained() -> None:
    gate = adjudicate_r3_only_attribution(
        R3OnlyAttributionConfig(run_id="e73-formal-test"),
        {"readiness_valid": True},
        {"source_valid": True},
        contract(),
        {"trained_rows": rows()},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_r3_only_neural_gain_attributed"
    assert gate["next_action"]["seed1"] is True


def test_r3_only_formal_gate_holds_when_full_prefix_quality_is_lost() -> None:
    gate = adjudicate_r3_only_attribution(
        R3OnlyAttributionConfig(run_id="e73-formal-test"),
        {"readiness_valid": True},
        {"source_valid": True},
        contract(),
        {"trained_rows": rows(true_auc=0.90)},
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_r3_only_quality_not_retained"


def test_plot_writes_chinese_e73_formal_svg(tmp_path: Path) -> None:
    result_rows = rows()
    gate = adjudicate_r3_only_attribution(
        R3OnlyAttributionConfig(run_id="e73-formal-test"),
        {"readiness_valid": True},
        {"source_valid": True},
        contract(),
        {"trained_rows": result_rows},
    )
    summary = {
        "trained_rows": result_rows,
        "contract": {"parameter_counts": {"true": 4_795}},
        "gate": gate,
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E73" in svg
    assert "30轮正式归因" in svg
