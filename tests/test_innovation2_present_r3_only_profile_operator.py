from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.cli.plot_innovation2_present_r3_only_profile_operator import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_r3_only_profile_operator import (
    R3OnlyProfileConfig,
    adjudicate_r3_only_readiness,
    make_r3_only_model,
)


def test_r3_only_model_reduces_parameters_and_outputs_64_logits() -> None:
    config = R3OnlyProfileConfig(run_id="e73-test")
    models = {
        mode: make_r3_only_model(config, mode)
        for mode in ("independent", "true", "corrupted")
    }
    features = torch.randn(3, 64, 13)
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models.values()]

    assert models["true"](features).shape == (3, 64)
    assert len(set(counts)) == 1
    assert counts[0] / 5_679 <= 0.90


def rows(true_auc: float = 0.82) -> list[dict[str, object]]:
    aucs = {"independent": true_auc - 0.08, "corrupted": true_auc - 0.06, "true": true_auc}
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


def contract() -> dict[str, object]:
    return {
        "output_shape": [4, 64],
        "input_dim": 13,
        "masked_loss_explicit_max_abs_error": 0.0,
        "parameter_counts_match": True,
        "parameter_ratio_to_e68": 0.85,
        "topology_logit_max_abs_difference": 0.03,
        "cell_relabel_max_abs_error": 1e-7,
        "forbidden_named_state_absent": True,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
    }


def test_r3_only_gate_opens_formal_after_absolute_and_control_margins() -> None:
    gate = adjudicate_r3_only_readiness(
        R3OnlyProfileConfig(run_id="e73-test"),
        {"source_valid": True},
        contract(),
        {"trained_rows": rows()},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_r3_only_profile_readiness_passed"


def test_r3_only_gate_holds_without_corrupted_p_margin() -> None:
    result_rows = rows()
    result_rows[1]["validation_auc"] = result_rows[2]["validation_auc"] - 0.01
    gate = adjudicate_r3_only_readiness(
        R3OnlyProfileConfig(run_id="e73-test"),
        {"source_valid": True},
        contract(),
        {"trained_rows": result_rows},
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_r3_only_profile_readiness_not_passed"


def test_plot_writes_chinese_e73_svg(tmp_path: Path) -> None:
    result_rows = rows()
    gate = adjudicate_r3_only_readiness(
        R3OnlyProfileConfig(run_id="e73-test"),
        {"source_valid": True},
        contract(),
        {"trained_rows": result_rows},
    )
    summary = {
        "trained_rows": result_rows,
        "contract": {
            "parameter_counts": {row["relation_mode"]: 4_800 for row in result_rows},
            "parameter_ratio_to_e68": 0.845,
            "cell_relabel_max_abs_error": 1e-7,
        },
        "gate": gate,
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E73" in svg
    assert "第3轮前缀" in svg
