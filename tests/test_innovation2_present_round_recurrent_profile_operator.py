from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.plot_innovation2_present_round_recurrent_profile_operator import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_round_recurrent_profile_operator import (
    MODE_SPECS,
    RoundRecurrentProfileConfig,
    adjudicate_round_recurrent_readiness,
    make_round_recurrent_model,
)


def copy_parameters(source: torch.nn.Module, target: torch.nn.Module) -> None:
    targets = dict(target.named_parameters())
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            targets[name].copy_(parameter)


def test_round_recurrent_operator_has_fair_controls_and_64_logits() -> None:
    config = RoundRecurrentProfileConfig(run_id="e71-test")
    models = {mode: make_round_recurrent_model(config, mode) for mode in MODE_SPECS}
    candidate = models["true_order_true_P"]
    for mode, model in models.items():
        if mode != "true_order_true_P":
            copy_parameters(candidate, model)
    features = torch.randn(3, 64, 39)

    with torch.no_grad():
        logits = {mode: model(features) for mode, model in models.items()}

    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models.values()]
    assert logits["true_order_true_P"].shape == (3, 64)
    assert len(set(counts)) == 1
    assert 0.90 <= counts[0] / 5_679 <= 1.10
    assert not torch.allclose(
        logits["true_order_true_P"], logits["wrong_order_true_P"]
    )
    assert not torch.allclose(
        logits["true_order_true_P"], logits["true_order_corrupted_P"]
    )


def readiness_rows(candidate_auc: float = 0.78) -> list[dict[str, object]]:
    aucs = {
        "true_order_true_P": candidate_auc,
        "wrong_order_true_P": candidate_auc - 0.05,
        "true_order_corrupted_P": candidate_auc - 0.06,
    }
    return [
        {
            "mode": mode,
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


def readiness_contract() -> dict[str, object]:
    return {
        "output_shape": [4, 64],
        "masked_loss_explicit_max_abs_error": 0.0,
        "parameter_counts_match": True,
        "parameter_ratio_to_e68": 0.95,
        "round_order_logit_max_abs_difference": 0.02,
        "topology_logit_max_abs_difference": 0.03,
        "cell_relabel_max_abs_error": 1e-7,
        "forbidden_named_state_absent": True,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
    }


def test_round_recurrent_gate_opens_formal_only_after_both_controls() -> None:
    gate = adjudicate_round_recurrent_readiness(
        RoundRecurrentProfileConfig(run_id="e71-test"),
        {"source_valid": True},
        readiness_contract(),
        {"trained_rows": readiness_rows()},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_round_recurrent_readiness_passed"
    assert gate["next_action"]["formal_seed0"] is True


def test_round_recurrent_gate_holds_without_order_gain() -> None:
    rows = readiness_rows()
    rows[1]["validation_auc"] = rows[0]["validation_auc"] - 0.01
    gate = adjudicate_round_recurrent_readiness(
        RoundRecurrentProfileConfig(run_id="e71-test"),
        {"source_valid": True},
        readiness_contract(),
        {"trained_rows": rows},
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_present_round_recurrent_readiness_not_passed"
    )


def test_plot_writes_chinese_e71_svg(tmp_path: Path) -> None:
    rows = readiness_rows()
    summary = {
        "trained_rows": rows,
        "contract": {
            "parameter_counts": {row["mode"]: 5_400 for row in rows},
            "parameter_ratio_to_e68": 0.951,
            "cell_relabel_max_abs_error": 1e-7,
        },
        "gate": {
            "decision": "innovation2_present_round_recurrent_readiness_passed",
            "metrics": {
                "candidate_minus_wrong_order": 0.05,
                "candidate_minus_corrupted": 0.06,
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E71" in svg
    assert "显式轮序" in svg
