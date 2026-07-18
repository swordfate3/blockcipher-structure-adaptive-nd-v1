from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_gift64_r3_only_profile_operator_attribution import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.gift64_r3_only_profile_operator_attribution import (
    Gift64R3AttributionConfig,
    adjudicate_gift_r3_attribution,
    validate_e77_source,
)


def _e77_source() -> dict[str, object]:
    rows = [
        {"family": "deterministic_ridge", "variant": variant}
        for variant in ("local", "corrupted_shift1", "corrupted_shift2", "corrupted_shift3", "true")
    ]
    rows.extend(
        {"family": "frozen_checkpoint_inference", "variant": variant}
        for variant in ("corrupted_shift1", "corrupted_shift2", "corrupted_shift3", "true")
    )
    return {
        "gate": {
            "run_id": "i2_gift64_r4_topology_interaction_readjudication_20260719",
            "status": "pass",
            "decision": "innovation2_gift64_topology_interaction_gate_repaired",
            "protocol_checks": {"valid": True},
            "deterministic_checks": {"valid": True},
            "checkpoint_checks": {"valid": True},
            "metrics": {
                "ridges": {"true": {"validation_auc": 0.7434963579604579}}
            },
            "next_action": {"training_performed": False},
        },
        "rows": rows,
        "hashes": {"gate": "a" * 64, "results": "b" * 64},
    }


def test_e77_source_validation_requires_passed_no_training_audit() -> None:
    assert all(validate_e77_source(_e77_source()).values())


def _rows(true_auc: float = 0.86) -> list[dict[str, object]]:
    aucs = {
        "independent": true_auc - 0.15,
        "corrupted": true_auc - 0.08,
        "true": true_auc,
    }
    return [
        {
            "relation_mode": mode,
            "epochs_completed": 30,
            "best_epoch": 24,
            "train_auc": auc + 0.04,
            "train_accuracy": 0.78,
            "train_loss": 0.48,
            "validation_auc": auc,
            "validation_accuracy": 0.74,
            "validation_loss": 0.54,
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


def test_e78_gate_opens_seed1_after_quality_and_topology_pass() -> None:
    gate = adjudicate_gift_r3_attribution(
        Gift64R3AttributionConfig(run_id="e78-test"),
        {"profile_valid": True},
        {"e77_valid": True},
        _contract(),
        {"trained_rows": _rows()},
        _e77_source(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_gift64_r3_only_neural_gain_attributed"


def test_e78_gate_holds_when_true_does_not_beat_corrupted() -> None:
    rows = _rows()
    rows[1]["validation_auc"] = rows[2]["validation_auc"] - 0.01
    gate = adjudicate_gift_r3_attribution(
        Gift64R3AttributionConfig(run_id="e78-test"),
        {"profile_valid": True},
        {"e77_valid": True},
        _contract(),
        {"trained_rows": rows},
        _e77_source(),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_gift64_r3_only_topology_not_attributed"


def test_plot_writes_clear_chinese_e78_svg(tmp_path: Path) -> None:
    rows = _rows()
    gate = adjudicate_gift_r3_attribution(
        Gift64R3AttributionConfig(run_id="e78-test"),
        {"profile_valid": True},
        {"e77_valid": True},
        _contract(),
        {"trained_rows": rows},
        _e77_source(),
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(
        json.dumps({"trained_rows": rows, "gate": gate}), encoding="utf-8"
    )

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E78" in svg
    assert "30轮seed0" in svg
    assert "确定性安全锚点" in svg


def test_e78_training_protocol_is_frozen() -> None:
    try:
        Gift64R3AttributionConfig(run_id="e78-test", seed=1)
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("E78 seed0 protocol must remain frozen")
