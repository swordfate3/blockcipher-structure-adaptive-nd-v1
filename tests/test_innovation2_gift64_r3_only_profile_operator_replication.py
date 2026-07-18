from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_gift64_r3_only_profile_operator_replication import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.gift64_r3_only_profile_operator_replication import (
    Gift64R3ReplicationConfig,
    adjudicate_gift_r3_replication,
    validate_e78_source,
)


def _rows(seed: int, true_auc: float) -> list[dict[str, object]]:
    aucs = {
        "independent": true_auc - 0.24,
        "corrupted": true_auc - 0.12,
        "true": true_auc,
    }
    return [
        {
            "relation_mode": mode,
            "seed": seed,
            "epochs_completed": 30,
            "best_epoch": 25,
            "train_auc": auc + 0.03,
            "train_accuracy": 0.78,
            "train_loss": 0.48,
            "validation_auc": auc,
            "validation_accuracy": 0.74,
            "validation_loss": 0.54,
        }
        for mode, auc in aucs.items()
    ]


def _e78_source() -> dict[str, object]:
    return {
        "gate": {
            "run_id": "i2_gift64_r4_r3_only_profile_operator_attribution_seed0_20260719",
            "status": "pass",
            "decision": "innovation2_gift64_r3_only_neural_gain_attributed",
            "protocol_checks": {"valid": True},
            "candidate_checks": {"valid": True},
            "relation_checks": {"valid": True},
            "metrics": {"e77_true_topology_ridge_auc": 0.7434963579604579},
        },
        "rows": _rows(0, 0.91),
        "hashes": {"gate": "a" * 64, "results": "b" * 64},
    }


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


def test_e78_source_validation_requires_seed0_pass() -> None:
    assert all(validate_e78_source(_e78_source()).values())


def test_e79_gate_confirms_two_seed_result() -> None:
    gate = adjudicate_gift_r3_replication(
        Gift64R3ReplicationConfig(run_id="e79-test"),
        {"profile_valid": True},
        {"e78_valid": True},
        _contract(),
        {"trained_rows": _rows(1, 0.90)},
        _e78_source(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_gift64_r3_only_two_seed_confirmed"


def test_e79_gate_holds_when_seed1_topology_margin_fails() -> None:
    rows = _rows(1, 0.90)
    rows[1]["validation_auc"] = rows[2]["validation_auc"] - 0.01
    gate = adjudicate_gift_r3_replication(
        Gift64R3ReplicationConfig(run_id="e79-test"),
        {"profile_valid": True},
        {"e78_valid": True},
        _contract(),
        {"trained_rows": rows},
        _e78_source(),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_gift64_r3_only_seed_not_replicated"


def test_plot_writes_clear_chinese_e79_svg(tmp_path: Path) -> None:
    seed1_rows = _rows(1, 0.90)
    gate = adjudicate_gift_r3_replication(
        Gift64R3ReplicationConfig(run_id="e79-test"),
        {"profile_valid": True},
        {"e78_valid": True},
        _contract(),
        {"trained_rows": seed1_rows},
        _e78_source(),
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(
        json.dumps({"trained_rows": seed1_rows, "gate": gate}), encoding="utf-8"
    )

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E79" in svg
    assert "seed1独立复现" in svg
    assert "平均值不能掩盖" in svg


def test_e79_training_protocol_is_frozen() -> None:
    try:
        Gift64R3ReplicationConfig(run_id="e79-test", seed=2)
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("E79 seed1 protocol must remain frozen")
