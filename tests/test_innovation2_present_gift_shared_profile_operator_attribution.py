from __future__ import annotations

import json
from pathlib import Path

import pytest

from blockcipher_nd.cli.plot_innovation2_present_gift_shared_profile_operator_attribution import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_gift_shared_profile_operator_attribution import (
    SharedProfileAttributionConfig,
    adjudicate_shared_profile_attribution,
)


def _contract() -> dict[str, object]:
    return {
        "output_shapes": {"present": [4, 64], "gift": [4, 64]},
        "masked_loss_explicit_max_abs_errors": {"present": 0.0, "gift": 0.0},
        "parameter_counts": {
            mode: 4_795 for mode in ("independent", "true", "corrupted")
        },
        "parameter_counts_match": True,
        "initial_parameter_max_abs_difference": 0.0,
        "runtime_topology_state_absent": True,
        "cipher_specific_named_state_absent": True,
        "topology_logit_max_abs_differences": {"present": 0.1, "gift": 0.2},
        "cell_relabel_max_abs_errors": {"present": 1e-7, "gift": 1e-7},
        "true_players_are_distinct": True,
        "corrupted_players_are_distinct_from_true": True,
        "all_players_are_permutations": True,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
    }


def _matrix(
    true_present: float = 0.94,
    true_gift: float = 0.91,
) -> dict[str, object]:
    values = {
        "independent": (0.66, 0.57),
        "corrupted": (0.82, 0.78),
        "true": (true_present, true_gift),
    }
    rows = []
    for mode, (present_auc, gift_auc) in values.items():
        row = {
            "relation_mode": mode,
            "epochs_completed": 30,
            "macro_validation_auc": (present_auc + gift_auc) / 2,
        }
        for cipher, auc in (("present", present_auc), ("gift", gift_auc)):
            row[f"{cipher}_train_auc"] = auc + 0.02
            row[f"{cipher}_train_accuracy"] = 0.80
            row[f"{cipher}_train_loss"] = 0.40
            row[f"{cipher}_validation_auc"] = auc
            row[f"{cipher}_validation_accuracy"] = 0.78
            row[f"{cipher}_validation_loss"] = 0.44
        rows.append(row)
    audits = {
        mode: {
            "epoch_batch_counts": [
                {"present": 7, "gift": 14} for _ in range(30)
            ],
            "total_updates": 630,
        }
        for mode in values
    }
    return {"rows": rows, "history": [], "schedule_audits": audits}


def test_e86_gate_passes_only_when_both_ciphers_retain_quality() -> None:
    gate = adjudicate_shared_profile_attribution(
        SharedProfileAttributionConfig(run_id="e86-test"),
        {"formal_source_valid": True},
        {"profile_source_valid": True},
        _contract(),
        _matrix(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_shared_profile_operator_seed0_attributed"
    assert gate["metrics"]["shared_parameter_count"] == 4_795
    assert gate["metrics"]["separate_parameter_count"] == 9_590

    held = adjudicate_shared_profile_attribution(
        SharedProfileAttributionConfig(run_id="e86-test"),
        {"formal_source_valid": True},
        {"profile_source_valid": True},
        _contract(),
        _matrix(true_gift=0.85),
    )
    assert held["status"] == "hold"
    assert held["decision"] == "innovation2_shared_profile_operator_quality_not_retained"


def test_e86_gate_separates_topology_failure_from_quality_failure() -> None:
    matrix = _matrix()
    true_row = next(
        row for row in matrix["rows"] if row["relation_mode"] == "true"
    )
    corrupted_row = next(
        row for row in matrix["rows"] if row["relation_mode"] == "corrupted"
    )
    corrupted_row["gift_validation_auc"] = true_row["gift_validation_auc"] - 0.01

    gate = adjudicate_shared_profile_attribution(
        SharedProfileAttributionConfig(run_id="e86-test"),
        {"formal_source_valid": True},
        {"profile_source_valid": True},
        _contract(),
        matrix,
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_shared_profile_operator_topology_not_attributed"


def test_plot_writes_clear_chinese_e86_svg(tmp_path: Path) -> None:
    gate = adjudicate_shared_profile_attribution(
        SharedProfileAttributionConfig(run_id="e86-test"),
        {"formal_source_valid": True},
        {"profile_source_valid": True},
        _contract(),
        _matrix(),
    )
    summary = tmp_path / "summary.json"
    output = tmp_path / "curves.svg"
    summary.write_text(json.dumps({"gate": gate}), encoding="utf-8")

    assert plot_main(["--summary", str(summary), "--output", str(output)]) == 0
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E86" in svg
    assert "不能用另一密码" in svg
    assert "不是零样本" in svg


def test_e86_protocol_is_frozen() -> None:
    with pytest.raises(ValueError, match="frozen"):
        SharedProfileAttributionConfig(run_id="e86-test", epochs=40)
