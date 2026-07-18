from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_balance_profile_operator_replication import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_replication import (
    ProfileOperatorReplicationConfig,
    adjudicate_profile_operator_replication,
)


def rows(seed: int, true_auc: float) -> list[dict[str, float | int | str]]:
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
            "best_epoch": 20,
            "seed": seed,
            "parameter_count": 5679,
        }
        for mode, validation_auc, train_auc in (
            ("independent", 0.76, 0.80),
            ("true", true_auc, true_auc + 0.04),
            ("corrupted", 0.74, 0.79),
        )
    ]


def contract() -> dict[str, object]:
    return {
        "output_shape": [4, 64],
        "masked_loss_explicit_max_abs_error": 0.0,
        "parameter_counts_match": True,
        "cell_relabel_max_abs_error": 1e-7,
        "true_corrupted_logit_max_abs_difference": 0.1,
    }


def test_replication_gate_requires_seed1_and_joint_margins() -> None:
    seed0 = {"results": rows(0, 0.95)}
    gate = adjudicate_profile_operator_replication(
        ProfileOperatorReplicationConfig(run_id="e68-test"),
        {"seed0_valid": True},
        {"profile_valid": True},
        contract(),
        seed0,
        {"trained_rows": rows(1, 0.92)},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_present_profile_operator_two_seed_confirmed"
    )
    assert all(gate["seed1_checks"].values())
    assert all(gate["joint_checks"].values())


def test_replication_gate_holds_when_seed1_loses_relation_margin() -> None:
    seed0 = {"results": rows(0, 0.95)}
    seed1 = rows(1, 0.80)
    next(row for row in seed1 if row["relation_mode"] == "corrupted")[
        "validation_auc"
    ] = 0.79
    gate = adjudicate_profile_operator_replication(
        ProfileOperatorReplicationConfig(run_id="e68-test"),
        {"seed0_valid": True},
        {"profile_valid": True},
        contract(),
        seed0,
        {"trained_rows": seed1},
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_present_profile_operator_seed_not_replicated"
    )


def test_plot_writes_chinese_e68_svg(tmp_path: Path) -> None:
    seed_metrics = {
        "seed0": {
            "true_auc": 0.95,
            "independent_auc": 0.76,
            "corrupted_auc": 0.80,
            "true_minus_independent": 0.19,
            "true_minus_corrupted": 0.15,
            "true_minus_ridge": 0.16,
            "train_validation_gap": 0.03,
        },
        "seed1": {
            "true_auc": 0.92,
            "independent_auc": 0.75,
            "corrupted_auc": 0.78,
            "true_minus_independent": 0.17,
            "true_minus_corrupted": 0.14,
            "true_minus_ridge": 0.13,
            "train_validation_gap": 0.05,
        },
    }
    summary = {
        "gate": {
            "decision": "innovation2_present_profile_operator_two_seed_confirmed",
            "metrics": {
                "seed_metrics": seed_metrics,
                "mean_metrics": {
                    key: (seed_metrics["seed0"][key] + seed_metrics["seed1"][key])
                    / 2
                    for key in seed_metrics["seed0"]
                },
            },
        }
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert (
        plot_main(["--summary", str(summary_path), "--output", str(output_path)])
        == 0
    )
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E68" in svg
    assert "跨随机种子复现" in svg
