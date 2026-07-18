from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_r3_only_profile_operator_replication import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_r3_only_profile_operator_replication import (
    R3OnlyReplicationConfig,
    adjudicate_r3_only_replication,
)


def rows(seed: int, true_auc: float) -> list[dict[str, object]]:
    aucs = {"independent": 0.66, "corrupted": 0.81, "true": true_auc}
    return [
        {
            "relation_mode": mode,
            "seed": seed,
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
        "parameter_counts_match": True,
        "parameter_ratio_to_e68": 0.844,
        "cell_relabel_max_abs_error": 1e-7,
    }


def test_r3_only_replication_confirms_two_seed_method() -> None:
    seed0_source = {"rows": rows(0, 0.9455555555555556)}
    gate = adjudicate_r3_only_replication(
        R3OnlyReplicationConfig(run_id="e73-seed1-test"),
        {"seed0_valid": True},
        {"source_valid": True},
        contract(),
        seed0_source,
        {"trained_rows": rows(1, 0.95)},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_r3_only_two_seed_confirmed"


def test_r3_only_replication_holds_when_seed1_loses_quality() -> None:
    seed0_source = {"rows": rows(0, 0.9455555555555556)}
    gate = adjudicate_r3_only_replication(
        R3OnlyReplicationConfig(run_id="e73-seed1-test"),
        {"seed0_valid": True},
        {"source_valid": True},
        contract(),
        seed0_source,
        {"trained_rows": rows(1, 0.90)},
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_r3_only_seed_not_replicated"


def test_plot_writes_chinese_e73_replication_svg(tmp_path: Path) -> None:
    seed0_source = {"rows": rows(0, 0.9455555555555556)}
    seed1_rows = rows(1, 0.95)
    gate = adjudicate_r3_only_replication(
        R3OnlyReplicationConfig(run_id="e73-seed1-test"),
        {"seed0_valid": True},
        {"source_valid": True},
        contract(),
        seed0_source,
        {"trained_rows": seed1_rows},
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps({"gate": gate}), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E73" in svg
    assert "双seed复核" in svg
