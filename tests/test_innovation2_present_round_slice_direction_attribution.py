from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_round_slice_direction import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_round_slice_direction_attribution import (
    RoundSliceDirectionConfig,
    adjudicate_round_slice_direction,
)


def source_checks() -> dict[str, object]:
    return {
        "source_valid": True,
        "seed0_expected_auc": 0.95,
        "seed1_expected_auc": 0.96,
    }


def evaluation(consensus: bool = True) -> dict[str, object]:
    seed1_drops = {"r1": 0.09, "r2": 0.03, "r3": 0.01}
    if not consensus:
        seed1_drops = {"r1": 0.02, "r2": 0.03, "r3": 0.10}
    return {
        "ridge_auc": {"r1": 0.76, "r2": 0.70, "r3": 0.66},
        "full_ridge_auc": 0.7936111111111112,
        "checkpoints": {
            "seed0": {
                "intact_auc": 0.95,
                "neutralized_auc": {"r1": 0.85, "r2": 0.92, "r3": 0.94},
                "ablation_drop": {"r1": 0.10, "r2": 0.03, "r3": 0.01},
            },
            "seed1": {
                "intact_auc": 0.96,
                "neutralized_auc": {
                    round_name: 0.96 - drop for round_name, drop in seed1_drops.items()
                },
                "ablation_drop": seed1_drops,
            },
        },
        "train_mean_source_structures": 50,
        "validation_structures": 18,
    }


def test_round_slice_gate_opens_only_consistent_r1_skip_route() -> None:
    gate = adjudicate_round_slice_direction(
        RoundSliceDirectionConfig(run_id="e72-test"),
        source_checks(),
        evaluation(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_early_round_skip_candidate_ready"
    assert gate["next_action"]["neural_readiness"] is True


def test_round_slice_gate_holds_on_cross_seed_direction_disagreement() -> None:
    gate = adjudicate_round_slice_direction(
        RoundSliceDirectionConfig(run_id="e72-test"),
        source_checks(),
        evaluation(consensus=False),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_round_direction_not_confirmed"


def test_plot_writes_chinese_e72_svg(tmp_path: Path) -> None:
    gate = adjudicate_round_slice_direction(
        RoundSliceDirectionConfig(run_id="e72-test"),
        source_checks(),
        evaluation(),
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps({"gate": gate}), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E72" in svg
    assert "前缀轮" in svg
