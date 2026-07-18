from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.cli.plot_innovation2_present_active_dimension_zero_shot_transfer import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_active_dimension_zero_shot_transfer import (
    ActiveDimensionTransferConfig,
    SupportGrowthCapExceeded,
    adjudicate_active_dimension_transfer,
    build_transfer_rows,
    compatible_prefix_features,
    variable_dimension_supports,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    anf_prefix_features,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    possible_active_monomials,
)


def test_variable_dimension_prefix_replays_eight_bit_contract() -> None:
    active_bits = tuple(range(8))
    variable = {
        rounds: variable_dimension_supports(active_bits, rounds)
        for rounds in (1, 2, 3)
    }
    compatible = compatible_prefix_features(variable, 8)
    original = {
        rounds: possible_active_monomials(active_bits, rounds)
        for rounds in (1, 2, 3)
    }

    assert compatible.shape == (64, 39)
    assert np.isfinite(compatible).all()
    for output_bit in range(8):
        expected = anf_prefix_features(np.asarray([output_bit]), original)
        assert np.array_equal(compatible[output_bit], expected)


def test_twelve_bit_prefix_keeps_39_dimensions_and_folds_high_degree() -> None:
    active_bits = tuple(range(12))
    supports = {
        rounds: variable_dimension_supports(active_bits, rounds)
        for rounds in (1, 2, 3)
    }

    features = compatible_prefix_features(supports, 12)

    assert features.shape == (64, 39)
    assert np.isfinite(features).all()
    assert np.all(features >= 0.0)


def test_variable_dimension_support_stops_before_unbounded_product() -> None:
    with pytest.raises(SupportGrowthCapExceeded) as error:
        variable_dimension_supports(
            tuple(range(12)), rounds=4, combination_cap=8
        )

    assert error.value.details["candidate_count"] > 8
    assert error.value.details["cap"] == 8


def test_transfer_checkerboard_balances_structure_and_output() -> None:
    labels = np.asarray(
        [
            [1, 0, -1, -1],
            [0, 1, -1, -1],
            [-1, -1, 1, 0],
            [-1, -1, 0, 1],
        ],
        dtype=np.int8,
    )

    transfer = build_transfer_rows(labels, dimension=4, attempts=4)

    assert transfer["metrics"]["positive"] == 4
    assert transfer["metrics"]["negative"] == 4
    assert transfer["balance"]["maximum_structure_class_delta"] == 0
    assert transfer["balance"]["maximum_mask_class_delta"] == 0


def ready_dimension_payload() -> dict[str, object]:
    labels = np.full((16, 64), -1, dtype=np.int8)
    labels.reshape(-1)[:128] = 1
    labels.reshape(-1)[128:256] = 0
    return {
        "data": {
            "labels": labels,
            "provider_complete": True,
            "provider_cap_events": [],
            "completed_structures": 16,
        },
        "transfer": {
            "metrics": {
                "rows": 80,
                "positive": 40,
                "negative": 40,
                "structures": 8,
                "output_bits": 16,
            },
            "balance": {
                "maximum_structure_class_delta": 0,
                "maximum_mask_class_delta": 0,
            },
        },
        "scalar_validation": {"all_pass": True, "checked": 16, "passed": 16},
    }


def transfer_reports(true_auc: float = 0.82) -> dict[str, dict[str, float]]:
    return {
        str(dimension): {
            "seed0_true_auc": true_auc,
            "seed1_true_auc": true_auc - 0.01,
            "seed0_independent_auc": 0.62,
            "seed1_independent_auc": 0.61,
            "seed0_corrupted_auc": 0.64,
            "seed1_corrupted_auc": 0.63,
            "e65_ridge_auc": 0.65,
        }
        for dimension in (4, 12)
    }


def test_transfer_gate_confirms_only_ready_labels_and_model_margins() -> None:
    gate = adjudicate_active_dimension_transfer(
        ActiveDimensionTransferConfig(run_id="e70-test"),
        {"source_valid": True, "replay_error": 0.0},
        {4: ready_dimension_payload(), 12: ready_dimension_payload()},
        transfer_reports(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_present_active_dimension_zero_shot_confirmed"
    )
    assert gate["next_action"]["dimension_conditioned_training"] is True


def test_transfer_gate_does_not_interpret_narrow_labels() -> None:
    narrow = ready_dimension_payload()
    narrow["data"]["labels"][:] = -1
    gate = adjudicate_active_dimension_transfer(
        ActiveDimensionTransferConfig(run_id="e70-test"),
        {"source_valid": True},
        {4: narrow, 12: ready_dimension_payload()},
        transfer_reports(),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_present_active_dimension_transfer_labels_not_ready"
    )


def test_transfer_gate_requires_complete_label_provider() -> None:
    incomplete = ready_dimension_payload()
    incomplete["data"]["provider_complete"] = False
    incomplete["data"]["provider_cap_events"] = [{"candidate_count": 9, "cap": 8}]
    incomplete["data"]["completed_structures"] = 15

    gate = adjudicate_active_dimension_transfer(
        ActiveDimensionTransferConfig(run_id="e70-test"),
        {"source_valid": True},
        {4: incomplete, 12: ready_dimension_payload()},
        transfer_reports(),
    )

    assert gate["status"] == "hold"
    assert gate["label_checks"]["d4_provider_complete"] is False


def test_plot_writes_chinese_e70_svg(tmp_path: Path) -> None:
    dimensions = {
        str(dimension): {
            "raw_positive": 300,
            "raw_negative": 500,
            "raw_unknown": 224,
            "transfer": report,
        }
        for dimension, report in (
            (4, transfer_reports()["4"]),
            (12, transfer_reports()["12"]),
        )
    }
    summary = {
        "gate": {
            "decision": "innovation2_present_active_dimension_zero_shot_confirmed",
            "metrics": {
                "dimensions": dimensions,
                "mean_deltas": {
                    "true_minus_independent": 0.20,
                    "true_minus_corrupted": 0.18,
                    "true_minus_ridge": 0.17,
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
    assert "创新2 E70" in svg
    assert "跨活动维度迁移" in svg


def test_plot_hides_placeholder_auc_when_labels_are_not_ready(tmp_path: Path) -> None:
    transfer = transfer_reports()["4"]
    dimensions = {
        "4": {
            "raw_positive": 0,
            "raw_negative": 0,
            "raw_unknown": 1024,
            "completed_structures": 16,
            "provider_cap_events": [],
            "transfer": transfer,
        },
        "12": {
            "raw_positive": 0,
            "raw_negative": 0,
            "raw_unknown": 1024,
            "completed_structures": 0,
            "provider_cap_events": [{"candidate_count": 4_741_632, "cap": 2_000_000}],
            "transfer": transfer,
        },
    }
    summary = {
        "gate": {
            "decision": "innovation2_present_active_dimension_transfer_labels_not_ready",
            "metrics": {
                "dimensions": dimensions,
                "mean_deltas": {
                    "true_minus_independent": 0.0,
                    "true_minus_corrupted": 0.0,
                    "true_minus_ridge": 0.0,
                },
            },
        }
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "标签提供器完成度" in svg
    assert "不展示、不计算也不解释" in svg
    assert "checkpoint零样本迁移" not in svg
