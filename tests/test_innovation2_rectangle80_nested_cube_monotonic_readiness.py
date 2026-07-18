from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.cli.plot_innovation2_rectangle80_nested_cube_monotonic_readiness import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.rectangle80_nested_cube_monotonic_readiness import (
    Rectangle80NestedCubeConfig,
    adjudicate_nested_cube_checks,
    apply_positive_monotonic_closure,
    make_nested_chains,
    nested_unary_baselines,
)


def test_e94_protocol_is_frozen() -> None:
    assert Rectangle80NestedCubeConfig().chain_count == 192
    with pytest.raises(ValueError, match="frozen"):
        Rectangle80NestedCubeConfig(chain_count=96)


def test_nested_chain_construction_is_deterministic_and_strict() -> None:
    structures = [
        {"active_bits": list(range(8))},
        {"active_bits": [0, 2, 4, 6, 8, 10, 12, 14]},
    ]

    first = make_nested_chains(structures)
    second = make_nested_chains(structures)

    assert first == second
    assert first[0].removed_bit == 0
    assert first[0].added_bit == 8
    assert first[0].split == "validation"
    assert first[1].removed_bit == 2
    assert set(first[1].active_bits_7) < set(first[1].active_bits_8)
    assert set(first[1].active_bits_8) < set(first[1].active_bits_9)


def test_positive_monotonic_closure_propagates_only_into_unknowns() -> None:
    direct = np.full((1, 3, 4), -1, dtype=np.int8)
    direct[0, :, 0] = (1, -1, -1)
    direct[0, :, 1] = (-1, 1, -1)
    direct[0, :, 2] = (1, 0, -1)
    direct[0, :, 3] = (0, -1, 0)

    closed, metrics = apply_positive_monotonic_closure(direct)

    np.testing.assert_array_equal(closed[0, :, 0], (1, 1, 1))
    np.testing.assert_array_equal(closed[0, :, 1], (-1, 1, 1))
    np.testing.assert_array_equal(closed[0, :, 2], (1, 0, 1))
    np.testing.assert_array_equal(closed[0, :, 3], (0, -1, 0))
    assert metrics["d7_positive_to_d8_negative"] == 1
    assert metrics["inherited_positive_d8"] == 1
    assert metrics["inherited_positive_d9"] == 3


def test_e94_gate_separates_protocol_from_label_width() -> None:
    passed = {"check": True}
    failed = {"check": False}

    assert adjudicate_nested_cube_checks(passed, passed, passed, passed)[:2] == (
        "pass",
        "innovation2_rectangle80_nested_cube_monotonic_labels_ready",
    )
    assert adjudicate_nested_cube_checks(passed, passed, failed, passed)[:2] == (
        "hold",
        "innovation2_rectangle80_nested_cube_monotonic_labels_not_ready",
    )
    assert adjudicate_nested_cube_checks(passed, failed, passed, passed)[:2] == (
        "fail",
        "innovation2_rectangle80_nested_cube_monotonic_protocol_invalid",
    )


def test_nested_unary_baselines_are_chance_for_balanced_chain_rows() -> None:
    rows = []
    for chain_index in range(8):
        split = "validation" if not chain_index % 4 else "train"
        for dimension in (7, 8, 9):
            labels = (0, 1) if chain_index % 2 == 0 else (1, 0)
            for output_bit, label in enumerate(labels):
                rows.append(
                    {
                        "split": split,
                        "chain_index": chain_index,
                        "dimension": dimension,
                        "output_bit": output_bit,
                        "removed_bit": chain_index,
                        "added_bit": 63 - chain_index,
                        "active_bits": list(range(dimension)),
                        "label": label,
                    }
                )

    baselines = nested_unary_baselines(rows)

    assert baselines["strongest_auc"] == pytest.approx(0.5)


def test_plot_writes_clear_chinese_e94_svg(tmp_path: Path) -> None:
    counts = {
        "d7": {"positive": 1000, "negative": 900, "unknown": 100},
        "d8": {"positive": 1400, "negative": 500, "unknown": 100},
        "d9": {"positive": 1700, "negative": 200, "unknown": 100},
    }
    matched = {
        name: {
            "train": {"positive": 120, "negative": 120},
            "validation": {"positive": 40, "negative": 40},
        }
        for name in ("d7", "d8", "d9")
    }
    summary = {
        "gate": {
            "decision": "innovation2_rectangle80_nested_cube_monotonic_labels_ready",
            "metrics": {
                "direct_counts": counts,
                "closed_positive_prevalence": {"d7": 0.52, "d8": 0.74, "d9": 0.89},
                "matched_dimension_metrics": matched,
                "matched_unary_baselines": {"strongest_auc": 0.5},
                "transition_chains": 100,
                "monotonicity": {
                    "d7_positive_to_d8_negative": 0,
                    "d8_positive_to_d9_negative": 0,
                    "d7_positive_to_d9_negative": 0,
                },
            },
        }
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E94" in svg
    assert "7/8/9-bit嵌套cube" in svg
    assert "不训练网络" in svg
