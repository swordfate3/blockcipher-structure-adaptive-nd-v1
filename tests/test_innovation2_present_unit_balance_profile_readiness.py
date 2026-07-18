from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_present_unit_balance_profile import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_unit_balance_profile_readiness import (
    UnitBalanceProfileConfig,
    adjudicate_unit_profile,
    assemble_profile_targets,
)


def test_profile_targets_keep_unobserved_entries_masked() -> None:
    rows = [
        {"structure_index": 0, "mask_index": 1, "label": 1},
        {"structure_index": 0, "mask_index": 2, "label": 0},
        {"structure_index": 1, "mask_index": 1, "label": 0},
        {"structure_index": 1, "mask_index": 2, "label": 1},
    ]

    targets, observed = assemble_profile_targets(rows, 2, 4)

    assert targets.tolist() == [[-1, 1, 0, -1], [-1, 0, 1, -1]]
    assert observed.tolist() == [
        [False, True, True, False],
        [False, True, True, False],
    ]


def test_profile_targets_reject_duplicate_edges() -> None:
    rows = [
        {"structure_index": 0, "mask_index": 0, "label": 1},
        {"structure_index": 0, "mask_index": 0, "label": 0},
    ]

    try:
        assemble_profile_targets(rows, 1, 1)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate profile edge must fail")


def test_adjudication_selects_prefix_guided_profile_route() -> None:
    targets = np.full((96, 64), -1, dtype=np.int8)
    observed = np.zeros((96, 64), dtype=np.bool_)
    rows = []
    for split, structure_range, outputs in (
        ("train", range(48), range(32)),
        ("validation", range(64, 80), range(24)),
    ):
        for structure in structure_range:
            for output in outputs:
                if (structure + output) % 8:
                    continue
                label = (structure + output) % 2
                rows.append(
                    {
                        "split": split,
                        "structure_index": structure,
                        "mask_index": output,
                        "label": label,
                    }
                )
                targets[structure, output] = label
                observed[structure, output] = True
    benchmark = {
        "rows": rows,
        "profile_targets": targets,
        "profile_observed": observed,
        "split_metrics": {
            "train": {"positive": 200, "negative": 200},
            "validation": {"positive": 60, "negative": 60},
        },
        "balance": {
            "duplicate_edges": 0,
            "maximum_structure_class_delta": 0,
            "maximum_mask_class_delta": 0,
        },
        "marginal_baselines": {"strongest_auc": 0.5},
    }
    true_player = np.arange(64)
    corrupted_player = np.roll(true_player, 1)
    table = {
        "matrices": {
            name: np.zeros((len(rows), 2))
            for name in (
                "static_set",
                "corrupted_topology",
                "true_topology",
                "anf_prefix",
            )
        },
        "true_player": true_player,
        "corrupted_player": corrupted_player,
    }
    reports = {
        "static_set": {"validation_auc": 0.50, "train_standardization_only": True},
        "corrupted_topology": {"validation_auc": 0.52, "train_standardization_only": True},
        "true_topology": {"validation_auc": 0.62, "train_standardization_only": True},
        "anf_prefix": {"validation_auc": 0.68, "train_standardization_only": True},
    }

    gate = adjudicate_unit_profile(
        UnitBalanceProfileConfig(run_id="e65-test"),
        {"source_valid": True},
        benchmark,
        table,
        reports,
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_unit_balance_profile_prefix_ready"
    assert gate["metrics"]["selected_route"] == "prefix_guided_profile_route"


def test_plot_writes_chinese_e65_svg(tmp_path: Path) -> None:
    summary = {
        "reports": {
            "static_set": {"validation_auc": 0.50},
            "corrupted_topology": {"validation_auc": 0.52},
            "true_topology": {"validation_auc": 0.62},
            "anf_prefix": {"validation_auc": 0.68},
        },
        "gate": {
            "decision": "innovation2_present_unit_balance_profile_prefix_ready",
            "metrics": {
                "split_metrics": {
                    "train": {"positive": 178, "negative": 178},
                    "validation": {"positive": 60, "negative": 60},
                },
                "train_structures": 50,
                "validation_structures": 18,
                "train_outputs": 32,
                "validation_outputs": 23,
                "shared_outputs": 23,
                "marginal_baselines": {"strongest_auc": 0.5},
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert (
        plot_main(["--summary", str(summary_path), "--output", str(output_path)])
        == 0
    )
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E65" in svg
    assert "单位输出积分平衡谱" in svg
