from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_present_multibit_mask_profile import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_multibit_mask_profile_readiness import (
    MULTIBIT_FAMILIES,
    MultibitMaskProfileConfig,
    adjudicate_multibit_profile,
    decompose_multibit_labels,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    LinearOutputMask,
)


def test_multibit_decomposition_detects_componentwise_and_nontrivial_positive() -> None:
    masks = tuple(
        [
            LinearOutputMask(index=index, mask_id=f"u{index}", family="unit", value=1 << index)
            for index in range(64)
        ]
        + [
            LinearOutputMask(index=64, mask_id="m64", family="nibble", value=0b11)
        ]
    )
    labels = np.full((2, 65), -1, dtype=np.int8)
    labels[0, 0] = labels[0, 1] = labels[0, 64] = 1
    labels[1, 0] = 1
    labels[1, 1] = 0
    labels[1, 64] = 1
    source = {"labels": labels, "masks": masks}
    benchmark = {
        "rows": [
            {
                "split": "validation",
                "structure_index": 0,
                "mask_index": 64,
                "mask_family": "nibble",
                "label": 1,
            },
            {
                "split": "validation",
                "structure_index": 1,
                "mask_index": 64,
                "mask_family": "nibble",
                "label": 1,
            },
        ]
    }

    result = decompose_multibit_labels(source, benchmark)

    assert result["rows"][0]["all_component_units_positive"] is True
    assert result["rows"][0]["nontrivial_positive"] is False
    assert result["rows"][1]["all_component_units_positive"] is False
    assert result["rows"][1]["nontrivial_positive"] is True


def test_multibit_gate_holds_componentwise_dominated_labels() -> None:
    family_metrics = {
        family: {
            "train": {"positive": 50, "negative": 50},
            "validation": {"positive": 20, "negative": 20},
        }
        for family in MULTIBIT_FAMILIES
    }
    rows = []
    for split, structures in (("train", range(48)), ("validation", range(64, 80))):
        for family_index, family in enumerate(MULTIBIT_FAMILIES):
            for structure in structures:
                for offset in range(2):
                    rows.append(
                        {
                            "split": split,
                            "structure_index": structure,
                            "mask_index": 64 + 10 * family_index + offset,
                            "mask_family": family,
                            "label": offset,
                        }
                    )
    benchmark = {
        "rows": rows,
        "split_metrics": {
            "train": {"positive": 300, "negative": 300},
            "validation": {"positive": 100, "negative": 100},
        },
        "family_metrics": family_metrics,
        "balance": {
            "duplicate_edges": 0,
            "maximum_structure_class_delta": 0,
            "maximum_mask_class_delta": 0,
        },
        "marginal_baselines": {"strongest_auc": 0.5},
        "family_benchmarks": {
            family: {
                "rows": [
                    {"split": "train", "mask_index": 64 + index},
                    {"split": "validation", "mask_index": 64 + index},
                ],
                "marginal_baselines": {"strongest_auc": 0.5},
            }
            for index, family in enumerate(MULTIBIT_FAMILIES)
        },
    }
    decomposition_rows = [
        {**row, "all_component_units_positive": bool(row["label"])} for row in rows
    ]
    decomposition = {
        "rows": decomposition_rows,
        "reports": {
            scope: {
                "train_nontrivial_positive": 0,
                "validation_nontrivial_positive": 0,
                "nontrivial_positive_fraction": 0.0,
                "validation_componentwise_auc": 1.0,
                "validation_positive": 20,
                "raw_positive": 100,
                "raw_nontrivial_positive": 0,
                "raw_nontrivial_positive_fraction": 0.0,
            }
            for scope in ("combined", *MULTIBIT_FAMILIES)
        },
    }
    table = {"matrices": {"x": np.zeros((len(rows), 1))}}
    feature_reports = {
        scope: {
            family: {"validation_auc": 0.70}
            for family in (
                "static_set",
                "corrupted_topology",
                "true_topology",
                "anf_prefix",
            )
        }
        for scope in ("combined", *MULTIBIT_FAMILIES)
    }

    gate = adjudicate_multibit_profile(
        MultibitMaskProfileConfig(run_id="e69-test"),
        {"source_valid": True},
        benchmark,
        decomposition,
        table,
        feature_reports,
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_present_multibit_profile_componentwise_dominated"
    )
    assert gate["next_action"]["mask_query_decoder"] is False


def test_plot_writes_chinese_e69_svg(tmp_path: Path) -> None:
    reports = {
        scope: {
            "validation_positive": 20,
            "validation_nontrivial_positive": 0,
            "nontrivial_positive_fraction": 0.0,
            "validation_componentwise_auc": 1.0,
            "raw_positive": 100,
            "raw_nontrivial_positive": 0,
            "raw_nontrivial_positive_fraction": 0.0,
        }
        for scope in ("combined", *MULTIBIT_FAMILIES)
    }
    summary = {
        "gate": {
            "decision": "innovation2_present_multibit_profile_componentwise_dominated",
            "metrics": {
                "decomposition_reports": reports,
                "feature_reports": {
                    "combined": {
                        family: {"validation_auc": value}
                        for family, value in (
                            ("static_set", 0.50),
                            ("corrupted_topology", 0.62),
                            ("true_topology", 0.65),
                            ("anf_prefix", 0.75),
                        )
                    }
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
    assert "创新2 E69" in svg
    assert "多bit linear mask" in svg
