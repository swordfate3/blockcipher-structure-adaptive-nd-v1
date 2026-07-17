from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.audit_innovation2_speck_position_labels import main
from blockcipher_nd.cli.plot_innovation2_speck_position_labels import (
    render_position_label_svg,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_position_labels import (
    EXPECTED_SOURCE_DECISION,
    EXPECTED_SOURCE_TASK,
    TARGET_MASK,
    SpeckPositionLabelConfig,
    adjudicate_position_label_audit,
    bitwise_design,
    bounded_flipping_masks,
    build_label_grid,
    evaluate_shortcuts,
    full_position_kernels,
    mask_in_kernel,
    run_position_label_audit,
)


def _source_rows(*, full_positions: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start in tuple(range(15)) + tuple(range(16, 31)):
        row: dict[str, object] = {
            "position_start": start,
            "fixed_bits": f"{start},{start + 1}",
            "evidence_keys": 64 if len(rows) < full_positions else 0,
        }
        if len(rows) < full_positions:
            row.update(
                {
                    "joint_nullity": 1,
                    "joint_kernel_basis": f"0x{TARGET_MASK:08X}",
                }
            )
        rows.append(row)
    return rows


def _source_gate(*, valid: bool = True) -> dict[str, object]:
    return {
        "run_id": "e27-source",
        "status": "pass" if valid else "hold",
        "decision": EXPECTED_SOURCE_DECISION if valid else "other",
    }


def _source_metadata() -> dict[str, object]:
    return {"task": EXPECTED_SOURCE_TASK, "training_performed": False}


def _metric_rows(value: float = 0.60) -> list[dict[str, object]]:
    keys = (
        "position_identity_marginal",
        "mask_identity_marginal",
        "position_mask_additive",
        "position_mask_bitwise_linear",
        "position_disjoint_bitwise",
        "mask_disjoint_bitwise",
        "dual_disjoint_bitwise",
        "label_shuffle_dual_disjoint",
    )
    return [
        {
            "baseline": key,
            "accuracy": 0.5,
            "brier": 0.25,
            "auc": value,
            "directional_auc": value,
        }
        for key in keys
    ]


def test_bounded_candidates_use_basis_and_pairwise_xor_only() -> None:
    kernels = {
        0: (0x01, 0x02, 0x04),
        1: (0x01, 0x08),
        2: (),
    }
    assert bounded_flipping_masks(kernels) == (
        0x01,
        0x02,
        0x03,
        0x04,
        0x05,
        0x06,
        0x08,
        0x09,
    )
    assert mask_in_kernel(0x03, kernels[0]) is True
    assert mask_in_kernel(0x08, kernels[0]) is False


def test_full_position_kernel_parser_and_label_grid() -> None:
    rows = _source_rows(full_positions=2)
    rows[0]["joint_nullity"] = 2
    rows[0]["joint_kernel_basis"] = "0x00000001;0x00000002"
    rows[1]["joint_kernel_basis"] = "0x00000002"
    kernels = full_position_kernels(rows)
    assert kernels == {0: (1, 2), 1: (2,)}
    config = SpeckPositionLabelConfig(run_id="e28-grid")
    grid = build_label_grid(config, kernels, (1, 2, 3))
    assert len(grid) == 6
    assert [row["balanced_label"] for row in grid] == [1, 1, 1, 0, 1, 0]


def test_same_kernel_everywhere_is_too_narrow_not_protocol_failure() -> None:
    result = run_position_label_audit(
        SpeckPositionLabelConfig(run_id="e28-narrow"),
        source_gate=_source_gate(),
        source_metadata=_source_metadata(),
        source_rows=_source_rows(),
    )
    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_speck_position_label_grid_too_narrow"
    )
    assert result["gate"]["full_evidence_positions"] == 8
    assert result["gate"]["flipping_masks"] == 0
    assert result["rows"] == []


def test_invalid_source_is_distinct_from_narrow_label_grid() -> None:
    result = run_position_label_audit(
        SpeckPositionLabelConfig(run_id="e28-invalid"),
        source_gate=_source_gate(valid=False),
        source_metadata=_source_metadata(),
        source_rows=_source_rows(),
    )
    assert result["gate"]["status"] == "fail"
    assert result["gate"]["decision"] == (
        "innovation2_speck_position_label_protocol_invalid"
    )


def test_adjudication_advance_and_shortcut_branches() -> None:
    config = SpeckPositionLabelConfig(run_id="e28-gate")
    checks = {"check": True}
    advance = adjudicate_position_label_audit(
        config,
        _metric_rows(),
        source_checks=checks,
        width_checks=checks,
        positive_rate=0.4,
        full_positions=8,
        flipping_masks=8,
        distinct_signatures=4,
    )
    assert advance["decision"] == "innovation2_speck_position_label_grid_advance"
    shortcut_rows = _metric_rows()
    next(row for row in shortcut_rows if row["baseline"] == "mask_disjoint_bitwise")[
        "directional_auc"
    ] = 0.80
    shortcut = adjudicate_position_label_audit(
        config,
        shortcut_rows,
        source_checks=checks,
        width_checks=checks,
        positive_rate=0.4,
        full_positions=8,
        flipping_masks=8,
        distinct_signatures=4,
    )
    assert shortcut["decision"] == (
        "innovation2_speck_position_label_grid_shortcut_dominated"
    )


def test_bitwise_design_is_32_plus_32_bits() -> None:
    design = bitwise_design([0x00000003], [0x80000001])
    assert design.shape == (1, 65)
    assert design[0, 0] == 1
    assert design[0, 1] == 1
    assert design[0, 2] == 1
    assert design[0, 33] == 1
    assert design[0, 64] == 1


def test_shortcut_evaluation_uses_deterministic_four_folds() -> None:
    config = SpeckPositionLabelConfig(run_id="e28-shortcuts")
    rows = []
    for position in range(8):
        for mask in range(8):
            rows.append(
                {
                    "balanced_label": (position + mask) % 2,
                    "position_start": position,
                    "position_weight": 2,
                    "position_mask_value": (1 << position) | (1 << (position + 1)),
                    "mask_index": mask,
                    "mask_hex": f"0x{(1 << mask):08X}",
                    "mask_weight": 1,
                    "mask_value": 1 << mask,
                }
            )
    metrics, diagnostics, position_folds, mask_folds = evaluate_shortcuts(config, rows)
    assert len(metrics) == 11
    assert position_folds == {value: value % 4 for value in range(8)}
    assert mask_folds == {value: value % 4 for value in range(8)}
    assert diagnostics["position_disjoint_bitwise"]["all_rows_assigned_once"]
    assert diagnostics["mask_disjoint_bitwise"]["all_rows_assigned_once"]
    assert diagnostics["dual_disjoint_bitwise"]["all_rows_assigned_once"]


def test_plot_handles_too_narrow_and_metric_results(tmp_path: Path) -> None:
    narrow_path = tmp_path / "narrow.svg"
    narrow_gate = {
        "decision": "innovation2_speck_position_label_grid_too_narrow",
        "full_evidence_positions": 8,
        "flipping_masks": 0,
        "distinct_position_signatures": 1,
        "positive_rate": 0.0,
    }
    render_position_label_svg([], narrow_gate, narrow_path)
    assert "标签宽度门未通过" in narrow_path.read_text(encoding="utf-8")

    metric_path = tmp_path / "metrics.svg"
    render_position_label_svg(_metric_rows(), narrow_gate, metric_path)
    text = metric_path.read_text(encoding="utf-8")
    assert "方向无关AUC" in text
    assert "mask组外位模式" in text


def test_cli_writes_too_narrow_gate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "gate.local.json").write_text(
        json.dumps(_source_gate()), encoding="utf-8"
    )
    (source / "metadata.json").write_text(
        json.dumps(_source_metadata()), encoding="utf-8"
    )
    (source / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in _source_rows()),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--run-id",
                "e28-cli",
                "--source-root",
                str(source),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    assert gate["decision"] == "innovation2_speck_position_label_grid_too_narrow"
    assert (output / "label_rows.csv").is_file()
