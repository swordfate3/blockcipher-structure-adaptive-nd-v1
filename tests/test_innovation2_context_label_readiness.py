from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli import audit_innovation2_context_label_readiness as cli
from blockcipher_nd.tasks.innovation2.integral_context_label_readiness import (
    ContextLabelReadinessConfig,
    adjudicate_context_label_readiness,
    ridge_loocv_predictions,
    run_context_label_readiness_audit,
)


def test_ridge_loocv_matches_explicit_refits() -> None:
    design = np.asarray(
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
        dtype=np.float64,
    )
    labels = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    actual = ridge_loocv_predictions(design, labels, alpha=1.0)
    expected = []
    for held_out in range(len(labels)):
        keep = np.arange(len(labels)) != held_out
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design[keep].T @ design[keep] + penalty,
            design[keep].T @ labels[keep],
        )
        expected.append(float(design[held_out] @ coefficients))

    np.testing.assert_allclose(
        actual,
        np.clip(expected, 0.0, 1.0),
        atol=1e-12,
    )


def test_context_label_gate_holds_for_bitwise_shortcut() -> None:
    config = ContextLabelReadinessConfig(run_id="test")
    baselines = [
        {"baseline": "context_identity_marginal", "accuracy": 0.75, "auc": 0.6},
        {"baseline": "context_weight_marginal", "accuracy": 0.75, "auc": 0.6},
        {"baseline": "mask_identity_marginal", "accuracy": 0.90, "auc": 0.9},
        {"baseline": "mask_weight_marginal", "accuracy": 0.75, "auc": 0.6},
        {"baseline": "context_mask_additive", "accuracy": 0.90, "auc": 0.9},
        {
            "baseline": "context_mask_bitwise_linear",
            "accuracy": 0.96,
            "auc": 0.9,
        },
    ]

    gate = adjudicate_context_label_readiness(
        config,
        baselines,
        {"ok": True},
        positive_rate=0.3,
        flipping_masks=5,
        distinct_context_label_signatures=5,
    )

    assert gate["status"] == "hold"
    assert gate["shortcut_checks"][
        "context_mask_bitwise_linear_accuracy_below_0p95"
    ] is False


def test_context_label_gate_holds_for_high_auc_shortcut() -> None:
    config = ContextLabelReadinessConfig(run_id="test")
    baselines = [
        {"baseline": "context_identity_marginal", "accuracy": 0.75, "auc": 0.6},
        {"baseline": "context_weight_marginal", "accuracy": 0.75, "auc": 0.6},
        {"baseline": "mask_identity_marginal", "accuracy": 0.94, "auc": 0.98},
        {"baseline": "mask_weight_marginal", "accuracy": 0.75, "auc": 0.7},
        {"baseline": "context_mask_additive", "accuracy": 0.94, "auc": 0.97},
        {
            "baseline": "context_mask_bitwise_linear",
            "accuracy": 0.94,
            "auc": 0.97,
        },
    ]

    gate = adjudicate_context_label_readiness(
        config,
        baselines,
        {"ok": True},
        positive_rate=0.3,
        flipping_masks=5,
        distinct_context_label_signatures=5,
    )

    assert gate["status"] == "hold"
    assert gate["shortcut_checks"][
        "mask_identity_marginal_auc_below_0p95"
    ] is False


def test_context_label_runner_builds_frozen_grid() -> None:
    common = (1, 2, 4, 8)
    extras = (16, 32, 64, 128, 256)
    basis_rows = []
    for context_id in range(16):
        basis = common if context_id == 0 else common + (extras[(context_id - 1) % 5],)
        for basis_index, vector in enumerate(basis):
            basis_rows.append(
                {
                    "context_id": str(context_id),
                    "fixed_plaintext": f"0x{context_id:016X}",
                    "basis_index": str(basis_index),
                    "vector_hex": f"0x{vector:016X}",
                    "vector_weight": str(vector.bit_count()),
                }
            )
    result = run_context_label_readiness_audit(
        ContextLabelReadinessConfig(run_id="runner"),
        source_gate={
            "run_id": "source",
            "status": "pass",
            "decision": "innovation2_inactive_context_kernel_diversity_ready",
        },
        source_metadata={
            "task": "innovation2_present_r7_inactive_context_kernel_diversity",
            "training_performed": False,
            "contexts": [f"0x{context_id:012X}" for context_id in range(16)],
        },
        source_basis_rows=basis_rows,
    )

    assert result["metadata"]["candidate_masks"] == 18
    assert result["metadata"]["label_rows"] == 288
    assert len(result["rows"]) == 9
    assert result["gate"]["readiness_checks"][
        "source_basis_union_has_nine_masks"
    ] is True


def test_context_label_cli_writes_expected_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "gate.json").write_text("{}")
    (source_root / "metadata.json").write_text("{}")
    (source_root / "kernel_basis.csv").write_text("context_id\n0\n")
    label_rows = [
        {
            "context_id": context_id,
            "balanced_label": int(context_id % 3 == 0),
        }
        for context_id in range(16)
    ]
    baselines = [
        {
            "run_id": "test",
            "baseline": "global_rate",
            "baseline_label": "全局正例率",
            "accuracy": 0.6,
            "brier": 0.2,
            "auc": 0.5,
            "rows": 288,
            "is_oracle": False,
        }
    ]
    gate = {
        "run_id": "test",
        "status": "pass",
        "decision": "innovation2_context_label_interaction_ready",
        "positive_rate": 0.3,
        "flipping_masks": 5,
        "distinct_context_label_signatures": 5,
        "baseline_accuracies": {"global_rate": 0.6},
        "baseline_aucs": {"global_rate": 0.5},
        "next_action": {"training": False, "remote_scale": False},
    }
    monkeypatch.setattr(
        cli,
        "run_context_label_readiness_audit",
        lambda config, **kwargs: {
            "rows": baselines,
            "label_rows": label_rows,
            "gate": gate,
            "metadata": {"run_id": "test"},
        },
    )
    output_root = tmp_path / "output"

    assert (
        cli.main(
            [
                "--run-id",
                "test",
                "--source-root",
                str(source_root),
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )

    assert {
        "results.jsonl",
        "labels.csv",
        "progress.jsonl",
        "gate.json",
        "metadata.json",
        "curves.svg",
    }.issubset(path.name for path in output_root.iterdir())
    assert json.loads((output_root / "gate.json").read_text())["status"] == "pass"
    svg = (output_root / "curves.svg").read_text(encoding="utf-8")
    assert "context-mask 输出平衡标签" in svg
    assert "不是积分/随机二分类" in svg
