from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1x import render_k1x_svg
from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1w import (
    read_tasks,
    task_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1x import (
    EXPECTED_GRADIENT_ROWS,
    EXPECTED_INFERENCE_ROWS,
    PLAN,
    adjudicate,
    audit_gradient_seed,
    source_binding_checks,
)


def test_k1x_source_panel_is_fully_bound() -> None:
    checks = source_binding_checks()

    assert checks
    assert all(checks.values())


def test_k1x_gradient_proof_recovers_exact_sixteen_fold_ratio(
    tmp_path: Path,
) -> None:
    task = task_map(read_tasks(PLAN))[("uknit64", 3, "compact_exact")]
    generator = np.random.default_rng(20260728)
    cache = DiskDifferentialDataset(
        features=generator.integers(0, 2, size=(64, 512), dtype=np.uint8),
        labels=np.tile(np.array([0, 1], dtype=np.uint8), 32),
        metadata={},
        cache_dir=tmp_path,
    )

    result = audit_gradient_seed(seed=3, task=task, dataset=cache)

    assert result["optimizer_steps"] == 0
    assert result["training_performed"] is False
    assert result["state_restored_exact"] is True
    assert result["slot_gradient_relative_error"] <= 1e-9
    assert result["folded_gradient_relative_error"] <= 1e-9
    assert 15.999 <= result["folded_effective_update_ratio"] <= 16.001


def test_k1x_gate_authorizes_only_complete_two_seed_mechanism() -> None:
    inference = synthetic_inference_rows()
    gradients = synthetic_gradient_rows()
    gate = adjudicate(
        inference_rows=inference,
        gradient_rows=gradients,
        source_checks={"sources": True},
    )

    assert len(inference) == EXPECTED_INFERENCE_ROWS
    assert len(gradients) == EXPECTED_GRADIENT_ROWS
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("16x_optimization_geometry_supported")
    assert all(gate["research_checks"].values())

    gradients[1]["folded_effective_update_ratio"] = 8.0
    held = adjudicate(
        inference_rows=inference,
        gradient_rows=gradients,
        source_checks={"sources": True},
    )
    assert held["status"] == "hold"
    assert held["remote_scale"] == "no"


def test_k1x_plot_names_zero_training_mechanism_and_gates(tmp_path: Path) -> None:
    gate = adjudicate(
        inference_rows=synthetic_inference_rows(),
        gradient_rows=synthetic_gradient_rows(),
        source_checks={"sources": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1x_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "只做推理和梯度读取" in svg
    assert "折叠有效更新比" in svg
    assert "关闭直方图分支" in svg


def synthetic_inference_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (3, 4):
        for checkpoint in ("k1w_compact", "k1t_folded"):
            exact = 0.51 if checkpoint == "k1w_compact" else 0.58
            for condition, auc in (
                ("exact", exact),
                ("zero_histogram_gate", exact - 0.005),
                ("wrong_sbox_same_checkpoint", exact - 0.005),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "checkpoint_kind": checkpoint,
                        "condition": condition,
                        "auc": auc,
                        "accuracy": 0.5,
                        "source_auc": exact,
                        "histogram_effective_gate": 0.05,
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
    return rows


def synthetic_gradient_rows() -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "old_compact_max_abs_logit_error": 1e-12,
            "old_compact_abs_loss_error": 1e-14,
            "slot_gradient_relative_error": 1e-12,
            "folded_gradient_relative_error": 1e-12,
            "folded_effective_update_ratio": 16.0,
            "state_restored_exact": True,
            "training_performed": False,
            "optimizer_steps": 0,
        }
        for seed in (3, 4)
    ]
