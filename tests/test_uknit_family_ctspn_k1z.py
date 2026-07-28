from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1z import render_k1z_svg
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1z import (
    ALPHAS,
    EXPECTED_CONFIRMATION_ROWS,
    EXPECTED_GRID_ROWS,
    adjudicate,
    select_alpha,
    source_binding_checks,
)


def test_k1z_sources_are_bound_to_held_k1x() -> None:
    checks = source_binding_checks()

    assert checks
    assert all(checks.values())


def test_k1z_selection_uses_auc_then_distance_from_one() -> None:
    metrics = {alpha: {"auc": 0.5} for alpha in ALPHAS}
    metrics[2.0]["auc"] = 0.7
    metrics[0.0]["auc"] = 0.7

    assert select_alpha(metrics) == 0.0


def test_k1z_gate_requires_both_seed_confirmation() -> None:
    grid = synthetic_grid_rows()
    confirmation = synthetic_confirmation_rows()
    gate = adjudicate(
        grid_rows=grid,
        confirmation_rows=confirmation,
        source_checks={"sources": True},
    )

    assert len(grid) == EXPECTED_GRID_ROWS
    assert len(confirmation) == EXPECTED_CONFIRMATION_ROWS
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("recoverable_signal_fusion_failure")

    confirmation[4]["auc"] = 0.50
    held = adjudicate(
        grid_rows=grid,
        confirmation_rows=confirmation,
        source_checks={"sources": True},
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith(
        "inference_rescaling_insufficient_optimization_unresolved"
    )


def test_k1z_plot_separates_discovery_and_confirmation(tmp_path: Path) -> None:
    grid = synthetic_grid_rows()
    gate = adjudicate(
        grid_rows=grid,
        confirmation_rows=synthetic_confirmation_rows(),
        source_checks={"sources": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1z_svg(gate, grid, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "训练缓存只用于选择倍率" in svg
    assert "倍率发现曲线" in svg
    assert "跨密钥验证确认" in svg


def synthetic_grid_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (3, 4):
        for alpha in ALPHAS:
            rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "selected": alpha == 2.0,
                    "auc": 0.70 if alpha == 2.0 else 0.50,
                    "accuracy": 0.50,
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )
    return rows


def synthetic_confirmation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, exact_auc, source_auc in (
        (3, 0.56, 0.508),
        (4, 0.58, 0.528),
    ):
        common = {
            "seed": seed,
            "selected_alpha": 2.0,
            "source_alpha1_auc": source_auc,
            "strict_state_dict_load": True,
            "state_unchanged": True,
            "training_performed": False,
            "optimizer_steps": 0,
            "accuracy": 0.5,
        }
        rows.extend(
            (
                {**common, "condition": "exact_selected", "auc": exact_auc},
                {
                    **common,
                    "condition": "wrong_sbox_selected",
                    "auc": exact_auc - 0.02,
                },
                {**common, "condition": "exact_alpha0", "auc": 0.50},
                {**common, "condition": "exact_alpha1", "auc": source_auc},
            )
        )
    return rows
