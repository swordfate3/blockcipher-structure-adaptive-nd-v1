from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_cgpr_attribution import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_cgpr_attribution import (
    CgprAttributionConfig,
    adjudicate_e51,
)


def config() -> CgprAttributionConfig:
    return CgprAttributionConfig(run_id="e51-test")


def test_e51_config_is_frozen_to_30_epochs() -> None:
    value = config()

    assert value.epochs == 30
    assert value.batch_size == 32
    assert value.residual_bound == 0.25


def test_e51_gate_passes_all_candidate_and_attribution_margins() -> None:
    gate = adjudicate_e51(
        config(),
        {"source_valid": True},
        valid_contract(),
        matrix_fixture(prefix_auc=0.70, true_auc=0.74, corrupted_auc=0.70),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_cgpr_topology_attributed"
    assert gate["next_action"]["seed1"] is True


def test_e51_gate_holds_when_candidate_does_not_beat_ridge() -> None:
    gate = adjudicate_e51(
        config(),
        {"source_valid": True},
        valid_contract(),
        matrix_fixture(prefix_auc=0.69, true_auc=0.69, corrupted_auc=0.66),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_cgpr_candidate_not_ready"


def test_e51_gate_separates_prefix_capacity_from_pair_residual() -> None:
    gate = adjudicate_e51(
        config(),
        {"source_valid": True},
        valid_contract(),
        matrix_fixture(prefix_auc=0.71, true_auc=0.72, corrupted_auc=0.68),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_present_cgpr_pair_residual_not_attributed"
    )


def test_e51_gate_separates_true_from_corrupted_topology() -> None:
    gate = adjudicate_e51(
        config(),
        {"source_valid": True},
        valid_contract(),
        matrix_fixture(prefix_auc=0.68, true_auc=0.72, corrupted_auc=0.71),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_cgpr_topology_not_attributed"


def test_plot_writes_chinese_e51_svg(tmp_path: Path) -> None:
    matrix = matrix_fixture(prefix_auc=0.70, true_auc=0.74, corrupted_auc=0.70)
    gate = adjudicate_e51(
        config(), {"source_valid": True}, valid_contract(), matrix
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(
        json.dumps({"rows": matrix["rows"], "gate": gate}), encoding="utf-8"
    )

    assert (
        plot_main(["--summary", str(summary_path), "--output", str(output_path)])
        == 0
    )
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E51" in svg
    assert "允许本地seed1" in svg


def valid_contract() -> dict[str, object]:
    return {
        "prefix_shape": [1036, 39],
        "ridge_validation_auc": 0.6860815857512209,
        "train_standardization_only": True,
        "parameter_counts": {
            "prefix": 10659,
            "pair_true": 10725,
            "pair_corrupted": 10725,
        },
        "parameter_relative_spread": 0.0062,
        "zero_residual_prefix_max_abs_error": 0.0,
        "zero_residual_true_max_abs_error": 0.0,
        "zero_residual_corrupted_max_abs_error": 0.0,
        "true_corrupted_pair_embedding_max_abs_difference": 0.05,
        "ridge_buffers_require_grad_false": True,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
        "forbidden_buffers_absent": True,
    }


def matrix_fixture(
    *, prefix_auc: float, true_auc: float, corrupted_auc: float
) -> dict[str, object]:
    rows = [
        {
            "row_id": "e45_anf_prefix_ridge_anchor",
            "residual_mode": "off",
            "topology_mode": "none",
            "train_auc": 0.777,
            "validation_auc": 0.6860815857512209,
            "train_loss": 0.65,
            "validation_loss": 0.67,
            "ridge_weight_max_delta": 0.0,
            "training_performed": False,
        },
        trained_row("cgpr_prefix_only_seed0", "prefix", "true", prefix_auc),
        trained_row("cgpr_pair_true_seed0", "pair", "true", true_auc),
        trained_row(
            "cgpr_pair_corrupted_seed0", "pair", "corrupted", corrupted_auc
        ),
    ]
    history = [
        {"row_id": row["row_id"], "epoch": epoch}
        for row in rows
        if row["training_performed"]
        for epoch in range(1, 31)
    ]
    return {"rows": rows, "history": history}


def trained_row(
    row_id: str, residual_mode: str, topology_mode: str, auc: float
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "residual_mode": residual_mode,
        "topology_mode": topology_mode,
        "train_auc": min(0.99, auc + 0.10),
        "validation_auc": auc,
        "train_loss": 0.60,
        "validation_loss": 0.66,
        "ridge_weight_max_delta": 0.0,
        "training_performed": True,
    }
