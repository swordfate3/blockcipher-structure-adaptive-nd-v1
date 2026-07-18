from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from blockcipher_nd.cli.plot_innovation2_skinny64_true_ridge_sparse_residual import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.skinny64_true_ridge_sparse_residual import (
    E83_TRUE_RIDGE_AUC,
    Skinny64TrueRidgeResidualConfig,
    adjudicate_true_ridge_residual,
    make_true_ridge_residual_model,
    measure_true_ridge_residual_contract,
)


def _ridge() -> dict[str, object]:
    weights = np.zeros(40, dtype=np.float64)
    weights[0] = 0.2
    weights[1:] = np.linspace(-0.1, 0.1, 39)
    return {
        "mean": np.zeros(39, dtype=np.float64),
        "scale": np.ones(39, dtype=np.float64),
        "weights": weights,
        "train_auc": 0.90,
        "validation_auc": E83_TRUE_RIDGE_AUC,
        "train_standardization_only": True,
    }


def test_true_ridge_residual_zero_initialization_and_bound() -> None:
    config = Skinny64TrueRidgeResidualConfig(run_id="e84-test")
    model = make_true_ridge_residual_model(config, _ridge(), "true", dropout=0.0)
    features = torch.randn(3, 64, 13)

    assert torch.equal(model(features), model.base_score(features))
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_795
    with torch.no_grad():
        model.output_head.bias.fill_(100.0)
        maximum_residual = float(torch.max(torch.abs(model.residual_score(features))))
    assert maximum_residual <= 0.25


def test_true_ridge_residual_contract_is_equivariant_and_frozen() -> None:
    rng = np.random.default_rng(84)
    sources = {
        "prefix_features": rng.normal(size=(4, 64, 13)).astype(np.float64),
        "profile_targets": rng.integers(0, 2, size=(4, 64), dtype=np.int8),
        "profile_observed": np.ones((4, 64), dtype=np.bool_),
        "matched_rows": [
            {
                "structure_index": structure,
                "output_bit": output,
                "label": 0,
                "split": "train",
            }
            for structure in range(4)
            for output in range(64)
        ],
    }
    contract = measure_true_ridge_residual_contract(
        Skinny64TrueRidgeResidualConfig(run_id="e84-test"),
        sources,
        _ridge(),
    )

    assert max(contract["zero_residual_max_abs_errors"].values()) == 0.0
    assert contract["ridge_buffers_require_grad_false"] is True
    assert set(contract["parameter_counts"].values()) == {4_795}
    assert contract["true_corrupted_embedding_max_abs_difference"] >= 1e-5
    assert contract["cell_relabel_max_abs_error"] <= 1e-6


def _matrix(true_auc: float = 0.89) -> dict[str, object]:
    rows = [
        {
            "relation_mode": "ridge_only",
            "training_performed": False,
            "validation_auc": E83_TRUE_RIDGE_AUC,
        }
    ]
    for mode, auc in (("independent", 0.84), ("true", true_auc), ("corrupted", 0.85)):
        rows.append(
            {
                "relation_mode": mode,
                "training_performed": True,
                "epochs_completed": 2,
                "train_auc": auc + 0.03,
                "train_accuracy": 0.75,
                "train_loss": 0.50,
                "validation_auc": auc,
                "validation_accuracy": 0.72,
                "validation_loss": 0.54,
                "ridge_weight_max_delta": 0.0,
            }
        )
    return {"rows": rows, "history": []}


def _contract() -> dict[str, object]:
    return {
        "ridge_validation_auc": E83_TRUE_RIDGE_AUC,
        "train_standardization_only": True,
        "zero_residual_max_abs_errors": {mode: 0.0 for mode in ("independent", "true", "corrupted")},
        "ridge_buffers_require_grad_false": True,
        "parameter_counts": {mode: 4_795 for mode in ("independent", "true", "corrupted")},
        "parameter_counts_match": True,
        "true_corrupted_embedding_max_abs_difference": 0.1,
        "cell_relabel_max_abs_error": 1e-7,
        "residual_bound": 0.25,
        "forbidden_buffers_absent": True,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
    }


def test_e84_gate_passes_only_when_true_residual_beats_strong_ridge() -> None:
    gate = adjudicate_true_ridge_residual(
        Skinny64TrueRidgeResidualConfig(run_id="e84-test"),
        {"source_valid": True},
        _contract(),
        _matrix(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_skinny64_true_ridge_residual_readiness_passed"


def test_e84_gate_holds_when_true_residual_only_matches_ridge() -> None:
    gate = adjudicate_true_ridge_residual(
        Skinny64TrueRidgeResidualConfig(run_id="e84-test"),
        {"source_valid": True},
        _contract(),
        _matrix(true_auc=E83_TRUE_RIDGE_AUC),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_skinny64_true_ridge_residual_not_ready"


def test_plot_writes_clear_chinese_e84_svg(tmp_path: Path) -> None:
    gate = adjudicate_true_ridge_residual(
        Skinny64TrueRidgeResidualConfig(run_id="e84-test"),
        {"source_valid": True},
        _contract(),
        _matrix(),
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps({"gate": gate}), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E84" in svg
    assert "epoch0严格等于ridge" in svg
    assert "不是正式增益" in svg


def test_e84_protocol_is_frozen() -> None:
    with pytest.raises(ValueError, match="frozen"):
        Skinny64TrueRidgeResidualConfig(run_id="e84-test", residual_bound=0.5)
