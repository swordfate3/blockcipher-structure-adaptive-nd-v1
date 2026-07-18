from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from blockcipher_nd.cli.plot_innovation2_skinny64_sparse_profile_operator_readiness import (
    main as plot_main,
)
from blockcipher_nd.models.structure.spn.sparse_linear_profile_operator import (
    SparseLinearProfileOperator,
    SparseLinearProfileOperatorSpec,
)
from blockcipher_nd.tasks.innovation2.skinny64_r4_only_sparse_profile_operator_readiness import (
    Skinny64SparseProfileReadinessConfig,
    adjudicate_sparse_profile_readiness,
    corrupted_skinny_linear_adjacency,
    linear_graph_contract,
    make_skinny_sparse_model,
    measure_sparse_operator_contract,
    skinny_linear_adjacency,
)


def test_skinny_sparse_linear_graph_has_expected_exact_contract() -> None:
    adjacency = skinny_linear_adjacency()
    corrupted = corrupted_skinny_linear_adjacency(adjacency)
    contract = linear_graph_contract()

    assert adjacency.shape == (64, 64)
    assert contract["true_edge_count"] == 128
    assert contract["true_degree_histogram"] == {"1": 16, "2": 32, "3": 16}
    assert contract["degrees_match"] is True
    assert contract["graphs_differ"] is True
    assert contract["true_lane_preserved"] is True
    assert contract["corrupted_lane_preserved"] is True
    assert np.array_equal(adjacency.sum(axis=1), corrupted.sum(axis=1))


def test_sparse_profile_operator_keeps_4795_parameter_budget() -> None:
    config = Skinny64SparseProfileReadinessConfig(run_id="e83-test")
    models = [make_skinny_sparse_model(config, mode) for mode in ("independent", "true", "corrupted")]
    features = torch.randn(3, 64, 13)

    assert {sum(parameter.numel() for parameter in model.parameters()) for model in models} == {4_795}
    assert all(model(features).shape == (3, 64) for model in models)


def test_sparse_profile_operator_rejects_invalid_adjacency() -> None:
    with pytest.raises(ValueError, match="predecessor"):
        SparseLinearProfileOperator(
            SparseLinearProfileOperatorSpec(), torch.zeros(64, 64)
        )


def test_sparse_operator_contract_is_cell_relabel_equivariant() -> None:
    rng = np.random.default_rng(83)
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
    contract = measure_sparse_operator_contract(
        Skinny64SparseProfileReadinessConfig(run_id="e83-test"), sources
    )

    assert contract["output_shape"] == [4, 64]
    assert contract["parameter_counts_match"] is True
    assert set(contract["parameter_counts"].values()) == {4_795}
    assert contract["cell_relabel_max_abs_error"] <= 1e-6
    assert contract["true_corrupted_logit_max_abs_difference"] >= 1e-6


def _training_rows() -> list[dict[str, object]]:
    return [
        {
            "relation_mode": mode,
            "epochs_completed": 2,
            "train_auc": auc + 0.03,
            "train_accuracy": 0.72,
            "train_loss": 0.55,
            "validation_auc": auc,
            "validation_accuracy": 0.70,
            "validation_loss": 0.57,
        }
        for mode, auc in (("independent", 0.62), ("true", 0.76), ("corrupted", 0.67))
    ]


def _ridges() -> dict[str, dict[str, object]]:
    return {
        "local13": {"validation_auc": 0.65, "train_standardization_only": True},
        "true_sparse39": {
            "validation_auc": 0.74,
            "train_standardization_only": True,
        },
        "corrupted_sparse39": {
            "validation_auc": 0.66,
            "train_standardization_only": True,
        },
    }


def _contract() -> dict[str, object]:
    return {
        "output_shape": [4, 64],
        "input_dim": 13,
        "masked_loss_explicit_max_abs_error": 0.0,
        "parameter_counts": {mode: 4_795 for mode in ("independent", "true", "corrupted")},
        "parameter_counts_match": True,
        "true_corrupted_logit_max_abs_difference": 0.02,
        "cell_relabel_max_abs_error": 1e-7,
        "forbidden_named_state_absent": True,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
        "linear_graph": linear_graph_contract(),
    }


def test_e83_gate_passes_only_with_deterministic_and_neural_attribution() -> None:
    gate = adjudicate_sparse_profile_readiness(
        Skinny64SparseProfileReadinessConfig(run_id="e83-test"),
        {"source_valid": True},
        _contract(),
        _ridges(),
        {"trained_rows": _training_rows()},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_skinny64_sparse_profile_readiness_passed"


def test_e83_gate_holds_when_true_sparse_ridge_loses_to_corrupted() -> None:
    ridges = _ridges()
    ridges["corrupted_sparse39"]["validation_auc"] = 0.73
    gate = adjudicate_sparse_profile_readiness(
        Skinny64SparseProfileReadinessConfig(run_id="e83-test"),
        {"source_valid": True},
        _contract(),
        ridges,
        {"trained_rows": _training_rows()},
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_skinny64_sparse_profile_topology_not_attributed"


def test_plot_writes_clear_chinese_e83_svg(tmp_path: Path) -> None:
    gate = adjudicate_sparse_profile_readiness(
        Skinny64SparseProfileReadinessConfig(run_id="e83-test"),
        {"source_valid": True},
        _contract(),
        _ridges(),
        {"trained_rows": _training_rows()},
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps({"gate": gate}), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E83" in svg
    assert "ShiftRows+MixColumns" in svg
    assert "不是正式增益" in svg


def test_e83_training_protocol_is_frozen() -> None:
    with pytest.raises(ValueError, match="frozen"):
        Skinny64SparseProfileReadinessConfig(run_id="e83-test", epochs=3)
