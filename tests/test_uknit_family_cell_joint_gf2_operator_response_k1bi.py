from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.cli.plot_uknit_family_cell_joint_gf2_operator_response_k1bi import (
    render_k1bi_svg,
)
from blockcipher_nd.models.structure.spn.cell_joint_gf2_operator_response import (
    cell_joint_response_feature_dim,
    extract_cell_joint_gf2_operator_features,
    response_bits_to_cell_values,
)
from blockcipher_nd.tasks.innovation1.uknit_family_cell_joint_gf2_operator_response_k1bi import (
    adjudicate_k1bi,
    evaluate_k1bi,
    load_and_validate_config,
    load_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_exact_gf2_operator_response_k1bh import (
    EXPECTED_FEATURE_ROWS,
    EXPECTED_RESULT_ROWS,
    EXPECTED_SCORER_ROWS,
)


@pytest.fixture(scope="module")
def k1bi_authority() -> tuple[object, ...]:
    config = load_and_validate_config()
    authority = load_authority(config)
    return config, *authority


def test_k1bi_reconstructs_native_cell_values_from_runtime_geometry(
    k1bi_authority: tuple[object, ...],
) -> None:
    structures = k1bi_authority[3]
    structure = structures["uknit64"]
    rng = np.random.default_rng(20260729)
    expected = rng.integers(
        0,
        16,
        size=(2, 3, structure.cells, 12),
        dtype=np.uint8,
    )
    response = np.zeros((2, 3, structure.block_bits, 12), dtype=np.uint8)
    for bit in range(structure.block_bits):
        cell = int(structure.cell_membership[bit])
        role = int(structure.bit_role[bit])
        response[..., bit, :] = (expected[..., cell, :] >> role) & 1

    observed = response_bits_to_cell_values(response, structure)

    assert np.array_equal(observed, expected)


def test_k1bi_features_are_position_preserved_four_pair_histograms(
    k1bi_authority: tuple[object, ...],
) -> None:
    datasets = k1bi_authority[2]
    structures = k1bi_authority[3]
    structure = structures["uknit64"]
    flat = datasets[("uknit64", 3, "same_key_fresh")].features[:7]

    features = extract_cell_joint_gf2_operator_features(flat, structure)
    histograms = features.reshape(7, structure.cells, 12, 16)

    assert features.shape == (7, cell_joint_response_feature_dim(structure.cells))
    assert np.allclose(histograms.sum(axis=-1), 1.0)
    assert np.allclose(features * 4.0, np.rint(features * 4.0))
    assert np.all((features >= 0.0) & (features <= 1.0))


def test_k1bi_real_authority_and_gate_are_protocol_complete(
    k1bi_authority: tuple[object, ...],
) -> None:
    (
        config,
        dataset_rows,
        datasets,
        structures,
        corrupted,
        cross,
        source_gate,
        source_checks,
    ) = k1bi_authority
    assert all(source_checks.values()), source_checks

    features, scorers, results = evaluate_k1bi(
        config=config,
        dataset_rows=dataset_rows,
        datasets=datasets,
        structures=structures,
        corrupted_structures=corrupted,
        cross_operators=cross,
    )
    gate = adjudicate_k1bi(
        config=config,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_gate=source_gate,
        source_checks=source_checks,
    )

    assert len(features) == EXPECTED_FEATURE_ROWS
    assert len(scorers) == EXPECTED_SCORER_ROWS
    assert len(results) == EXPECTED_RESULT_ROWS
    assert all(gate["protocol_checks"].values()), gate["failed_protocol_checks"]
    assert gate["status"] in {"pass", "hold"}

    passing_results = deepcopy(results)
    source_panels = {
        (int(panel["replica"]), str(panel["cipher_key"]), str(panel["split"])): panel
        for panel in source_gate["panels"]
    }
    correct_auc: dict[tuple[int, str, str], float] = {}
    for row in passing_results:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        if row["cipher_key"] in {"midori64", "dialga128"}:
            correct_auc[key] = float(source_panels[key]["correct_auc"])
        else:
            correct_auc[key] = 0.70
    for row in passing_results:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        if row["condition"] == "correct_operator":
            row["auc"] = correct_auc[key]
        elif row["condition"] == "label_shuffled_correct_operator":
            row["auc"] = 0.50
        else:
            row["auc"] = correct_auc[key] - 0.05

    passed = adjudicate_k1bi(
        config=config,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=passing_results,
        source_gate=source_gate,
        source_checks=source_checks,
    )
    assert passed["status"] == "pass"
    assert passed["decision"].endswith("cell_joint_topology_signal_supported")
    assert all(passed["research_checks"].values())

    reversed_shuffle = deepcopy(passing_results)
    row = next(
        item
        for item in reversed_shuffle
        if item["replica"] == 0
        and item["cipher_key"] == "dialga128"
        and item["split"] == "same_key_fresh"
        and item["condition"] == "label_shuffled_correct_operator"
    )
    row["auc"] = 0.30
    held = adjudicate_k1bi(
        config=config,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=reversed_shuffle,
        source_gate=source_gate,
        source_checks=source_checks,
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("shuffle_attribution_not_supported")


def test_k1bi_plot_names_cell_joint_variable(tmp_path: Path) -> None:
    panels = []
    for replica in (0, 1):
        for cipher in ("uknit64", "midori64", "dialga128"):
            for split in ("same_key_fresh", "cross_key_validation"):
                panels.append(
                    {
                        "replica": replica,
                        "cipher_key": cipher,
                        "split": split,
                        "correct_auc": 0.65,
                        "same_summary_wrong_auc": 0.60,
                        "cross_cipher_wrong_auc": 0.59,
                        "identity_auc": 0.60,
                        "label_shuffle_auc": 0.50,
                        "correct_minus_same_summary_wrong": 0.05,
                        "correct_minus_cross_cipher_wrong": 0.06,
                        "correct_minus_identity": 0.05,
                        "correct_minus_label_shuffle": 0.15,
                    }
                )
    gate = {
        "status": "pass",
        "decision": (
            "innovation1_uknit_family_k1bi_cell_joint_topology_signal_supported"
        ),
        "panels": panels,
    }
    output = tmp_path / "curves.svg"

    report = render_k1bi_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert report["experiment"] == "K1-BI"
    assert "保留 cell 内四比特联合取值" in svg
    assert "运行时cell重建0–15取值直方图" in svg
    assert "唯一变化是独立bit均值改为运行时cell的16类联合响应" in svg
    assert "双向控制：标签打乱 AUC 是否仍显著偏离随机" in svg
