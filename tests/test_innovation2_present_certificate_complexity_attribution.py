from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_present_certificate_complexity import (
    main as plot_main,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    CertificateAttributionConfig,
    adjudicate_e45,
    anf_prefix_features,
    fit_train_only_ridge,
    prefix_feature_names,
    static_feature_names,
    static_set_features,
    topology_feature_names,
    topology_reachability_features,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    possible_active_monomials,
)


def present_player() -> np.ndarray:
    return np.asarray(
        [(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)],
        dtype=np.int64,
    )


def test_feature_family_dimensions_are_frozen_and_finite() -> None:
    active = np.arange(8, dtype=np.int64)
    selected = np.asarray([0, 1, 4, 5], dtype=np.int64)
    supports = {
        rounds: possible_active_monomials(tuple(active), rounds)
        for rounds in (1, 2, 3)
    }

    static = static_set_features(active, selected)
    topology = topology_reachability_features(active, selected, present_player())
    prefix = anf_prefix_features(selected, supports)

    assert len(static) == len(static_feature_names()) == 20
    assert len(topology) == len(topology_feature_names()) == 18
    assert len(prefix) == len(prefix_feature_names()) == 39
    assert np.isfinite(static).all()
    assert np.isfinite(topology).all()
    assert np.isfinite(prefix).all()


def test_true_and_corrupted_topology_features_share_shape_but_can_differ() -> None:
    active = np.asarray([0, 1, 4, 5, 20, 21, 40, 41], dtype=np.int64)
    selected = np.asarray([3, 19, 35, 51], dtype=np.int64)
    true = present_player()
    corrupted = topology_players(true[None, :], "corrupted")[0]

    true_features = topology_reachability_features(active, selected, true)
    corrupted_features = topology_reachability_features(active, selected, corrupted)

    assert true_features.shape == corrupted_features.shape
    assert not np.array_equal(true_features, corrupted_features)


def test_ridge_weights_do_not_depend_on_validation_values() -> None:
    train_x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    train_y = np.asarray([0.0, 0.0, 1.0, 1.0])

    first = fit_train_only_ridge(train_x, train_y, np.asarray([[4.0]]), 1e-3)
    second = fit_train_only_ridge(train_x, train_y, np.asarray([[400.0]]), 1e-3)

    assert np.allclose(first["weights"], second["weights"])
    assert np.allclose(first["mean"], second["mean"])
    assert np.allclose(first["scale"], second["scale"])
    assert not np.allclose(first["validation_scores"], second["validation_scores"])


def test_adjudication_selects_mspn_when_prefix_dominates() -> None:
    reports = {
        "static_set": {"validation_auc": 0.50, "train_standardization_only": True},
        "true_topology": {
            "validation_auc": 0.648,
            "train_standardization_only": True,
        },
        "corrupted_topology": {
            "validation_auc": 0.459,
            "train_standardization_only": True,
        },
        "anf_prefix": {"validation_auc": 0.686, "train_standardization_only": True},
        "final_oracle": {"validation_auc": 1.0},
    }
    table = {
        "matrices": {
            "static_set": np.zeros((4, 2)),
            "true_topology": np.zeros((4, 2)),
            "corrupted_topology": np.zeros((4, 2)),
            "anf_prefix": np.zeros((4, 2)),
            "final_oracle": np.zeros((4, 1)),
        },
        "true_player": present_player(),
        "corrupted_player": topology_players(present_player()[None, :], "corrupted")[0],
    }

    gate = adjudicate_e45(
        CertificateAttributionConfig(run_id="e45-test"),
        {"source_valid": True},
        table,
        {"reports": reports},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_mspn_route_ready"
    assert gate["metrics"]["selected_route"] == "certificate_prefix_route"


def test_plot_writes_chinese_e45_svg(tmp_path: Path) -> None:
    summary = {
        "reports": {
            "static_set": {"validation_auc": 0.504},
            "corrupted_topology": {"validation_auc": 0.459},
            "true_topology": {"validation_auc": 0.648},
            "anf_prefix": {"validation_auc": 0.686},
            "final_oracle": {"validation_auc": 1.0},
        },
        "gate": {
            "decision": "innovation2_present_mspn_route_ready",
            "metrics": {
                "true_minus_corrupted_topology": 0.189,
                "prefix_minus_true_topology": 0.038,
                "prefix_minus_static": 0.182,
                "selected_route": "certificate_prefix_route",
                "e44_triangle_validation_auc": 0.562,
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert (
        plot_main(
            ["--summary", str(summary_path), "--output", str(output_path)]
        )
        == 0
    )
    assert "创新2 E45" in output_path.read_text(encoding="utf-8")
