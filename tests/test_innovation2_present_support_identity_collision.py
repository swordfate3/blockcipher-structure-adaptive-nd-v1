from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_present_support_identity_collision import (
    main as plot_main,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.present_support_identity_collision import (
    SupportIdentityAuditConfig,
    adjudicate_e48,
    collision_metrics,
    degree_only_vector,
    permute_identity_vector,
    support_identity_vector,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    possible_active_monomials,
)


def present_player() -> np.ndarray:
    return np.asarray(
        [(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)],
        dtype=np.int64,
    )


def test_possible_active_monomials_accepts_distinct_valid_player() -> None:
    active_bits = (0, 1, 4, 5, 20, 21, 40, 41)
    true_player = present_player()
    corrupted_player = topology_players(true_player[None, :], "corrupted")[0]

    true_supports = possible_active_monomials(
        active_bits, 3, player=true_player
    )
    corrupted_supports = possible_active_monomials(
        active_bits, 3, player=corrupted_player
    )

    assert np.array_equal(np.sort(corrupted_player), np.arange(64))
    assert true_supports != corrupted_supports


def test_identity_permutation_preserves_degree_but_changes_support() -> None:
    supports = {
        rounds: possible_active_monomials(tuple(range(8)), rounds)
        for rounds in (1, 2, 3)
    }
    identity = support_identity_vector(np.asarray([0, 1, 4, 5]), supports)
    permutation = np.asarray([1, 0, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    permuted = permute_identity_vector(identity, permutation)

    assert not np.array_equal(identity, permuted)
    assert np.allclose(degree_only_vector(identity), degree_only_vector(permuted))


def test_collision_metrics_counts_mixed_label_signatures() -> None:
    matrix = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0], [1, 1]], dtype=np.uint8)
    labels = np.asarray([0, 1, 0, 0, 1], dtype=np.float64)

    metrics = collision_metrics(matrix, labels)

    assert metrics == {
        "rows": 5,
        "unique_signatures": 3,
        "conflicting_signatures": 1,
        "conflicting_rows": 2,
        "conflicting_row_rate": 0.4,
    }


def test_e48_gate_rejects_identity_when_degree_only_is_stronger() -> None:
    reports = {
        "degree_only": _report(0.689),
        "exact_identity": _report(0.599),
        "sketch16": _report(0.632),
        "sketch32": _report(0.680),
        "sketch64": _report(0.671, feature_count=64),
        "permuted_sketch64": _report(0.408, feature_count=64),
        "corrupted_sketch64": _report(0.599, feature_count=64),
    }
    collisions = {
        "degree_only": _collision(0.026),
        "exact_identity": _collision(0.002),
        "sketch16": _collision(0.004),
        "sketch32": _collision(0.002),
        "sketch64": _collision(0.002),
    }
    table = {
        "rows": [{"label": index % 2} for index in range(10)],
        "degree": np.zeros((10, 27)),
        "exact": np.zeros((10, 768)),
        "sketches": {
            "sketch16": np.zeros((10, 16)),
            "sketch32": np.zeros((10, 32)),
            "sketch64": np.zeros((10, 64)),
            "permuted_sketch64": np.zeros((10, 64)),
            "corrupted_sketch64": np.zeros((10, 64)),
        },
        "true_player": present_player(),
        "corrupted_player": topology_players(
            present_player()[None, :], "corrupted"
        )[0],
        "rademacher_sha256": "a" * 64,
        "binary_projection_sha256": "b" * 64,
    }

    gate = adjudicate_e48(
        SupportIdentityAuditConfig(run_id="e48-test"),
        {"source_valid": True},
        table,
        {"reports": reports, "collisions": collisions},
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_support_identity_not_supported"
    assert np.isclose(gate["metrics"]["sketch64_minus_degree"], -0.018)
    assert gate["metrics"]["selected_route"] is None
    assert gate["next_action"]["network_smoke"] is False
    assert gate["next_action"]["action"] == (
        "stop identity-network route and audit intermediate degree-spectrum "
        "distillation before any new certificate network"
    )


def test_plot_writes_chinese_e48_svg(tmp_path: Path) -> None:
    reports = {
        "degree_only": {"validation_auc": 0.689},
        "exact_identity": {"validation_auc": 0.599},
        "sketch16": {"validation_auc": 0.632},
        "sketch32": {"validation_auc": 0.680},
        "sketch64": {"validation_auc": 0.671},
        "permuted_sketch64": {"validation_auc": 0.408},
        "corrupted_sketch64": {"validation_auc": 0.599},
    }
    collisions = {
        "degree_only": {"conflicting_row_rate": 0.026},
        "exact_identity": {"conflicting_row_rate": 0.002},
        "sketch16": {"conflicting_row_rate": 0.004},
        "sketch32": {"conflicting_row_rate": 0.002},
        "sketch64": {"conflicting_row_rate": 0.002},
    }
    summary = {
        "reports": reports,
        "collisions": collisions,
        "gate": {
            "decision": "innovation2_present_support_identity_not_supported",
            "metrics": {
                "sketch64_minus_degree": -0.018,
                "sketch64_minus_permuted": 0.263,
                "sketch64_minus_corrupted": 0.072,
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
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E48" in svg
    assert "关闭identity网络路线" in svg


def _report(auc: float, *, feature_count: int = 27) -> dict[str, object]:
    return {
        "validation_auc": auc,
        "feature_count": feature_count,
        "train_standardization_only": True,
    }


def _collision(rate: float) -> dict[str, object]:
    return {
        "rows": 10,
        "unique_signatures": 10,
        "conflicting_signatures": 0,
        "conflicting_rows": round(rate * 10),
        "conflicting_row_rate": rate,
    }
