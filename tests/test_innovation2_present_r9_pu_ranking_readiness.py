from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_r9_pu_ranking_readiness import (
    render_pu_readiness,
)
from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import (
    ATM_EXPECTED_RESULT_FILES,
)
from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import (
    PuRankingReadinessConfig,
    adjudicate_pu_readiness,
    build_ranking_audit,
)


def _relation(input_bit: int, output_bit: int) -> frozenset[tuple[int, int]]:
    left_a = (1 << input_bit) | (1 << ((input_bit + 3) % 64))
    left_b = (1 << input_bit) | (1 << ((input_bit + 5) % 64))
    right_a = 1 << output_bit
    right_b = 1 << ((output_bit + 11) % 64)
    return frozenset(
        {
            (left_a, right_a),
            (left_a, right_b),
            (left_b, right_a),
            (left_b, right_b),
        }
    )


def _groups() -> dict[str, set[frozenset[tuple[int, int]]]]:
    common = frozenset({(3, 1)})
    return {
        name: {common, _relation(index * 7, index * 5 + 1)}
        for index, name in enumerate(ATM_EXPECTED_RESULT_FILES)
    }


def test_pu_candidate_pools_are_group_disjoint_and_marginal_matched() -> None:
    audit = build_ranking_audit(_groups(), minimum_unlabeled_per_positive=31)

    assert audit["metrics"]["total_heldout_positives"] == 8
    assert audit["metrics"]["groups_with_any_heldout_positive"] == 8
    assert audit["metrics"]["unlabeled_pools_overlapping_known_positives"] == 0
    assert audit["metrics"]["candidate_marginal_mismatches"] == 0
    assert audit["metrics"]["nondeterministic_candidate_pools"] == 0
    assert audit["metrics"]["minimum_unlabeled_candidates"] >= 31


def test_e98_holds_when_group_width_and_rank_contract_are_not_ready() -> None:
    ranking = build_ranking_audit(_groups(), minimum_unlabeled_per_positive=31)
    source = {
        "checks": {"frozen_source": True},
        "atm": {
            "metrics": {
                "unique_serialized_basis_elements": 470,
                "union_gf2_rank": 468,
                "published_dimension": 470,
            }
        },
    }

    gate = adjudicate_pu_readiness(
        PuRankingReadinessConfig(),
        source_audit=source,
        ranking_audit=ranking,
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_r9_pu_ranking_benchmark_not_ready"
    assert not gate["readiness_checks"][
        "published_dimension_matches_recomputed_union_rank"
    ]
    assert not gate["readiness_checks"]["at_least_64_total_heldout_positives"]
    assert gate["next_action"]["e99_open"] is False
    assert gate["next_action"]["remote_scale"] is False


def test_e98_plot_explains_positive_unlabeled_scope(tmp_path: Path) -> None:
    summary = {
        "folds": [
            {"heldout_file": name, "heldout_relations": count}
            for name, count in zip(ATM_EXPECTED_RESULT_FILES, (0, 3, 18, 0, 0, 3, 0, 0), strict=True)
        ],
        "ranking_baselines": [
            {
                "baseline": name,
                "recall_at_5": 0.1,
                "mean_reciprocal_rank": 0.08,
            }
            for name in (
                "deterministic_hash_random",
                "file_id",
                "relation_size",
                "exponent_weight",
                "exact_training_frequency",
                "training_coordinate_frequency",
                "training_support_overlap",
                "absolute_bit_position",
            )
        ],
        "gate": {
            "decision": "innovation2_present_r9_pu_ranking_benchmark_not_ready",
            "metrics": {
                "union_gf2_rank": 468,
                "published_dimension": 470,
                "total_heldout_positives": 24,
                "eligible_heldout_groups": 1,
            },
        },
    }
    output = tmp_path / "curves.svg"

    render_pu_readiness(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "创新2 E98" in svg
    assert "未标注" in svg
    assert "不是负例" in svg
    assert "不训练E99" in svg
