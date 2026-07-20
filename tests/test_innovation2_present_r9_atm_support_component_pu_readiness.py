from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.audit_innovation2_present_r9_atm_support_component_pu_readiness import (
    _pool_row,
)
from blockcipher_nd.cli.plot_innovation2_present_r9_atm_support_component_pu_readiness import (
    render_support_component_readiness,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_support_component_pu_readiness import (
    E98A_DECISION,
    E98A_GATE_SHA256,
    SupportComponentPuConfig,
    _pack_components,
    _support_components,
    adjudicate_support_component_readiness,
)


def _relation(*coordinates: tuple[int, int]) -> frozenset[tuple[int, int]]:
    return frozenset(coordinates)


def _passing_audit() -> dict[str, object]:
    return {
        "metrics": {
            "canonical_independent_relations": 468,
            "dependent_relations_removed": 2,
            "groups": 6,
            "minimum_group_positives": 78,
            "maximum_group_positives": 78,
            "maximum_relation_overlap": 0,
            "maximum_component_overlap": 0,
            "maximum_support_coordinate_overlap": 0,
            "minimum_unlabeled_candidates": 51,
            "candidate_marginal_mismatches": 0,
            "candidate_known_positive_overlap": 0,
            "nondeterministic_candidate_pools": 0,
        },
        "baseline_rows": [
            {
                "baseline": "deterministic_hash_random",
                "recall_at_5": 0.08,
                "mean_reciprocal_rank": 0.07,
            },
            {
                "baseline": "training_support_overlap",
                "recall_at_5": 0.12,
                "mean_reciprocal_rank": 0.09,
            },
        ],
    }


def _e98a_gate() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": E98A_DECISION,
        "next_action": {"e99_open": False},
    }


def test_support_components_are_not_split_during_balanced_packing() -> None:
    linked_a = _relation((1, 2), (3, 4))
    linked_b = _relation((3, 4), (5, 6))
    singleton_a = _relation((7, 8))
    singleton_b = _relation((9, 10))

    components = _support_components((linked_a, linked_b, singleton_a, singleton_b))
    packed = _pack_components(components, 2)

    assert sorted(len(component) for component in components) == [1, 1, 2]
    assert any(set(component) == {linked_a, linked_b} for group in packed for component in group)
    assert not any(
        (linked_a in component) != (linked_b in component)
        for group in packed
        for component in group
    )


def test_e98b_pass_opens_only_local_e99() -> None:
    gate = adjudicate_support_component_readiness(
        SupportComponentPuConfig(),
        audit=_passing_audit(),
        e98a_gate=_e98a_gate(),
        e98a_gate_hash=E98A_GATE_SHA256,
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_r9_atm_support_component_pu_ready"
    assert gate["next_action"]["e99_local_open"] is True
    assert gate["next_action"]["remote_scale"] is False


def test_e98b_shortcut_dominance_holds_neural_training() -> None:
    audit = _passing_audit()
    audit["baseline_rows"][1]["recall_at_5"] = 0.75

    gate = adjudicate_support_component_readiness(
        SupportComponentPuConfig(),
        audit=audit,
        e98a_gate=_e98a_gate(),
        e98a_gate_hash=E98A_GATE_SHA256,
    )

    assert gate["status"] == "hold"
    assert gate["next_action"]["e99_local_open"] is False


def test_e98b_source_replay_mismatch_is_protocol_failure() -> None:
    gate = adjudicate_support_component_readiness(
        SupportComponentPuConfig(),
        audit=_passing_audit(),
        e98a_gate=_e98a_gate(),
        e98a_gate_hash="wrong",
    )

    assert gate["status"] == "fail"
    assert gate["decision"].endswith("protocol_invalid")


def test_pool_row_never_calls_candidates_negatives() -> None:
    row = _pool_row(
        {
            "heldout_group": 2,
            "positive_id": "positive",
            "unlabeled_count": 2,
            "minimum_unlabeled_met": True,
            "unlabeled_ids": ("candidate-a", "candidate-b"),
        }
    )

    assert row["heldout_group"] == "group_2"
    assert row["unlabeled_ids"] == "candidate-a|candidate-b"
    assert not any("negative" in key for key in row)


def test_e98b_plot_explains_pu_semantics_and_local_only_gate(tmp_path: Path) -> None:
    groups = [
        {
            "group_id": f"group_{index}",
            "heldout_relations": 78,
            "minimum_unlabeled_candidates": 51 + index,
        }
        for index in range(6)
    ]
    baselines = [
        {
            "baseline": "deterministic_hash_random",
            "recall_at_5": 0.08,
            "mean_reciprocal_rank": 0.07,
        },
        {
            "baseline": "training_support_overlap",
            "recall_at_5": 0.12,
            "mean_reciprocal_rank": 0.09,
        },
    ]
    summary = {
        "groups": groups,
        "ranking_baselines": baselines,
        "gate": {
            "status": "pass",
            "metrics": {
                "canonical_independent_relations": 468,
                "support_components": 452,
                "minimum_group_positives": 78,
                "minimum_unlabeled_candidates": 51,
                "maximum_relation_overlap": 0,
                "maximum_component_overlap": 0,
                "maximum_support_coordinate_overlap": 0,
            },
        },
    }
    output = tmp_path / "curves.svg"

    render_support_component_readiness(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "正例/未标注排序" in svg
    assert "三层泄漏检查" in svg
    assert "只开放E99本地神经排序门" in svg
    assert "远程仍关闭" in svg
