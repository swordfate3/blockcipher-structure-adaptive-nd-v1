from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_r9_atm_support_rotation_orbit_pu_readiness import (
    render_support_rotation_orbit_readiness,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_support_rotation_orbit_pu_readiness import (
    E98B_DECISION,
    E98B_GATE_SHA256,
    SupportRotationOrbitPuConfig,
    _rotation_orbit_signature,
    _support_rotation_components,
    adjudicate_support_rotation_orbit_readiness,
)
from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import _rotate64


def _relation(*coordinates: tuple[int, int]) -> frozenset[tuple[int, int]]:
    return frozenset(coordinates)


def test_rotation_orbit_signature_is_synchronous_rotation_invariant() -> None:
    relation = _relation((0x81, 0x42), (0x18, 0x24))
    rotated = frozenset((_rotate64(left, 17), _rotate64(right, 17)) for left, right in relation)
    assert _rotation_orbit_signature(relation) == _rotation_orbit_signature(rotated)


def test_components_join_shared_support_and_same_rotation_orbit() -> None:
    first = _relation((0x01, 0x02))
    rotated = _relation((_rotate64(0x01, 5), _rotate64(0x02, 5)))
    support_link = _relation((0x01, 0x02), (0x08, 0x10))
    isolated = _relation((0x100, 0x400))
    components, orbits = _support_rotation_components((first, rotated, support_link, isolated))
    assert any(set(component) == {first, rotated, support_link} for component in components)
    assert sorted(map(len, components)) == [1, 3]
    assert any(set(orbit) == {first, rotated} for orbit in orbits.values())


def _audit() -> dict[str, object]:
    return {
        "metrics": {
            "canonical_independent_relations": 468,
            "dependent_relations_removed": 2,
            "rotation_orbits": 368,
            "groups": 6,
            "minimum_group_positives": 78,
            "maximum_group_positives": 78,
            "rotation_orbits_split_across_groups": 0,
            "maximum_train_test_all_relation_overlap": 0,
            "candidate_positive_support_overlap": 0,
            "candidate_known_positive_overlap": 0,
            "minimum_train_unlabeled": 55,
            "minimum_test_unlabeled": 51,
            "candidate_marginal_mismatches": 0,
        },
        "baseline_rows": [
            {"baseline": "deterministic_hash_random", "recall_at_5": 0.08, "mean_reciprocal_rank": 0.07},
            {"baseline": "absolute_bit_position", "recall_at_5": 0.13, "mean_reciprocal_rank": 0.12},
        ],
    }


def _source_gate() -> dict[str, object]:
    return {"status": "pass", "decision": E98B_DECISION}


def test_e98c_pass_opens_only_revised_local_e99() -> None:
    gate = adjudicate_support_rotation_orbit_readiness(
        SupportRotationOrbitPuConfig(),
        audit=_audit(),
        e98b_gate=_source_gate(),
        e98b_gate_hash=E98B_GATE_SHA256,
    )
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_r9_atm_support_orbit_pu_ready"
    assert gate["next_action"]["e99_local_open"] is True
    assert gate["next_action"]["remote_scale"] is False


def test_e98c_exact_candidate_relation_leakage_holds_route() -> None:
    audit = _audit()
    audit["metrics"]["maximum_train_test_all_relation_overlap"] = 1
    gate = adjudicate_support_rotation_orbit_readiness(
        SupportRotationOrbitPuConfig(),
        audit=audit,
        e98b_gate=_source_gate(),
        e98b_gate_hash=E98B_GATE_SHA256,
    )
    assert gate["status"] == "hold"
    assert gate["next_action"]["e99_local_open"] is False


def test_e98c_plot_explains_orbit_fix_and_unlabeled_semantics(tmp_path: Path) -> None:
    summary = {
        "groups": [
            {"minimum_train_unlabeled": 55 + index % 2, "minimum_test_unlabeled": 51 + index % 3}
            for index in range(6)
        ],
        "gate": {
            "status": "pass",
            "metrics": {
                "combined_component_size_histogram": {"1": 267, "2": 67, "3": 11, "4": 3, "5": 2, "6": 2},
                "rotation_orbits": 368,
                "combined_components": 352,
                "minimum_train_unlabeled": 55,
                "minimum_test_unlabeled": 51,
                "rotation_orbits_split_across_groups": 0,
                "maximum_train_test_all_relation_overlap": 0,
                "candidate_positive_support_overlap": 0,
                "candidate_known_positive_overlap": 0,
            },
        },
    }
    output = tmp_path / "curves.svg"
    render_support_rotation_orbit_readiness(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "候选旋转轨道泄漏修复门" in svg
    assert "四层泄漏检查全部为0" in svg
    assert "不是密码学负例" in svg
    assert "恢复E99本地神经排序" in svg
