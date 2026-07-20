from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import Property
from blockcipher_nd.tasks.innovation2.present_r9_atm_support_component_pu_readiness import (
    _build_filtered_pool,
    _canonical_independent_basis,
    _pack_components,
)
from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import (
    _canonical_coordinates,
    _evaluate_baselines,
    _marginals_match,
    _relation_id,
    _rotate64,
    _rotation_candidates,
)


RUN_ID = "i2_present_r9_atm_support_rotation_orbit_pu_readiness_20260720"
E98B_GATE_SHA256 = "2f3f3d0cce46d3e786a39899ed87949eddb6c614deb52e16c8aaca623a5c0cb9"
E98B_DECISION = "innovation2_present_r9_atm_support_component_pu_ready"


@dataclass(frozen=True)
class SupportRotationOrbitPuConfig:
    run_id: str = RUN_ID
    group_count: int = 6
    minimum_group_positives: int = 64
    minimum_unlabeled_per_positive: int = 31
    maximum_shortcut_recall_at_5: float = 0.50
    maximum_shortcut_mrr: float = 0.35

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError("E98-C run_id is frozen")
        if (
            self.group_count != 6
            or self.minimum_group_positives != 64
            or self.minimum_unlabeled_per_positive != 31
        ):
            raise ValueError("E98-C width protocol is frozen")


def build_support_rotation_orbit_audit(
    groups: dict[str, set[Property]],
    config: SupportRotationOrbitPuConfig,
) -> dict[str, Any]:
    all_known = set().union(*groups.values())
    basis, dropped = _canonical_independent_basis(all_known)
    components, orbit_groups = _support_rotation_components(basis)
    packed = _pack_components(components, config.group_count)
    relation_groups = tuple(
        tuple(relation for component in component_group for relation in component)
        for component_group in packed
    )
    relation_group_index = {
        relation: group_index
        for group_index, relation_group in enumerate(relation_groups)
        for relation in relation_group
    }
    component_rows = [
        {
            "component_id": f"component_{component_index:03d}",
            "relations": len(component),
            "support_coordinates": len(
                {coordinate for relation in component for coordinate in relation}
            ),
            "rotation_orbits": len({_rotation_orbit_signature(relation) for relation in component}),
            "relation_ids": "|".join(_relation_id(relation) for relation in component),
        }
        for component_index, component in enumerate(components)
    ]
    group_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    test_pools: list[dict[str, Any]] = []
    for heldout_index, heldout_group in enumerate(relation_groups):
        heldout = set(heldout_group)
        train = set().union(
            *(set(group) for index, group in enumerate(relation_groups) if index != heldout_index)
        )
        train_support = Counter(coordinate for relation in train for coordinate in relation)
        heldout_support = Counter(coordinate for relation in heldout for coordinate in relation)
        fold_test_pools = [
            _build_filtered_pool(
                heldout_group=heldout_index,
                positive=relation,
                all_known=all_known,
                train_relations=train,
                train_coordinates=train_support,
                minimum_unlabeled=config.minimum_unlabeled_per_positive,
            )
            for relation in sorted(heldout, key=_canonical_coordinates)
        ]
        fold_train_pools = [
            {
                "positive": relation,
                "candidates": tuple(
                    candidate
                    for candidate in _rotation_candidates(relation, all_known)
                    if all(coordinate not in heldout_support for coordinate in candidate)
                ),
            }
            for relation in sorted(train, key=_canonical_coordinates)
        ]
        train_examples = {
            relation
            for pool in fold_train_pools
            for relation in (pool["positive"], *pool["candidates"])
        }
        test_examples = {
            relation
            for pool in fold_test_pools
            for relation in (pool["positive"], *pool["unlabeled_relations"])
        }
        row = {
            "fold": heldout_index,
            "train_positives": len(train),
            "test_positives": len(heldout),
            "minimum_train_unlabeled": min(len(pool["candidates"]) for pool in fold_train_pools),
            "minimum_test_unlabeled": min(pool["unlabeled_count"] for pool in fold_test_pools),
            "train_test_all_relation_overlap": len(train_examples & test_examples),
            "train_candidate_test_positive_support_overlap": sum(
                any(coordinate in heldout_support for coordinate in candidate)
                for pool in fold_train_pools
                for candidate in pool["candidates"]
            ),
            "test_candidate_train_positive_support_overlap": sum(
                any(coordinate in train_support for coordinate in candidate)
                for pool in fold_test_pools
                for candidate in pool["unlabeled_relations"]
            ),
            "known_positive_candidate_overlap": sum(
                candidate in all_known
                for pool in fold_train_pools
                for candidate in pool["candidates"]
            )
            + sum(
                candidate in all_known
                for pool in fold_test_pools
                for candidate in pool["unlabeled_relations"]
            ),
        }
        fold_rows.append(row)
        test_pools.extend(fold_test_pools)
        group_rows.append(
            {
                "group_id": f"group_{heldout_index}",
                "components": len(packed[heldout_index]),
                "positives": len(heldout),
                "support_coordinates": len(heldout_support),
                "minimum_train_unlabeled": row["minimum_train_unlabeled"],
                "minimum_test_unlabeled": row["minimum_test_unlabeled"],
            }
        )
    baseline_rows = _evaluate_baselines(test_pools)
    orbit_split_count = sum(
        len({relation_group_index[relation] for relation in orbit}) > 1
        for orbit in orbit_groups.values()
    )
    component_histogram = Counter(map(len, components))
    orbit_histogram = Counter(map(len, orbit_groups.values()))
    metrics = {
        "known_relations": len(all_known),
        "canonical_independent_relations": len(basis),
        "dependent_relations_removed": len(dropped),
        "rotation_orbits": len(orbit_groups),
        "rotation_orbit_size_histogram": dict(sorted(orbit_histogram.items())),
        "combined_components": len(components),
        "combined_component_size_histogram": dict(sorted(component_histogram.items())),
        "maximum_component_size": max(map(len, components)),
        "groups": len(relation_groups),
        "minimum_group_positives": min(map(len, relation_groups)),
        "maximum_group_positives": max(map(len, relation_groups)),
        "rotation_orbits_split_across_groups": orbit_split_count,
        "maximum_train_test_all_relation_overlap": max(
            row["train_test_all_relation_overlap"] for row in fold_rows
        ),
        "candidate_positive_support_overlap": sum(
            row["train_candidate_test_positive_support_overlap"]
            + row["test_candidate_train_positive_support_overlap"]
            for row in fold_rows
        ),
        "candidate_known_positive_overlap": sum(
            row["known_positive_candidate_overlap"] for row in fold_rows
        ),
        "minimum_train_unlabeled": min(row["minimum_train_unlabeled"] for row in fold_rows),
        "minimum_test_unlabeled": min(row["minimum_test_unlabeled"] for row in fold_rows),
        "candidate_marginal_mismatches": sum(
            not _marginals_match(pool["positive"], candidate)
            for pool in test_pools
            for candidate in pool["unlabeled_relations"]
        ),
    }
    return {
        "basis": basis,
        "dropped": dropped,
        "components": components,
        "component_rows": component_rows,
        "group_rows": group_rows,
        "fold_rows": fold_rows,
        "test_pools": test_pools,
        "baseline_rows": baseline_rows,
        "metrics": metrics,
    }


def adjudicate_support_rotation_orbit_readiness(
    config: SupportRotationOrbitPuConfig,
    *,
    audit: dict[str, Any],
    e98b_gate: dict[str, Any],
    e98b_gate_hash: str,
) -> dict[str, Any]:
    metrics = dict(audit["metrics"])
    shortcuts = [
        row for row in audit["baseline_rows"] if row["baseline"] != "deterministic_hash_random"
    ]
    metrics["best_shortcut_recall_at_5"] = max(row["recall_at_5"] for row in shortcuts)
    metrics["best_shortcut_mrr"] = max(row["mean_reciprocal_rank"] for row in shortcuts)
    protocol_checks = {
        "e98b_gate_hash_matches": e98b_gate_hash == E98B_GATE_SHA256,
        "e98b_status_pass": e98b_gate.get("status") == "pass",
        "e98b_decision_matches": e98b_gate.get("decision") == E98B_DECISION,
        "canonical_basis_replays": metrics["canonical_independent_relations"] == 468
        and metrics["dependent_relations_removed"] == 2,
        "rotation_orbit_canonicalization_replays": metrics["rotation_orbits"] == 368,
        "candidate_marginals_all_match": metrics["candidate_marginal_mismatches"] == 0,
        "unlabeled_contains_no_known_positive": metrics["candidate_known_positive_overlap"] == 0,
    }
    readiness_checks = {
        "six_groups_created": metrics["groups"] == config.group_count,
        "every_group_has_at_least_64_positives": metrics["minimum_group_positives"]
        >= config.minimum_group_positives,
        "group_size_delta_at_most_one": metrics["maximum_group_positives"]
        - metrics["minimum_group_positives"]
        <= 1,
        "rotation_orbits_are_group_disjoint": metrics["rotation_orbits_split_across_groups"] == 0,
        "all_train_test_relations_are_disjoint": metrics[
            "maximum_train_test_all_relation_overlap"
        ]
        == 0,
        "candidate_positive_support_is_disjoint": metrics[
            "candidate_positive_support_overlap"
        ]
        == 0,
        "every_train_positive_has_at_least_31_unlabeled": metrics[
            "minimum_train_unlabeled"
        ]
        >= config.minimum_unlabeled_per_positive,
        "every_test_positive_has_at_least_31_unlabeled": metrics[
            "minimum_test_unlabeled"
        ]
        >= config.minimum_unlabeled_per_positive,
        "shortcut_recall_at_5_below_stop_line": metrics["best_shortcut_recall_at_5"]
        <= config.maximum_shortcut_recall_at_5,
        "shortcut_mrr_below_stop_line": metrics["best_shortcut_mrr"]
        <= config.maximum_shortcut_mrr,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_r9_atm_support_orbit_pu_protocol_invalid"
        action = "repair source replay, orbit canonicalization, components, or candidate invariants"
    elif all(readiness_checks.values()):
        status = "pass"
        decision = "innovation2_present_r9_atm_support_orbit_pu_ready"
        action = "amend E99 to consume this gate and run the local neural ranking matrix"
    else:
        status = "hold"
        decision = "innovation2_present_r9_atm_support_orbit_pu_not_ready"
        action = "stop the current public-corpus nine-round neural route; do not weaken leakage checks"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "readiness_checks": readiness_checks,
        "metrics": metrics,
        "baseline_metrics": audit["baseline_rows"],
        "claim_scope": (
            "deterministic support-and-rotation-orbit-disjoint positive-unlabeled ranking "
            "readiness inside the corrected public independent-round-key PRESENT r9 ATM corpus; "
            "unlabeled candidates are not negatives and this is not neural training, a new "
            "relation, PRESENT-80 key-schedule validation, a distinguisher, an attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "e99_local_open": status == "pass",
            "remote_scale": False,
            "training": False,
        },
    }


def result_rows(
    config: SupportRotationOrbitPuConfig,
    audit: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_atm_support_rotation_orbit_pu_readiness",
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": False,
    }
    return [
        {**common, "result_kind": "fold", **row} for row in audit["fold_rows"]
    ] + [
        {**common, "result_kind": "baseline", **row} for row in audit["baseline_rows"]
    ]


def serializable_config(config: SupportRotationOrbitPuConfig) -> dict[str, Any]:
    return asdict(config)


def _rotation_orbit_signature(relation: Property) -> tuple[tuple[int, int], ...]:
    return min(
        tuple(
            sorted(
                (_rotate64(left, shift), _rotate64(right, shift))
                for left, right in relation
            )
        )
        for shift in range(64)
    )


def _support_rotation_components(
    basis: tuple[Property, ...],
) -> tuple[
    tuple[tuple[Property, ...], ...],
    dict[tuple[tuple[int, int], ...], tuple[Property, ...]],
]:
    parent = list(range(len(basis)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    coordinate_members: dict[tuple[int, int], list[int]] = defaultdict(list)
    orbit_members: dict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
    for relation_index, relation in enumerate(basis):
        for coordinate in relation:
            coordinate_members[coordinate].append(relation_index)
        orbit_members[_rotation_orbit_signature(relation)].append(relation_index)
    for members in (*coordinate_members.values(), *orbit_members.values()):
        for relation_index in members[1:]:
            union(members[0], relation_index)
    grouped: dict[int, list[Property]] = defaultdict(list)
    for relation_index, relation in enumerate(basis):
        grouped[find(relation_index)].append(relation)
    components = tuple(
        sorted(
            (
                tuple(sorted(component, key=_canonical_coordinates))
                for component in grouped.values()
            ),
            key=lambda component: (
                -len(component),
                tuple(_canonical_coordinates(relation) for relation in component),
            ),
        )
    )
    orbit_relations = {
        signature: tuple(basis[index] for index in members)
        for signature, members in orbit_members.items()
    }
    return components, orbit_relations
