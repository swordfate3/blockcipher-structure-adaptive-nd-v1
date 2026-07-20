from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import Property
from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import (
    BASELINES,
    _absolute_position,
    _baseline_score,
    _canonical_coordinates,
    _evaluate_baselines,
    _marginals_match,
    _relation_id,
    _rotation_candidates,
)


RUN_ID = "i2_present_r9_atm_support_component_pu_readiness_20260720"
E98A_GATE_SHA256 = "756801674ec77dca1b78b504fa96c031e3418673b93ce475edbab73af3f874ed"
E98A_DECISION = "innovation2_present_r9_atm_public_merge_count_not_rank"


@dataclass(frozen=True)
class SupportComponentPuConfig:
    run_id: str = RUN_ID
    group_count: int = 6
    minimum_group_positives: int = 64
    minimum_unlabeled_per_positive: int = 31
    maximum_shortcut_recall_at_5: float = 0.50
    maximum_shortcut_mrr: float = 0.35

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError("E98-B run_id is frozen")
        if (
            self.group_count != 6
            or self.minimum_group_positives != 64
            or self.minimum_unlabeled_per_positive != 31
        ):
            raise ValueError("E98-B width protocol is frozen")


def build_support_component_audit(
    groups: dict[str, set[Property]],
    config: SupportComponentPuConfig,
) -> dict[str, Any]:
    all_known = set().union(*groups.values())
    basis, dropped = _canonical_independent_basis(all_known)
    components = _support_components(basis)
    packed = _pack_components(components, config.group_count)
    relation_groups = [
        tuple(relation for component in component_group for relation in component)
        for component_group in packed
    ]
    component_rows: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        component_rows.append(
            {
                "component_id": f"component_{component_index:03d}",
                "relations": len(component),
                "support_coordinates": len(
                    {coordinate for relation in component for coordinate in relation}
                ),
                "relation_ids": "|".join(_relation_id(relation) for relation in component),
            }
        )

    group_rows: list[dict[str, Any]] = []
    pools: list[dict[str, Any]] = []
    for heldout_index, heldout_relations in enumerate(relation_groups):
        heldout = set(heldout_relations)
        train = set().union(
            *(set(group) for index, group in enumerate(relation_groups) if index != heldout_index)
        )
        train_coordinates = Counter(
            coordinate for relation in train for coordinate in relation
        )
        heldout_coordinates = {
            coordinate for relation in heldout for coordinate in relation
        }
        fold_pools = [
            _build_filtered_pool(
                heldout_group=heldout_index,
                positive=relation,
                all_known=all_known,
                train_relations=train,
                train_coordinates=train_coordinates,
                minimum_unlabeled=config.minimum_unlabeled_per_positive,
            )
            for relation in sorted(heldout, key=_canonical_coordinates)
        ]
        pools.extend(fold_pools)
        group_rows.append(
            {
                "group_id": f"group_{heldout_index}",
                "components": len(packed[heldout_index]),
                "heldout_relations": len(heldout),
                "train_relations": len(train),
                "heldout_support_coordinates": len(heldout_coordinates),
                "train_support_coordinates": len(train_coordinates),
                "relation_overlap": len(heldout & train),
                "component_overlap": 0,
                "support_coordinate_overlap": len(
                    heldout_coordinates & set(train_coordinates)
                ),
                "minimum_unlabeled_candidates": min(
                    pool["unlabeled_count"] for pool in fold_pools
                ),
                "maximum_unlabeled_candidates": max(
                    pool["unlabeled_count"] for pool in fold_pools
                ),
            }
        )

    baseline_rows = _evaluate_baselines(pools)
    candidate_known_overlap = sum(
        candidate in all_known
        for pool in pools
        for candidate in pool["unlabeled_relations"]
    )
    marginal_mismatches = sum(
        not _marginals_match(pool["positive"], candidate)
        for pool in pools
        for candidate in pool["unlabeled_relations"]
    )
    nondeterministic_pools = sum(
        pool["unlabeled_ids"]
        != tuple(
            _relation_id(candidate)
            for candidate in _filtered_candidates(
                pool["positive"], all_known, pool["train_coordinates"]
            )
        )
        for pool in pools
    )
    component_sizes = Counter(len(component) for component in components)
    metrics = {
        "known_relations": len(all_known),
        "canonical_independent_relations": len(basis),
        "dependent_relations_removed": len(dropped),
        "support_components": len(components),
        "component_size_histogram": dict(sorted(component_sizes.items())),
        "groups": len(relation_groups),
        "minimum_group_positives": min(len(group) for group in relation_groups),
        "maximum_group_positives": max(len(group) for group in relation_groups),
        "total_group_positives": sum(len(group) for group in relation_groups),
        "maximum_relation_overlap": max(row["relation_overlap"] for row in group_rows),
        "maximum_component_overlap": max(row["component_overlap"] for row in group_rows),
        "maximum_support_coordinate_overlap": max(
            row["support_coordinate_overlap"] for row in group_rows
        ),
        "ranking_pools": len(pools),
        "minimum_unlabeled_candidates": min(pool["unlabeled_count"] for pool in pools),
        "maximum_unlabeled_candidates": max(pool["unlabeled_count"] for pool in pools),
        "candidate_known_positive_overlap": candidate_known_overlap,
        "candidate_marginal_mismatches": marginal_mismatches,
        "nondeterministic_candidate_pools": nondeterministic_pools,
    }
    return {
        "basis": basis,
        "dropped": dropped,
        "components": components,
        "component_rows": component_rows,
        "group_rows": group_rows,
        "pools": pools,
        "baseline_rows": baseline_rows,
        "metrics": metrics,
    }


def adjudicate_support_component_readiness(
    config: SupportComponentPuConfig,
    *,
    audit: dict[str, Any],
    e98a_gate: dict[str, Any],
    e98a_gate_hash: str,
) -> dict[str, Any]:
    metrics = dict(audit["metrics"])
    shortcuts = [
        row
        for row in audit["baseline_rows"]
        if row["baseline"] != "deterministic_hash_random"
    ]
    metrics["best_shortcut_recall_at_5"] = max(
        row["recall_at_5"] for row in shortcuts
    )
    metrics["best_shortcut_mrr"] = max(
        row["mean_reciprocal_rank"] for row in shortcuts
    )
    protocol_checks = {
        "e98a_gate_hash_matches": e98a_gate_hash == E98A_GATE_SHA256,
        "e98a_status_pass": e98a_gate["status"] == "pass",
        "e98a_decision_matches": e98a_gate["decision"] == E98A_DECISION,
        "e98a_e99_remained_closed": not e98a_gate["next_action"]["e99_open"],
        "candidate_generation_deterministic": metrics[
            "nondeterministic_candidate_pools"
        ]
        == 0,
        "candidate_marginals_all_match": metrics["candidate_marginal_mismatches"] == 0,
        "unlabeled_contains_no_known_positive": metrics[
            "candidate_known_positive_overlap"
        ]
        == 0,
    }
    readiness_checks = {
        "canonical_basis_has_468_relations": metrics[
            "canonical_independent_relations"
        ]
        == 468,
        "exactly_two_dependent_relations_removed": metrics[
            "dependent_relations_removed"
        ]
        == 2,
        "six_groups_created": metrics["groups"] == config.group_count,
        "every_group_has_at_least_64_positives": metrics["minimum_group_positives"]
        >= config.minimum_group_positives,
        "group_size_delta_at_most_one": metrics["maximum_group_positives"]
        - metrics["minimum_group_positives"]
        <= 1,
        "train_heldout_relation_overlap_zero": metrics["maximum_relation_overlap"] == 0,
        "train_heldout_component_overlap_zero": metrics["maximum_component_overlap"] == 0,
        "train_heldout_support_overlap_zero": metrics[
            "maximum_support_coordinate_overlap"
        ]
        == 0,
        "every_positive_has_at_least_31_unlabeled": metrics[
            "minimum_unlabeled_candidates"
        ]
        >= config.minimum_unlabeled_per_positive,
        "shortcut_recall_at_5_below_stop_line": metrics[
            "best_shortcut_recall_at_5"
        ]
        <= config.maximum_shortcut_recall_at_5,
        "shortcut_mrr_below_stop_line": metrics["best_shortcut_mrr"]
        <= config.maximum_shortcut_mrr,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_r9_atm_support_component_pu_protocol_invalid"
        action = "repair E98-A replay, canonical basis, components, or candidate invariants"
    elif all(readiness_checks.values()):
        status = "pass"
        decision = "innovation2_present_r9_atm_support_component_pu_ready"
        action = (
            "preregister E99 local six-fold positive-unlabeled neural ranking with summary, "
            "coordinate-set, and topology-aware rows; keep remote scale closed"
        )
    else:
        status = "hold"
        decision = "innovation2_present_r9_atm_support_component_pu_not_ready"
        action = "do not train E99; stop the current public-corpus high-round neural route"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "readiness_checks": readiness_checks,
        "metrics": metrics,
        "baseline_metrics": audit["baseline_rows"],
        "claim_scope": (
            "deterministic support-component-disjoint positive-unlabeled ranking readiness "
            "inside the corrected 468-dimensional public independent-round-key PRESENT r9 "
            "ATM corpus; groups are not independent publications, rotated candidates are "
            "unlabeled rather than negatives, and this is not neural training, a new relation, "
            "PRESENT-80 schedule validation, a distinguisher, an attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "e99_local_open": status == "pass",
            "remote_scale": False,
        },
    }


def result_rows(
    config: SupportComponentPuConfig,
    audit: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_atm_support_component_pu_readiness",
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": False,
    }
    return [
        {**common, "result_kind": "group", **row} for row in audit["group_rows"]
    ] + [
        {**common, "result_kind": "baseline", **row}
        for row in audit["baseline_rows"]
    ]


def serializable_config(config: SupportComponentPuConfig) -> dict[str, Any]:
    return asdict(config)


def _canonical_independent_basis(
    relations: set[Property],
) -> tuple[tuple[Property, ...], tuple[Property, ...]]:
    ordered = tuple(sorted(relations, key=_canonical_coordinates))
    coordinates = tuple(
        sorted({coordinate for relation in ordered for coordinate in relation})
    )
    coordinate_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
    pivots: dict[int, int] = {}
    basis: list[Property] = []
    dropped: list[Property] = []
    for relation in ordered:
        row = sum(1 << coordinate_index[coordinate] for coordinate in relation)
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                basis.append(relation)
                break
        if not row:
            dropped.append(relation)
    return tuple(basis), tuple(dropped)


def _support_components(basis: tuple[Property, ...]) -> tuple[tuple[Property, ...], ...]:
    coordinate_to_relations: dict[tuple[int, int], list[int]] = defaultdict(list)
    for relation_index, relation in enumerate(basis):
        for coordinate in relation:
            coordinate_to_relations[coordinate].append(relation_index)
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

    for members in coordinate_to_relations.values():
        for relation_index in members[1:]:
            union(members[0], relation_index)
    grouped: dict[int, list[Property]] = defaultdict(list)
    for relation_index, relation in enumerate(basis):
        grouped[find(relation_index)].append(relation)
    return tuple(
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


def _pack_components(
    components: tuple[tuple[Property, ...], ...], group_count: int
) -> tuple[tuple[tuple[Property, ...], ...], ...]:
    groups: list[list[tuple[Property, ...]]] = [[] for _ in range(group_count)]
    sizes = [0] * group_count
    for component in components:
        target = min(range(group_count), key=lambda index: (sizes[index], index))
        groups[target].append(component)
        sizes[target] += len(component)
    return tuple(tuple(group) for group in groups)


def _build_filtered_pool(
    *,
    heldout_group: int,
    positive: Property,
    all_known: set[Property],
    train_relations: set[Property],
    train_coordinates: Counter[tuple[int, int]],
    minimum_unlabeled: int,
) -> dict[str, Any]:
    candidates = _filtered_candidates(positive, all_known, train_coordinates)
    position_target = mean(_absolute_position(relation) for relation in train_relations)
    entries = (positive, *candidates)
    scores = {
        baseline: tuple(
            _baseline_score(
                baseline,
                relation,
                train_relations=train_relations,
                train_coordinates=train_coordinates,
                position_target=position_target,
            )
            for relation in entries
        )
        for baseline in BASELINES
    }
    return {
        "heldout_group": heldout_group,
        "positive": positive,
        "positive_id": _relation_id(positive),
        "unlabeled_relations": candidates,
        "unlabeled_ids": tuple(_relation_id(candidate) for candidate in candidates),
        "unlabeled_count": len(candidates),
        "minimum_unlabeled_met": len(candidates) >= minimum_unlabeled,
        "train_coordinates": train_coordinates,
        "scores": scores,
    }


def _filtered_candidates(
    positive: Property,
    all_known: set[Property],
    train_coordinates: Counter[tuple[int, int]],
) -> tuple[Property, ...]:
    return tuple(
        candidate
        for candidate in _rotation_candidates(positive, all_known)
        if all(coordinate not in train_coordinates for coordinate in candidate)
    )
