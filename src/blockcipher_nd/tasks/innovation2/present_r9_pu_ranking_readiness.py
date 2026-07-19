from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import (
    ATM_COMMIT,
    ATM_EXPECTED_RESULT_FILES,
    ATM_PUBLISHED_DIMENSION,
    Property,
    audit_atm_results,
    load_builtin_property_pickle,
)


RUN_ID = "i2_present_r9_generalized_relation_pu_ranking_readiness_20260719"
EXPECTED_UNION_RANK = 468
EXPECTED_SERIALIZED_RELATIONS = 470
EXPECTED_FILE_HASHES = {
    "R9-complex-oracle-1-5-3.pkl": (
        "47d369f890e4f39933ea165ea81ac9507bafbf3cf579083331d646495b5e5fdc"
    ),
    "R9-complex-oracle-1-6-2.pkl": (
        "5e6cd689edc1e726418b1865a47ca71ec2de770cf54bd95069f8d1d62729e63c"
    ),
    "R9-complex-oracle-1-7-1.pkl": (
        "11e161f88a3392bf76e0f631ea585fa068666b3492398f83fb256cf5253c22dd"
    ),
    "R9-complex-oracle-2-4-3.pkl": (
        "ea73c0441dc9ee7ab8dea1d1cfc4cc39123e51d9fcb61ef8860fc5a2976415f2"
    ),
    "R9-complex-oracle-2-5-2.pkl": (
        "1907fa8790b331036ba506d3be1f7b025e5a7729f9dbeba6127db51de7395c41"
    ),
    "R9-complex-oracle-2-6-1.pkl": (
        "cfaf0df5b31d3a59950c3fae91583ed886fa551ca07e4b1b275f274038792d84"
    ),
    "R9-complex-oracle-3-4-2.pkl": (
        "7393cd1d24a9437dd92c4ec2168aa23e6b42d7247d8bd1ff7b11e0bec3f76ec6"
    ),
    "R9-complex-oracle-3-5-1.pkl": (
        "e0ba626d6a8eb86d573dbc4307c0842bb2af184306296ea4d71ed4d618261290"
    ),
}
MASK64 = (1 << 64) - 1
BASELINES = (
    "deterministic_hash_random",
    "file_id",
    "relation_size",
    "exponent_weight",
    "exact_training_frequency",
    "training_coordinate_frequency",
    "training_support_overlap",
    "absolute_bit_position",
)


@dataclass(frozen=True)
class PuRankingReadinessConfig:
    run_id: str = RUN_ID
    minimum_eligible_groups: int = 6
    minimum_positives_per_group: int = 8
    minimum_total_heldout_positives: int = 64
    minimum_unlabeled_per_positive: int = 31
    maximum_shortcut_recall_at_5: float = 0.50
    maximum_shortcut_mrr: float = 0.35

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError("E98 run_id is frozen")
        if min(
            self.minimum_eligible_groups,
            self.minimum_positives_per_group,
            self.minimum_total_heldout_positives,
            self.minimum_unlabeled_per_positive,
        ) <= 0:
            raise ValueError("E98 width thresholds must be positive")
        if not 0.0 <= self.maximum_shortcut_recall_at_5 <= 1.0:
            raise ValueError("maximum_shortcut_recall_at_5 must be in [0, 1]")
        if not 0.0 <= self.maximum_shortcut_mrr <= 1.0:
            raise ValueError("maximum_shortcut_mrr must be in [0, 1]")


def load_relation_groups(results_root: Path) -> dict[str, set[Property]]:
    return {
        name: load_builtin_property_pickle(results_root / name)
        for name in ATM_EXPECTED_RESULT_FILES
    }


def audit_sources(
    results_root: Path,
    *,
    actual_commit: str,
) -> dict[str, Any]:
    actual_names = tuple(sorted(path.name for path in results_root.glob("*.pkl")))
    hashes = {
        name: _sha256(results_root / name)
        for name in actual_names
    }
    atm = audit_atm_results(results_root)
    checks = {
        "commit_matches_frozen_version": actual_commit == ATM_COMMIT,
        "exact_eight_file_set_matches": actual_names
        == tuple(sorted(ATM_EXPECTED_RESULT_FILES)),
        "all_file_hashes_match": hashes == EXPECTED_FILE_HASHES,
        "safe_builtin_pickle_shapes": bool(
            atm["checks"]["safe_builtin_pickle_shapes"]
        ),
        "serialized_relation_count_replays": (
            atm["metrics"]["unique_serialized_basis_elements"]
            == EXPECTED_SERIALIZED_RELATIONS
        ),
        "union_rank_replays": atm["metrics"]["union_gf2_rank"]
        == EXPECTED_UNION_RANK,
    }
    return {
        "actual_commit": actual_commit,
        "hashes": hashes,
        "checks": checks,
        "atm": atm,
    }


def build_ranking_audit(
    groups: dict[str, set[Property]],
    *,
    minimum_unlabeled_per_positive: int,
) -> dict[str, Any]:
    all_known = set().union(*groups.values())
    provenance = Counter(
        relation for relations in groups.values() for relation in relations
    )
    folds: list[dict[str, Any]] = []
    pools: list[dict[str, Any]] = []

    for heldout_file in ATM_EXPECTED_RESULT_FILES:
        train_files = tuple(name for name in ATM_EXPECTED_RESULT_FILES if name != heldout_file)
        train_relations = set().union(*(groups[name] for name in train_files))
        train_coordinates = Counter(
            coordinate for relation in train_relations for coordinate in relation
        )
        heldout_relations = sorted(
            groups[heldout_file] - train_relations,
            key=_canonical_coordinates,
        )
        leaking = [
            relation
            for relation in heldout_relations
            if any(coordinate in train_coordinates for coordinate in relation)
        ]
        fold_pools: list[dict[str, Any]] = []
        for relation in heldout_relations:
            pool = _build_pool(
                heldout_file=heldout_file,
                positive=relation,
                all_known=all_known,
                train_relations=train_relations,
                train_coordinates=train_coordinates,
                minimum_unlabeled=minimum_unlabeled_per_positive,
            )
            fold_pools.append(pool)
            pools.append(pool)
        folds.append(
            {
                "heldout_file": heldout_file,
                "source_relations": len(groups[heldout_file]),
                "train_relations": len(train_relations),
                "heldout_relations": len(heldout_relations),
                "heldout_singletons": sum(len(relation) == 1 for relation in heldout_relations),
                "heldout_multi_term": sum(len(relation) > 1 for relation in heldout_relations),
                "heldout_with_coordinate_leakage": len(leaking),
                "heldout_coordinate_disjoint": len(heldout_relations) - len(leaking),
                "minimum_unlabeled_candidates": min(
                    (pool["unlabeled_count"] for pool in fold_pools), default=0
                ),
                "eligible_width": len(heldout_relations) >= 8 and not leaking,
            }
        )

    baseline_rows = _evaluate_baselines(pools)
    unlabeled_known_overlap = sum(
        bool(set(pool["unlabeled_relations"]) & all_known) for pool in pools
    )
    marginal_mismatches = sum(
        not _marginals_match(pool["positive"], candidate)
        for pool in pools
        for candidate in pool["unlabeled_relations"]
    )
    nondeterministic_pools = sum(
        pool["unlabeled_ids"]
        != tuple(
            _relation_id(relation)
            for relation in _rotation_candidates(pool["positive"], all_known)
        )
        for pool in pools
    )
    metrics = {
        "source_groups": len(groups),
        "serialized_relations_across_files": sum(len(group) for group in groups.values()),
        "deduplicated_known_relations": len(all_known),
        "relations_unique_to_one_file": sum(value == 1 for value in provenance.values()),
        "relations_shared_across_files": sum(value > 1 for value in provenance.values()),
        "eligible_heldout_groups": sum(row["eligible_width"] for row in folds),
        "groups_with_any_heldout_positive": sum(row["heldout_relations"] > 0 for row in folds),
        "groups_with_at_least_eight_heldout_positives": sum(
            row["heldout_relations"] >= 8 for row in folds
        ),
        "total_heldout_positives": sum(row["heldout_relations"] for row in folds),
        "total_heldout_singletons": sum(row["heldout_singletons"] for row in folds),
        "total_heldout_multi_term": sum(row["heldout_multi_term"] for row in folds),
        "heldout_relations_with_coordinate_leakage": sum(
            row["heldout_with_coordinate_leakage"] for row in folds
        ),
        "ranking_pools": len(pools),
        "minimum_unlabeled_candidates": min(
            (pool["unlabeled_count"] for pool in pools), default=0
        ),
        "maximum_unlabeled_candidates": max(
            (pool["unlabeled_count"] for pool in pools), default=0
        ),
        "unlabeled_pools_overlapping_known_positives": unlabeled_known_overlap,
        "candidate_marginal_mismatches": marginal_mismatches,
        "nondeterministic_candidate_pools": nondeterministic_pools,
    }
    return {
        "folds": folds,
        "pools": pools,
        "baseline_rows": baseline_rows,
        "metrics": metrics,
    }


def adjudicate_pu_readiness(
    config: PuRankingReadinessConfig,
    *,
    source_audit: dict[str, Any],
    ranking_audit: dict[str, Any],
) -> dict[str, Any]:
    metrics = {
        **source_audit["atm"]["metrics"],
        **ranking_audit["metrics"],
    }
    baselines = ranking_audit["baseline_rows"]
    shortcuts = [row for row in baselines if row["baseline"] != "deterministic_hash_random"]
    best_shortcut_recall = max((row["recall_at_5"] for row in shortcuts), default=0.0)
    best_shortcut_mrr = max((row["mean_reciprocal_rank"] for row in shortcuts), default=0.0)
    metrics["best_shortcut_recall_at_5"] = best_shortcut_recall
    metrics["best_shortcut_mrr"] = best_shortcut_mrr

    protocol_checks = {
        "all_heldout_relations_are_train_disjoint": all(
            row["heldout_relations"]
            == row["heldout_coordinate_disjoint"]
            + row["heldout_with_coordinate_leakage"]
            for row in ranking_audit["folds"]
        ),
        "unlabeled_contains_no_known_positive": metrics[
            "unlabeled_pools_overlapping_known_positives"
        ]
        == 0,
        "all_candidate_marginals_match": metrics["candidate_marginal_mismatches"] == 0,
        "candidate_generation_is_deterministic": metrics[
            "nondeterministic_candidate_pools"
        ]
        == 0,
    }
    readiness_checks = {
        "published_dimension_matches_recomputed_union_rank": metrics["union_gf2_rank"]
        == metrics["published_dimension"],
        "at_least_six_eligible_heldout_groups": metrics["eligible_heldout_groups"]
        >= config.minimum_eligible_groups,
        "each_eligible_group_has_at_least_eight_positives": metrics[
            "groups_with_at_least_eight_heldout_positives"
        ]
        >= config.minimum_eligible_groups,
        "at_least_64_total_heldout_positives": metrics["total_heldout_positives"]
        >= config.minimum_total_heldout_positives,
        "heldout_support_coordinates_are_train_disjoint": metrics[
            "heldout_relations_with_coordinate_leakage"
        ]
        == 0,
        "at_least_31_unlabeled_per_positive": metrics["ranking_pools"] > 0
        and metrics["minimum_unlabeled_candidates"]
        >= config.minimum_unlabeled_per_positive,
        "shortcut_recall_at_5_below_stop_line": best_shortcut_recall
        <= config.maximum_shortcut_recall_at_5,
        "shortcut_mrr_below_stop_line": best_shortcut_mrr
        <= config.maximum_shortcut_mrr,
    }
    if not all(source_audit["checks"].values()) or not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_r9_pu_ranking_protocol_invalid"
        action = "repair frozen source, safe parsing, or candidate-pool invariants"
    elif all(readiness_checks.values()):
        status = "pass"
        decision = "innovation2_present_r9_pu_ranking_ready_for_local_neural_gate"
        action = (
            "preregister E99 local group-disjoint positive-unlabeled neural ranking; "
            "keep remote scale closed until the local gate beats all shortcuts"
        )
    else:
        status = "hold"
        decision = "innovation2_present_r9_pu_ranking_benchmark_not_ready"
        action = (
            "do not train E99; resolve the 470-versus-468 basis contract and obtain "
            "additional independent nine-round positive-relation groups or a sound generator"
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_audit["checks"],
        "protocol_checks": protocol_checks,
        "readiness_checks": readiness_checks,
        "metrics": metrics,
        "baseline_metrics": baselines,
        "claim_scope": (
            "local deterministic readiness audit of known PRESENT-round-function r9 ATM "
            "relations under independent round keys; unlabeled candidates are not negatives, "
            "constants are not known balanced-zero labels, and this is not neural training, "
            "a new distinguisher, an attack, PRESENT-80 schedule validation, or SOTA"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "e99_open": status == "pass",
            "remote_scale": False,
            "closed_routes": [
                "calling unlabeled relations strict negatives",
                "binary accuracy or AUC on positive-versus-unlabeled rows",
                "training E99 while the group-width gate is held",
                "remote GPU scale-up before a passing local neural ranking gate",
            ],
        },
    }


def result_rows(
    config: PuRankingReadinessConfig,
    ranking_audit: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_generalized_relation_pu_ranking_readiness",
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": False,
    }
    return [
        {**common, "result_kind": "fold", **row}
        for row in ranking_audit["folds"]
    ] + [
        {**common, "result_kind": "baseline", **row}
        for row in ranking_audit["baseline_rows"]
    ]


def serializable_config(config: PuRankingReadinessConfig) -> dict[str, Any]:
    return asdict(config)


def _build_pool(
    *,
    heldout_file: str,
    positive: Property,
    all_known: set[Property],
    train_relations: set[Property],
    train_coordinates: Counter[tuple[int, int]],
    minimum_unlabeled: int,
) -> dict[str, Any]:
    candidates = _rotation_candidates(positive, all_known)
    position_target = mean(_absolute_position(relation) for relation in train_relations)
    entries = (positive, *candidates)
    scores: dict[str, tuple[float, ...]] = {}
    for baseline in BASELINES:
        scores[baseline] = tuple(
            _baseline_score(
                baseline,
                relation,
                train_relations=train_relations,
                train_coordinates=train_coordinates,
                position_target=position_target,
            )
            for relation in entries
        )
    return {
        "heldout_file": heldout_file,
        "positive": positive,
        "positive_id": _relation_id(positive),
        "unlabeled_relations": candidates,
        "unlabeled_ids": tuple(_relation_id(relation) for relation in candidates),
        "unlabeled_count": len(candidates),
        "minimum_unlabeled_met": len(candidates) >= minimum_unlabeled,
        "relation_size": len(positive),
        "scores": scores,
    }


def _evaluate_baselines(pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline in BASELINES:
        ranks: list[int] = []
        pool_sizes: list[int] = []
        for pool in pools:
            ids = (pool["positive_id"], *pool["unlabeled_ids"])
            ranked = sorted(
                zip(pool["scores"][baseline], ids, strict=True),
                key=lambda item: (-item[0], item[1]),
            )
            rank = next(
                index
                for index, (_, relation_id) in enumerate(ranked, start=1)
                if relation_id == pool["positive_id"]
            )
            ranks.append(rank)
            pool_sizes.append(len(ids))
        recall_at_1 = mean(rank <= 1 for rank in ranks) if ranks else 0.0
        recall_at_5 = mean(rank <= 5 for rank in ranks) if ranks else 0.0
        random_top5 = mean(min(5 / size, 1.0) for size in pool_sizes) if ranks else 0.0
        rows.append(
            {
                "baseline": baseline,
                "ranking_pools": len(ranks),
                "recall_at_1": recall_at_1,
                "recall_at_5": recall_at_5,
                "mean_reciprocal_rank": mean(1 / rank for rank in ranks) if ranks else 0.0,
                "top5_enrichment": recall_at_5 / random_top5 if random_top5 else 0.0,
                "minimum_rank": min(ranks, default=0),
                "maximum_rank": max(ranks, default=0),
            }
        )
    return rows


def _baseline_score(
    baseline: str,
    relation: Property,
    *,
    train_relations: set[Property],
    train_coordinates: Counter[tuple[int, int]],
    position_target: float,
) -> float:
    if baseline == "deterministic_hash_random":
        return int(_relation_id(relation)[:16], 16) / float(1 << 64)
    if baseline == "file_id":
        return 0.0
    if baseline == "relation_size":
        target = mean(len(item) for item in train_relations)
        return -abs(len(relation) - target)
    if baseline == "exponent_weight":
        target = mean(
            sum(left.bit_count() + right.bit_count() for left, right in item)
            for item in train_relations
        )
        value = sum(left.bit_count() + right.bit_count() for left, right in relation)
        return -abs(value - target)
    if baseline == "exact_training_frequency":
        return float(relation in train_relations)
    if baseline == "training_coordinate_frequency":
        return float(sum(train_coordinates[coordinate] for coordinate in relation))
    if baseline == "training_support_overlap":
        return float(sum(coordinate in train_coordinates for coordinate in relation))
    if baseline == "absolute_bit_position":
        return -abs(_absolute_position(relation) - position_target)
    raise ValueError(f"unknown E98 baseline: {baseline}")


def _rotation_candidates(
    positive: Property,
    all_known: set[Property],
) -> tuple[Property, ...]:
    variants = {
        frozenset((_rotate64(left, shift), _rotate64(right, shift)) for left, right in positive)
        for shift in range(1, 64)
    }
    variants.discard(positive)
    variants.difference_update(all_known)
    return tuple(sorted(variants, key=_canonical_coordinates))


def _marginals_match(left: Property, right: Property) -> bool:
    return _relation_marginals(left) == _relation_marginals(right)


def _relation_marginals(relation: Property) -> tuple[Any, ...]:
    inputs = sorted({left for left, _ in relation})
    outputs = sorted({right for _, right in relation})
    return (
        len(relation),
        tuple(sorted(left.bit_count() for left, _ in relation)),
        tuple(sorted(right.bit_count() for _, right in relation)),
        len(inputs),
        len(outputs),
        tuple(sorted((left ^ right).bit_count() for left, right in _pairs(inputs))),
        tuple(sorted((left ^ right).bit_count() for left, right in _pairs(outputs))),
        relation == frozenset((left, right) for left in inputs for right in outputs),
    )


def _pairs(values: list[int]) -> Iterable[tuple[int, int]]:
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            yield left, right


def _absolute_position(relation: Property) -> float:
    positions = [
        bit
        for left, right in relation
        for value in (left, right)
        for bit in range(64)
        if value & (1 << bit)
    ]
    return mean(positions) if positions else 0.0


def _rotate64(value: int, shift: int) -> int:
    normalized = shift % 64
    return ((value << normalized) | (value >> (64 - normalized))) & MASK64


def _canonical_coordinates(relation: Property) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(relation))


def _relation_id(relation: Property) -> str:
    payload = json.dumps(_canonical_coordinates(relation), separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
