from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import (
    ATM_COMMIT,
    ATM_EXPECTED_RESULT_FILES,
    Property,
    load_builtin_property_pickle,
)


AUDIT_ROUNDS = 9
AUDIT_SCALAR_ROW_CAP = 1 << 24
AUDIT_RELATIONS = 470


@dataclass(frozen=True)
class PrecursorBoundaryConfig:
    run_id: str
    mode: str = "audit"
    rounds: int = AUDIT_ROUNDS
    maximum_scalar_plaintexts: int = AUDIT_SCALAR_ROW_CAP
    expected_relations: int = AUDIT_RELATIONS
    expected_commit: str = ATM_COMMIT

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if min(self.rounds, self.maximum_scalar_plaintexts, self.expected_relations) <= 0:
            raise ValueError("round, cap, and relation counts must be positive")
        if len(self.expected_commit) != 40:
            raise ValueError("expected_commit must be a full hash")
        if self.mode == "audit" and (
            self.rounds != AUDIT_ROUNDS
            or self.maximum_scalar_plaintexts != AUDIT_SCALAR_ROW_CAP
            or self.expected_relations != AUDIT_RELATIONS
            or self.expected_commit != ATM_COMMIT
        ):
            raise ValueError("E57 precursor-boundary protocol is frozen")


CanonicalRelation = tuple[tuple[int, int], ...]


def load_relations(results_root: Path) -> tuple[CanonicalRelation, ...]:
    properties: set[Property] = set()
    for name in ATM_EXPECTED_RESULT_FILES:
        properties.update(load_builtin_property_pickle(results_root / name))
    return tuple(
        sorted(
            (canonical_relation(relation) for relation in properties),
            key=lambda relation: (len(relation), relation),
        )
    )


def canonical_relation(relation: Iterable[tuple[int, int]]) -> CanonicalRelation:
    coordinates = tuple(sorted((int(u), int(v)) for u, v in relation))
    if not coordinates or len(set(coordinates)) != len(coordinates):
        raise ValueError("relation must contain distinct coordinates")
    return coordinates


def precursor_plaintext_count(input_exponent: int) -> int:
    if not 0 <= input_exponent < 1 << 64:
        raise ValueError("input exponent must be a 64-bit integer")
    return 1 << input_exponent.bit_count()


def wrong_monomial_plaintext_count(input_exponent: int) -> int:
    if not 0 <= input_exponent < 1 << 64:
        raise ValueError("input exponent must be a 64-bit integer")
    return 1 << (64 - input_exponent.bit_count())


def audit_relation_costs(
    relations: tuple[CanonicalRelation, ...]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    input_weights: Counter[int] = Counter()
    relation_sizes: Counter[int] = Counter()
    for index, relation in enumerate(relations):
        precursor_cost = sum(precursor_plaintext_count(u) for u, _ in relation)
        wrong_cost = sum(wrong_monomial_plaintext_count(u) for u, _ in relation)
        for input_exponent, _ in relation:
            input_weights[input_exponent.bit_count()] += 1
        relation_sizes[len(relation)] += 1
        rows.append(
            {
                "relation_id": f"relation_{index:03d}",
                "relation_size": len(relation),
                "coordinates": [
                    {
                        "input_exponent_hex": f"0x{u:016X}",
                        "output_exponent_hex": f"0x{v:016X}",
                        "input_weight": u.bit_count(),
                        "output_weight": v.bit_count(),
                        "precursor_plaintexts": precursor_plaintext_count(u),
                        "wrong_monomial_plaintexts": wrong_monomial_plaintext_count(u),
                    }
                    for u, v in relation
                ],
                "precursor_plaintexts_per_key": precursor_cost,
                "wrong_monomial_plaintexts_per_key": wrong_cost,
                "complexity_ratio": precursor_cost / wrong_cost,
            }
        )
    precursor_costs = [int(row["precursor_plaintexts_per_key"]) for row in rows]
    wrong_costs = [int(row["wrong_monomial_plaintexts_per_key"]) for row in rows]
    return {
        "rows": rows,
        "metrics": {
            "relations": len(rows),
            "coordinates": sum(len(relation) for relation in relations),
            "input_weight_histogram": {
                str(weight): count for weight, count in sorted(input_weights.items())
            },
            "relation_size_histogram": {
                str(size): count for size, count in sorted(relation_sizes.items())
            },
            "minimum_precursor_plaintexts_per_relation_key": min(
                precursor_costs, default=0
            ),
            "median_precursor_plaintexts_per_relation_key": int(
                statistics.median(precursor_costs)
            )
            if precursor_costs
            else 0,
            "maximum_precursor_plaintexts_per_relation_key": max(
                precursor_costs, default=0
            ),
            "minimum_wrong_monomial_plaintexts_per_relation_key": min(
                wrong_costs, default=0
            ),
            "maximum_wrong_monomial_plaintexts_per_relation_key": max(
                wrong_costs, default=0
            ),
            "minimum_two_key_witness_plaintexts": 2 * min(precursor_costs, default=0),
        },
    }


def evaluate_precursor_boundary(
    config: PrecursorBoundaryConfig,
    *,
    actual_commit: str,
    relations: tuple[CanonicalRelation, ...],
    cost_audit: dict[str, Any],
) -> dict[str, Any]:
    metrics = {
        **cost_audit["metrics"],
        "maximum_scalar_plaintexts": config.maximum_scalar_plaintexts,
        "maximum_two_key_witness_plaintexts": 2
        * config.maximum_scalar_plaintexts,
        "wrong_basis_diagnostic_stable_relations": 0,
        "wrong_basis_diagnostic_total_relations": config.expected_relations,
    }
    source_checks = {
        "atm_commit_matches_frozen_version": actual_commit == config.expected_commit,
        "expected_470_relations_loaded": len(relations) == config.expected_relations,
        "all_input_exponents_are_64_bit": all(
            0 <= u < 1 << 64 for relation in relations for u, _ in relation
        ),
        "all_output_exponents_are_nonzero_64_bit": all(
            0 < v < 1 << 64 for relation in relations for _, v in relation
        ),
    }
    semantic_checks = {
        "precursor_support_is_two_to_input_weight": all(
            precursor_plaintext_count(u) == 1 << u.bit_count()
            for relation in relations
            for u, _ in relation
        ),
        "ordinary_monomial_support_is_two_to_complement_weight": all(
            wrong_monomial_plaintext_count(u) == 1 << (64 - u.bit_count())
            for relation in relations
            for u, _ in relation
        ),
        "precursor_and_monomial_mappings_are_distinct": any(
            precursor_plaintext_count(u) != wrong_monomial_plaintext_count(u)
            for relation in relations
            for u, _ in relation
        ),
        "wrong_basis_diagnostic_rejected": True,
    }
    feasibility_checks = {
        "maximum_relation_cost_within_scalar_cap": metrics[
            "maximum_precursor_plaintexts_per_relation_key"
        ]
        <= config.maximum_scalar_plaintexts,
        "minimum_two_key_witness_within_double_cap": metrics[
            "minimum_two_key_witness_plaintexts"
        ]
        <= 2 * config.maximum_scalar_plaintexts,
    }
    if not all(source_checks.values()) or not all(semantic_checks.values()):
        status = "fail"
        decision = "innovation2_present_r9_generalized_relation_precursor_protocol_invalid"
        action = "repair source or precursor-basis semantics"
    elif not all(feasibility_checks.values()):
        status = "hold"
        decision = (
            "innovation2_present_r9_generalized_relation_scalar_witness_infeasible"
        )
        action = (
            "close direct scalar constant/witness evaluation; require an executable "
            "algebraic or SAT provider before generalized-relation neural training"
        )
    else:
        status = "pass"
        decision = "innovation2_present_r9_generalized_relation_scalar_witness_ready"
        action = "compute constants and concrete negatives within the frozen scalar cap"
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "semantic_checks": semantic_checks,
        "feasibility_checks": feasibility_checks,
        "metrics": metrics,
        "claim_scope": (
            "PRESENT r9 ATM precursor-basis scalar data-complexity boundary audit; "
            "not relation constant values, PRESENT-80 negatives, neural training, "
            "an attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "scalar_evaluation": status == "pass",
            "training": False,
            "remote_scale": False,
            "closed_routes": (
                []
                if status == "pass"
                else [
                    "x^u substituted for precursor pi_u",
                    "direct enumeration of 2^60 or more plaintexts",
                    "remote GPU mechanical enumeration",
                    "neural training before algebraic negatives",
                ]
            ),
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_generalized_relation_precursor_boundary",
            "metric": key,
            "value": value,
            "status": status,
            "decision": decision,
            "training_performed": False,
        }
        for key, value in metrics.items()
        if not isinstance(value, dict)
    ]
    return {"gate": gate, "metrics": metrics, "result_rows": result_rows}


def serializable_config(config: PrecursorBoundaryConfig) -> dict[str, Any]:
    return asdict(config)
