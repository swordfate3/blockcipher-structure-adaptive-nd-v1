from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, Present80
from blockcipher_nd.tasks.innovation2.integral_property_prediction import (
    IntegralStructure,
    integral_mask_parity,
    make_keys,
)


SELECTORS = (
    "structure_mlp",
    "linear_same_input",
    "p_layer_reachability",
    "fixed_random",
)
SELECTOR_LABELS = {
    "structure_mlp": "结构 MLP",
    "linear_same_input": "同输入线性基线",
    "p_layer_reachability": "P层可达性启发式",
    "fixed_random": "固定随机",
}
SOURCE_DECISION = "innovation2_integral_geometry_holdout_passed"
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class FreshKeyValidationConfig:
    run_id: str
    top_k: int
    fresh_keys: int
    key_seed: int
    random_selector_seed: int
    rounds: int = 5
    gate_mode: str = "fresh-key-enrichment"
    key_chunk_size: int = 256

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not 1 <= self.top_k <= 128:
            raise ValueError("top_k must be between 1 and 128")
        if self.fresh_keys <= 0:
            raise ValueError("fresh_keys must be positive")
        if self.rounds != 5:
            raise ValueError("the frozen E5 protocol requires PRESENT r5")
        if self.gate_mode not in {"fresh-key-smoke", "fresh-key-enrichment"}:
            raise ValueError(
                "gate_mode must be fresh-key-smoke or fresh-key-enrichment"
            )
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")


@dataclass(frozen=True)
class FreshKeyThresholds:
    minimum_candidate_linear_mean_advantage: float = 0.03
    minimum_candidate_reachability_mean_advantage: float = 0.03
    minimum_candidate_random_mean_advantage: float = 0.10
    minimum_candidate_structure_balance_rate: float = 0.75
    minimum_candidate_zero_observed_failure_structures: int = 1


def evaluate_fresh_key_enrichment(
    config: FreshKeyValidationConfig,
    *,
    ranking_rows: list[dict[str, str]],
    ranking_gate: dict[str, Any],
    source_summary: dict[str, Any],
    thresholds: FreshKeyThresholds = FreshKeyThresholds(),
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    readiness = validate_source(ranking_rows, ranking_gate, source_summary)
    structures_by_signature = {
        structure.signature: structure
        for structure in (structure_from_ranking_row(row) for row in ranking_rows)
    }
    selections = select_structures(
        ranking_rows,
        top_k=config.top_k,
        random_selector_seed=config.random_selector_seed,
    )
    selection_counts_valid = all(
        len(selected) == config.top_k for selected in selections.values()
    )
    historical_keys = _historical_keys(source_summary)
    fresh_keys = make_keys(count=config.fresh_keys, seed=config.key_seed)
    key_sets_disjoint = set(fresh_keys).isdisjoint(historical_keys)
    round_keys = present_round_key_matrix(fresh_keys, rounds=config.rounds)

    selected_signatures = {
        signature
        for selector_rows in selections.values()
        for signature in selector_rows
    }
    parities_by_signature: dict[str, np.ndarray] = {}
    for completed, signature in enumerate(sorted(selected_signatures), start=1):
        parities_by_signature[signature] = present_integral_parities(
            structures_by_signature[signature],
            round_keys,
            key_chunk_size=config.key_chunk_size,
        )
        _emit(
            progress_callback,
            "structure_evaluated",
            {
                "structure_signature": signature,
                "completed_structures": completed,
                "total_structures": len(selected_signatures),
                "fresh_keys": config.fresh_keys,
            },
        )

    vectorized_matches_scalar = _crosscheck_vectorized(
        structures_by_signature,
        parities_by_signature,
        fresh_keys,
        selected_signatures,
        rounds=config.rounds,
    )
    readiness.update(
        {
            "four_selector_counts_match_top_k": selection_counts_valid,
            "fresh_keys_disjoint_from_all_source_keys": key_sets_disjoint,
            "vectorized_parity_matches_scalar": vectorized_matches_scalar,
        }
    )

    rate_rows = _build_rate_rows(
        selections,
        structures_by_signature,
        parities_by_signature,
        fresh_keys=config.fresh_keys,
    )
    selector_rows = _summarize_selectors(
        config,
        rate_rows,
        thresholds=thresholds,
    )
    overlap_rows = _selector_overlaps(selections)
    gate = adjudicate_fresh_key_enrichment(
        config,
        selector_rows=selector_rows,
        readiness_checks=readiness,
        thresholds=thresholds,
        source_run_id=str(ranking_gate["run_id"]),
    )
    return {
        "rows": selector_rows,
        "rate_rows": rate_rows,
        "overlap_rows": overlap_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "source_run_id": str(ranking_gate["run_id"]),
            "top_k": config.top_k,
            "fresh_keys": config.fresh_keys,
            "key_seed": config.key_seed,
            "random_selector_seed": config.random_selector_seed,
            "fresh_key_fingerprints": [f"{key:020X}" for key in fresh_keys],
            "historical_key_count": len(historical_keys),
            "unique_selected_structures": len(selected_signatures),
            "claim_scope": gate["claim_scope"],
        },
    }


def validate_source(
    ranking_rows: list[dict[str, str]],
    ranking_gate: dict[str, Any],
    source_summary: dict[str, Any],
) -> dict[str, bool]:
    required_columns = {
        "structure_id",
        "signature",
        "active_nibble",
        "output_nibble",
        "output_mask",
        "anchor_rank",
        "candidate_rank",
    }
    rows_complete = bool(ranking_rows) and all(
        required_columns.issubset(row) for row in ranking_rows
    )
    structures: list[IntegralStructure] = []
    if rows_complete:
        structures = [structure_from_ranking_row(row) for row in ranking_rows]
    signatures = [structure.signature for structure in structures]
    geometries = [structure.geometry_id for structure in structures]
    split_mode = str(source_summary.get("structure_split_mode", ""))
    return {
        "source_gate_is_geometry_holdout_pass": (
            ranking_gate.get("status") == "pass"
            and ranking_gate.get("decision") == SOURCE_DECISION
            and ranking_gate.get("structure_split_mode") == "geometry-disjoint"
        ),
        "source_has_exactly_128_rows": len(ranking_rows) == 128,
        "source_columns_complete": rows_complete,
        "source_signatures_unique": len(signatures) == len(set(signatures)),
        "source_geometries_unique": len(geometries) == len(set(geometries)),
        "source_summary_is_geometry_disjoint": (
            split_mode == "geometry-disjoint"
            and source_summary.get("geometry_splits_disjoint") is True
            and source_summary.get("one_structure_per_geometry") is True
        ),
    }


def select_structures(
    ranking_rows: list[dict[str, str]],
    *,
    top_k: int,
    random_selector_seed: int,
) -> dict[str, tuple[str, ...]]:
    if top_k > len(ranking_rows):
        raise ValueError("top_k cannot exceed the number of ranking rows")
    structures = [structure_from_ranking_row(row) for row in ranking_rows]
    by_signature = {
        structure.signature: (row, structure)
        for row, structure in zip(ranking_rows, structures, strict=True)
    }
    candidate = sorted(
        by_signature,
        key=lambda signature: (
            float(by_signature[signature][0]["candidate_rank"]),
            by_signature[signature][1].structure_id,
        ),
    )[:top_k]
    linear = sorted(
        by_signature,
        key=lambda signature: (
            float(by_signature[signature][0]["anchor_rank"]),
            by_signature[signature][1].structure_id,
        ),
    )[:top_k]
    reachability = sorted(
        by_signature,
        key=lambda signature: _reachability_sort_key(by_signature[signature][1]),
    )[:top_k]
    rng = np.random.default_rng(random_selector_seed)
    random_indices = rng.choice(len(structures), size=top_k, replace=False)
    random_selected = sorted(
        (structures[int(index)] for index in random_indices),
        key=lambda structure: structure.structure_id,
    )
    return {
        "structure_mlp": tuple(candidate),
        "linear_same_input": tuple(linear),
        "p_layer_reachability": tuple(reachability),
        "fixed_random": tuple(
            structure.signature for structure in random_selected
        ),
    }


def p_layer_reachability_score(
    structure: IntegralStructure,
    *,
    rounds: int = 5,
) -> tuple[int, float, int]:
    reachable = {4 * structure.active_nibble + bit for bit in range(4)}
    earliest = [-1] * 64
    for bit in reachable:
        earliest[bit] = 0
    for round_index in range(1, rounds + 1):
        expanded = {
            4 * (bit // 4) + output_bit
            for bit in reachable
            for output_bit in range(4)
        }
        reachable = {
            (16 * bit) % 63 if bit < 63 else 63
            for bit in expanded
        }
        for bit in reachable:
            if earliest[bit] < 0:
                earliest[bit] = round_index
    selected_rounds = [
        earliest[4 * structure.output_nibble + bit]
        for bit in range(4)
        if structure.output_mask & (1 << bit)
    ]
    return (
        min(selected_rounds),
        float(sum(selected_rounds) / len(selected_rounds)),
        structure.output_mask.bit_count(),
    )


def present_round_key_matrix(
    keys: tuple[int, ...],
    *,
    rounds: int,
) -> np.ndarray:
    matrix = np.zeros((rounds + 1, len(keys)), dtype=np.uint64)
    for key_index, key in enumerate(keys):
        key_register = key & ((1 << 80) - 1)
        for round_counter in range(1, rounds + 1):
            matrix[round_counter - 1, key_index] = np.uint64(
                key_register >> 16
            )
            key_register = Present80._update_key(key_register, round_counter)
        matrix[rounds, key_index] = np.uint64(key_register >> 16)
    return matrix


def present_integral_parities(
    structure: IntegralStructure,
    round_keys: np.ndarray,
    *,
    key_chunk_size: int = 256,
) -> np.ndarray:
    if round_keys.ndim != 2 or round_keys.shape[0] < 2:
        raise ValueError("round_keys must have shape (rounds + 1, key_count)")
    rounds = round_keys.shape[0] - 1
    plaintexts = np.asarray(
        [structure.plaintext(value) for value in range(16)],
        dtype=np.uint64,
    )
    parities = np.zeros(round_keys.shape[1], dtype=np.uint8)
    sbox = np.asarray(PRESENT_SBOX, dtype=np.uint64)
    nibble_parity = np.asarray(
        [(value & structure.output_mask).bit_count() & 1 for value in range(16)],
        dtype=np.uint8,
    )
    output_shift = np.uint64(4 * structure.output_nibble)
    for start in range(0, round_keys.shape[1], key_chunk_size):
        stop = min(start + key_chunk_size, round_keys.shape[1])
        states = np.broadcast_to(
            plaintexts,
            (stop - start, len(plaintexts)),
        ).copy()
        for round_index in range(rounds):
            states ^= round_keys[round_index, start:stop, None]
            states = _present_sbox_layer_words(states, sbox)
            states = _present_permutation_layer_words(states)
        states ^= round_keys[rounds, start:stop, None]
        xor_words = np.bitwise_xor.reduce(states, axis=1)
        output_nibbles = ((xor_words >> output_shift) & np.uint64(0xF)).astype(
            np.uint8
        )
        parities[start:stop] = nibble_parity[output_nibbles]
    return parities


def present_integral_parity_matrix(
    structures: tuple[IntegralStructure, ...],
    round_keys: np.ndarray,
    *,
    structure_chunk_size: int = 16,
) -> np.ndarray:
    if not structures:
        raise ValueError("structures must be non-empty")
    if round_keys.ndim != 2 or round_keys.shape[0] < 2:
        raise ValueError("round_keys must have shape (rounds + 1, key_count)")
    if structure_chunk_size <= 0:
        raise ValueError("structure_chunk_size must be positive")
    rounds = round_keys.shape[0] - 1
    result = np.zeros((len(structures), round_keys.shape[1]), dtype=np.uint8)
    sbox = np.asarray(PRESENT_SBOX, dtype=np.uint64)
    parity_lookup = np.asarray(
        [value.bit_count() & 1 for value in range(16)],
        dtype=np.uint8,
    )
    for start in range(0, len(structures), structure_chunk_size):
        stop = min(start + structure_chunk_size, len(structures))
        chunk = structures[start:stop]
        plaintexts = np.asarray(
            [
                [structure.plaintext(value) for value in range(16)]
                for structure in chunk
            ],
            dtype=np.uint64,
        )
        states = np.broadcast_to(
            plaintexts,
            (round_keys.shape[1], *plaintexts.shape),
        ).copy()
        for round_index in range(rounds):
            states ^= round_keys[round_index, :, None, None]
            states = _present_sbox_layer_words(states, sbox)
            states = _present_permutation_layer_words(states)
        states ^= round_keys[rounds, :, None, None]
        xor_words = np.bitwise_xor.reduce(states, axis=2)
        shifts = np.asarray(
            [4 * structure.output_nibble for structure in chunk],
            dtype=np.uint64,
        )
        masks = np.asarray(
            [structure.output_mask for structure in chunk],
            dtype=np.uint64,
        )
        nibbles = (xor_words >> shifts[None, :]) & np.uint64(0xF)
        masked = (nibbles & masks[None, :]).astype(np.uint8)
        result[start:stop] = parity_lookup[masked].T
    return result


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - radius, center + radius


def zero_failure_upper_bound(trials: int, *, alpha: float = 0.05) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    return 1.0 - alpha ** (1.0 / trials)


def adjudicate_fresh_key_enrichment(
    config: FreshKeyValidationConfig,
    *,
    selector_rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
    thresholds: FreshKeyThresholds,
    source_run_id: str,
) -> dict[str, Any]:
    by_selector = {str(row["selector"]): row for row in selector_rows}
    selectors_complete = set(by_selector) == set(SELECTORS)
    candidate = by_selector.get("structure_mlp", {})
    linear = by_selector.get("linear_same_input", {})
    reachability = by_selector.get("p_layer_reachability", {})
    random = by_selector.get("fixed_random", {})
    finite_metrics = selectors_complete and all(
        math.isfinite(float(row[metric]))
        for row in selector_rows
        for metric in (
            "mean_balance_rate",
            "minimum_balance_rate",
            "maximum_balance_rate",
        )
    )
    readiness = {
        **readiness_checks,
        "four_selector_summaries_present": selectors_complete,
        "selector_metrics_finite": finite_metrics,
    }
    candidate_linear = _difference(
        candidate, linear, "mean_balance_rate"
    )
    candidate_reachability = _difference(
        candidate, reachability, "mean_balance_rate"
    )
    candidate_random = _difference(
        candidate, random, "mean_balance_rate"
    )
    enrichment_checks = {
        "candidate_linear_mean_advantage_at_least_0_03": (
            candidate_linear >= thresholds.minimum_candidate_linear_mean_advantage
        ),
        "candidate_reachability_mean_advantage_at_least_0_03": (
            candidate_reachability
            >= thresholds.minimum_candidate_reachability_mean_advantage
        ),
        "candidate_random_mean_advantage_at_least_0_10": (
            candidate_random >= thresholds.minimum_candidate_random_mean_advantage
        ),
        "candidate_minimum_structure_balance_at_least_0_75": (
            float(candidate.get("minimum_balance_rate", float("-inf")))
            >= thresholds.minimum_candidate_structure_balance_rate
        ),
    }
    zero_failure_check = (
        int(candidate.get("zero_observed_failure_structures", 0))
        >= thresholds.minimum_candidate_zero_observed_failure_structures
    )
    readiness_passed = all(readiness.values())
    if config.gate_mode == "fresh-key-smoke":
        status = "pass" if readiness_passed else "fail"
        decision = (
            "innovation2_integral_fresh_key_implementation_ready"
            if readiness_passed
            else "innovation2_integral_fresh_key_smoke_invalid"
        )
        next_action = (
            "Run the frozen 4096-fresh-key E5 enrichment matrix."
            if readiness_passed
            else "Repair source ownership, selectors, or vectorized parity before E5."
        )
    elif not readiness_passed:
        status = "fail"
        decision = "innovation2_integral_fresh_key_protocol_invalid"
        next_action = "Repair E5 protocol validity; do not interpret selector metrics."
    elif all(enrichment_checks.values()) and zero_failure_check:
        status = "pass"
        decision = "innovation2_integral_fresh_key_enrichment_passed"
        next_action = (
            "Freeze Innovation 2 experimental evidence and draft the thesis method, "
            "same-budget controls, geometry-generalization result, fresh-key result, "
            "and deterministic-proof limitation. Do not launch remote scale."
        )
    elif all(enrichment_checks.values()):
        status = "hold"
        decision = "innovation2_integral_fresh_key_ranking_only"
        next_action = (
            "Keep the empirical ranking contribution, state that no zero-failure "
            "candidate was observed, and move to thesis writing without more scaling."
        )
    else:
        status = "hold"
        decision = "innovation2_integral_fresh_key_enrichment_not_confirmed"
        next_action = (
            "Keep E4 as a 256-key geometry diagnostic only. Do not claim 4096-key "
            "robustness and do not tune selectors on the observed E5 keys."
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "gate_mode": config.gate_mode,
        "source_run_id": source_run_id,
        "readiness_checks": readiness,
        "enrichment_checks": enrichment_checks,
        "zero_failure_check": zero_failure_check,
        "metrics": {
            "candidate_mean_balance_rate": candidate.get("mean_balance_rate"),
            "linear_mean_balance_rate": linear.get("mean_balance_rate"),
            "reachability_mean_balance_rate": reachability.get("mean_balance_rate"),
            "random_mean_balance_rate": random.get("mean_balance_rate"),
            "candidate_linear_mean_advantage": candidate_linear,
            "candidate_reachability_mean_advantage": candidate_reachability,
            "candidate_random_mean_advantage": candidate_random,
            "candidate_minimum_balance_rate": candidate.get("minimum_balance_rate"),
            "candidate_zero_observed_failure_structures": candidate.get(
                "zero_observed_failure_structures"
            ),
        },
        "thresholds": {
            "minimum_candidate_linear_mean_advantage": (
                thresholds.minimum_candidate_linear_mean_advantage
            ),
            "minimum_candidate_reachability_mean_advantage": (
                thresholds.minimum_candidate_reachability_mean_advantage
            ),
            "minimum_candidate_random_mean_advantage": (
                thresholds.minimum_candidate_random_mean_advantage
            ),
            "minimum_candidate_structure_balance_rate": (
                thresholds.minimum_candidate_structure_balance_rate
            ),
            "minimum_candidate_zero_observed_failure_structures": (
                thresholds.minimum_candidate_zero_observed_failure_structures
            ),
        },
        "next_action": next_action,
        "claim_scope": (
            f"local PRESENT-r5 empirical selector enrichment on {config.fresh_keys} "
            "uniform independent fresh keys; zero observed failures provide a binomial "
            "confidence bound, not a deterministic all-key integral proof"
        ),
    }


def structure_from_ranking_row(row: dict[str, str]) -> IntegralStructure:
    signature = str(row["signature"])
    try:
        fixed_plaintext = int(signature.rsplit("-p", 1)[1], 16)
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid structure signature: {signature}") from error
    structure = IntegralStructure(
        structure_id=str(row["structure_id"]),
        active_nibble=int(row["active_nibble"]),
        output_nibble=int(row["output_nibble"]),
        output_mask=int(str(row["output_mask"]), 2),
        fixed_plaintext=fixed_plaintext,
    )
    if structure.signature != signature:
        raise ValueError(
            f"ranking row does not match its structure signature: {signature}"
        )
    return structure


def _reachability_sort_key(structure: IntegralStructure) -> tuple[Any, ...]:
    minimum_round, mean_round, mask_weight = p_layer_reachability_score(structure)
    return (-minimum_round, -mean_round, mask_weight, structure.structure_id)


def _present_sbox_layer_words(
    states: np.ndarray,
    sbox: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(states)
    for nibble in range(16):
        shift = np.uint64(4 * nibble)
        values = ((states >> shift) & np.uint64(0xF)).astype(np.uint8)
        output |= sbox[values] << shift
    return output


def _present_permutation_layer_words(states: np.ndarray) -> np.ndarray:
    output = np.zeros_like(states)
    for bit in range(64):
        target = (16 * bit) % 63 if bit < 63 else 63
        output |= ((states >> np.uint64(bit)) & np.uint64(1)) << np.uint64(
            target
        )
    return output


def _crosscheck_vectorized(
    structures: dict[str, IntegralStructure],
    parities: dict[str, np.ndarray],
    fresh_keys: tuple[int, ...],
    selected_signatures: set[str],
    *,
    rounds: int,
) -> bool:
    for signature in sorted(selected_signatures)[:3]:
        structure = structures[signature]
        for key_index, key in enumerate(fresh_keys[:8]):
            expected = integral_mask_parity(
                Present80(rounds=rounds, key=key),
                structure,
            )
            if int(parities[signature][key_index]) != expected:
                return False
    return True


def _historical_keys(source_summary: dict[str, Any]) -> set[int]:
    keys: set[int] = set()
    for split in source_summary.get("splits", {}).values():
        for key in split.get("keys", []):
            keys.add(int(str(key), 16))
    stability = source_summary.get("stability", {})
    for key in stability.get("keys", []):
        keys.add(int(str(key), 16))
    return keys


def _build_rate_rows(
    selections: dict[str, tuple[str, ...]],
    structures: dict[str, IntegralStructure],
    parities: dict[str, np.ndarray],
    *,
    fresh_keys: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    upper_bound = zero_failure_upper_bound(fresh_keys)
    for selector in SELECTORS:
        for selection_rank, signature in enumerate(selections[selector], start=1):
            structure = structures[signature]
            q1_count = int(parities[signature].sum())
            balance_count = fresh_keys - q1_count
            balance_rate = balance_count / fresh_keys
            lower, upper = wilson_interval(balance_count, fresh_keys)
            reachability = p_layer_reachability_score(structure)
            rows.append(
                {
                    "selector": selector,
                    "selector_label": SELECTOR_LABELS[selector],
                    "selection_rank": selection_rank,
                    "structure_id": structure.structure_id,
                    "signature": signature,
                    "geometry_id": structure.geometry_id,
                    "active_nibble": structure.active_nibble,
                    "output_nibble": structure.output_nibble,
                    "output_mask": f"{structure.output_mask:04b}",
                    "fresh_keys": fresh_keys,
                    "q1_count": q1_count,
                    "balance_count": balance_count,
                    "balance_rate": balance_rate,
                    "balance_wilson95_lower": lower,
                    "balance_wilson95_upper": upper,
                    "zero_observed_failure": q1_count == 0,
                    "zero_failure_q1_upper95": (
                        upper_bound if q1_count == 0 else None
                    ),
                    "reachability_minimum_round": reachability[0],
                    "reachability_mean_round": reachability[1],
                    "output_mask_weight": reachability[2],
                }
            )
    return rows


def _summarize_selectors(
    config: FreshKeyValidationConfig,
    rate_rows: list[dict[str, Any]],
    *,
    thresholds: FreshKeyThresholds,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for selector in SELECTORS:
        selected = [row for row in rate_rows if row["selector"] == selector]
        rates = np.asarray([row["balance_rate"] for row in selected], dtype=np.float64)
        summaries.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_integral_fresh_key_enrichment",
                "selector": selector,
                "selector_label": SELECTOR_LABELS[selector],
                "top_k": config.top_k,
                "fresh_keys": config.fresh_keys,
                "key_seed": config.key_seed,
                "random_selector_seed": config.random_selector_seed,
                "mean_balance_rate": float(rates.mean()),
                "median_balance_rate": float(np.median(rates)),
                "minimum_balance_rate": float(rates.min()),
                "maximum_balance_rate": float(rates.max()),
                "zero_observed_failure_structures": sum(
                    bool(row["zero_observed_failure"]) for row in selected
                ),
                "minimum_candidate_linear_mean_advantage": (
                    thresholds.minimum_candidate_linear_mean_advantage
                ),
                "minimum_candidate_reachability_mean_advantage": (
                    thresholds.minimum_candidate_reachability_mean_advantage
                ),
                "minimum_candidate_random_mean_advantage": (
                    thresholds.minimum_candidate_random_mean_advantage
                ),
                "claim_scope": (
                    "fresh-key empirical selector validation only; not an all-key proof"
                ),
            }
        )
    return summaries


def _selector_overlaps(
    selections: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(SELECTORS, 2):
        shared = sorted(set(selections[left]) & set(selections[right]))
        rows.append(
            {
                "left_selector": left,
                "right_selector": right,
                "overlap_count": len(shared),
                "shared_signatures": ";".join(shared),
            }
        )
    return rows


def _difference(
    left: dict[str, Any],
    right: dict[str, Any],
    metric: str,
) -> float:
    try:
        return float(left[metric]) - float(right[metric])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _emit(
    callback: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event, payload)
