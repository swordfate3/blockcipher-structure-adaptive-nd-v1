from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    SOURCE_DECISION,
    present_integral_parity_matrix,
    present_round_key_matrix,
    structure_from_ranking_row,
    validate_source,
    wilson_interval,
    zero_failure_upper_bound,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import (
    IntegralStructure,
    integral_mask_parity,
    make_keys,
    make_structure_splits,
)


SELECTORS = (
    "structure_mlp",
    "train_output_position_prior",
    "position_matched_linear",
    "position_matched_random",
)
SELECTOR_LABELS = {
    "structure_mlp": "结构 MLP",
    "train_output_position_prior": "训练集输出位置先验",
    "position_matched_linear": "位置匹配线性基线",
    "position_matched_random": "位置匹配随机基线",
}
SPLIT_NAMES = ("train", "validation", "calibration", "test")
STRUCTURE_SEED_OFFSETS = {
    "train": 101,
    "validation": 301,
    "calibration": 701,
    "test": 501,
}
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class PositionPriorAuditConfig:
    run_id: str
    top_k: int
    fresh_keys: int
    key_seed: int
    matched_random_seed: int
    experiment_seed: int = 0
    rounds: int = 5
    gate_mode: str = "position-prior-audit"
    structure_chunk_size: int = 16

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not 1 <= self.top_k <= 128:
            raise ValueError("top_k must be between 1 and 128")
        if self.fresh_keys <= 0:
            raise ValueError("fresh_keys must be positive")
        if self.rounds != 5:
            raise ValueError("the frozen E6 protocol requires PRESENT r5")
        if self.gate_mode not in {"position-prior-smoke", "position-prior-audit"}:
            raise ValueError(
                "gate_mode must be position-prior-smoke or position-prior-audit"
            )
        if self.structure_chunk_size <= 0:
            raise ValueError("structure_chunk_size must be positive")


@dataclass(frozen=True)
class PositionPriorThresholds:
    minimum_candidate_position_prior_advantage: float = 0.03
    minimum_candidate_matched_linear_advantage: float = 0.02
    minimum_candidate_matched_random_advantage: float = 0.03


def evaluate_position_prior_audit(
    config: PositionPriorAuditConfig,
    *,
    ranking_rows: list[dict[str, str]],
    ranking_gate: dict[str, Any],
    source_summary: dict[str, Any],
    thresholds: PositionPriorThresholds = PositionPriorThresholds(),
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    readiness = validate_source(ranking_rows, ranking_gate, source_summary)
    reconstructed = reconstruct_source_structures(
        source_summary,
        experiment_seed=config.experiment_seed,
    )
    geometry_reconstruction_matches = all(
        {structure.geometry_id for structure in reconstructed[name]}
        == set(source_summary["splits"][name]["geometry_ids"])
        for name in SPLIT_NAMES
    )
    train_key_count = int(source_summary["splits"]["train"]["keys_per_structure"])
    train_keys = make_keys(
        count=train_key_count,
        seed=config.experiment_seed + 201,
    )
    train_keys_match = [f"{key:020X}" for key in train_keys] == list(
        source_summary["splits"]["train"]["keys"]
    )
    train_round_keys = present_round_key_matrix(train_keys, rounds=config.rounds)
    train_parities = present_integral_parity_matrix(
        reconstructed["train"],
        train_round_keys,
        structure_chunk_size=config.structure_chunk_size,
    )
    position_priors = training_output_position_priors(
        reconstructed["train"],
        train_parities,
    )
    _emit(
        progress_callback,
        "training_position_priors_built",
        {
            "train_structures": len(reconstructed["train"]),
            "train_keys": len(train_keys),
            "position_priors": position_priors,
        },
    )

    structures_by_signature = {
        structure.signature: structure
        for structure in (
            structure_from_ranking_row(row) for row in ranking_rows
        )
    }
    ranking_test_geometries_match = {
        structure.geometry_id for structure in structures_by_signature.values()
    } == {structure.geometry_id for structure in reconstructed["test"]}
    selections = select_position_prior_controls(
        ranking_rows,
        position_priors=position_priors,
        top_k=config.top_k,
        matched_random_seed=config.matched_random_seed,
    )
    candidate_histogram = _position_histogram(
        selections["structure_mlp"],
        structures_by_signature,
    )
    matched_histograms_valid = all(
        _position_histogram(selections[selector], structures_by_signature)
        == candidate_histogram
        for selector in ("position_matched_linear", "position_matched_random")
    )
    selection_counts_valid = all(
        len(selected) == config.top_k for selected in selections.values()
    )

    fresh_keys = make_keys(count=config.fresh_keys, seed=config.key_seed)
    historical_keys = _historical_keys(source_summary)
    fresh_keys_disjoint = set(fresh_keys).isdisjoint(historical_keys)
    selected_signatures = sorted(
        {
            signature
            for selected in selections.values()
            for signature in selected
        }
    )
    selected_structures = tuple(
        structures_by_signature[signature] for signature in selected_signatures
    )
    fresh_round_keys = present_round_key_matrix(fresh_keys, rounds=config.rounds)
    fresh_parities = present_integral_parity_matrix(
        selected_structures,
        fresh_round_keys,
        structure_chunk_size=config.structure_chunk_size,
    )
    parities_by_signature = {
        signature: fresh_parities[index]
        for index, signature in enumerate(selected_signatures)
    }
    vectorized_matches_scalar = _crosscheck(
        selected_structures,
        fresh_parities,
        fresh_keys,
        rounds=config.rounds,
    )
    _emit(
        progress_callback,
        "fresh_key_matrix_built",
        {
            "unique_selected_structures": len(selected_structures),
            "fresh_keys": len(fresh_keys),
        },
    )

    readiness.update(
        {
            "source_geometry_reconstruction_matches": (
                geometry_reconstruction_matches
            ),
            "source_train_keys_reconstruction_matches": train_keys_match,
            "ranking_test_geometries_match_reconstruction": (
                ranking_test_geometries_match
            ),
            "all_output_positions_present_in_training": (
                set(position_priors) == set(range(16))
            ),
            "four_selector_counts_match_top_k": selection_counts_valid,
            "matched_selector_position_histograms_match_candidate": (
                matched_histograms_valid
            ),
            "fresh_keys_disjoint_from_all_source_keys": fresh_keys_disjoint,
            "vectorized_parity_matches_scalar": vectorized_matches_scalar,
        }
    )
    rate_rows = _rate_rows(
        selections,
        structures_by_signature,
        parities_by_signature,
        fresh_keys=config.fresh_keys,
    )
    selector_rows = _selector_rows(config, rate_rows)
    overlap_rows = _overlap_rows(selections)
    position_rows = [
        {
            "output_nibble": position,
            "train_q1_rate": position_priors[position],
            "train_balance_rate": 1.0 - position_priors[position],
            "train_observations": sum(
                len(train_keys)
                for structure in reconstructed["train"]
                if structure.output_nibble == position
            ),
        }
        for position in range(16)
    ]
    gate = adjudicate_position_prior_audit(
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
        "position_rows": position_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "source_run_id": str(ranking_gate["run_id"]),
            "source_decision": SOURCE_DECISION,
            "top_k": config.top_k,
            "fresh_keys": config.fresh_keys,
            "key_seed": config.key_seed,
            "matched_random_seed": config.matched_random_seed,
            "experiment_seed": config.experiment_seed,
            "candidate_position_histogram": candidate_histogram,
            "position_priors": position_priors,
            "unique_selected_structures": len(selected_structures),
            "claim_scope": gate["claim_scope"],
        },
    }


def reconstruct_source_structures(
    source_summary: dict[str, Any],
    *,
    experiment_seed: int,
) -> dict[str, tuple[IntegralStructure, ...]]:
    split_counts = {
        name: int(source_summary["splits"][name]["structures"])
        for name in SPLIT_NAMES
    }
    return make_structure_splits(
        split_counts=split_counts,
        seed=experiment_seed,
        structure_split_mode="geometry-disjoint",
        random_seed_offsets=STRUCTURE_SEED_OFFSETS,
    )


def training_output_position_priors(
    structures: tuple[IntegralStructure, ...],
    parity_matrix: np.ndarray,
) -> dict[int, float]:
    if parity_matrix.shape[0] != len(structures):
        raise ValueError("parity matrix rows must align with structures")
    priors: dict[int, float] = {}
    for output_nibble in range(16):
        indices = [
            index
            for index, structure in enumerate(structures)
            if structure.output_nibble == output_nibble
        ]
        if not indices:
            raise ValueError(
                f"training structures do not cover output nibble {output_nibble}"
            )
        priors[output_nibble] = float(parity_matrix[indices].mean())
    return priors


def select_position_prior_controls(
    ranking_rows: list[dict[str, str]],
    *,
    position_priors: dict[int, float],
    top_k: int,
    matched_random_seed: int,
) -> dict[str, tuple[str, ...]]:
    structures = [structure_from_ranking_row(row) for row in ranking_rows]
    records = list(zip(ranking_rows, structures, strict=True))
    candidate_records = sorted(
        records,
        key=lambda item: (float(item[0]["candidate_rank"]), item[1].structure_id),
    )[:top_k]
    candidate = tuple(item[1].signature for item in candidate_records)
    position_prior = tuple(
        item[1].signature
        for item in sorted(
            records,
            key=lambda item: (
                position_priors[item[1].output_nibble],
                item[1].structure_id,
            ),
        )[:top_k]
    )
    candidate_counts = Counter(item[1].output_nibble for item in candidate_records)
    matched_linear: list[IntegralStructure] = []
    matched_random: list[IntegralStructure] = []
    rng = np.random.default_rng(matched_random_seed)
    for output_nibble in sorted(candidate_counts):
        count = candidate_counts[output_nibble]
        pool = [
            item for item in records if item[1].output_nibble == output_nibble
        ]
        matched_linear.extend(
            item[1]
            for item in sorted(
                pool,
                key=lambda item: (
                    float(item[0]["anchor_rank"]),
                    item[1].structure_id,
                ),
            )[:count]
        )
        random_indices = rng.choice(len(pool), size=count, replace=False)
        matched_random.extend(pool[int(index)][1] for index in random_indices)
    return {
        "structure_mlp": candidate,
        "train_output_position_prior": position_prior,
        "position_matched_linear": tuple(
            structure.signature
            for structure in sorted(
                matched_linear,
                key=lambda structure: structure.structure_id,
            )
        ),
        "position_matched_random": tuple(
            structure.signature
            for structure in sorted(
                matched_random,
                key=lambda structure: structure.structure_id,
            )
        ),
    }


def adjudicate_position_prior_audit(
    config: PositionPriorAuditConfig,
    *,
    selector_rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
    thresholds: PositionPriorThresholds,
    source_run_id: str,
) -> dict[str, Any]:
    by_selector = {str(row["selector"]): row for row in selector_rows}
    selectors_complete = set(by_selector) == set(SELECTORS)
    finite_metrics = selectors_complete and all(
        math.isfinite(float(row["mean_balance_rate"])) for row in selector_rows
    )
    readiness = {
        **readiness_checks,
        "four_selector_summaries_present": selectors_complete,
        "selector_metrics_finite": finite_metrics,
    }
    candidate = by_selector.get("structure_mlp", {})
    position_prior = by_selector.get("train_output_position_prior", {})
    matched_linear = by_selector.get("position_matched_linear", {})
    matched_random = by_selector.get("position_matched_random", {})
    candidate_position = _mean_difference(candidate, position_prior)
    candidate_linear = _mean_difference(candidate, matched_linear)
    candidate_random = _mean_difference(candidate, matched_random)
    checks = {
        "candidate_position_prior_advantage_at_least_0_03": (
            candidate_position
            >= thresholds.minimum_candidate_position_prior_advantage
        ),
        "candidate_matched_linear_advantage_at_least_0_02": (
            candidate_linear
            >= thresholds.minimum_candidate_matched_linear_advantage
        ),
        "candidate_matched_random_advantage_at_least_0_03": (
            candidate_random
            >= thresholds.minimum_candidate_matched_random_advantage
        ),
    }
    readiness_passed = all(readiness.values())
    matched_checks_passed = (
        checks["candidate_matched_linear_advantage_at_least_0_02"]
        and checks["candidate_matched_random_advantage_at_least_0_03"]
    )
    if config.gate_mode == "position-prior-smoke":
        status = "pass" if readiness_passed else "fail"
        decision = (
            "innovation2_integral_position_prior_audit_ready"
            if readiness_passed
            else "innovation2_integral_position_prior_smoke_invalid"
        )
        next_action = (
            "Run the frozen 4096-key E6 output-position prior audit."
            if readiness_passed
            else "Repair reconstruction, matching, or parity before E6."
        )
    elif not readiness_passed:
        status = "fail"
        decision = "innovation2_integral_position_prior_protocol_invalid"
        next_action = "Repair E6 protocol validity; do not interpret attribution metrics."
    elif all(checks.values()):
        status = "pass"
        decision = "innovation2_integral_neural_interaction_residual_supported"
        next_action = (
            "Freeze Innovation 2 as a position-aware neural interaction result and "
            "draft the thesis chapter. Do not add more PRESENT-r5 selectors."
        )
    elif matched_checks_passed:
        status = "hold"
        decision = (
            "innovation2_integral_position_prior_dominant_with_conditional_residual"
        )
        next_action = (
            "Write the global enrichment as output-position-prior dominated and retain "
            "only the within-position MLP residual as a conditional contribution."
        )
    else:
        status = "hold"
        decision = "innovation2_integral_position_prior_explains_enrichment"
        next_action = (
            "Report E5 as explained by a simple output-position prior. Keep E0-E6 as "
            "a bounded diagnostic study and stop model/selector tuning on these keys."
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "gate_mode": config.gate_mode,
        "source_run_id": source_run_id,
        "readiness_checks": readiness,
        "attribution_checks": checks,
        "metrics": {
            "candidate_mean_balance_rate": candidate.get("mean_balance_rate"),
            "position_prior_mean_balance_rate": position_prior.get(
                "mean_balance_rate"
            ),
            "matched_linear_mean_balance_rate": matched_linear.get(
                "mean_balance_rate"
            ),
            "matched_random_mean_balance_rate": matched_random.get(
                "mean_balance_rate"
            ),
            "candidate_position_prior_advantage": candidate_position,
            "candidate_matched_linear_advantage": candidate_linear,
            "candidate_matched_random_advantage": candidate_random,
        },
        "thresholds": {
            "minimum_candidate_position_prior_advantage": (
                thresholds.minimum_candidate_position_prior_advantage
            ),
            "minimum_candidate_matched_linear_advantage": (
                thresholds.minimum_candidate_matched_linear_advantage
            ),
            "minimum_candidate_matched_random_advantage": (
                thresholds.minimum_candidate_matched_random_advantage
            ),
        },
        "next_action": next_action,
        "claim_scope": (
            f"post-hoc local PRESENT-r5 position-prior attribution on {config.fresh_keys} "
            "uniform independent fresh keys; no retraining and no deterministic proof"
        ),
    }


def _rate_rows(
    selections: dict[str, tuple[str, ...]],
    structures: dict[str, IntegralStructure],
    parities: dict[str, np.ndarray],
    *,
    fresh_keys: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    zero_upper = zero_failure_upper_bound(fresh_keys)
    for selector in SELECTORS:
        for selection_rank, signature in enumerate(selections[selector], start=1):
            structure = structures[signature]
            q1_count = int(parities[signature].sum())
            balance_count = fresh_keys - q1_count
            balance_rate = balance_count / fresh_keys
            lower, upper = wilson_interval(balance_count, fresh_keys)
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
                    "balance_rate": balance_rate,
                    "balance_wilson95_lower": lower,
                    "balance_wilson95_upper": upper,
                    "zero_observed_failure": q1_count == 0,
                    "zero_failure_q1_upper95": (
                        zero_upper if q1_count == 0 else None
                    ),
                }
            )
    return rows


def _selector_rows(
    config: PositionPriorAuditConfig,
    rate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selector in SELECTORS:
        selected = [row for row in rate_rows if row["selector"] == selector]
        rates = np.asarray([row["balance_rate"] for row in selected], dtype=np.float64)
        rows.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_integral_position_prior_audit",
                "selector": selector,
                "selector_label": SELECTOR_LABELS[selector],
                "top_k": config.top_k,
                "fresh_keys": config.fresh_keys,
                "mean_balance_rate": float(rates.mean()),
                "median_balance_rate": float(np.median(rates)),
                "minimum_balance_rate": float(rates.min()),
                "maximum_balance_rate": float(rates.max()),
                "zero_observed_failure_structures": sum(
                    bool(row["zero_observed_failure"]) for row in selected
                ),
                "output_position_histogram": _histogram_text(selected),
                "claim_scope": "post-hoc position attribution only",
            }
        )
    return rows


def _position_histogram(
    signatures: tuple[str, ...],
    structures: dict[str, IntegralStructure],
) -> dict[int, int]:
    return dict(
        sorted(Counter(structures[item].output_nibble for item in signatures).items())
    )


def _histogram_text(rows: list[dict[str, Any]]) -> str:
    counts = Counter(int(row["output_nibble"]) for row in rows)
    return ";".join(f"{position}:{counts[position]}" for position in sorted(counts))


def _overlap_rows(
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


def _historical_keys(source_summary: dict[str, Any]) -> set[int]:
    return {
        int(str(key), 16)
        for split in source_summary["splits"].values()
        for key in split.get("keys", [])
    }


def _crosscheck(
    structures: tuple[IntegralStructure, ...],
    matrix: np.ndarray,
    keys: tuple[int, ...],
    *,
    rounds: int,
) -> bool:
    for structure_index, structure in enumerate(structures[:3]):
        for key_index, key in enumerate(keys[:8]):
            expected = integral_mask_parity(
                Present80(rounds=rounds, key=key),
                structure,
            )
            if int(matrix[structure_index, key_index]) != expected:
                return False
    return True


def _mean_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    try:
        return float(left["mean_balance_rate"]) - float(
            right["mean_balance_rate"]
        )
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _emit(
    callback: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event, payload)
