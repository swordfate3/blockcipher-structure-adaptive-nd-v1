from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.ciphers.spn.uknit import uknit_round_keys
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.exact_operator_composition import (
    exact_operator_composition_views,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    DiagonalFisherScorer,
    deterministic_label_shuffle,
    fit_diagonal_fisher,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i1_uknit_r6_last_round_key_hypothesis_k1bp_seed2_seed3_seed4_20260730"
EXPECTED_PAIRS = 4
EXPECTED_BLOCK_BITS = 64
EXPECTED_CELLS = 16
EXPECTED_CONE_SOURCE_BITS = 12
EXPECTED_EFFECTIVE_KEY_BITS = 4
EXPECTED_CANDIDATES = 1 << EXPECTED_EFFECTIVE_KEY_BITS
EXPECTED_SPLITS = ("train_seen", "same_key_fresh", "cross_key_validation")
FRESH_SPLITS = ("same_key_fresh", "cross_key_validation")
CONFIRMATION_SEEDS = (3, 4)


@dataclass(frozen=True)
class CellDependencyCone:
    target_cell: int
    target_bits: tuple[int, ...]
    source_bits: tuple[int, ...]
    source_cells: tuple[int, ...]
    source_key_bits: int
    effective_key_bits: int
    candidate_count: int
    source_key_equivalence_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_cell": self.target_cell,
            "target_bits": list(self.target_bits),
            "source_bits": list(self.source_bits),
            "source_cells": list(self.source_cells),
            "source_key_bits": self.source_key_bits,
            "effective_key_bits": self.effective_key_bits,
            "candidate_count": self.candidate_count,
            "source_key_equivalence_size": self.source_key_equivalence_size,
        }


@dataclass(frozen=True)
class KeyRankResult:
    true_hypothesis: int
    true_rank: int
    candidate_count: int
    query_rows: int
    query_pairs: int
    correct_auc: float
    correct_positive_mean: float
    best_wrong_positive_mean: float
    correct_minus_best_wrong: float
    zero_key_rank: int
    zero_key_positive_mean: float
    top_hypotheses: tuple[tuple[int, float], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "true_hypothesis": self.true_hypothesis,
            "true_hypothesis_hex": f"0x{self.true_hypothesis:x}",
            "true_rank": self.true_rank,
            "candidate_count": self.candidate_count,
            "rank_percentile": 1.0
            - (self.true_rank - 1) / max(1, self.candidate_count - 1),
            "query_rows": self.query_rows,
            "query_pairs": self.query_pairs,
            "correct_auc": self.correct_auc,
            "correct_positive_mean": self.correct_positive_mean,
            "best_wrong_positive_mean": self.best_wrong_positive_mean,
            "correct_minus_best_wrong": self.correct_minus_best_wrong,
            "zero_key_rank": self.zero_key_rank,
            "zero_key_positive_mean": self.zero_key_positive_mean,
            "top_hypotheses": [
                {"hypothesis": guess, "hex": f"0x{guess:x}", "score": score}
                for guess, score in self.top_hypotheses
            ],
        }


def runtime_pairs(features: np.ndarray | torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(np.asarray(features).copy(), dtype=torch.float32)
    if values.ndim != 2 or values.shape[1] != EXPECTED_PAIRS * 2 * EXPECTED_BLOCK_BITS:
        raise ValueError("K1-BP requires four 64-bit ciphertext pairs per sample")
    if not torch.all((values == 0) | (values == 1)):
        raise ValueError("K1-BP ciphertext features must be binary")
    return values.reshape(
        values.shape[0], EXPECTED_PAIRS, 2, EXPECTED_BLOCK_BITS
    ).flip(-1)


def round_key_runtime_bits(master_key: int, round_index: int = 5) -> torch.Tensor:
    if round_index != 5:
        raise ValueError("K1-BP is frozen to the added sixth transition key K5")
    value = uknit_round_keys(master_key)[round_index]
    return torch.tensor(
        [(value >> bit) & 1 for bit in range(EXPECTED_BLOCK_BITS)],
        dtype=torch.float32,
    )


def strip_last_round(
    features: np.ndarray | torch.Tensor,
    *,
    last_transition: RuntimeSpnStructure,
    round_key_bits: torch.Tensor,
) -> np.ndarray:
    if last_transition.rounds != 1 or last_transition.block_bits != EXPECTED_BLOCK_BITS:
        raise ValueError("K1-BP requires the exact one-transition uKNIT r6 suffix")
    key = torch.as_tensor(round_key_bits, dtype=torch.float32)
    if key.shape != (EXPECTED_BLOCK_BITS,) or not torch.all((key == 0) | (key == 1)):
        raise ValueError("K1-BP round key must contain 64 binary runtime-order bits")
    runtime = runtime_pairs(features)
    views = exact_operator_composition_views(runtime, last_transition)
    recovered_with_mask = torch.stack((views[..., -3], views[..., -2]), dim=2)
    recovered = torch.remainder(recovered_with_mask + key, 2.0)
    return recovered.flip(-1).reshape(recovered.shape[0], -1).numpy().astype(
        np.uint8, copy=False
    )


def dependency_cones(
    transition: RuntimeSpnStructure,
) -> tuple[CellDependencyCone, ...]:
    if transition.rounds != 1 or transition.block_bits != EXPECTED_BLOCK_BITS:
        raise ValueError("K1-BP dependency audit requires one 64-bit transition")
    matrix = transition.inverse_linear_matrices[0].to(torch.bool)
    cones: list[CellDependencyCone] = []
    for cell in range(transition.cells):
        targets = torch.nonzero(transition.cell_membership == cell).flatten()
        sources = torch.nonzero(matrix[targets].any(dim=0)).flatten()
        source_cells = sorted(
            {int(transition.cell_membership[index]) for index in sources}
        )
        local_matrix = matrix[targets][:, sources].to(torch.uint8)
        effective_bits = _gf2_rank(local_matrix)
        source_bits = int(len(sources))
        cones.append(
            CellDependencyCone(
                target_cell=cell,
                target_bits=tuple(int(value) for value in targets),
                source_bits=tuple(int(value) for value in sources),
                source_cells=tuple(source_cells),
                source_key_bits=source_bits,
                effective_key_bits=effective_bits,
                candidate_count=1 << effective_bits,
                source_key_equivalence_size=1 << (source_bits - effective_bits),
            )
        )
    return tuple(cones)


def build_cell_linear_lookup(
    transition: RuntimeSpnStructure,
    cone: CellDependencyCone,
) -> np.ndarray:
    if cone.source_key_bits > 20:
        raise ValueError("K1-BP refuses an unbounded cell lookup")
    source_pattern_count = 1 << cone.source_key_bits
    codes = torch.arange(source_pattern_count, dtype=torch.long)
    source_values = (
        (codes[:, None] >> torch.arange(cone.source_key_bits, dtype=torch.long)) & 1
    ).to(torch.float32)
    states = torch.zeros(source_pattern_count, transition.block_bits, dtype=torch.float32)
    states[:, list(cone.source_bits)] = source_values
    inverse_linear = transition.exact_inverse(states, 0)
    target_indices = _ordered_cell_indices(transition, cone.target_cell)
    target_bits = inverse_linear[:, target_indices].to(torch.long)
    weights = 1 << torch.arange(3, -1, -1, dtype=torch.long)
    lookup = torch.sum(target_bits * weights, dim=1).numpy().astype(np.uint8)
    if lookup.shape != (source_pattern_count,):
        raise ValueError("K1-BP cell linear lookup geometry drifted")
    if len(np.unique(lookup)) != cone.candidate_count:
        raise ValueError("K1-BP effective-key rank does not match the lookup image")
    return lookup


def inverse_sbox_table(
    transition: RuntimeSpnStructure, target_cell: int
) -> np.ndarray:
    table = transition.inverse_sbox_tables(0)[target_cell].numpy().astype(np.uint8)
    if table.shape != (16,) or set(table.tolist()) != set(range(16)):
        raise ValueError("K1-BP target inverse S-box table is invalid")
    return table


def sparse_histogram_from_r5(
    dataset: DifferentialDataset,
    *,
    cone: CellDependencyCone,
    linear_lookup: np.ndarray,
    inverse_table: np.ndarray,
) -> np.ndarray:
    runtime = runtime_pairs(dataset.features)
    left_codes = _source_codes(runtime[:, :, 0], cone.source_bits)
    right_codes = _source_codes(runtime[:, :, 1], cone.source_bits)
    left_values = linear_lookup[left_codes]
    right_values = linear_lookup[right_codes]
    deltas = inverse_table[left_values] ^ inverse_table[right_values]
    histogram = np.eye(16, dtype=np.float32)[deltas].mean(axis=1)
    if histogram.shape != (len(dataset.labels), 16):
        raise ValueError("K1-BP sparse r5 histogram geometry drifted")
    return histogram


def masked_r5_cell_values_from_r6(
    dataset: DifferentialDataset,
    *,
    last_transition: RuntimeSpnStructure,
    cone: CellDependencyCone,
    linear_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    runtime = runtime_pairs(dataset.features)
    views = exact_operator_composition_views(runtime, last_transition)
    masked = torch.stack((views[..., -3], views[..., -2]), dim=2)
    left_codes = _source_codes(masked[:, :, 0], cone.source_bits)
    right_codes = _source_codes(masked[:, :, 1], cone.source_bits)
    return linear_lookup[left_codes], linear_lookup[right_codes]


def true_effective_hypothesis(
    master_key: int,
    cone: CellDependencyCone,
    linear_lookup: np.ndarray,
) -> int:
    key = round_key_runtime_bits(master_key)
    source_code = int(
        sum(int(key[bit]) << index for index, bit in enumerate(cone.source_bits))
    )
    return int(linear_lookup[source_code])


def rank_sparse_hypotheses(
    *,
    left_values: np.ndarray,
    right_values: np.ndarray,
    labels: np.ndarray,
    inverse_table: np.ndarray,
    scorer: DiagonalFisherScorer,
    true_hypothesis: int,
    candidate_batch_size: int = 128,
) -> KeyRankResult:
    left = np.asarray(left_values, dtype=np.uint8)
    right = np.asarray(right_values, dtype=np.uint8)
    targets = np.asarray(labels, dtype=np.uint8)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != EXPECTED_PAIRS:
        raise ValueError("K1-BP masked source-code geometry is invalid")
    if targets.shape != (left.shape[0],) or set(np.unique(targets).tolist()) != {0, 1}:
        raise ValueError("K1-BP ranking requires aligned balanced labels")
    candidate_count = EXPECTED_CANDIDATES
    if not 0 <= true_hypothesis < candidate_count:
        raise ValueError("K1-BP true hypothesis is outside the candidate set")
    if candidate_batch_size <= 0:
        raise ValueError("K1-BP candidate batch size must be positive")

    positive = targets == 1
    constant = float(np.asarray(scorer.midpoint) @ np.asarray(scorer.weights))
    weights = np.asarray(scorer.weights, dtype=np.float64)
    candidate_means = np.empty(candidate_count, dtype=np.float64)
    correct_scores: np.ndarray | None = None
    for start in range(0, candidate_count, candidate_batch_size):
        stop = min(start + candidate_batch_size, candidate_count)
        guesses = np.arange(start, stop, dtype=np.uint8)[:, None, None]
        deltas = inverse_table[np.bitwise_xor(left[None], guesses)] ^ inverse_table[
            np.bitwise_xor(right[None], guesses)
        ]
        sample_scores = weights[deltas].mean(axis=2) - constant
        candidate_means[start:stop] = sample_scores[:, positive].mean(axis=1)
        if start <= true_hypothesis < stop:
            correct_scores = sample_scores[true_hypothesis - start].copy()
    if correct_scores is None or not np.all(np.isfinite(candidate_means)):
        raise ValueError("K1-BP ranking produced incomplete scores")

    order = np.argsort(-candidate_means, kind="mergesort")
    ranks = np.empty(candidate_count, dtype=np.int64)
    ranks[order] = np.arange(1, candidate_count + 1)
    wrong = candidate_means.copy()
    wrong[true_hypothesis] = -math.inf
    top = tuple((int(index), float(candidate_means[index])) for index in order[:16])
    return KeyRankResult(
        true_hypothesis=true_hypothesis,
        true_rank=int(ranks[true_hypothesis]),
        candidate_count=candidate_count,
        query_rows=int(positive.sum()),
        query_pairs=int(positive.sum()) * EXPECTED_PAIRS,
        correct_auc=binary_auc(targets, correct_scores),
        correct_positive_mean=float(candidate_means[true_hypothesis]),
        best_wrong_positive_mean=float(wrong.max()),
        correct_minus_best_wrong=float(
            candidate_means[true_hypothesis] - wrong.max()
        ),
        zero_key_rank=int(ranks[0]),
        zero_key_positive_mean=float(candidate_means[0]),
        top_hypotheses=top,
    )


def evaluate_sparse_anchor(
    *,
    datasets: Mapping[str, DifferentialDataset],
    transition: RuntimeSpnStructure,
    target_cell: int,
    shuffle_labels: bool = False,
    seed: int = 0,
) -> tuple[
    DiagonalFisherScorer,
    dict[str, float],
    CellDependencyCone,
    np.ndarray,
    np.ndarray,
]:
    if set(datasets) != set(EXPECTED_SPLITS):
        raise ValueError("K1-BP sparse anchor requires all three splits")
    cones = dependency_cones(transition)
    cone = cones[target_cell]
    linear_lookup = build_cell_linear_lookup(transition, cone)
    inverse_table = inverse_sbox_table(transition, target_cell)
    features = {
        split: sparse_histogram_from_r5(
            dataset,
            cone=cone,
            linear_lookup=linear_lookup,
            inverse_table=inverse_table,
        )
        for split, dataset in datasets.items()
    }
    labels = np.asarray(datasets["train_seen"].labels, dtype=np.uint8)
    if shuffle_labels:
        labels, _ = deterministic_label_shuffle(labels, seed=seed)
    scorer = fit_diagonal_fisher(features["train_seen"], labels)
    aucs = {
        split: binary_auc(
            np.asarray(dataset.labels, dtype=np.uint8), scorer.score(features[split])
        )
        for split, dataset in datasets.items()
    }
    return scorer, aucs, cone, linear_lookup, inverse_table


def adjudicate_k1bp(
    *,
    protocol_checks: Mapping[str, bool],
    discovery_rows: Sequence[Mapping[str, Any]],
    full_oracle_rows: Sequence[Mapping[str, Any]],
    sparse_rank_rows: Sequence[Mapping[str, Any]],
    selected_cell: int,
    thresholds: Mapping[str, float | int],
) -> dict[str, Any]:
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    weak_signal_auc_floor = float(thresholds["weak_signal_auc_floor"])
    strong_signal_auc_floor = float(thresholds["discovery_sparse_auc_floor"])
    discovery = next(
        row for row in discovery_rows if int(row["target_cell"]) == selected_cell
    )
    discovery_min_auc = float(discovery["minimum_fresh_auc"])
    discovery_pass = discovery_min_auc >= strong_signal_auc_floor
    if discovery_pass:
        discovery_signal_tier = "strong"
    elif discovery_min_auc >= weak_signal_auc_floor:
        discovery_signal_tier = "weak"
    else:
        discovery_signal_tier = "none"
    oracle_checks: dict[str, bool] = {}
    for row in full_oracle_rows:
        prefix = f"seed{row['seed']}_{row['split']}"
        oracle_checks[f"{prefix}_auc_floor"] = float(row["correct_key_auc"]) >= float(
            thresholds["full_oracle_auc_floor"]
        )
        oracle_checks[f"{prefix}_wrong_key_margin"] = float(
            row["correct_minus_best_wrong_auc"]
        ) >= float(thresholds["full_oracle_wrong_key_margin"])
    sparse_checks: dict[str, bool] = {}
    for row in sparse_rank_rows:
        prefix = f"seed{row['seed']}_{row['split']}"
        sparse_checks[f"{prefix}_r5_auc_floor"] = float(row["r5_sparse_auc"]) >= float(
            thresholds["confirmation_sparse_auc_floor"]
        )
        sparse_checks[f"{prefix}_r6_auc_floor"] = float(row["correct_auc"]) >= float(
            thresholds["confirmation_sparse_auc_floor"]
        )
        sparse_checks[f"{prefix}_true_rank_exact"] = int(row["true_rank"]) == int(
            thresholds["required_true_key_rank"]
        )
        sparse_checks[f"{prefix}_beats_wrong_sbox_rank"] = int(
            row["true_rank"]
        ) < int(row["wrong_sbox_true_rank"])
        sparse_checks[f"{prefix}_beats_label_shuffle_rank"] = int(
            row["true_rank"]
        ) < int(row["label_shuffle_true_rank"])

    confirmation_min_r5_auc = min(
        (float(row["r5_sparse_auc"]) for row in sparse_rank_rows), default=float("nan")
    )
    confirmation_min_r6_auc = min(
        (float(row["correct_auc"]) for row in sparse_rank_rows), default=float("nan")
    )
    confirmation_min_auc = min(confirmation_min_r5_auc, confirmation_min_r6_auc)
    confirmation_strong = bool(sparse_rank_rows) and confirmation_min_auc >= float(
        thresholds["confirmation_sparse_auc_floor"]
    )
    confirmation_weak = bool(sparse_rank_rows) and confirmation_min_auc >= weak_signal_auc_floor
    confirmation_signal_tier = (
        "strong" if confirmation_strong else "weak" if confirmation_weak else "none"
    )
    if confirmation_signal_tier != "none":
        evidence_tier = f"{confirmation_signal_tier}_confirmed"
    elif discovery_signal_tier == "weak":
        evidence_tier = "weak_discovery_only_unconfirmed"
    elif discovery_signal_tier == "strong":
        evidence_tier = "strong_discovery_only_unconfirmed"
    else:
        evidence_tier = "no_supported_sparse_signal"

    oracle_pass = bool(oracle_checks) and all(oracle_checks.values())
    sparse_pass = discovery_pass and bool(sparse_checks) and all(sparse_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_r6_k1bp_protocol_invalid"
        next_action = (
            "repair only the failed inverse-round, source, checkpoint, bit-order or "
            "candidate-enumeration binding and rerun K1-BP"
        )
    elif oracle_pass and sparse_pass:
        status = "pass"
        decision = "innovation1_uknit_r6_k1bp_bounded_key_hypothesis_supported"
        next_action = (
            "freeze the selected four-bit effective cell key and run K1-BQ on larger "
            "independent multi-key query panels; keep difference, four pairs, scorer and 16 "
            "guesses fixed and report query/guess complexity separately from AUC"
        )
    elif oracle_pass and discovery_pass:
        status = "hold"
        decision = "innovation1_uknit_r6_k1bp_sparse_anchor_without_key_rank"
        next_action = (
            "do not scale; train at most one small r5 specialist on the same frozen "
            "16-bin, four-bit-effective feature and repeat the local 16-guess rank gate"
        )
    elif oracle_pass:
        status = "hold"
        decision = "innovation1_uknit_r6_k1bp_single_cell_sparse_anchor_not_supported"
        next_action = (
            "keep the full-key oracle only as mechanism evidence and preregister a "
            "two- versus three-cell effective-key audit; compute joint GF(2) rank first, "
            "then cap enumeration at 12 effective bits (4096 candidates)"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_r6_k1bp_oracle_not_supported"
        next_action = (
            "do not interpret the sparse rank; audit checkpoint compatibility and "
            "the exact r6-to-r5 inverse mapping before any further experiment"
        )

    research_checks = {
        "discovery_sparse_anchor_auc_floor": discovery_pass,
        **oracle_checks,
        **sparse_checks,
    }
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "selected_cell": selected_cell,
        "full_model_required_key_bits": 64,
        "full_model_candidate_count": 1 << 64,
        "sparse_source_key_bits": EXPECTED_CONE_SOURCE_BITS,
        "sparse_effective_key_bits": EXPECTED_EFFECTIVE_KEY_BITS,
        "sparse_candidate_count": EXPECTED_CANDIDATES,
        "oracle_pass": oracle_pass,
        "bounded_route_pass": sparse_pass,
        "evidence_tier": evidence_tier,
        "weak_signal_observed": discovery_signal_tier == "weak",
        "weak_signal_confirmed": confirmation_signal_tier in {"weak", "strong"},
        "discovery_signal_tier": discovery_signal_tier,
        "discovery_minimum_fresh_auc": discovery_min_auc,
        "confirmation_signal_tier": confirmation_signal_tier,
        "confirmation_minimum_r5_auc": confirmation_min_r5_auc,
        "confirmation_minimum_r6_auc": confirmation_min_r6_auc,
        "remote_scale": "no",
        "neural_training_authorized": False,
        "protocol_checks": dict(protocol_checks),
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "thresholds": dict(thresholds),
        "next_action": next_action,
        "signal_tier_policy": (
            "AUC >= 0.51 is a weak signal worth local confirmation; AUC >= 0.55 is "
            "a strong candidate floor. Neither tier overrides required rank and control gates."
        ),
        "claim_scope": (
            "local zero-neural-training uKNIT r6 last-round feasibility audit using "
            "reused 2048/class train and 1024/class fresh caches, one frozen K1-U "
            "r5 checkpoint per seed, a full-key oracle and exhaustive four-bit "
            "effective one-cell key hypotheses derived from a 12-source-bit cone; "
            "not formal scale, full-key recovery, attack, SOTA or universal r6 evidence"
        ),
        "blocked_actions": [
            "calling the 64-bit true-key oracle a six-round attack",
            "remote r6 scale without a passed bounded local key-rank gate",
            "changing the selected cell, difference, pairs, keys or thresholds after results",
            "hiding guessed-bit, candidate or query complexity",
        ],
    }


def _source_codes(values: torch.Tensor, source_bits: Sequence[int]) -> np.ndarray:
    selected = values[..., list(source_bits)].to(torch.long)
    weights = 1 << torch.arange(len(source_bits), dtype=torch.long)
    return torch.sum(selected * weights, dim=-1).numpy().astype(np.uint16)


def _ordered_cell_indices(
    structure: RuntimeSpnStructure, cell: int
) -> torch.Tensor:
    indices = torch.empty(4, dtype=torch.long)
    bit_indices = torch.arange(structure.block_bits)
    mask = structure.cell_membership == cell
    indices[structure.bit_role[mask]] = bit_indices[mask]
    return indices


def _gf2_rank(matrix: torch.Tensor) -> int:
    values = torch.as_tensor(matrix, dtype=torch.uint8).clone()
    rows, columns = values.shape
    rank = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(rank, rows) if int(values[row, column]) == 1),
            None,
        )
        if pivot is None:
            continue
        values[[rank, pivot]] = values[[pivot, rank]]
        for row in range(rows):
            if row != rank and int(values[row, column]) == 1:
                values[row] ^= values[rank]
        rank += 1
        if rank == rows:
            break
    return rank


__all__ = [
    "CONFIRMATION_SEEDS",
    "EXPECTED_CANDIDATES",
    "EXPECTED_CONE_SOURCE_BITS",
    "EXPECTED_EFFECTIVE_KEY_BITS",
    "EXPECTED_SPLITS",
    "FRESH_SPLITS",
    "RUN_ID",
    "CellDependencyCone",
    "KeyRankResult",
    "adjudicate_k1bp",
    "build_cell_linear_lookup",
    "dependency_cones",
    "evaluate_sparse_anchor",
    "inverse_sbox_table",
    "masked_r5_cell_values_from_r6",
    "rank_sparse_hypotheses",
    "round_key_runtime_bits",
    "runtime_pairs",
    "sparse_histogram_from_r5",
    "strip_last_round",
    "true_effective_hypothesis",
]
