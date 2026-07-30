from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.spn.uknit import (
    UknitBc,
    uknit_linear_layer,
    uknit_round_keys,
    uknit_substitution_layer,
)
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
    gf2_rank,
    kernel_basis_valid,
)


CALIBRATION_ROUND = 1
TARGET_ROUNDS = tuple(range(3, 12))
ROUNDS = (CALIBRATION_ROUND, *TARGET_ROUNDS)
ACTIVE_CELLS = tuple(range(16))
OUTPUT_BITS = 64
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class UknitLinearIntegralCensusConfig:
    run_id: str
    seed: int = 0
    discovery_trials: int = 128
    validation_trials: int = 128
    trial_chunk_size: int = 8

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.discovery_trials < 4 or self.validation_trials < 4:
            raise ValueError("discovery and validation trials must each be at least four")
        if self.trial_chunk_size <= 0:
            raise ValueError("trial_chunk_size must be positive")

    @property
    def total_trials(self) -> int:
        return self.discovery_trials + self.validation_trials


def run_uknit_linear_integral_census(
    config: UknitLinearIntegralCensusConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = _make_unique_keys(config.total_trials, seed=config.seed + 12_701)
    base_plaintexts = _make_unique_u64_values(
        config.total_trials,
        seed=config.seed + 12_702,
    )
    parity_rows = _collect_parity_rows(
        config,
        keys=keys,
        base_plaintexts=base_plaintexts,
        progress_callback=progress_callback,
    )
    random_control_rows = _make_random_control_rows(
        config,
        shape=parity_rows.shape,
    )
    return evaluate_uknit_linear_integral_census(
        config,
        keys=keys,
        base_plaintexts=base_plaintexts,
        parity_rows=parity_rows,
        random_control_rows=random_control_rows,
    )


def evaluate_uknit_linear_integral_census(
    config: UknitLinearIntegralCensusConfig,
    *,
    keys: tuple[int, ...],
    base_plaintexts: np.ndarray,
    parity_rows: np.ndarray,
    random_control_rows: np.ndarray,
) -> dict[str, Any]:
    rows_array = np.asarray(parity_rows, dtype=np.uint64)
    controls_array = np.asarray(random_control_rows, dtype=np.uint64)
    expected_shape = (len(ROUNDS), len(ACTIVE_CELLS), config.total_trials)
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    all_basis_valid = True

    for round_index, rounds in enumerate(ROUNDS):
        for cell_index, active_cell in enumerate(ACTIVE_CELLS):
            words = rows_array[round_index, cell_index]
            control_words = controls_array[round_index, cell_index]
            split_words = {
                "discovery": words[: config.discovery_trials],
                "validation": words[config.discovery_trials :],
                "joint": words,
            }
            metrics: dict[str, Any] = {}
            bases: dict[str, tuple[int, ...]] = {}
            for split, matrix in split_words.items():
                basis = gf2_kernel_basis(matrix, width=OUTPUT_BITS)
                rank = gf2_rank(matrix, width=OUTPUT_BITS)
                valid = kernel_basis_valid(matrix, basis)
                bases[split] = basis
                all_basis_valid = (
                    all_basis_valid and valid and rank + len(basis) == OUTPUT_BITS
                )
                metrics[f"{split}_rank"] = rank
                metrics[f"{split}_nullity"] = len(basis)
                metrics[f"{split}_basis_valid"] = valid
                for basis_index, mask in enumerate(basis):
                    basis_rows.append(
                        {
                            "run_id": config.run_id,
                            "rounds": rounds,
                            "active_cell": active_cell,
                            "split": split,
                            "basis_index": basis_index,
                            "mask_hex": f"0x{mask:016X}",
                            "mask_weight": mask.bit_count(),
                            "output_bits_lsb_first": ",".join(
                                str(bit)
                                for bit in range(OUTPUT_BITS)
                                if (mask >> bit) & 1
                            ),
                            "basis_valid": valid,
                        }
                    )

            discovery_basis = bases["discovery"]
            validation_matrix = split_words["validation"]
            survivors = tuple(
                mask
                for mask in discovery_basis
                if kernel_basis_valid(validation_matrix, (mask,))
            )
            joint_basis = bases["joint"]
            joint_valid_both_halves = kernel_basis_valid(
                split_words["discovery"], joint_basis
            ) and kernel_basis_valid(split_words["validation"], joint_basis)
            control_rank = gf2_rank(control_words, width=OUTPUT_BITS)
            control_nullity = OUTPUT_BITS - control_rank
            stable = (
                bool(joint_basis)
                and bool(discovery_basis)
                and bool(survivors)
                and joint_valid_both_halves
                and control_nullity == 0
            )
            false_accept_log2_bound = (
                len(discovery_basis) - config.validation_trials
                if discovery_basis
                else None
            )
            rows.append(
                {
                    "run_id": config.run_id,
                    "task": "innovation2_uknit_linear_integral_round_census",
                    "rounds": rounds,
                    "active_cell": active_cell,
                    "plaintexts_per_multiset": 16,
                    "discovery_trials": config.discovery_trials,
                    "validation_trials": config.validation_trials,
                    **metrics,
                    "discovery_basis_validation_survivors": len(survivors),
                    "discovery_basis_validation_survival_fraction": (
                        len(survivors) / len(discovery_basis)
                        if discovery_basis
                        else 0.0
                    ),
                    "joint_basis_valid_both_halves": joint_valid_both_halves,
                    "joint_basis_masks": ";".join(
                        f"0x{mask:016X}" for mask in joint_basis
                    ),
                    "joint_minimum_mask_weight": (
                        min(mask.bit_count() for mask in joint_basis)
                        if joint_basis
                        else 0
                    ),
                    "random_control_joint_rank": control_rank,
                    "random_control_joint_nullity": control_nullity,
                    "stable_nontrivial_kernel": stable,
                    "post_selection_false_accept_log2_bound": (
                        false_accept_log2_bound
                    ),
                }
            )

    round_summaries = _summarize_rounds(config, rows)
    readiness_checks = {
        "official_full_round_vector_matches": (
            UknitBc(
                rounds=12,
                key=0x0123456789ABCDEF0123456789ABCDEF,
            ).encrypt(0x0123456789ABCDEF)
            == 0x7D4EF882C1F42DBA
        ),
        "exact_key_count": len(keys) == config.total_trials,
        "keys_unique": len(set(keys)) == config.total_trials,
        "key_splits_disjoint": set(keys[: config.discovery_trials]).isdisjoint(
            keys[config.discovery_trials :]
        ),
        "base_plaintexts_shape_and_dtype": (
            np.asarray(base_plaintexts).shape == (config.total_trials,)
            and np.asarray(base_plaintexts).dtype == np.uint64
        ),
        "parity_rows_shape_and_dtype": (
            rows_array.shape == expected_shape and rows_array.dtype == np.uint64
        ),
        "random_control_shape_and_dtype": (
            controls_array.shape == expected_shape
            and controls_array.dtype == np.uint64
        ),
        "all_computed_bases_validate": all_basis_valid,
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if key.endswith("_rank")
            or key.endswith("_nullity")
            or key.endswith("_fraction")
        ),
        "r1_all_cells_rank0_nullity64": all(
            row["joint_rank"] == 0 and row["joint_nullity"] == 64
            for row in rows
            if row["rounds"] == CALIBRATION_ROUND
        ),
        "target_random_controls_full_rank": all(
            row["random_control_joint_nullity"] == 0
            for row in rows
            if row["rounds"] in TARGET_ROUNDS
        ),
    }
    gate = adjudicate_uknit_linear_integral_census(
        config,
        round_summaries,
        readiness_checks,
    )
    return {
        "rows": rows,
        "round_summaries": round_summaries,
        "basis_rows": basis_rows,
        "keys": keys,
        "base_plaintexts": np.asarray(base_plaintexts, dtype=np.uint64),
        "parity_rows": rows_array,
        "random_control_rows": controls_array,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_uknit_linear_integral_round_census",
            "cipher": "uKNIT-BC",
            "block_bits": 64,
            "key_bits": 128,
            "round_semantics": "prefix AddRoundKey-Sbox-GF2LinearLayer",
            "calibration_round": CALIBRATION_ROUND,
            "target_rounds": list(TARGET_ROUNDS),
            "active_cells": list(ACTIVE_CELLS),
            "cell_order": "MSB-first nibble index 0..15",
            "output_bit_order": "integer LSB-first bit 0..63",
            "feature": "raw 64-bit ciphertext XOR parity",
            "discovery_trials": config.discovery_trials,
            "validation_trials": config.validation_trials,
            "total_trials": config.total_trials,
            "plaintexts_per_multiset": 16,
            "seed": config.seed,
            "key_generation_seed": config.seed + 12_701,
            "base_plaintext_seed": config.seed + 12_702,
            "random_control_seed": config.seed + 12_703,
            "fresh_key_and_context_per_parity_row": True,
            "training_performed": False,
            "paper_baseline": "Hwang et al. 2026 parity-matrix right kernel",
            "claim_scope": gate["claim_scope"],
        },
    }


def adjudicate_uknit_linear_integral_census(
    config: UknitLinearIntegralCensusConfig,
    round_summaries: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
) -> dict[str, Any]:
    target = [row for row in round_summaries if row["rounds"] in TARGET_ROUNDS]
    passing = [row for row in target if row["stable_cells"] > 0]
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    highest_round = max((int(row["rounds"]) for row in passing), default=None)
    highest_row = next(
        (row for row in passing if row["rounds"] == highest_round),
        None,
    )
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_uknit_linear_integral_protocol_invalid"
        next_action = {
            "action": "repair uKNIT vector, multiset enumeration, split ownership, or GF(2) basis",
            "training": False,
            "remote_scale": False,
        }
    elif highest_round is None:
        status = "hold"
        decision = "innovation2_uknit_single_cell_linear_kernel_exhausted"
        next_action = {
            "action": "scan adjacent two-cell 256-plaintext multisets with the same 128+128 protocol",
            "one_variable_change": "active integral dimension: one cell -> two adjacent cells",
            "training": False,
            "remote_scale": False,
        }
    elif highest_round >= 7:
        status = "pass"
        decision = "innovation2_uknit_high_round_linear_kernel_candidate"
        next_action = {
            "action": "confirm only the highest round and its predecessor with 1000+1000 trials and seed0/seed1",
            "training": False,
            "remote_scale": False,
        }
    elif highest_round <= 4:
        status = "pass"
        decision = "innovation2_uknit_linear_kernel_candidate"
        next_action = {
            "action": "scan topology-coherent two-cell 256-plaintext multisets with the same 128+128 protocol",
            "reason": "the validated single-cell boundary is too low for confirmation to answer the high-round question",
            "one_variable_change": "active integral dimension: one cell -> two topology-coherent cells",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "pass"
        decision = "innovation2_uknit_linear_kernel_candidate"
        next_action = {
            "action": "confirm the highest round with 1000+1000 trials, then test topology-coherent two-cell extension",
            "one_variable_change_after_confirmation": (
                "active integral dimension: one cell -> two topology-coherent cells"
            ),
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "highest_supported_round": highest_round,
        "highest_supported_cells": (
            highest_row["stable_cell_indices"] if highest_row else []
        ),
        "claim_scope": (
            f"local {config.discovery_trials}-discovery plus "
            f"{config.validation_trials}-validation fresh-key/fresh-context "
            "single-active-cell uKNIT empirical raw-bit kernel census; not an "
            "all-key proof, paper-default 1000+1000 confirmation, neural result, "
            "or complete key-recovery conclusion"
        ),
        "next_action": next_action,
    }


def _collect_parity_rows(
    config: UknitLinearIntegralCensusConfig,
    *,
    keys: tuple[int, ...],
    base_plaintexts: np.ndarray,
    progress_callback: ProgressCallback | None,
) -> np.ndarray:
    return collect_uknit_integral_parity_rows(
        rounds=ROUNDS,
        structures=tuple((cell,) for cell in ACTIVE_CELLS),
        keys=keys,
        base_plaintexts=base_plaintexts,
        trial_chunk_size=config.trial_chunk_size,
        progress_callback=progress_callback,
    )


def collect_uknit_integral_parity_rows(
    *,
    rounds: tuple[int, ...],
    structures: tuple[tuple[int, ...], ...],
    keys: tuple[int, ...],
    base_plaintexts: np.ndarray,
    trial_chunk_size: int,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    if not rounds or min(rounds) < 1 or max(rounds) > 11:
        raise ValueError("rounds must be non-empty and within uKNIT prefix rounds 1..11")
    if len(set(rounds)) != len(rounds):
        raise ValueError("rounds must be unique")
    if not structures:
        raise ValueError("structures must be non-empty")
    for structure in structures:
        if not structure or len(set(structure)) != len(structure):
            raise ValueError("each active-cell structure must be non-empty and unique")
        if any(cell < 0 or cell >= 16 for cell in structure):
            raise ValueError("active cells must be between 0 and 15")
    if len(keys) != len(base_plaintexts):
        raise ValueError("keys and base_plaintexts must have equal length")
    if trial_chunk_size <= 0:
        raise ValueError("trial_chunk_size must be positive")
    rows = np.empty(
        (len(rounds), len(structures), len(keys)),
        dtype=np.uint64,
    )
    round_to_index = {round_count: index for index, round_count in enumerate(rounds)}
    maximum_round = max(rounds)
    for start in range(0, len(keys), trial_chunk_size):
        stop = min(start + trial_chunk_size, len(keys))
        _emit(progress_callback, "trial_chunk_start", {"start": start, "stop": stop})
        for trial_index in range(start, stop):
            round_keys = uknit_round_keys(keys[trial_index])
            base = int(base_plaintexts[trial_index])
            for structure_index, active_cells in enumerate(structures):
                cleared = base
                for active_cell in active_cells:
                    cleared &= ~(0xF << (4 * (15 - active_cell)))
                states: list[int] = []
                for assignment in range(16 ** len(active_cells)):
                    plaintext = cleared
                    value = assignment
                    for active_cell in reversed(active_cells):
                        plaintext |= (value & 0xF) << (4 * (15 - active_cell))
                        value >>= 4
                    states.append(plaintext)
                for round_index in range(maximum_round):
                    for state_index, state in enumerate(states):
                        state ^= round_keys[round_index]
                        state = uknit_substitution_layer(state, round_index)
                        states[state_index] = uknit_linear_layer(state, round_index)
                    round_count = round_index + 1
                    output_index = round_to_index.get(round_count)
                    if output_index is not None:
                        parity = 0
                        for state in states:
                            parity ^= state
                        rows[output_index, structure_index, trial_index] = parity
        _emit(
            progress_callback,
            "trial_chunk_done",
            {"completed": stop, "total": len(keys)},
        )
    return rows


def _summarize_rounds(
    config: UknitLinearIntegralCensusConfig,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for rounds in ROUNDS:
        round_rows = [row for row in rows if row["rounds"] == rounds]
        stable_rows = [row for row in round_rows if row["stable_nontrivial_kernel"]]
        summaries.append(
            {
                "run_id": config.run_id,
                "rounds": rounds,
                "stable_cells": len(stable_rows),
                "stable_cell_indices": ",".join(
                    str(row["active_cell"]) for row in stable_rows
                ),
                "maximum_joint_nullity": max(
                    int(row["joint_nullity"]) for row in round_rows
                ),
                "minimum_joint_rank": min(
                    int(row["joint_rank"]) for row in round_rows
                ),
                "random_control_nontrivial_cells": sum(
                    int(row["random_control_joint_nullity"] > 0)
                    for row in round_rows
                ),
                "minimum_false_accept_log2_bound": min(
                    (
                        int(row["post_selection_false_accept_log2_bound"])
                        for row in stable_rows
                        if row["post_selection_false_accept_log2_bound"] is not None
                    ),
                    default=None,
                ),
            }
        )
    return summaries


def _make_unique_keys(count: int, *, seed: int) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    keys: list[int] = []
    used: set[int] = set()
    while len(keys) < count:
        high = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        low = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        key = (high << 64) | low
        if key not in used:
            keys.append(key)
            used.add(key)
    return tuple(keys)


def _make_unique_u64_values(count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values: list[int] = []
    used: set[int] = set()
    while len(values) < count:
        value = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        if value not in used:
            values.append(value)
            used.add(value)
    return np.asarray(values, dtype=np.uint64)


def _make_random_control_rows(
    config: UknitLinearIntegralCensusConfig,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 12_703)
    return rng.integers(0, 1 << 64, size=shape, dtype=np.uint64)


def _emit(
    callback: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event, payload)


__all__ = [
    "ACTIVE_CELLS",
    "CALIBRATION_ROUND",
    "ROUNDS",
    "TARGET_ROUNDS",
    "UknitLinearIntegralCensusConfig",
    "adjudicate_uknit_linear_integral_census",
    "collect_uknit_integral_parity_rows",
    "evaluate_uknit_linear_integral_census",
    "run_uknit_linear_integral_census",
]
