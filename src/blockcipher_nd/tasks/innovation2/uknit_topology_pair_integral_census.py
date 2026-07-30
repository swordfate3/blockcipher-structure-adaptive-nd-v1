from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.spn.uknit import UknitBc
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
    gf2_rank,
    kernel_basis_valid,
)
from blockcipher_nd.tasks.innovation2.uknit_linear_integral_census import (
    _make_unique_keys,
    _make_unique_u64_values,
    collect_uknit_integral_parity_rows,
)


CALIBRATION_ROUND = 1
TARGET_ROUNDS = tuple(range(4, 9))
ROUNDS = (CALIBRATION_ROUND, *TARGET_ROUNDS)
TOPOLOGY_GROUPS = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
)
TOPOLOGY_PAIRS = tuple(
    pair for group in TOPOLOGY_GROUPS for pair in combinations(group, 2)
)
CONTROL_PAIRS = ((3, 4), (7, 8), (11, 12), (15, 0))
PAIR_STRUCTURES = (*TOPOLOGY_PAIRS, *CONTROL_PAIRS)
OUTPUT_BITS = 64
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class UknitTopologyPairCensusConfig:
    run_id: str
    seed: int = 0
    discovery_trials: int = 128
    validation_trials: int = 128
    trial_chunk_size: int = 2

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


def run_uknit_topology_pair_integral_census(
    config: UknitTopologyPairCensusConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = _make_unique_keys(config.total_trials, seed=config.seed + 13_701)
    base_plaintexts = _make_unique_u64_values(
        config.total_trials,
        seed=config.seed + 13_702,
    )
    parity_rows = collect_uknit_integral_parity_rows(
        rounds=ROUNDS,
        structures=PAIR_STRUCTURES,
        keys=keys,
        base_plaintexts=base_plaintexts,
        trial_chunk_size=config.trial_chunk_size,
        progress_callback=progress_callback,
    )
    rng = np.random.default_rng(config.seed + 13_703)
    random_control_rows = rng.integers(
        0,
        1 << 64,
        size=parity_rows.shape,
        dtype=np.uint64,
    )
    return evaluate_uknit_topology_pair_integral_census(
        config,
        keys=keys,
        base_plaintexts=base_plaintexts,
        parity_rows=parity_rows,
        random_control_rows=random_control_rows,
    )


def evaluate_uknit_topology_pair_integral_census(
    config: UknitTopologyPairCensusConfig,
    *,
    keys: tuple[int, ...],
    base_plaintexts: np.ndarray,
    parity_rows: np.ndarray,
    random_control_rows: np.ndarray,
) -> dict[str, Any]:
    rows_array = np.asarray(parity_rows, dtype=np.uint64)
    controls_array = np.asarray(random_control_rows, dtype=np.uint64)
    expected_shape = (len(ROUNDS), len(PAIR_STRUCTURES), config.total_trials)
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    all_basis_valid = True

    for round_index, rounds in enumerate(ROUNDS):
        for structure_index, pair in enumerate(PAIR_STRUCTURES):
            role = "topology" if pair in TOPOLOGY_PAIRS else "cross_group_control"
            words = rows_array[round_index, structure_index]
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
                            "role": role,
                            "active_pair": _pair_label(pair),
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
            control_words = controls_array[round_index, structure_index]
            random_rank = gf2_rank(control_words, width=OUTPUT_BITS)
            random_nullity = OUTPUT_BITS - random_rank
            stable = (
                bool(joint_basis)
                and bool(discovery_basis)
                and bool(survivors)
                and joint_valid_both_halves
                and random_nullity == 0
            )
            rows.append(
                {
                    "run_id": config.run_id,
                    "task": "innovation2_uknit_topology_pair_integral_round_census",
                    "rounds": rounds,
                    "role": role,
                    "active_pair": _pair_label(pair),
                    "active_cells": list(pair),
                    "plaintexts_per_multiset": 256,
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
                    "random_control_joint_rank": random_rank,
                    "random_control_joint_nullity": random_nullity,
                    "stable_nontrivial_kernel": stable,
                    "post_selection_false_accept_log2_bound": (
                        len(discovery_basis) - config.validation_trials
                        if discovery_basis
                        else None
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
        "exact_structure_ownership": (
            len(TOPOLOGY_PAIRS) == 24
            and len(CONTROL_PAIRS) == 4
            and len(set(PAIR_STRUCTURES)) == 28
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
        "r1_all_pairs_rank0_nullity64": all(
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
    gate = adjudicate_uknit_topology_pair_integral_census(
        config,
        rows,
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
            "task": "innovation2_uknit_topology_pair_integral_round_census",
            "cipher": "uKNIT-BC",
            "block_bits": 64,
            "key_bits": 128,
            "round_semantics": "prefix AddRoundKey-Sbox-GF2LinearLayer",
            "calibration_round": CALIBRATION_ROUND,
            "target_rounds": list(TARGET_ROUNDS),
            "topology_groups": [list(group) for group in TOPOLOGY_GROUPS],
            "topology_pairs": [list(pair) for pair in TOPOLOGY_PAIRS],
            "cross_group_controls": [list(pair) for pair in CONTROL_PAIRS],
            "feature": "raw 64-bit ciphertext XOR parity",
            "discovery_trials": config.discovery_trials,
            "validation_trials": config.validation_trials,
            "total_trials": config.total_trials,
            "plaintexts_per_multiset": 256,
            "seed": config.seed,
            "key_generation_seed": config.seed + 13_701,
            "base_plaintext_seed": config.seed + 13_702,
            "random_control_seed": config.seed + 13_703,
            "fresh_key_and_context_per_parity_row": True,
            "training_performed": False,
            "paper_baseline": "Hwang et al. 2026 parity-matrix right kernel",
            "claim_scope": gate["claim_scope"],
        },
    }


def adjudicate_uknit_topology_pair_integral_census(
    config: UknitTopologyPairCensusConfig,
    rows: list[dict[str, Any]],
    round_summaries: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
) -> dict[str, Any]:
    stable_target_rows = [
        row
        for row in rows
        if row["rounds"] in TARGET_ROUNDS
        and row["role"] == "topology"
        and row["stable_nontrivial_kernel"]
    ]
    stable_any_rows = [
        row
        for row in rows
        if row["rounds"] in TARGET_ROUNDS and row["stable_nontrivial_kernel"]
    ]
    highest_round = max(
        (int(row["rounds"]) for row in stable_any_rows),
        default=None,
    )
    highest_rows = [
        row for row in stable_any_rows if int(row["rounds"]) == highest_round
    ]
    strongest = sorted(
        highest_rows,
        key=lambda row: (
            -int(row["joint_nullity"]),
            str(row["active_pair"]),
        ),
    )
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    topology_extended = any(int(row["rounds"]) >= 5 for row in stable_target_rows)
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_uknit_topology_pair_integral_protocol_invalid"
        next_action = {
            "action": "repair pair enumeration, uKNIT vector, split ownership, cache, or GF(2) basis",
            "training": False,
            "remote_scale": False,
        }
    elif highest_round is not None and highest_round >= 5:
        status = "pass"
        decision = "innovation2_uknit_pair_linear_kernel_round_extension"
        next_action = {
            "action": "confirm the strongest highest-round pairs with 1000+1000 trials and seed0/seed1",
            "selected_pairs": [str(row["active_pair"]) for row in strongest[:4]],
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_uknit_pair_linear_kernel_no_round_extension"
        next_action = {
            "action": "rank three-active-cell versus 256-dimensional cell-VDS routes before another run",
            "do_not": "mechanically add keys, seeds, neural training, or remote GPU budget",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "highest_supported_round": highest_round,
        "highest_supported_pairs": [str(row["active_pair"]) for row in strongest],
        "highest_pair_nullities": {
            str(row["active_pair"]): int(row["joint_nullity"])
            for row in strongest
        },
        "topology_coherent_pair_extended_beyond_r4": topology_extended,
        "round_summaries": round_summaries,
        "claim_scope": (
            f"local {config.discovery_trials}-discovery plus "
            f"{config.validation_trials}-validation fresh-key/fresh-context "
            "uKNIT two-active-cell empirical raw-bit kernel census; not an all-key "
            "proof, paper-default confirmation, neural result, or complete key-recovery conclusion"
        ),
        "next_action": next_action,
    }


def _summarize_rounds(
    config: UknitTopologyPairCensusConfig,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for rounds in ROUNDS:
        round_rows = [row for row in rows if row["rounds"] == rounds]
        topology = [row for row in round_rows if row["role"] == "topology"]
        controls = [
            row for row in round_rows if row["role"] == "cross_group_control"
        ]
        stable_topology = [row for row in topology if row["stable_nontrivial_kernel"]]
        stable_controls = [row for row in controls if row["stable_nontrivial_kernel"]]
        summaries.append(
            {
                "run_id": config.run_id,
                "rounds": rounds,
                "stable_topology_pairs": len(stable_topology),
                "stable_topology_pair_labels": ";".join(
                    str(row["active_pair"]) for row in stable_topology
                ),
                "stable_control_pairs": len(stable_controls),
                "stable_control_pair_labels": ";".join(
                    str(row["active_pair"]) for row in stable_controls
                ),
                "topology_stable_fraction": len(stable_topology) / len(topology),
                "control_stable_fraction": len(stable_controls) / len(controls),
                "maximum_joint_nullity": max(
                    int(row["joint_nullity"]) for row in round_rows
                ),
                "random_control_nontrivial_pairs": sum(
                    int(row["random_control_joint_nullity"] > 0)
                    for row in round_rows
                ),
            }
        )
    return summaries


def _pair_label(pair: tuple[int, int]) -> str:
    return f"{pair[0]}+{pair[1]}"


__all__ = [
    "CONTROL_PAIRS",
    "PAIR_STRUCTURES",
    "ROUNDS",
    "TARGET_ROUNDS",
    "TOPOLOGY_GROUPS",
    "TOPOLOGY_PAIRS",
    "UknitTopologyPairCensusConfig",
    "adjudicate_uknit_topology_pair_integral_census",
    "evaluate_uknit_topology_pair_integral_census",
    "run_uknit_topology_pair_integral_census",
]
