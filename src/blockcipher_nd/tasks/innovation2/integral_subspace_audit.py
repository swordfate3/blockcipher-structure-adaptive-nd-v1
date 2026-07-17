from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from blockcipher_nd.tasks.innovation2.integral_bit_transition_audit import (
    ACTIVE_BIT_WIDTHS,
    BitIntegralStructure,
    bit_integral_output_xor_matrix,
    make_bit_transition_structures,
    scalar_bit_integral_output_xor,
)
from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    present_round_key_matrix,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import make_keys


ROUNDS = (5, 6)
CALIBRATION_ROUNDS = 4
OUTPUT_BITS = 64
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class StableSubspaceAuditConfig:
    run_id: str
    seed: int = 0
    structures_per_width: int = 64
    keys_per_structure: int = 256
    structure_chunk_size: int = 4

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.structures_per_width < 64 or self.structures_per_width % 16:
            raise ValueError("structures_per_width must be a multiple of 16 and at least 64")
        if self.keys_per_structure < 8 or self.keys_per_structure % 2:
            raise ValueError("keys_per_structure must be even and at least eight")
        if self.structure_chunk_size <= 0:
            raise ValueError("structure_chunk_size must be positive")


def gf2_kernel_basis(words: np.ndarray, *, width: int = OUTPUT_BITS) -> tuple[int, ...]:
    matrix, pivots = _gf2_rref(words, width=width)
    pivot_set = set(pivots)
    free_columns = [column for column in range(width) if column not in pivot_set]
    basis: list[int] = []
    for free_column in free_columns:
        vector = 1 << free_column
        for row_index, pivot_column in enumerate(pivots):
            if int(matrix[row_index, free_column]):
                vector |= 1 << pivot_column
        basis.append(vector)
    return tuple(basis)


def gf2_rank(words: np.ndarray, *, width: int = OUTPUT_BITS) -> int:
    _, pivots = _gf2_rref(words, width=width)
    return len(pivots)


def kernel_basis_valid(words: np.ndarray, basis: tuple[int, ...]) -> bool:
    return all(
        all((int(word) & vector).bit_count() % 2 == 0 for word in words)
        for vector in basis
    )


def run_stable_subspace_audit(
    config: StableSubspaceAuditConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = make_keys(count=config.keys_per_structure, seed=config.seed + 1401)
    structure_sets = {
        width: make_bit_transition_structures(
            active_bit_width=width,
            count=config.structures_per_width,
            seed=config.seed + width * 100 + 1701,
        )
        for width in ACTIVE_BIT_WIDTHS
    }
    calibration_structures = _make_r4_calibration_structures(config.seed)
    rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    xor_matrices: dict[tuple[int, int], np.ndarray] = {}
    all_basis_valid = True
    calibration_round_keys = present_round_key_matrix(
        keys,
        rounds=CALIBRATION_ROUNDS,
    )
    calibration_xor_matrix = bit_integral_output_xor_matrix(
        calibration_structures,
        calibration_round_keys,
        structure_chunk_size=config.structure_chunk_size,
    )
    xor_matrices[(CALIBRATION_ROUNDS, 4)] = calibration_xor_matrix
    calibration_summary, calibration_details, calibration_valid = (
        _summarize_round_width(
            config,
            rounds=CALIBRATION_ROUNDS,
            width=4,
            structures=calibration_structures,
            xor_matrix=calibration_xor_matrix,
        )
    )
    rows.append(calibration_summary)
    subspace_rows.extend(calibration_details)
    all_basis_valid = all_basis_valid and calibration_valid
    for rounds in ROUNDS:
        round_keys = present_round_key_matrix(keys, rounds=rounds)
        for width in ACTIVE_BIT_WIDTHS:
            _emit(
                progress_callback,
                "xor_matrix_start",
                {
                    "rounds": rounds,
                    "active_bit_width": width,
                    "structures": config.structures_per_width,
                    "keys": config.keys_per_structure,
                },
            )
            xor_matrix = bit_integral_output_xor_matrix(
                structure_sets[width],
                round_keys,
                structure_chunk_size=config.structure_chunk_size,
            )
            xor_matrices[(rounds, width)] = xor_matrix
            summary, details, valid = _summarize_round_width(
                config,
                rounds=rounds,
                width=width,
                structures=structure_sets[width],
                xor_matrix=xor_matrix,
            )
            rows.append(summary)
            subspace_rows.extend(details)
            all_basis_valid = all_basis_valid and valid
            _emit(
                progress_callback,
                "xor_matrix_done",
                {
                    "rounds": rounds,
                    "active_bit_width": width,
                    "nontrivial_joint_kernel_fraction": summary[
                        "nontrivial_joint_kernel_fraction"
                    ],
                    "maximum_joint_kernel_dimension": summary[
                        "maximum_joint_kernel_dimension"
                    ],
                },
            )

    scalar_matches = _crosscheck_scalar(
        calibration_structures,
        structure_sets,
        xor_matrices,
        keys,
    )
    half = config.keys_per_structure // 2
    readiness = {
        "calibration_and_six_round_width_rows_present": (
            len(rows) == 1 + len(ROUNDS) * len(ACTIVE_BIT_WIDTHS)
        ),
        "same_structures_used_for_r5_and_r6": True,
        "same_keys_used_for_all_rounds_and_widths": True,
        "key_halves_nonempty_and_disjoint": set(keys[:half]).isdisjoint(keys[half:]),
        "all_joint_kernel_bases_validate_both_halves": all_basis_valid,
        "vectorized_xor_words_match_scalar": scalar_matches,
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "nontrivial_joint_kernel_fraction",
                "mean_joint_kernel_dimension",
                "maximum_joint_kernel_dimension",
                "mean_discovery_basis_validation_survival_fraction",
            )
        ),
    }
    gate = adjudicate_stable_subspace_audit(config, rows, readiness)
    return {
        "rows": rows,
        "subspace_rows": subspace_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_stable_output_balance_subspace_readiness",
            "cipher": "PRESENT-80",
            "rounds": list(ROUNDS),
            "calibration_rounds": CALIBRATION_ROUNDS,
            "active_bit_widths": list(ACTIVE_BIT_WIDTHS),
            "structures_per_width": config.structures_per_width,
            "keys_per_structure": config.keys_per_structure,
            "key_half_size": half,
            "seed": config.seed,
            "output_bits": OUTPUT_BITS,
            "training_performed": False,
            "baseline": "Hwang et al. 2026 empirical parity-matrix kernel",
            "keys": [f"{key:020X}" for key in keys],
        },
    }


def adjudicate_stable_subspace_audit(
    config: StableSubspaceAuditConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
) -> dict[str, Any]:
    by_setting = {
        (int(row["rounds"]), int(row["active_bit_width"])): row for row in rows
    }
    calibration = by_setting.get((CALIBRATION_ROUNDS, 4), {})
    r4_calibration_pass = (
        _metric(calibration, "nontrivial_joint_kernel_fraction") == 1.0
        and _metric(calibration, "minimum_joint_kernel_dimension") == OUTPUT_BITS
        and _metric(calibration, "maximum_joint_kernel_dimension") == OUTPUT_BITS
    )
    target_checks: dict[str, dict[str, bool]] = {}
    passing_widths: list[int] = []
    for width in ACTIVE_BIT_WIDTHS:
        row = by_setting.get((6, width), {})
        checks = {
            "nontrivial_joint_kernel_fraction_at_least_0p10": (
                _metric(row, "nontrivial_joint_kernel_fraction") >= 0.10
            ),
            "nontrivial_joint_kernel_structures_at_least_8": (
                _metric(row, "nontrivial_joint_kernel_structures") >= 8
            ),
            "distinct_nontrivial_subspace_signatures_at_least_4": (
                _metric(row, "distinct_nontrivial_joint_kernel_signatures") >= 4
            ),
            "mean_discovery_basis_validation_survival_at_least_0p50": (
                _metric(
                    row,
                    "mean_discovery_basis_validation_survival_fraction",
                )
                >= 0.50
            ),
        }
        target_checks[str(width)] = checks
        if all(checks.values()):
            passing_widths.append(width)

    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    selected_width: int | None = None
    if not readiness_pass or not r4_calibration_pass:
        status = "fail"
        decision = "innovation2_stable_balance_subspace_protocol_invalid"
        next_action = {
            "action": "repair parity words, GF(2) elimination, or r4 known fixture",
            "training": False,
            "remote_scale": False,
        }
    elif passing_widths:
        best_fraction = max(
            _metric(
                by_setting[(6, width)],
                "nontrivial_joint_kernel_fraction",
            )
            for width in passing_widths
        )
        selected_width = min(
            width
            for width in passing_widths
            if best_fraction
            - _metric(
                by_setting[(6, width)],
                "nontrivial_joint_kernel_fraction",
            )
            < 0.05
        )
        status = "pass"
        decision = "innovation2_r6_stable_balance_subspace_ready"
        next_action = {
            "action": "freeze structure-conditioned kernel-property prediction",
            "selected_active_bit_width": selected_width,
            "targets": ["joint_kernel_dimension", "balanced_mask_candidates"],
            "required_baseline": "direct GF(2) empirical kernel",
            "training": "local readiness only",
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_r6_stable_balance_subspace_not_found"
        next_action = {
            "action": "stop the current r6 structure family and audit literature VDS structures",
            "reason": "r4 calibration passes but r6 joint kernels are not usable",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "r4_known_fixture_calibration_pass": r4_calibration_pass,
        "r6_target_checks": target_checks,
        "r6_passing_widths": passing_widths,
        "selected_active_bit_width": selected_width,
        "claim_scope": (
            "local empirical GF(2) balance-subspace readiness; not an all-key proof, "
            "neural result, or claim of inventing the kernel method"
        ),
        "next_action": next_action,
    }


def _summarize_round_width(
    config: StableSubspaceAuditConfig,
    *,
    rounds: int,
    width: int,
    structures: tuple[BitIntegralStructure, ...],
    xor_matrix: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    half = xor_matrix.shape[1] // 2
    details: list[dict[str, Any]] = []
    all_valid = True
    for index, structure in enumerate(structures):
        discovery = xor_matrix[index, :half]
        validation = xor_matrix[index, half:]
        joint = xor_matrix[index]
        discovery_basis = gf2_kernel_basis(discovery)
        validation_basis = gf2_kernel_basis(validation)
        joint_basis = gf2_kernel_basis(joint)
        discovery_rank = OUTPUT_BITS - len(discovery_basis)
        validation_rank = OUTPUT_BITS - len(validation_basis)
        joint_rank = OUTPUT_BITS - len(joint_basis)
        survivors = tuple(
            vector
            for vector in discovery_basis
            if kernel_basis_valid(validation, (vector,))
        )
        survival_fraction = (
            len(survivors) / len(discovery_basis) if discovery_basis else 0.0
        )
        valid = (
            kernel_basis_valid(discovery, discovery_basis)
            and kernel_basis_valid(validation, validation_basis)
            and kernel_basis_valid(joint, joint_basis)
            and discovery_rank == gf2_rank(discovery)
            and validation_rank == gf2_rank(validation)
            and joint_rank == gf2_rank(joint)
        )
        all_valid = all_valid and valid
        weights = [vector.bit_count() for vector in joint_basis]
        details.append(
            {
                "run_id": config.run_id,
                "rounds": rounds,
                "active_bit_width": width,
                "structure_id": structure.structure_id,
                "signature": structure.signature,
                "discovery_rank": discovery_rank,
                "validation_rank": validation_rank,
                "joint_rank": joint_rank,
                "discovery_kernel_dimension": len(discovery_basis),
                "validation_kernel_dimension": len(validation_basis),
                "joint_kernel_dimension": len(joint_basis),
                "discovery_basis_validation_survivors": len(survivors),
                "discovery_basis_validation_survival_fraction": survival_fraction,
                "joint_kernel_basis": ";".join(
                    f"{vector:016X}" for vector in joint_basis
                ),
                "joint_kernel_signature": _basis_signature(joint_basis),
                "minimum_joint_mask_weight": min(weights) if weights else 0,
                "mean_joint_mask_weight": float(np.mean(weights)) if weights else 0.0,
                "maximum_joint_mask_weight": max(weights) if weights else 0,
                "basis_validation_pass": valid,
            }
        )

    dimensions = np.asarray(
        [row["joint_kernel_dimension"] for row in details],
        dtype=np.float64,
    )
    nontrivial = [row for row in details if row["joint_kernel_dimension"] > 0]
    discovery_nontrivial = [
        row for row in details if row["discovery_kernel_dimension"] > 0
    ]
    signatures = {
        str(row["joint_kernel_signature"])
        for row in nontrivial
    }
    summary = {
        "run_id": config.run_id,
        "task": "innovation2_stable_output_balance_subspace_readiness",
        "rounds": rounds,
        "role": (
            "calibration"
            if rounds == CALIBRATION_ROUNDS
            else "reference"
            if rounds == 5
            else "target"
        ),
        "active_bit_width": width,
        "set_size": 1 << width,
        "structures": len(structures),
        "keys_per_structure": xor_matrix.shape[1],
        "key_half_size": half,
        "nontrivial_joint_kernel_structures": len(nontrivial),
        "nontrivial_joint_kernel_fraction": len(nontrivial) / len(structures),
        "mean_joint_kernel_dimension": float(dimensions.mean()),
        "median_joint_kernel_dimension": float(np.median(dimensions)),
        "minimum_joint_kernel_dimension": int(dimensions.min()),
        "maximum_joint_kernel_dimension": int(dimensions.max()),
        "distinct_nontrivial_joint_kernel_signatures": len(signatures),
        "mean_discovery_basis_validation_survival_fraction": (
            float(
                np.mean(
                    [
                        row["discovery_basis_validation_survival_fraction"]
                        for row in discovery_nontrivial
                    ]
                )
            )
            if discovery_nontrivial
            else 0.0
        ),
        "all_basis_validation_pass": all(
            bool(row["basis_validation_pass"]) for row in details
        ),
        "training_performed": False,
    }
    return summary, details, all_valid


def _gf2_rref(words: np.ndarray, *, width: int) -> tuple[np.ndarray, tuple[int, ...]]:
    if width <= 0 or width > 64:
        raise ValueError("width must be between 1 and 64")
    values = np.asarray(words, dtype=np.uint64).reshape(-1)
    shifts = np.arange(width, dtype=np.uint64)
    matrix = ((values[:, None] >> shifts[None, :]) & np.uint64(1)).astype(
        np.uint8
    )
    pivot_row = 0
    pivots: list[int] = []
    for column in range(width):
        candidates = np.flatnonzero(matrix[pivot_row:, column])
        if not len(candidates):
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            matrix[[pivot_row, selected]] = matrix[[selected, pivot_row]]
        other_rows = np.flatnonzero(matrix[:, column])
        other_rows = other_rows[other_rows != pivot_row]
        if len(other_rows):
            matrix[other_rows] ^= matrix[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return matrix, tuple(pivots)


def _basis_signature(basis: tuple[int, ...]) -> str:
    return ";".join(f"{vector:016X}" for vector in basis)


def _crosscheck_scalar(
    calibration_structures: tuple[BitIntegralStructure, ...],
    structure_sets: dict[int, tuple[BitIntegralStructure, ...]],
    xor_matrices: dict[tuple[int, int], np.ndarray],
    keys: tuple[int, ...],
) -> bool:
    key_indices = (0, len(keys) // 2)
    calibration = calibration_structures[0]
    for key_index in key_indices:
        expected = scalar_bit_integral_output_xor(
            calibration,
            rounds=CALIBRATION_ROUNDS,
            key=keys[key_index],
        )
        if expected != int(
            xor_matrices[(CALIBRATION_ROUNDS, 4)][0, key_index]
        ):
            return False
    for rounds in ROUNDS:
        for width in ACTIVE_BIT_WIDTHS:
            structure = structure_sets[width][0]
            for key_index in key_indices:
                expected = scalar_bit_integral_output_xor(
                    structure,
                    rounds=rounds,
                    key=keys[key_index],
                )
                if expected != int(xor_matrices[(rounds, width)][0, key_index]):
                    return False
    return True


def _make_r4_calibration_structures(
    seed: int,
) -> tuple[BitIntegralStructure, ...]:
    rng = np.random.default_rng(seed + 2301)
    structures: list[BitIntegralStructure] = []
    for nibble in range(16):
        active_bits = tuple(range(4 * nibble, 4 * nibble + 4))
        fixed_plaintext = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        for bit in active_bits:
            fixed_plaintext &= ~(1 << bit)
        structures.append(
            BitIntegralStructure(
                structure_id=f"r4-cal-n{nibble:02d}",
                active_bits=active_bits,
                output_nibble=0,
                output_mask=1,
                fixed_plaintext=fixed_plaintext,
            )
        )
    return tuple(structures)


def _metric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _emit(
    callback: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event, payload)
