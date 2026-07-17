from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from blockcipher_nd.tasks.innovation2.integral_bit_transition_audit import (
    BitIntegralStructure,
    scalar_bit_integral_output_xor,
)
from blockcipher_nd.tasks.innovation2.integral_hwang_readiness import (
    _collect_xor_words,
    paper_basis_masks,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import make_keys
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
    kernel_basis_valid,
)


WINDOW_STARTS = tuple(range(16))
ANCHOR_STARTS = (0, 4, 8, 12)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class CyclicGeometryDiversityConfig:
    run_id: str
    seed: int = 0
    rounds: int = 7
    keys: int = 128
    key_chunk_size: int = 4

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 7:
            raise ValueError("the frozen geometry audit requires PRESENT r7")
        if self.keys != 128:
            raise ValueError("the frozen geometry audit requires exactly 128 keys")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")


def cyclic_nibble_window(start_nibble: int) -> tuple[int, ...]:
    if not 0 <= start_nibble < 16:
        raise ValueError("start_nibble must be in [0, 15]")
    nibbles = tuple((start_nibble + offset) % 16 for offset in range(4))
    return tuple(sorted(4 * nibble + bit for nibble in nibbles for bit in range(4)))


def run_cyclic_geometry_diversity_audit(
    config: CyclicGeometryDiversityConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = make_keys(count=config.keys, seed=config.seed + 3301)
    half = config.keys // 2
    structures = {
        start: BitIntegralStructure(
            structure_id=f"present-r7-cyclic-window-{start:02d}",
            active_bits=cyclic_nibble_window(start),
            output_nibble=0,
            output_mask=1,
            fixed_plaintext=0,
        )
        for start in WINDOW_STARTS
    }
    xor_words_by_start: dict[int, np.ndarray] = {}
    for start, structure in structures.items():
        xor_words_by_start[start] = _collect_xor_words(
            structure,
            keys,
            rounds=config.rounds,
            key_chunk_size=config.key_chunk_size,
            progress_callback=progress_callback,
        )
    scalar_matches = all(
        int(xor_words_by_start[start][0])
        == scalar_bit_integral_output_xor(
            structures[start],
            rounds=config.rounds,
            key=keys[0],
        )
        for start in ANCHOR_STARTS
    )
    rows, basis_rows, readiness, gate = evaluate_cyclic_geometry_diversity(
        config,
        xor_words_by_start=xor_words_by_start,
        scalar_matches=scalar_matches,
        key_halves_disjoint=set(keys[:half]).isdisjoint(keys[half:]),
    )
    return {
        "rows": rows,
        "basis_rows": basis_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_present_r7_cyclic_geometry_kernel_diversity",
            "cipher": "PRESENT-80",
            "rounds": config.rounds,
            "seed": config.seed,
            "key_generation_seed": config.seed + 3301,
            "keys": config.keys,
            "key_half_size": half,
            "key_chunk_size": config.key_chunk_size,
            "structures": len(WINDOW_STARTS),
            "plaintexts_per_structure_per_key": 1 << 16,
            "window_nibbles": {
                str(start): [((start + offset) % 16) for offset in range(4)]
                for start in WINDOW_STARTS
            },
            "active_bits": {
                str(start): list(structure.active_bits)
                for start, structure in structures.items()
            },
            "fixed_plaintext": "0x0000000000000000",
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def evaluate_cyclic_geometry_diversity(
    config: CyclicGeometryDiversityConfig,
    *,
    xor_words_by_start: dict[int, np.ndarray],
    scalar_matches: bool,
    key_halves_disjoint: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
    dict[str, Any],
]:
    if set(xor_words_by_start) != set(WINDOW_STARTS):
        raise ValueError("xor_words_by_start must contain starts 0 through 15")
    half = config.keys // 2
    paper_masks = paper_basis_masks(output_mapping="direct")
    paper_span = _bounded_span(paper_masks, max_dimension=4)
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    all_joint_bases_validate = True
    for start in WINDOW_STARTS:
        words = np.asarray(xor_words_by_start[start], dtype=np.uint64)
        if words.shape != (config.keys,):
            raise ValueError(f"xor words for start {start} must have shape ({config.keys},)")
        discovery = words[:half]
        validation = words[half:]
        discovery_basis = gf2_kernel_basis(discovery)
        validation_basis = gf2_kernel_basis(validation)
        joint_basis = gf2_kernel_basis(words)
        validates = kernel_basis_valid(discovery, joint_basis) and kernel_basis_valid(
            validation, joint_basis
        )
        all_joint_bases_validate &= validates
        signature = ":".join(f"{vector:016X}" for vector in joint_basis)
        rows.append(
            {
                "run_id": config.run_id,
                "start_nibble": start,
                "active_nibbles": "-".join(
                    str((start + offset) % 16) for offset in range(4)
                ),
                "discovery_kernel_dimension": len(discovery_basis),
                "validation_kernel_dimension": len(validation_basis),
                "joint_kernel_dimension": len(joint_basis),
                "joint_rank": 64 - len(joint_basis),
                "joint_basis_signature": signature,
                "joint_basis_valid_both_halves": validates,
                "joint_kernel_equals_hwang_span": (
                    len(joint_basis) == 4
                    and all(vector in paper_span for vector in joint_basis)
                ),
                "nonzero_output_parity_words": int(np.count_nonzero(words)),
            }
        )
        for basis_index, vector in enumerate(joint_basis):
            basis_rows.append(
                {
                    "run_id": config.run_id,
                    "start_nibble": start,
                    "basis_index": basis_index,
                    "vector_hex": f"0x{vector:016X}",
                    "vector_weight": vector.bit_count(),
                    "in_hwang_paper_span": vector in paper_span,
                }
            )
    anchor = next(row for row in rows if int(row["start_nibble"]) == 12)
    signatures = {str(row["joint_basis_signature"]) for row in rows}
    nontrivial = sum(int(row["joint_kernel_dimension"]) > 0 for row in rows)
    readiness = {
        "sixteen_cyclic_windows_present": len(rows) == 16,
        "key_halves_nonempty_and_disjoint": key_halves_disjoint,
        "four_embedded_anchors_match_scalar": scalar_matches,
        "all_output_parity_words_nonzero_somewhere": all(
            int(row["nonzero_output_parity_words"]) > 0 for row in rows
        ),
        "all_joint_bases_validate_both_halves": all_joint_bases_validate,
        "hwang_start12_anchor_exact": bool(anchor["joint_kernel_equals_hwang_span"]),
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "discovery_kernel_dimension",
                "validation_kernel_dimension",
                "joint_kernel_dimension",
                "joint_rank",
                "nonzero_output_parity_words",
            )
        ),
    }
    gate = adjudicate_cyclic_geometry_diversity(
        config,
        rows,
        readiness,
        distinct_signatures=len(signatures),
        nontrivial_structures=nontrivial,
    )
    return rows, basis_rows, readiness, gate


def adjudicate_cyclic_geometry_diversity(
    config: CyclicGeometryDiversityConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
    *,
    distinct_signatures: int,
    nontrivial_structures: int,
) -> dict[str, Any]:
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    diversity_pass = distinct_signatures >= 4 and nontrivial_structures >= 8
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_cyclic_geometry_diversity_protocol_invalid"
        next_action = {
            "action": "repair cyclic windows, scalar anchors, or Hwang calibration",
            "training": False,
            "remote_scale": False,
        }
    elif diversity_pass:
        status = "pass"
        decision = "innovation2_cyclic_geometry_kernel_diversity_ready"
        next_action = {
            "action": "rebuild structure-mask labels with geometry-disjoint controls",
            "next_adjudication": "E15 expanded output-label shortcut audit",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_cyclic_geometry_kernel_diversity_insufficient"
        next_action = {
            "action": "vary inactive context or use non-contiguous active geometries",
            "reason": "cyclic nibble windows do not create enough stable kernel signatures",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "distinct_joint_kernel_signatures": distinct_signatures,
        "nontrivial_joint_kernel_structures": nontrivial_structures,
        "claim_scope": (
            "16-structure local cyclic-geometry kernel readiness under 128 sampled keys; "
            "not a neural result or all-key proof"
        ),
        "next_action": next_action,
    }


def _bounded_span(basis: tuple[int, ...], *, max_dimension: int) -> set[int]:
    if len(basis) > max_dimension:
        raise ValueError("basis exceeds bounded span dimension")
    values = {0}
    for vector in basis:
        values |= {value ^ vector for value in tuple(values)}
    return values
