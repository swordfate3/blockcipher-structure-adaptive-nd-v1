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


BASE_STARTS = (0, 4, 8, 12)
P_POWERS = (0, 1, 2)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class TopologyGeometryConfig:
    run_id: str
    seed: int = 0
    rounds: int = 7
    keys: int = 128
    key_chunk_size: int = 4

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 7:
            raise ValueError("the frozen topology audit requires PRESENT r7")
        if self.keys != 128:
            raise ValueError("the frozen topology audit requires exactly 128 keys")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")


def present_p_bit(bit: int) -> int:
    if not 0 <= bit < 64:
        raise ValueError("bit must be in [0, 63]")
    return (16 * bit) % 63 if bit < 63 else 63


def topology_geometries() -> dict[str, tuple[int, ...]]:
    geometries: dict[str, tuple[int, ...]] = {}
    for start in BASE_STARTS:
        base = tuple(range(4 * start, 4 * (start + 4)))
        current = base
        for power in P_POWERS:
            geometry_id = f"block{start:02d}_p{power}"
            geometries[geometry_id] = tuple(sorted(current))
            current = tuple(present_p_bit(bit) for bit in current)
    if len(geometries) != 12 or len(set(geometries.values())) != 12:
        raise AssertionError("topology geometry family must contain 12 unique sets")
    return geometries


def run_topology_geometry_audit(
    config: TopologyGeometryConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = make_keys(count=config.keys, seed=config.seed + 3301)
    half = config.keys // 2
    geometry_bits = topology_geometries()
    structures = {
        geometry_id: BitIntegralStructure(
            structure_id=f"present-r7-{geometry_id}",
            active_bits=active_bits,
            output_nibble=0,
            output_mask=1,
            fixed_plaintext=0,
        )
        for geometry_id, active_bits in geometry_bits.items()
    }
    xor_words_by_geometry: dict[str, np.ndarray] = {}
    for geometry_id, structure in structures.items():
        xor_words_by_geometry[geometry_id] = _collect_xor_words(
            structure,
            keys,
            rounds=config.rounds,
            key_chunk_size=config.key_chunk_size,
            progress_callback=progress_callback,
        )
    scalar_matches = all(
        int(xor_words_by_geometry[f"block{start:02d}_p0"][0])
        == scalar_bit_integral_output_xor(
            structures[f"block{start:02d}_p0"],
            rounds=config.rounds,
            key=keys[0],
        )
        for start in BASE_STARTS
    )
    rows, basis_rows, readiness, gate = evaluate_topology_geometry_audit(
        config,
        xor_words_by_geometry=xor_words_by_geometry,
        scalar_matches=scalar_matches,
        key_halves_disjoint=set(keys[:half]).isdisjoint(keys[half:]),
    )
    return {
        "rows": rows,
        "basis_rows": basis_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_present_r7_topology_geometry_kernel_diversity",
            "cipher": "PRESENT-80",
            "rounds": config.rounds,
            "seed": config.seed,
            "key_generation_seed": config.seed + 3301,
            "keys": config.keys,
            "key_half_size": half,
            "key_chunk_size": config.key_chunk_size,
            "structures": len(structures),
            "plaintexts_per_structure_per_key": 1 << 16,
            "geometries": {
                geometry_id: list(active_bits)
                for geometry_id, active_bits in geometry_bits.items()
            },
            "fixed_plaintext": "0x0000000000000000",
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def evaluate_topology_geometry_audit(
    config: TopologyGeometryConfig,
    *,
    xor_words_by_geometry: dict[str, np.ndarray],
    scalar_matches: bool,
    key_halves_disjoint: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
    dict[str, Any],
]:
    geometries = topology_geometries()
    if set(xor_words_by_geometry) != set(geometries):
        raise ValueError("xor_words_by_geometry must contain the 12 frozen geometries")
    half = config.keys // 2
    paper_masks = paper_basis_masks(output_mapping="direct")
    paper_span = _bounded_span(paper_masks, max_dimension=4)
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    all_joint_bases_validate = True
    for geometry_id, active_bits in geometries.items():
        words = np.asarray(xor_words_by_geometry[geometry_id], dtype=np.uint64)
        if words.shape != (config.keys,):
            raise ValueError(
                f"xor words for {geometry_id} must have shape ({config.keys},)"
            )
        discovery = words[:half]
        validation = words[half:]
        discovery_basis = gf2_kernel_basis(discovery)
        validation_basis = gf2_kernel_basis(validation)
        joint_basis = gf2_kernel_basis(words)
        validates = kernel_basis_valid(discovery, joint_basis) and kernel_basis_valid(
            validation, joint_basis
        )
        all_joint_bases_validate &= validates
        start = int(geometry_id[5:7])
        power = int(geometry_id[-1])
        signature = ":".join(f"{vector:016X}" for vector in joint_basis)
        rows.append(
            {
                "run_id": config.run_id,
                "geometry_id": geometry_id,
                "base_start_nibble": start,
                "p_power": power,
                "active_bits": "-".join(str(bit) for bit in active_bits),
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
                    "geometry_id": geometry_id,
                    "base_start_nibble": start,
                    "p_power": power,
                    "basis_index": basis_index,
                    "vector_hex": f"0x{vector:016X}",
                    "vector_weight": vector.bit_count(),
                    "in_hwang_paper_span": vector in paper_span,
                }
            )
    anchor = next(row for row in rows if row["geometry_id"] == "block12_p0")
    signatures = {str(row["joint_basis_signature"]) for row in rows}
    nontrivial = sum(int(row["joint_kernel_dimension"]) > 0 for row in rows)
    readiness = {
        "twelve_unique_topology_geometries_present": len(rows) == 12,
        "key_halves_nonempty_and_disjoint": key_halves_disjoint,
        "four_base_blocks_match_scalar": scalar_matches,
        "all_output_parity_words_nonzero_somewhere": all(
            int(row["nonzero_output_parity_words"]) > 0 for row in rows
        ),
        "all_joint_bases_validate_both_halves": all_joint_bases_validate,
        "hwang_block12_p0_anchor_exact": bool(
            anchor["joint_kernel_equals_hwang_span"]
        ),
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
    gate = adjudicate_topology_geometry_audit(
        config,
        rows,
        readiness,
        distinct_signatures=len(signatures),
        nontrivial_structures=nontrivial,
    )
    return rows, basis_rows, readiness, gate


def adjudicate_topology_geometry_audit(
    config: TopologyGeometryConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
    *,
    distinct_signatures: int,
    nontrivial_structures: int,
) -> dict[str, Any]:
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    diversity_pass = distinct_signatures >= 4 and nontrivial_structures >= 6
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_topology_geometry_protocol_invalid"
        next_action = {
            "action": "repair P-layer geometries, scalar anchors, or Hwang calibration",
            "training": False,
            "remote_scale": False,
        }
    elif diversity_pass:
        status = "pass"
        decision = "innovation2_topology_geometry_kernel_diversity_ready"
        next_action = {
            "action": "rebuild structure-mask labels on topology geometries",
            "next_adjudication": "E16 topology output-label shortcut audit",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_topology_geometry_kernel_diversity_insufficient"
        next_action = {
            "action": "vary inactive context or stop PRESENT r7 multi-structure route",
            "reason": "P-layer orbit geometries do not create enough stable kernels",
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
            "12-structure local PRESENT-topology kernel readiness under 128 sampled "
            "keys; not a neural result or all-key proof"
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
