from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from blockcipher_nd.tasks.innovation2.integral_bit_transition_audit import (
    BitIntegralStructure,
    bit_integral_output_xor_matrix,
    scalar_bit_integral_output_xor,
)
from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    present_round_key_matrix,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import make_keys
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
    kernel_basis_valid,
)


PAPER_BASIS_BITS = ((0,), (4, 12), (16, 48), (20, 28, 52, 60))
INPUT_ORIENTATIONS = {
    "low_0_15": tuple(range(16)),
    "high_48_63": tuple(range(48, 64)),
}
OUTPUT_MAPPINGS = ("direct", "reflected")
CONTIGUOUS_ACTIVE_BLOCKS = {
    "block_0_15": tuple(range(0, 16)),
    "block_16_31": tuple(range(16, 32)),
    "block_32_47": tuple(range(32, 48)),
    "block_48_63": tuple(range(48, 64)),
}
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class HwangReadinessConfig:
    run_id: str
    seed: int = 0
    rounds: int = 7
    keys: int = 8
    key_chunk_size: int = 1

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 7:
            raise ValueError("the frozen Hwang readiness audit requires PRESENT r7")
        if self.keys < 4 or self.keys % 2:
            raise ValueError("keys must be even and at least four")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")


@dataclass(frozen=True)
class HwangKernelConvergenceConfig:
    run_id: str
    seed: int = 0
    rounds: int = 7
    keys: int = 128
    key_chunk_size: int = 1
    input_orientation: str = "low_0_15"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 7:
            raise ValueError("the frozen convergence audit requires PRESENT r7")
        if self.keys != 128:
            raise ValueError("the frozen convergence audit requires exactly 128 keys")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")
        if self.input_orientation not in INPUT_ORIENTATIONS:
            raise ValueError(f"unknown input orientation: {self.input_orientation}")


@dataclass(frozen=True)
class ActiveBlockKernelDiversityConfig:
    run_id: str
    seed: int = 0
    rounds: int = 7
    keys: int = 128
    key_chunk_size: int = 1

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 7:
            raise ValueError("the frozen diversity audit requires PRESENT r7")
        if self.keys != 128:
            raise ValueError("the frozen diversity audit requires exactly 128 keys")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")


def paper_basis_masks(*, output_mapping: str) -> tuple[int, ...]:
    if output_mapping not in OUTPUT_MAPPINGS:
        raise ValueError(f"unknown output mapping: {output_mapping}")
    return tuple(
        sum(1 << _map_output_bit(bit, output_mapping) for bit in bits)
        for bits in PAPER_BASIS_BITS
    )


def control_masks(*, seed: int, output_mapping: str) -> tuple[int, ...]:
    if output_mapping not in OUTPUT_MAPPINGS:
        raise ValueError(f"unknown output mapping: {output_mapping}")
    paper_coordinates = paper_basis_masks(output_mapping="direct")
    paper_span = _span(paper_coordinates)
    rng = np.random.default_rng(seed + 1979)
    controls: list[int] = []
    used = set(paper_span)
    for weight in (1, 2, 2, 4):
        while True:
            bits = tuple(
                sorted(
                    int(value)
                    for value in rng.choice(64, size=weight, replace=False)
                )
            )
            mask = sum(1 << bit for bit in bits)
            if mask in used:
                continue
            used.add(mask)
            controls.append(mask)
            break
    return tuple(_map_mask(mask, output_mapping) for mask in controls)


def run_hwang_readiness_audit(
    config: HwangReadinessConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = make_keys(count=config.keys, seed=config.seed + 2301)
    xor_words_by_input: dict[str, np.ndarray] = {}
    structures: dict[str, BitIntegralStructure] = {}
    for input_orientation, active_bits in INPUT_ORIENTATIONS.items():
        structure = BitIntegralStructure(
            structure_id=f"hwang-r7-{input_orientation}",
            active_bits=active_bits,
            output_nibble=0,
            output_mask=1,
            fixed_plaintext=0,
        )
        structures[input_orientation] = structure
        xor_words_by_input[input_orientation] = _collect_xor_words(
            structure,
            keys,
            rounds=config.rounds,
            key_chunk_size=config.key_chunk_size,
            progress_callback=progress_callback,
        )

    scalar_matches = all(
        int(xor_words_by_input[input_orientation][0])
        == scalar_bit_integral_output_xor(
            structure,
            rounds=config.rounds,
            key=keys[0],
        )
        for input_orientation, structure in structures.items()
    )
    rows, mask_rows = summarize_protocols(
        config,
        xor_words_by_input=xor_words_by_input,
    )
    half = config.keys // 2
    readiness = {
        "four_protocol_candidates_present": (
            len(rows) == len(INPUT_ORIENTATIONS) * len(OUTPUT_MAPPINGS)
        ),
        "key_halves_nonempty_and_disjoint": (
            half > 0 and set(keys[:half]).isdisjoint(keys[half:])
        ),
        "vectorized_output_xor_matches_scalar": scalar_matches,
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "paper_mask_failures_discovery",
                "paper_mask_failures_validation",
                "paper_mask_failures_joint",
                "control_mask_failures_joint",
                "joint_kernel_dimension",
            )
        ),
    }
    gate = adjudicate_hwang_readiness(config, rows, readiness)
    return {
        "rows": rows,
        "mask_rows": mask_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_present_r7_hwang_kernel_bitorder_readiness",
            "cipher": "PRESENT-80",
            "rounds": config.rounds,
            "seed": config.seed,
            "keys": config.keys,
            "key_half_size": half,
            "key_chunk_size": config.key_chunk_size,
            "plaintexts_per_structure": 1 << 16,
            "input_orientations": list(INPUT_ORIENTATIONS),
            "output_mappings": list(OUTPUT_MAPPINGS),
            "paper_basis_bits": [list(bits) for bits in PAPER_BASIS_BITS],
            "key_fingerprints": [f"{key:020X}" for key in keys],
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def run_hwang_kernel_convergence_audit(
    config: HwangKernelConvergenceConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = make_keys(count=config.keys, seed=config.seed + 3301)
    e10_keys = make_keys(count=8, seed=config.seed + 2301)
    structure = BitIntegralStructure(
        structure_id=f"hwang-r7-{config.input_orientation}",
        active_bits=INPUT_ORIENTATIONS[config.input_orientation],
        output_nibble=0,
        output_mask=1,
        fixed_plaintext=0,
    )
    xor_words = _collect_xor_words(
        structure,
        keys,
        rounds=config.rounds,
        key_chunk_size=config.key_chunk_size,
        progress_callback=progress_callback,
    )
    half = config.keys // 2
    scalar_indices = (0, half)
    scalar_matches = all(
        int(xor_words[index])
        == scalar_bit_integral_output_xor(
            structure,
            rounds=config.rounds,
            key=keys[index],
        )
        for index in scalar_indices
    )
    rows, basis_rows, readiness, gate = evaluate_hwang_kernel_convergence(
        config,
        xor_words=xor_words,
        scalar_matches=scalar_matches,
        key_halves_disjoint=set(keys[:half]).isdisjoint(keys[half:]),
        e10_keys_disjoint=set(keys).isdisjoint(e10_keys),
    )
    return {
        "rows": rows,
        "basis_rows": basis_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_present_r7_hwang_kernel_convergence",
            "cipher": "PRESENT-80",
            "rounds": config.rounds,
            "seed": config.seed,
            "key_generation_seed": config.seed + 3301,
            "keys": config.keys,
            "key_half_size": half,
            "key_chunk_size": config.key_chunk_size,
            "plaintexts_per_structure": structure.set_size,
            "active_bits": list(structure.active_bits),
            "input_orientation": config.input_orientation,
            "fixed_plaintext": f"0x{structure.fixed_plaintext:016X}",
            "output_mapping": "direct",
            "paper_basis_bits": [list(bits) for bits in PAPER_BASIS_BITS],
            "key_fingerprints": [f"{key:020X}" for key in keys],
            "e10_keys_disjoint": readiness["all_e11_keys_disjoint_from_e10"],
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def evaluate_hwang_kernel_convergence(
    config: HwangKernelConvergenceConfig,
    *,
    xor_words: np.ndarray,
    scalar_matches: bool,
    key_halves_disjoint: bool,
    e10_keys_disjoint: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
    dict[str, Any],
]:
    xor_words = np.asarray(xor_words, dtype=np.uint64)
    if xor_words.shape != (config.keys,):
        raise ValueError(f"xor_words must have shape ({config.keys},)")
    half = config.keys // 2
    paper_masks = paper_basis_masks(output_mapping="direct")
    nonpaper_masks = control_masks(seed=config.seed, output_mapping="direct")
    paper_span = _span(paper_masks)
    split_words = {
        "discovery": xor_words[:half],
        "validation": xor_words[half:],
        "joint": xor_words,
    }
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    for split, words in split_words.items():
        basis = gf2_kernel_basis(words)
        kernel_equals_paper_span = (
            len(basis) == 4 and all(vector in paper_span for vector in basis)
        )
        row = {
            "run_id": config.run_id,
            "input_orientation": config.input_orientation,
            "split": split,
            "keys": len(words),
            "rank": 64 - len(basis),
            "kernel_dimension": len(basis),
            "paper_mask_failures": _mask_failure_count(words, paper_masks),
            "control_mask_failures": _mask_failure_count(words, nonpaper_masks),
            "paper_masks_in_kernel": kernel_basis_valid(words, paper_masks),
            "kernel_equals_paper_span": kernel_equals_paper_span,
            "nonzero_output_parity_words": int(np.count_nonzero(words)),
        }
        rows.append(row)
        for basis_index, vector in enumerate(basis):
            basis_rows.append(
                {
                    "run_id": config.run_id,
                    "split": split,
                    "basis_index": basis_index,
                    "vector_hex": f"0x{vector:016X}",
                    "vector_weight": vector.bit_count(),
                    "in_paper_span": vector in paper_span,
                }
            )
    readiness = {
        "three_kernel_splits_present": len(rows) == 3,
        "key_halves_nonempty_and_disjoint": key_halves_disjoint,
        "all_e11_keys_disjoint_from_e10": e10_keys_disjoint,
        "vectorized_output_xor_matches_scalar_in_both_halves": scalar_matches,
        "paper_basis_has_dimension_four": len(paper_span) == 16,
        "output_parity_words_not_all_zero": bool(np.count_nonzero(xor_words)),
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "rank",
                "kernel_dimension",
                "paper_mask_failures",
                "control_mask_failures",
                "nonzero_output_parity_words",
            )
        ),
    }
    gate = adjudicate_hwang_kernel_convergence(config, rows, readiness)
    return rows, basis_rows, readiness, gate


def adjudicate_hwang_kernel_convergence(
    config: HwangKernelConvergenceConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
) -> dict[str, Any]:
    by_split = {str(row["split"]): row for row in rows}
    discovery = by_split.get("discovery", {})
    validation = by_split.get("validation", {})
    joint = by_split.get("joint", {})
    paper_masks_stable = all(
        row.get("paper_masks_in_kernel") is True
        and int(row.get("paper_mask_failures", -1)) == 0
        for row in (discovery, validation, joint)
    )
    exact_joint_kernel = (
        int(joint.get("kernel_dimension", -1)) == 4
        and joint.get("kernel_equals_paper_span") is True
    )
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_present_r7_hwang_convergence_protocol_invalid"
        next_action = {
            "action": "repair key separation, scalar parity, or kernel computation",
            "training": False,
            "remote_scale": False,
        }
    elif paper_masks_stable and exact_joint_kernel:
        status = "pass"
        decision = "innovation2_present_r7_hwang_kernel_reproduced"
        next_action = {
            "action": "freeze calibrated kernel and audit output-property structure diversity",
            "next_adjudication": (
                "E12 contiguous 16-bit active-block kernel diversity readiness"
            ),
            "required_controls": [
                "direct GF(2) kernel",
                "training-only field marginals",
                "mask-matched controls",
            ],
            "training": False,
            "remote_scale": False,
        }
    elif paper_masks_stable:
        status = "hold"
        decision = "innovation2_present_r7_hwang_kernel_underconstrained"
        next_action = {
            "action": "freeze a 256-key convergence audit",
            "reason": "paper masks survive but empirical kernel dimension remains above four",
            "training": False,
            "remote_scale": "evaluate after cache/readiness audit",
        }
    else:
        status = "hold"
        decision = "innovation2_present_r7_hwang_kernel_not_reproduced"
        next_action = {
            "action": "audit plaintext context, round boundary, and author implementation",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "input_orientation": config.input_orientation,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "paper_masks_stable_both_halves": paper_masks_stable,
        "joint_kernel_dimension": int(joint.get("kernel_dimension", -1)),
        "joint_kernel_equals_paper_span": bool(
            joint.get("kernel_equals_paper_span", False)
        ),
        "claim_scope": (
            "128 fresh-key empirical reproduction of Hwang et al. PRESENT r7 "
            "linear balance-mask kernel; not an all-key proof or neural result"
        ),
        "next_action": next_action,
    }


def run_active_block_kernel_diversity_audit(
    config: ActiveBlockKernelDiversityConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = make_keys(count=config.keys, seed=config.seed + 3301)
    half = config.keys // 2
    structures = {
        block_id: BitIntegralStructure(
            structure_id=f"present-r7-{block_id}",
            active_bits=active_bits,
            output_nibble=0,
            output_mask=1,
            fixed_plaintext=0,
        )
        for block_id, active_bits in CONTIGUOUS_ACTIVE_BLOCKS.items()
    }
    xor_words_by_block: dict[str, np.ndarray] = {}
    for block_id, structure in structures.items():
        xor_words_by_block[block_id] = _collect_xor_words(
            structure,
            keys,
            rounds=config.rounds,
            key_chunk_size=config.key_chunk_size,
            progress_callback=progress_callback,
        )
    scalar_matches = all(
        int(xor_words_by_block[block_id][0])
        == scalar_bit_integral_output_xor(
            structure,
            rounds=config.rounds,
            key=keys[0],
        )
        for block_id, structure in structures.items()
    )
    rows, basis_rows, readiness, gate = evaluate_active_block_kernel_diversity(
        config,
        xor_words_by_block=xor_words_by_block,
        scalar_matches=scalar_matches,
        key_halves_disjoint=set(keys[:half]).isdisjoint(keys[half:]),
    )
    return {
        "rows": rows,
        "basis_rows": basis_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_present_r7_active_block_kernel_diversity",
            "cipher": "PRESENT-80",
            "rounds": config.rounds,
            "seed": config.seed,
            "key_generation_seed": config.seed + 3301,
            "keys": config.keys,
            "key_half_size": half,
            "key_chunk_size": config.key_chunk_size,
            "plaintexts_per_structure_per_key": 1 << 16,
            "active_blocks": {
                block_id: list(active_bits)
                for block_id, active_bits in CONTIGUOUS_ACTIVE_BLOCKS.items()
            },
            "fixed_plaintext": "0x0000000000000000",
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def evaluate_active_block_kernel_diversity(
    config: ActiveBlockKernelDiversityConfig,
    *,
    xor_words_by_block: dict[str, np.ndarray],
    scalar_matches: bool,
    key_halves_disjoint: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
    dict[str, Any],
]:
    if set(xor_words_by_block) != set(CONTIGUOUS_ACTIVE_BLOCKS):
        raise ValueError("xor_words_by_block must contain the four frozen blocks")
    half = config.keys // 2
    paper_masks = paper_basis_masks(output_mapping="direct")
    nonpaper_masks = control_masks(seed=config.seed, output_mapping="direct")
    paper_span = _span(paper_masks)
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    all_joint_bases_validate_halves = True
    for block_id, active_bits in CONTIGUOUS_ACTIVE_BLOCKS.items():
        words = np.asarray(xor_words_by_block[block_id], dtype=np.uint64)
        if words.shape != (config.keys,):
            raise ValueError(f"xor words for {block_id} must have shape ({config.keys},)")
        discovery = words[:half]
        validation = words[half:]
        discovery_basis = gf2_kernel_basis(discovery)
        validation_basis = gf2_kernel_basis(validation)
        joint_basis = gf2_kernel_basis(words)
        basis_valid_both_halves = kernel_basis_valid(
            discovery, joint_basis
        ) and kernel_basis_valid(validation, joint_basis)
        all_joint_bases_validate_halves &= basis_valid_both_halves
        signature = ":".join(f"{vector:016X}" for vector in joint_basis)
        rows.append(
            {
                "run_id": config.run_id,
                "block_id": block_id,
                "active_start": active_bits[0],
                "active_stop": active_bits[-1],
                "discovery_kernel_dimension": len(discovery_basis),
                "validation_kernel_dimension": len(validation_basis),
                "joint_kernel_dimension": len(joint_basis),
                "joint_rank": 64 - len(joint_basis),
                "joint_basis_signature": signature,
                "joint_basis_valid_both_halves": basis_valid_both_halves,
                "joint_kernel_equals_paper_span": (
                    len(joint_basis) == 4
                    and all(vector in paper_span for vector in joint_basis)
                ),
                "paper_mask_failures": _mask_failure_count(words, paper_masks),
                "control_mask_failures": _mask_failure_count(
                    words, nonpaper_masks
                ),
                "nonzero_output_parity_words": int(np.count_nonzero(words)),
            }
        )
        for basis_index, vector in enumerate(joint_basis):
            basis_rows.append(
                {
                    "run_id": config.run_id,
                    "block_id": block_id,
                    "basis_index": basis_index,
                    "vector_hex": f"0x{vector:016X}",
                    "vector_weight": vector.bit_count(),
                    "in_hwang_paper_span": vector in paper_span,
                }
            )
    high_anchor = next(row for row in rows if row["block_id"] == "block_48_63")
    signatures = {str(row["joint_basis_signature"]) for row in rows}
    nontrivial_structures = sum(
        int(row["joint_kernel_dimension"]) > 0 for row in rows
    )
    readiness = {
        "four_frozen_blocks_present": len(rows) == 4,
        "key_halves_nonempty_and_disjoint": key_halves_disjoint,
        "vectorized_output_xor_matches_scalar_all_blocks": scalar_matches,
        "all_output_parity_words_nonzero_somewhere": all(
            int(row["nonzero_output_parity_words"]) > 0 for row in rows
        ),
        "all_joint_bases_validate_both_halves": all_joint_bases_validate_halves,
        "hwang_high16_anchor_exact_four_dimensional_span": bool(
            high_anchor["joint_kernel_equals_paper_span"]
        ),
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "discovery_kernel_dimension",
                "validation_kernel_dimension",
                "joint_kernel_dimension",
                "joint_rank",
                "paper_mask_failures",
                "control_mask_failures",
                "nonzero_output_parity_words",
            )
        ),
    }
    gate = adjudicate_active_block_kernel_diversity(
        config,
        rows,
        readiness,
        distinct_signatures=len(signatures),
        nontrivial_structures=nontrivial_structures,
    )
    return rows, basis_rows, readiness, gate


def adjudicate_active_block_kernel_diversity(
    config: ActiveBlockKernelDiversityConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
    *,
    distinct_signatures: int,
    nontrivial_structures: int,
) -> dict[str, Any]:
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    diversity_pass = distinct_signatures >= 2 and nontrivial_structures >= 2
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_present_r7_active_block_diversity_protocol_invalid"
        next_action = {
            "action": "repair block generation, scalar parity, or Hwang anchor",
            "training": False,
            "remote_scale": False,
        }
    elif diversity_pass:
        status = "pass"
        decision = "innovation2_present_r7_active_block_kernel_diversity_ready"
        next_action = {
            "action": "build structure-mask label table and audit marginal predictability",
            "next_adjudication": (
                "E13 structure-mask output-property benchmark readiness"
            ),
            "required_controls": [
                "direct GF(2) kernel",
                "active-block marginal prior",
                "mask-weight marginal prior",
                "label shuffle",
            ],
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_present_r7_active_block_kernel_not_diverse"
        next_action = {
            "action": "hold high16 active bits fixed and vary inactive context",
            "reason": "contiguous active-block position does not diversify labels",
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
            "four-structure local kernel-diversity readiness under 128 sampled keys; "
            "not a neural result or all-key proof"
        ),
        "next_action": next_action,
    }
def summarize_protocols(
    config: HwangReadinessConfig,
    *,
    xor_words_by_input: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    half = config.keys // 2
    for input_orientation in INPUT_ORIENTATIONS:
        xor_words = np.asarray(xor_words_by_input[input_orientation], dtype=np.uint64)
        if xor_words.shape != (config.keys,):
            raise ValueError(
                f"xor words for {input_orientation} must have shape ({config.keys},)"
            )
        discovery = xor_words[:half]
        validation = xor_words[half:]
        joint_basis = gf2_kernel_basis(xor_words)
        for output_mapping in OUTPUT_MAPPINGS:
            paper_masks = paper_basis_masks(output_mapping=output_mapping)
            nonpaper_masks = control_masks(
                seed=config.seed,
                output_mapping=output_mapping,
            )
            paper_discovery_failures = _mask_failure_count(discovery, paper_masks)
            paper_validation_failures = _mask_failure_count(validation, paper_masks)
            paper_joint_failures = _mask_failure_count(xor_words, paper_masks)
            control_joint_failures = _mask_failure_count(xor_words, nonpaper_masks)
            paper_in_kernel = kernel_basis_valid(xor_words, paper_masks)
            nonzero_words = int(np.count_nonzero(xor_words))
            candidate_pass = (
                paper_discovery_failures == 0
                and paper_validation_failures == 0
                and paper_joint_failures == 0
                and paper_in_kernel
                and nonzero_words > 0
                and control_joint_failures > 0
            )
            candidate_id = f"{input_orientation}__{output_mapping}"
            rows.append(
                {
                    "run_id": config.run_id,
                    "candidate_id": candidate_id,
                    "input_orientation": input_orientation,
                    "output_mapping": output_mapping,
                    "keys": config.keys,
                    "paper_mask_failures_discovery": paper_discovery_failures,
                    "paper_mask_failures_validation": paper_validation_failures,
                    "paper_mask_failures_joint": paper_joint_failures,
                    "control_mask_failures_joint": control_joint_failures,
                    "paper_masks_in_joint_kernel": paper_in_kernel,
                    "nonzero_output_parity_words": nonzero_words,
                    "joint_kernel_dimension": len(joint_basis),
                    "candidate_pass": candidate_pass,
                }
            )
            for role, masks in (("paper", paper_masks), ("control", nonpaper_masks)):
                for mask_index, mask in enumerate(masks):
                    mask_rows.append(
                        {
                            "run_id": config.run_id,
                            "candidate_id": candidate_id,
                            "input_orientation": input_orientation,
                            "output_mapping": output_mapping,
                            "mask_role": role,
                            "mask_index": mask_index,
                            "mask_hex": f"0x{mask:016X}",
                            "mask_weight": mask.bit_count(),
                            "discovery_failures": _single_mask_failure_count(
                                discovery, mask
                            ),
                            "validation_failures": _single_mask_failure_count(
                                validation, mask
                            ),
                            "joint_failures": _single_mask_failure_count(
                                xor_words, mask
                            ),
                        }
                    )
    return rows, mask_rows


def adjudicate_hwang_readiness(
    config: HwangReadinessConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
) -> dict[str, Any]:
    passing = [str(row["candidate_id"]) for row in rows if row["candidate_pass"]]
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_present_r7_hwang_protocol_invalid"
        selected = None
        next_action = {
            "action": "repair PRESENT vectorization, scalar cross-check, or key split",
            "training": False,
            "remote_scale": False,
        }
    elif len(passing) == 1:
        status = "pass"
        decision = "innovation2_present_r7_hwang_bitorder_ready"
        selected = passing[0]
        next_action = {
            "action": "freeze bit order and plan larger disjoint-key kernel reproduction",
            "selected_candidate": selected,
            "required_baselines": ["direct GF(2) empirical kernel"],
            "training": False,
            "remote_scale": False,
        }
    elif len(passing) > 1:
        status = "hold"
        decision = "innovation2_present_r7_hwang_bitorder_ambiguous"
        selected = None
        next_action = {
            "action": "increase calibration keys and audit paper state layout",
            "passing_candidates": passing,
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_present_r7_hwang_bitorder_not_reproduced"
        selected = None
        next_action = {
            "action": "audit paper rounds, final whitening, bit numbering, and active set",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "passing_candidates": passing,
        "selected_candidate": selected,
        "claim_scope": (
            "eight-key local bit-order readiness against Hwang et al. Table 8; "
            "not an all-key proof, full kernel reproduction, or neural result"
        ),
        "next_action": next_action,
    }


def _collect_xor_words(
    structure: BitIntegralStructure,
    keys: tuple[int, ...],
    *,
    rounds: int,
    key_chunk_size: int,
    progress_callback: ProgressCallback | None,
) -> np.ndarray:
    result = np.zeros(len(keys), dtype=np.uint64)
    for start in range(0, len(keys), key_chunk_size):
        stop = min(start + key_chunk_size, len(keys))
        round_keys = present_round_key_matrix(keys[start:stop], rounds=rounds)
        result[start:stop] = bit_integral_output_xor_matrix(
            (structure,),
            round_keys,
            structure_chunk_size=1,
        )[0]
        _emit(
            progress_callback,
            "key_chunk_done",
            {
                "input_orientation": structure.structure_id,
                "keys_done": stop,
                "keys": len(keys),
                "plaintexts_per_structure": structure.set_size,
            },
        )
    return result


def _mask_failure_count(words: np.ndarray, masks: tuple[int, ...]) -> int:
    return sum(_single_mask_failure_count(words, mask) for mask in masks)


def _single_mask_failure_count(words: np.ndarray, mask: int) -> int:
    return sum((int(word) & mask).bit_count() & 1 for word in words)


def _map_output_bit(bit: int, output_mapping: str) -> int:
    return bit if output_mapping == "direct" else 63 - bit


def _map_mask(mask: int, output_mapping: str) -> int:
    if output_mapping == "direct":
        return mask
    mapped = 0
    for bit in range(64):
        if mask & (1 << bit):
            mapped |= 1 << (63 - bit)
    return mapped


def _span(basis: tuple[int, ...]) -> set[int]:
    values = {0}
    for vector in basis:
        values |= {value ^ vector for value in tuple(values)}
    return values


def _emit(
    callback: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event, payload)
