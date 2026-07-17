from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.spn.skinny import Skinny64
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
    gf2_rank,
    kernel_basis_valid,
)
from blockcipher_nd.tasks.innovation2.skinny_hwang_readiness import (
    mask_to_paper_bits,
)
from blockcipher_nd.tasks.innovation2.skinny_hwang_r8_readiness import (
    _active_cells_parity_word,
    _make_unique_keys,
    _subspaces_equal,
)


BOTTOM_ROW_PAIRS = tuple(combinations(range(12, 16), 2))
CONTROL_PAIR = (0, 1)
AUDIT_PAIRS = BOTTOM_ROW_PAIRS + (CONTROL_PAIR,)
EXPECTED_ANCHOR_MASKS = {
    (12, 13): 0x0000080008000800,
    (13, 14): 0x0000008000800080,
    (14, 15): 0x0000000800080008,
}
OUTPUT_BITS = 64
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SkinnyR8BottomRowClosureConfig:
    run_id: str
    seed: int = 0
    rounds: int = 8
    discovery_keys: int = 64
    validation_keys: int = 64
    key_chunk_size: int = 8

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 8:
            raise ValueError("the frozen E23 audit requires SKINNY-64/64 r8")
        if self.discovery_keys != 64 or self.validation_keys != 64:
            raise ValueError("the frozen E23 audit requires 64+64 keys")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")

    @property
    def total_keys(self) -> int:
        return self.discovery_keys + self.validation_keys


def run_skinny_r8_bottom_row_closure_audit(
    config: SkinnyR8BottomRowClosureConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = _make_unique_keys(config.total_keys, seed=config.seed + 11201)
    prior_key_sets = {
        "e20": _make_unique_keys(768, seed=config.seed + 8201),
        "e21": _make_unique_keys(768, seed=config.seed + 9201),
        "e22": _make_unique_keys(128, seed=config.seed + 10201),
    }
    rng = np.random.default_rng(config.seed + 11202)
    base_plaintext = int(rng.integers(0, 1 << 64, dtype=np.uint64))
    parity_rows = np.empty((len(AUDIT_PAIRS), config.total_keys), dtype=np.uint64)
    for pair_index, active_cells in enumerate(AUDIT_PAIRS):
        parity_rows[pair_index] = _collect_pair_rows(
            config,
            keys=keys,
            base_plaintext=base_plaintext,
            active_cells=active_cells,
            pair_index=pair_index,
            progress_callback=progress_callback,
        )
    return evaluate_skinny_r8_bottom_row_closure(
        config,
        keys=keys,
        prior_key_sets=prior_key_sets,
        base_plaintext=base_plaintext,
        parity_rows=parity_rows,
    )


def evaluate_skinny_r8_bottom_row_closure(
    config: SkinnyR8BottomRowClosureConfig,
    *,
    keys: tuple[int, ...],
    prior_key_sets: dict[str, tuple[int, ...]],
    base_plaintext: int,
    parity_rows: np.ndarray,
) -> dict[str, Any]:
    rows_array = np.asarray(parity_rows, dtype=np.uint64)
    if rows_array.shape != (len(AUDIT_PAIRS), config.total_keys):
        raise ValueError(
            f"parity_rows must have shape ({len(AUDIT_PAIRS)}, {config.total_keys})"
        )

    summaries: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    bases: dict[tuple[int, str], tuple[int, ...]] = {}
    matrices: dict[tuple[int, str], np.ndarray] = {}
    all_bases_valid = True
    all_joint_bases_validate_halves = True
    for pair_index, active_cells in enumerate(AUDIT_PAIRS):
        pair_rows = rows_array[pair_index]
        split_rows = {
            "discovery": pair_rows[: config.discovery_keys],
            "validation": pair_rows[config.discovery_keys :],
            "joint": pair_rows,
        }
        split_metrics: dict[str, Any] = {}
        for split, matrix in split_rows.items():
            matrices[(pair_index, split)] = matrix
            basis = gf2_kernel_basis(matrix, width=OUTPUT_BITS)
            bases[(pair_index, split)] = basis
            rank = gf2_rank(matrix, width=OUTPUT_BITS)
            valid = kernel_basis_valid(matrix, basis)
            all_bases_valid &= valid and rank == OUTPUT_BITS - len(basis)
            split_metrics[f"{split}_rank"] = rank
            split_metrics[f"{split}_nullity"] = len(basis)
            split_metrics[f"{split}_basis_valid"] = valid
            for basis_index, vector in enumerate(basis):
                basis_rows.append(
                    {
                        "run_id": config.run_id,
                        "pair_index": pair_index,
                        "active_cells": _pair_label(active_cells),
                        "split": split,
                        "basis_index": basis_index,
                        "mask_hex": f"0x{vector:016X}",
                        "paper_bits": ",".join(
                            str(bit) for bit in mask_to_paper_bits(vector)
                        ),
                        "mask_weight": vector.bit_count(),
                        "basis_valid": valid,
                    }
                )

        discovery_basis = bases[(pair_index, "discovery")]
        validation_matrix = split_rows["validation"]
        joint_basis = bases[(pair_index, "joint")]
        survivors = sum(
            kernel_basis_valid(validation_matrix, (vector,))
            for vector in discovery_basis
        )
        joint_valid_discovery = kernel_basis_valid(
            split_rows["discovery"], joint_basis
        )
        joint_valid_validation = kernel_basis_valid(validation_matrix, joint_basis)
        all_joint_bases_validate_halves &= (
            joint_valid_discovery and joint_valid_validation
        )
        expected_mask = EXPECTED_ANCHOR_MASKS.get(active_cells)
        summaries.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_skinny_r8_bottom_row_pair_closure",
                "pair_index": pair_index,
                "active_cells": _pair_label(active_cells),
                "is_bottom_row_pair": active_cells in BOTTOM_ROW_PAIRS,
                "is_known_e22_anchor": expected_mask is not None,
                "is_same_budget_control": active_cells == CONTROL_PAIR,
                "expected_mask_hex": (
                    f"0x{expected_mask:016X}" if expected_mask is not None else ""
                ),
                "base_plaintext": f"0x{base_plaintext:016X}",
                "discovery_keys": config.discovery_keys,
                "validation_keys": config.validation_keys,
                **split_metrics,
                "discovery_basis_validation_survivors": survivors,
                "discovery_basis_validation_survival_fraction": (
                    survivors / len(discovery_basis) if discovery_basis else 0.0
                ),
                "joint_basis_valid_discovery": joint_valid_discovery,
                "joint_basis_valid_validation": joint_valid_validation,
                "joint_kernel_signature": _basis_signature(joint_basis),
                "expected_direction_valid_discovery": (
                    kernel_basis_valid(split_rows["discovery"], (expected_mask,))
                    if expected_mask is not None
                    else None
                ),
                "expected_direction_valid_validation": (
                    kernel_basis_valid(validation_matrix, (expected_mask,))
                    if expected_mask is not None
                    else None
                ),
                "joint_span_equals_expected": (
                    _subspaces_equal(joint_basis, (expected_mask,))
                    if expected_mask is not None
                    else None
                ),
            }
        )

    bottom_rows = [row for row in summaries if row["is_bottom_row_pair"]]
    nontrivial = [row for row in bottom_rows if int(row["joint_nullity"]) > 0]
    signatures = {
        str(row["joint_kernel_signature"])
        for row in nontrivial
        if str(row["joint_kernel_signature"])
    }
    mean_survival = (
        float(
            np.mean(
                [
                    float(row["discovery_basis_validation_survival_fraction"])
                    for row in nontrivial
                ]
            )
        )
        if nontrivial
        else 0.0
    )
    anchor_checks = _anchor_reproduction_checks(summaries)
    control_index = AUDIT_PAIRS.index(CONTROL_PAIR)
    control_joint = matrices[(control_index, "joint")]
    control_contains_anchor_family = any(
        kernel_basis_valid(control_joint, (mask,))
        for mask in EXPECTED_ANCHOR_MASKS.values()
    )
    keys_set = set(keys)
    readiness_checks = {
        "public_appendix_b_vector_matches": (
            Skinny64(rounds=32, key=0xF5269826FC681238).encrypt(
                0x06034F957724D19D
            )
            == 0xBB39DFB2429B8AC7
        ),
        "exact_bottom_row_pair_and_control_list": (
            BOTTOM_ROW_PAIRS
            == ((12, 13), (12, 14), (12, 15), (13, 14), (13, 15), (14, 15))
            and AUDIT_PAIRS[-1] == CONTROL_PAIR
            and len(set(AUDIT_PAIRS)) == 7
        ),
        "exact_key_count": len(keys) == config.total_keys,
        "keys_unique": len(keys_set) == config.total_keys,
        "key_splits_disjoint": set(keys[: config.discovery_keys]).isdisjoint(
            keys[config.discovery_keys :]
        ),
        "prior_key_sets_exact": (
            set(prior_key_sets) == {"e20", "e21", "e22"}
            and len(prior_key_sets["e20"]) == 768
            and len(prior_key_sets["e21"]) == 768
            and len(prior_key_sets["e22"]) == 128
        ),
        "keys_disjoint_from_e20": keys_set.isdisjoint(prior_key_sets["e20"]),
        "keys_disjoint_from_e21": keys_set.isdisjoint(prior_key_sets["e21"]),
        "keys_disjoint_from_e22": keys_set.isdisjoint(prior_key_sets["e22"]),
        "parity_rows_shape_and_dtype": (
            rows_array.shape == (7, config.total_keys)
            and rows_array.dtype == np.uint64
        ),
        "all_computed_bases_validate": all_bases_valid,
        "all_joint_bases_validate_both_halves": all_joint_bases_validate_halves,
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for row in summaries
            for key, value in row.items()
            if key.endswith("_rank")
            or key.endswith("_nullity")
            or key.endswith("_fraction")
        )
        and math.isfinite(mean_survival),
    }
    gate = adjudicate_skinny_r8_bottom_row_closure(
        config,
        readiness_checks=readiness_checks,
        anchor_checks=anchor_checks,
        nontrivial_structures=len(nontrivial),
        distinct_signatures=len(signatures),
        mean_survival=mean_survival,
        control_contains_anchor_family=control_contains_anchor_family,
    )
    return {
        "rows": summaries,
        "basis_rows": basis_rows,
        "keys": np.asarray(keys, dtype=np.uint64),
        "parity_rows": rows_array,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_skinny_r8_bottom_row_pair_closure",
            "cipher": "SKINNY-64/64",
            "rounds": config.rounds,
            "structures": [list(pair) for pair in AUDIT_PAIRS],
            "bottom_row_pairs": [list(pair) for pair in BOTTOM_ROW_PAIRS],
            "known_e22_anchor_masks": {
                _pair_label(pair): f"0x{mask:016X}"
                for pair, mask in EXPECTED_ANCHOR_MASKS.items()
            },
            "same_budget_control_pair": list(CONTROL_PAIR),
            "state_layout": "4x4 row-major nibbles",
            "output_bit_order": "MSB-first",
            "base_plaintext": f"0x{base_plaintext:016X}",
            "discovery_keys": config.discovery_keys,
            "validation_keys": config.validation_keys,
            "total_keys": config.total_keys,
            "plaintexts_per_structure_per_key": 256,
            "seed": config.seed,
            "key_generation_seed": config.seed + 11201,
            "base_plaintext_seed": config.seed + 11202,
            "prior_key_generation_seeds": {
                "e20": config.seed + 8201,
                "e21": config.seed + 9201,
                "e22": config.seed + 10201,
            },
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def adjudicate_skinny_r8_bottom_row_closure(
    config: SkinnyR8BottomRowClosureConfig,
    *,
    readiness_checks: dict[str, bool],
    anchor_checks: dict[str, bool],
    nontrivial_structures: int,
    distinct_signatures: int,
    mean_survival: float,
    control_contains_anchor_family: bool,
) -> dict[str, Any]:
    diversity_checks = {
        "bottom_row_nontrivial_joint_structures_at_least_five": (
            nontrivial_structures >= 5
        ),
        "bottom_row_distinct_joint_signatures_at_least_five": (
            distinct_signatures >= 5
        ),
        "mean_discovery_basis_validation_survival_at_least_0p50": (
            mean_survival >= 0.50
        ),
        "control_excludes_anchor_family": not control_contains_anchor_family,
    }
    if not readiness_checks or not all(readiness_checks.values()):
        status = "fail"
        decision = "innovation2_skinny_r8_bottom_row_protocol_invalid"
        next_action = {
            "action": "repair pair ownership, key ownership, cache, public vector, or GF(2) logic",
            "training": False,
            "remote_scale": False,
        }
    elif not all(anchor_checks.values()):
        status = "hold"
        decision = "innovation2_skinny_r8_bottom_row_anchor_not_reproduced"
        next_action = _stop_route_action("one or more E22 anchors failed on fresh keys")
    elif all(diversity_checks.values()):
        status = "pass"
        decision = "innovation2_skinny_r8_bottom_row_pair_family_ready"
        next_action = {
            "action": "construct structure-mask labels and audit shortcuts",
            "next_adjudication": "E24 SKINNY r8 structure-mask shortcut readiness",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_skinny_r8_bottom_row_pair_family_not_closed"
        next_action = _stop_route_action(
            "bottom-row kernel diversity, stability, or control specificity failed"
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "anchor_reproduction_checks": anchor_checks,
        "diversity_checks": diversity_checks,
        "metrics": {
            "bottom_row_nontrivial_joint_kernel_structures": nontrivial_structures,
            "bottom_row_distinct_joint_kernel_signatures": distinct_signatures,
            "mean_discovery_basis_validation_survival_fraction": mean_survival,
            "control_contains_anchor_family": control_contains_anchor_family,
        },
        "claim_scope": (
            "local six-bottom-row-pair plus one-control by 128 fresh-key "
            "SKINNY-64/64 r8 raw-bit kernel-closure audit; not paper-scale, "
            "an all-key proof, neural training, or a new balance-property claim"
        ),
        "next_action": next_action,
    }


def _anchor_reproduction_checks(rows: list[dict[str, Any]]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    by_pair = {str(row["active_cells"]): row for row in rows}
    for pair in EXPECTED_ANCHOR_MASKS:
        label = _pair_label(pair)
        key = f"pair_{pair[0]}_{pair[1]}"
        row = by_pair[label]
        checks[f"{key}_expected_direction_valid_discovery"] = bool(
            row["expected_direction_valid_discovery"]
        )
        checks[f"{key}_expected_direction_valid_validation"] = bool(
            row["expected_direction_valid_validation"]
        )
        checks[f"{key}_joint_rank_nullity_63_1"] = (
            row["joint_rank"] == 63 and row["joint_nullity"] == 1
        )
        checks[f"{key}_joint_span_equals_expected"] = bool(
            row["joint_span_equals_expected"]
        )
    return checks


def _collect_pair_rows(
    config: SkinnyR8BottomRowClosureConfig,
    *,
    keys: tuple[int, ...],
    base_plaintext: int,
    active_cells: tuple[int, int],
    pair_index: int,
    progress_callback: ProgressCallback | None,
) -> np.ndarray:
    rows = np.empty(len(keys), dtype=np.uint64)
    for start in range(0, len(keys), config.key_chunk_size):
        stop = min(start + config.key_chunk_size, len(keys))
        _emit(
            progress_callback,
            "bottom_row_chunk_start",
            {
                "pair_index": pair_index,
                "active_cells": list(active_cells),
                "start": start,
                "stop": stop,
            },
        )
        for key_index in range(start, stop):
            rows[key_index] = _active_cells_parity_word(
                key=keys[key_index],
                base_plaintext=base_plaintext,
                active_cells=active_cells,
                rounds=config.rounds,
            )
        _emit(
            progress_callback,
            "bottom_row_chunk_done",
            {
                "pair_index": pair_index,
                "active_cells": list(active_cells),
                "completed": stop,
                "total": len(keys),
            },
        )
    return rows


def _stop_route_action(reason: str) -> dict[str, Any]:
    return {
        "action": "run E24 SKINNY r7 all-single-cell geometry diversity audit",
        "next_adjudication": "E24 SKINNY r7 single-active-cell kernel diversity",
        "reason": reason,
        "stopped_route": "SKINNY r8 bottom-row pair geometry",
        "training": False,
        "remote_scale": False,
    }


def _pair_label(pair: tuple[int, int]) -> str:
    return f"{pair[0]},{pair[1]}"


def _basis_signature(basis: tuple[int, ...]) -> str:
    return ";".join(f"{vector:016X}" for vector in basis)


def _emit(
    callback: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event, payload)
