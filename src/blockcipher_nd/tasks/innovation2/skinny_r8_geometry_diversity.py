from __future__ import annotations

import math
from dataclasses import dataclass
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
    paper_basis_masks,
)


ADJACENT_PAIRS = tuple((cell, (cell + 1) % 16) for cell in range(16))
ANCHOR_PAIR = (14, 15)
CONTROL_PAIR = (0, 1)
OUTPUT_BITS = 64
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SkinnyR8GeometryDiversityConfig:
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
            raise ValueError("the frozen E22 audit requires SKINNY-64/64 r8")
        if self.discovery_keys != 64 or self.validation_keys != 64:
            raise ValueError("the frozen E22 audit requires 64+64 keys")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")

    @property
    def total_keys(self) -> int:
        return self.discovery_keys + self.validation_keys


def run_skinny_r8_geometry_diversity_audit(
    config: SkinnyR8GeometryDiversityConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = _make_unique_keys(config.total_keys, seed=config.seed + 10201)
    e20_keys = _make_unique_keys(768, seed=config.seed + 8201)
    e21_keys = _make_unique_keys(768, seed=config.seed + 9201)
    rng = np.random.default_rng(config.seed + 10202)
    base_plaintext = int(rng.integers(0, 1 << 64, dtype=np.uint64))
    parity_rows = np.empty(
        (len(ADJACENT_PAIRS), config.total_keys), dtype=np.uint64
    )
    for pair_index, active_cells in enumerate(ADJACENT_PAIRS):
        parity_rows[pair_index] = _collect_pair_rows(
            config,
            keys=keys,
            base_plaintext=base_plaintext,
            active_cells=active_cells,
            pair_index=pair_index,
            progress_callback=progress_callback,
        )
    return evaluate_skinny_r8_geometry_diversity(
        config,
        keys=keys,
        e20_keys=e20_keys,
        e21_keys=e21_keys,
        base_plaintext=base_plaintext,
        parity_rows=parity_rows,
    )


def evaluate_skinny_r8_geometry_diversity(
    config: SkinnyR8GeometryDiversityConfig,
    *,
    keys: tuple[int, ...],
    e20_keys: tuple[int, ...],
    e21_keys: tuple[int, ...],
    base_plaintext: int,
    parity_rows: np.ndarray,
) -> dict[str, Any]:
    rows_array = np.asarray(parity_rows, dtype=np.uint64)
    summaries: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    bases: dict[tuple[int, str], tuple[int, ...]] = {}
    matrices: dict[tuple[int, str], np.ndarray] = {}
    all_bases_valid = True
    all_joint_bases_validate_halves = True
    for pair_index, active_cells in enumerate(ADJACENT_PAIRS):
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
            all_bases_valid = all_bases_valid and valid and rank == 64 - len(basis)
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
        all_joint_bases_validate_halves = (
            all_joint_bases_validate_halves
            and joint_valid_discovery
            and joint_valid_validation
        )
        summaries.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_skinny_r8_adjacent_pair_kernel_diversity",
                "pair_index": pair_index,
                "active_cells": _pair_label(active_cells),
                "is_paper_anchor": active_cells == ANCHOR_PAIR,
                "is_same_budget_control": active_cells == CONTROL_PAIR,
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
            }
        )

    nontrivial = [row for row in summaries if int(row["joint_nullity"]) > 0]
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
    anchor_index = ADJACENT_PAIRS.index(ANCHOR_PAIR)
    paper_basis = paper_basis_masks(mapping="msb_first")
    anchor_span_equalities = {
        split: _subspaces_equal(bases[(anchor_index, split)], paper_basis)
        for split in ("discovery", "validation", "joint")
    }
    anchor_paper_validity = {
        split: kernel_basis_valid(matrices[(anchor_index, split)], paper_basis)
        for split in ("discovery", "validation", "joint")
    }
    keys_set = set(keys)
    readiness_checks = {
        "public_appendix_b_vector_matches": (
            Skinny64(rounds=32, key=0xF5269826FC681238).encrypt(
                0x06034F957724D19D
            )
            == 0xBB39DFB2429B8AC7
        ),
        "exact_adjacent_pair_list": (
            len(ADJACENT_PAIRS) == 16
            and len(set(ADJACENT_PAIRS)) == 16
            and ADJACENT_PAIRS[0] == CONTROL_PAIR
            and ADJACENT_PAIRS[14] == ANCHOR_PAIR
            and ADJACENT_PAIRS[-1] == (15, 0)
        ),
        "exact_key_count": len(keys) == config.total_keys,
        "keys_unique": len(keys_set) == config.total_keys,
        "key_splits_disjoint": set(keys[: config.discovery_keys]).isdisjoint(
            keys[config.discovery_keys :]
        ),
        "keys_disjoint_from_e20": keys_set.isdisjoint(e20_keys),
        "keys_disjoint_from_e21": keys_set.isdisjoint(e21_keys),
        "parity_rows_shape_and_dtype": (
            rows_array.shape == (16, config.total_keys)
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
    gate = adjudicate_skinny_r8_geometry_diversity(
        config,
        summaries,
        readiness_checks,
        anchor_span_equalities=anchor_span_equalities,
        anchor_paper_validity=anchor_paper_validity,
        nontrivial_structures=len(nontrivial),
        distinct_signatures=len(signatures),
        mean_survival=mean_survival,
    )
    return {
        "rows": summaries,
        "basis_rows": basis_rows,
        "keys": np.asarray(keys, dtype=np.uint64),
        "parity_rows": rows_array,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_skinny_r8_adjacent_pair_kernel_diversity",
            "cipher": "SKINNY-64/64",
            "rounds": config.rounds,
            "structures": [list(pair) for pair in ADJACENT_PAIRS],
            "paper_anchor_pair": list(ANCHOR_PAIR),
            "same_budget_control_pair": list(CONTROL_PAIR),
            "state_layout": "4x4 row-major nibbles",
            "output_bit_order": "MSB-first",
            "base_plaintext": f"0x{base_plaintext:016X}",
            "discovery_keys": config.discovery_keys,
            "validation_keys": config.validation_keys,
            "total_keys": config.total_keys,
            "plaintexts_per_structure_per_key": 256,
            "seed": config.seed,
            "key_generation_seed": config.seed + 10201,
            "base_plaintext_seed": config.seed + 10202,
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def adjudicate_skinny_r8_geometry_diversity(
    config: SkinnyR8GeometryDiversityConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
    *,
    anchor_span_equalities: dict[str, bool],
    anchor_paper_validity: dict[str, bool],
    nontrivial_structures: int,
    distinct_signatures: int,
    mean_survival: float,
) -> dict[str, Any]:
    anchor = next(
        (row for row in rows if bool(row.get("is_paper_anchor"))), {}
    )
    anchor_checks = {
        "paper_span_valid_discovery_half": anchor_paper_validity.get(
            "discovery", False
        ),
        "paper_span_valid_validation_half": anchor_paper_validity.get(
            "validation", False
        ),
        "joint_rank_nullity_63_1": (
            anchor.get("joint_rank") == 63
            and anchor.get("joint_nullity") == 1
        ),
        "joint_span_equals_paper": anchor_span_equalities.get("joint", False),
    }
    diversity_checks = {
        "nontrivial_joint_structures_at_least_four": nontrivial_structures >= 4,
        "distinct_joint_signatures_at_least_four": distinct_signatures >= 4,
        "mean_discovery_basis_validation_survival_at_least_0p50": (
            mean_survival >= 0.50
        ),
    }
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    if not readiness_pass or not all(anchor_checks.values()):
        status = "fail"
        decision = "innovation2_skinny_r8_geometry_protocol_invalid"
        next_action = {
            "action": "repair anchor, key ownership, pair enumeration, cache, or GF(2) logic",
            "training": False,
            "remote_scale": False,
        }
    elif all(diversity_checks.values()):
        status = "pass"
        decision = "innovation2_skinny_r8_geometry_kernel_diversity_ready"
        next_action = {
            "action": "construct structure-mask labels and audit shortcuts",
            "next_adjudication": "E23 SKINNY r8 structure-mask shortcut readiness",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_skinny_r8_geometry_kernel_not_diverse"
        next_action = {
            "action": "stop cyclic adjacent-pair expansion and rank literature-backed alternatives",
            "reason": "too few stable nontrivial kernels or signatures",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "anchor_checks": anchor_checks,
        "anchor_span_equalities": anchor_span_equalities,
        "anchor_paper_validity": anchor_paper_validity,
        "diversity_checks": diversity_checks,
        "metrics": {
            "nontrivial_joint_kernel_structures": nontrivial_structures,
            "distinct_nontrivial_joint_kernel_signatures": distinct_signatures,
            "mean_discovery_basis_validation_survival_fraction": mean_survival,
        },
        "claim_scope": (
            "local 16-geometry by 128 fresh-key SKINNY-64/64 r8 raw-bit "
            "kernel-diversity readiness; not paper-scale, an all-key proof, "
            "neural training, or a new balance-property claim"
        ),
        "next_action": next_action,
    }


def _collect_pair_rows(
    config: SkinnyR8GeometryDiversityConfig,
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
            "geometry_chunk_start",
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
            "geometry_chunk_done",
            {
                "pair_index": pair_index,
                "active_cells": list(active_cells),
                "completed": stop,
                "total": len(keys),
            },
        )
    return rows


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
