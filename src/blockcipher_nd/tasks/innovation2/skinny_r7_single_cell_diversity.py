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
    _active_cell_parity_word,
    _make_unique_keys,
    _subspaces_equal,
    mask_to_paper_bits,
    paper_basis_masks,
)


ACTIVE_CELLS = tuple(range(16))
ANCHOR_CELL = 15
CONTROL_CELL = 0
OUTPUT_BITS = 64
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SkinnyR7SingleCellDiversityConfig:
    run_id: str
    seed: int = 0
    rounds: int = 7
    discovery_keys: int = 64
    validation_keys: int = 64
    key_chunk_size: int = 8

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 7:
            raise ValueError("the frozen E24 audit requires SKINNY-64/64 r7")
        if self.discovery_keys != 64 or self.validation_keys != 64:
            raise ValueError("the frozen E24 audit requires 64+64 keys")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")

    @property
    def total_keys(self) -> int:
        return self.discovery_keys + self.validation_keys


def run_skinny_r7_single_cell_diversity_audit(
    config: SkinnyR7SingleCellDiversityConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = _make_unique_keys(config.total_keys, seed=config.seed + 12201)
    prior_key_sets = {
        "e20": _make_unique_keys(768, seed=config.seed + 8201),
        "e21": _make_unique_keys(768, seed=config.seed + 9201),
        "e22": _make_unique_keys(128, seed=config.seed + 10201),
        "e23": _make_unique_keys(128, seed=config.seed + 11201),
    }
    rng = np.random.default_rng(config.seed + 12202)
    base_plaintext = int(rng.integers(0, 1 << 64, dtype=np.uint64))
    parity_rows = np.empty((len(ACTIVE_CELLS), config.total_keys), dtype=np.uint64)
    for active_cell in ACTIVE_CELLS:
        parity_rows[active_cell] = _collect_cell_rows(
            config,
            keys=keys,
            base_plaintext=base_plaintext,
            active_cell=active_cell,
            progress_callback=progress_callback,
        )
    return evaluate_skinny_r7_single_cell_diversity(
        config,
        keys=keys,
        prior_key_sets=prior_key_sets,
        base_plaintext=base_plaintext,
        parity_rows=parity_rows,
    )


def evaluate_skinny_r7_single_cell_diversity(
    config: SkinnyR7SingleCellDiversityConfig,
    *,
    keys: tuple[int, ...],
    prior_key_sets: dict[str, tuple[int, ...]],
    base_plaintext: int,
    parity_rows: np.ndarray,
) -> dict[str, Any]:
    rows_array = np.asarray(parity_rows, dtype=np.uint64)
    if rows_array.shape != (len(ACTIVE_CELLS), config.total_keys):
        raise ValueError(
            f"parity_rows must have shape ({len(ACTIVE_CELLS)}, {config.total_keys})"
        )

    summaries: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    bases: dict[tuple[int, str], tuple[int, ...]] = {}
    matrices: dict[tuple[int, str], np.ndarray] = {}
    all_bases_valid = True
    all_joint_bases_validate_halves = True
    for active_cell in ACTIVE_CELLS:
        cell_rows = rows_array[active_cell]
        split_rows = {
            "discovery": cell_rows[: config.discovery_keys],
            "validation": cell_rows[config.discovery_keys :],
            "joint": cell_rows,
        }
        split_metrics: dict[str, Any] = {}
        for split, matrix in split_rows.items():
            matrices[(active_cell, split)] = matrix
            basis = gf2_kernel_basis(matrix, width=OUTPUT_BITS)
            bases[(active_cell, split)] = basis
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
                        "active_cell": active_cell,
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

        discovery_basis = bases[(active_cell, "discovery")]
        validation_matrix = split_rows["validation"]
        joint_basis = bases[(active_cell, "joint")]
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
        summaries.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_skinny_r7_single_cell_kernel_diversity",
                "active_cell": active_cell,
                "is_hwang_anchor": active_cell == ANCHOR_CELL,
                "is_same_budget_control": active_cell == CONTROL_CELL,
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
    paper_basis = paper_basis_masks(mapping="msb_first")
    anchor_span_equalities = {
        split: _subspaces_equal(bases[(ANCHOR_CELL, split)], paper_basis)
        for split in ("discovery", "validation", "joint")
    }
    anchor_paper_validity = {
        split: kernel_basis_valid(matrices[(ANCHOR_CELL, split)], paper_basis)
        for split in ("discovery", "validation", "joint")
    }
    control_joint = matrices[(CONTROL_CELL, "joint")]
    control_contains_anchor_direction = any(
        kernel_basis_valid(control_joint, (mask,)) for mask in paper_basis
    )
    keys_set = set(keys)
    readiness_checks = {
        "public_appendix_b_vector_matches": (
            Skinny64(rounds=32, key=0xF5269826FC681238).encrypt(
                0x06034F957724D19D
            )
            == 0xBB39DFB2429B8AC7
        ),
        "exact_active_cell_list_and_ownership": (
            ACTIVE_CELLS == tuple(range(16))
            and ANCHOR_CELL == 15
            and CONTROL_CELL == 0
        ),
        "paper_basis_rank_is_eighteen": (
            gf2_rank(np.asarray(paper_basis, dtype=np.uint64), width=OUTPUT_BITS)
            == 18
        ),
        "exact_key_count": len(keys) == config.total_keys,
        "keys_unique": len(keys_set) == config.total_keys,
        "key_splits_disjoint": set(keys[: config.discovery_keys]).isdisjoint(
            keys[config.discovery_keys :]
        ),
        "prior_key_sets_exact": (
            set(prior_key_sets) == {"e20", "e21", "e22", "e23"}
            and len(prior_key_sets["e20"]) == 768
            and len(prior_key_sets["e21"]) == 768
            and len(prior_key_sets["e22"]) == 128
            and len(prior_key_sets["e23"]) == 128
        ),
        "keys_disjoint_from_e20": keys_set.isdisjoint(prior_key_sets["e20"]),
        "keys_disjoint_from_e21": keys_set.isdisjoint(prior_key_sets["e21"]),
        "keys_disjoint_from_e22": keys_set.isdisjoint(prior_key_sets["e22"]),
        "keys_disjoint_from_e23": keys_set.isdisjoint(prior_key_sets["e23"]),
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
    gate = adjudicate_skinny_r7_single_cell_diversity(
        config,
        readiness_checks=readiness_checks,
        anchor_span_equalities=anchor_span_equalities,
        anchor_paper_validity=anchor_paper_validity,
        anchor_row=summaries[ANCHOR_CELL],
        nontrivial_structures=len(nontrivial),
        distinct_signatures=len(signatures),
        mean_survival=mean_survival,
        control_contains_anchor_direction=control_contains_anchor_direction,
    )
    return {
        "rows": summaries,
        "basis_rows": basis_rows,
        "keys": np.asarray(keys, dtype=np.uint64),
        "parity_rows": rows_array,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_skinny_r7_single_cell_kernel_diversity",
            "cipher": "SKINNY-64/64",
            "rounds": config.rounds,
            "active_cells": list(ACTIVE_CELLS),
            "hwang_anchor_cell": ANCHOR_CELL,
            "same_budget_control_cell": CONTROL_CELL,
            "state_layout": "4x4 row-major nibbles",
            "output_bit_order": "MSB-first",
            "base_plaintext": f"0x{base_plaintext:016X}",
            "discovery_keys": config.discovery_keys,
            "validation_keys": config.validation_keys,
            "total_keys": config.total_keys,
            "plaintexts_per_structure_per_key": 16,
            "seed": config.seed,
            "key_generation_seed": config.seed + 12201,
            "base_plaintext_seed": config.seed + 12202,
            "prior_key_generation_seeds": {
                "e20": config.seed + 8201,
                "e21": config.seed + 9201,
                "e22": config.seed + 10201,
                "e23": config.seed + 11201,
            },
            "paper_basis_bits": [
                list(mask_to_paper_bits(mask)) for mask in paper_basis
            ],
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def adjudicate_skinny_r7_single_cell_diversity(
    config: SkinnyR7SingleCellDiversityConfig,
    *,
    readiness_checks: dict[str, bool],
    anchor_span_equalities: dict[str, bool],
    anchor_paper_validity: dict[str, bool],
    anchor_row: dict[str, Any],
    nontrivial_structures: int,
    distinct_signatures: int,
    mean_survival: float,
    control_contains_anchor_direction: bool,
) -> dict[str, Any]:
    anchor_checks = {
        "paper_span_valid_discovery_half": anchor_paper_validity.get(
            "discovery", False
        ),
        "paper_span_valid_validation_half": anchor_paper_validity.get(
            "validation", False
        ),
        "joint_rank_nullity_46_18": (
            anchor_row.get("joint_rank") == 46
            and anchor_row.get("joint_nullity") == 18
        ),
        "joint_span_equals_hwang": anchor_span_equalities.get("joint", False),
    }
    diversity_checks = {
        "nontrivial_joint_structures_at_least_six": nontrivial_structures >= 6,
        "distinct_joint_signatures_at_least_four": distinct_signatures >= 4,
        "mean_discovery_basis_validation_survival_at_least_0p50": (
            mean_survival >= 0.50
        ),
        "control_excludes_every_hwang_direction": (
            not control_contains_anchor_direction
        ),
    }
    if not readiness_checks or not all(readiness_checks.values()):
        status = "fail"
        decision = "innovation2_skinny_r7_single_cell_protocol_invalid"
        next_action = {
            "action": "repair vector, active-cell ownership, keys, cache, or GF(2) logic",
            "training": False,
            "remote_scale": False,
        }
    elif not all(anchor_checks.values()):
        status = "hold"
        decision = "innovation2_skinny_r7_single_cell_anchor_not_reproduced"
        next_action = {
            "action": "audit fresh-key anchor stability without training",
            "training": False,
            "remote_scale": False,
        }
    elif all(diversity_checks.values()):
        status = "pass"
        decision = "innovation2_skinny_r7_single_cell_kernel_diversity_ready"
        next_action = {
            "action": "construct structure-mask labels and audit shortcuts",
            "next_adjudication": "E25 SKINNY r7 structure-mask shortcut readiness",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_skinny_r7_single_cell_kernel_not_diverse"
        next_action = {
            "action": "stop SKINNY position geometry and reproduce Hwang SPECK7 main case",
            "next_adjudication": "E25 Hwang SPECK32/64 r7 kernel readiness",
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
            "control_contains_anchor_direction": control_contains_anchor_direction,
        },
        "claim_scope": (
            "local 16-single-cell by 128 fresh-key SKINNY-64/64 r7 raw-bit "
            "kernel-diversity readiness; not paper-scale, an all-key proof, "
            "neural training, or a new balance-property claim"
        ),
        "next_action": next_action,
    }


def _collect_cell_rows(
    config: SkinnyR7SingleCellDiversityConfig,
    *,
    keys: tuple[int, ...],
    base_plaintext: int,
    active_cell: int,
    progress_callback: ProgressCallback | None,
) -> np.ndarray:
    rows = np.empty(len(keys), dtype=np.uint64)
    for start in range(0, len(keys), config.key_chunk_size):
        stop = min(start + config.key_chunk_size, len(keys))
        _emit(
            progress_callback,
            "single_cell_chunk_start",
            {"active_cell": active_cell, "start": start, "stop": stop},
        )
        for key_index in range(start, stop):
            rows[key_index] = _active_cell_parity_word(
                key=keys[key_index],
                base_plaintext=base_plaintext,
                active_cell=active_cell,
                rounds=config.rounds,
            )
        _emit(
            progress_callback,
            "single_cell_chunk_done",
            {"active_cell": active_cell, "completed": stop, "total": len(keys)},
        )
    return rows


def _basis_signature(basis: tuple[int, ...]) -> str:
    return ";".join(f"{vector:016X}" for vector in basis)


def _emit(
    callback: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event, payload)
