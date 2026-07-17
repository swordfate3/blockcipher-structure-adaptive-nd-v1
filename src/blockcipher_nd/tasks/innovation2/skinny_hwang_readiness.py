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


PAPER_BASIS_BITS = (
    (4, 52),
    (5, 53),
    (6, 54),
    (7, 55),
    (8, 44, 56, 60),
    (9, 45, 57, 61),
    (16, 32, 48),
    (17, 33, 49),
    (18, 34, 50),
    (19, 35, 51),
    (24, 56),
    (25, 57),
    (26, 58),
    (27, 59),
    (28, 44, 60),
    (29, 45, 61),
    (30, 46, 62),
    (31, 47, 63),
)
TARGET_ACTIVE_CELL = 15
CONTROL_ACTIVE_CELL = 0
OUTPUT_BITS = 64
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SkinnyHwangReadinessConfig:
    run_id: str
    seed: int = 0
    rounds: int = 7
    discovery_keys: int = 512
    validation_keys: int = 256
    key_chunk_size: int = 64

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 7:
            raise ValueError("the frozen E20 audit requires SKINNY-64/64 r7")
        if self.discovery_keys != 512 or self.validation_keys != 256:
            raise ValueError("the frozen E20 audit requires 512+256 keys")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")

    @property
    def total_keys(self) -> int:
        return self.discovery_keys + self.validation_keys


def paper_basis_masks(*, mapping: str = "msb_first") -> tuple[int, ...]:
    if mapping not in {
        "msb_first",
        "lsb_first",
        "cell_reverse",
        "bit_reverse_in_cell",
    }:
        raise ValueError(f"unknown paper bit mapping: {mapping}")
    return tuple(
        sum(1 << _integer_bit_for_paper_bit(bit, mapping=mapping) for bit in bits)
        for bits in PAPER_BASIS_BITS
    )


def mask_to_paper_bits(mask: int) -> tuple[int, ...]:
    if mask < 0 or mask >= (1 << OUTPUT_BITS):
        raise ValueError("mask must fit in 64 bits")
    return tuple(bit for bit in range(OUTPUT_BITS) if mask & (1 << (63 - bit)))


def run_skinny_hwang_readiness_audit(
    config: SkinnyHwangReadinessConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = _make_unique_keys(config.total_keys, seed=config.seed + 8201)
    rng = np.random.default_rng(config.seed + 8202)
    base_plaintext = int(rng.integers(0, 1 << 64, dtype=np.uint64))
    parity_rows = np.empty((2, config.total_keys), dtype=np.uint64)
    for role_index, (role, active_cell) in enumerate(
        (("target", TARGET_ACTIVE_CELL), ("control", CONTROL_ACTIVE_CELL))
    ):
        parity_rows[role_index] = _collect_role_rows(
            config,
            keys=keys,
            base_plaintext=base_plaintext,
            active_cell=active_cell,
            role=role,
            progress_callback=progress_callback,
        )
    return evaluate_skinny_hwang_readiness(
        config,
        keys=keys,
        base_plaintext=base_plaintext,
        parity_rows=parity_rows,
    )


def evaluate_skinny_hwang_readiness(
    config: SkinnyHwangReadinessConfig,
    *,
    keys: tuple[int, ...],
    base_plaintext: int,
    parity_rows: np.ndarray,
) -> dict[str, Any]:
    rows_array = np.asarray(parity_rows, dtype=np.uint64)
    summaries: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    bases: dict[tuple[str, str], tuple[int, ...]] = {}
    matrices: dict[tuple[str, str], np.ndarray] = {}
    all_basis_valid = True
    for role_index, (role, active_cell) in enumerate(
        (("target", TARGET_ACTIVE_CELL), ("control", CONTROL_ACTIVE_CELL))
    ):
        role_rows = rows_array[role_index]
        split_rows = {
            "discovery": role_rows[: config.discovery_keys],
            "validation": role_rows[config.discovery_keys :],
            "joint": role_rows,
        }
        split_metrics: dict[str, Any] = {}
        for split, matrix in split_rows.items():
            matrices[(role, split)] = matrix
            basis = gf2_kernel_basis(matrix, width=OUTPUT_BITS)
            bases[(role, split)] = basis
            rank = gf2_rank(matrix, width=OUTPUT_BITS)
            valid = kernel_basis_valid(matrix, basis)
            all_basis_valid = all_basis_valid and valid and rank == 64 - len(basis)
            split_metrics[f"{split}_rank"] = rank
            split_metrics[f"{split}_nullity"] = len(basis)
            split_metrics[f"{split}_basis_valid"] = valid
            for basis_index, vector in enumerate(basis):
                basis_rows.append(
                    {
                        "run_id": config.run_id,
                        "role": role,
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
        discovery_basis = bases[(role, "discovery")]
        validation_matrix = split_rows["validation"]
        survivors = sum(
            kernel_basis_valid(validation_matrix, (vector,))
            for vector in discovery_basis
        )
        summaries.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_skinny_r7_hwang_kernel_readiness",
                "role": role,
                "active_cell": active_cell,
                "base_plaintext": f"0x{base_plaintext:016X}",
                "discovery_keys": config.discovery_keys,
                "validation_keys": config.validation_keys,
                **split_metrics,
                "discovery_basis_validation_survivors": survivors,
                "discovery_basis_validation_survival_fraction": (
                    survivors / len(discovery_basis) if discovery_basis else 0.0
                ),
            }
        )

    paper_basis = paper_basis_masks(mapping="msb_first")
    target_span_equalities = {
        split: _subspaces_equal(bases[("target", split)], paper_basis)
        for split in ("discovery", "validation", "joint")
    }
    wrong_mapping_equalities = {
        mapping: _subspaces_equal(
            bases[("target", "joint")], paper_basis_masks(mapping=mapping)
        )
        for mapping in ("lsb_first", "cell_reverse", "bit_reverse_in_cell")
    }
    paper_masks_valid = {
        role: {
            split: kernel_basis_valid(
                matrices[(role, split)],
                paper_basis,
            )
            for split in ("discovery", "validation", "joint")
        }
        for role in ("target", "control")
    }
    keys_array = np.asarray(keys, dtype=np.uint64)
    readiness_checks = {
        "public_appendix_b_vector_matches": (
            Skinny64(rounds=32, key=0xF5269826FC681238).encrypt(
                0x06034F957724D19D
            )
            == 0xBB39DFB2429B8AC7
        ),
        "exact_key_count": len(keys) == config.total_keys,
        "keys_unique": len(set(keys)) == config.total_keys,
        "key_splits_disjoint": set(keys[: config.discovery_keys]).isdisjoint(
            keys[config.discovery_keys :]
        ),
        "parity_rows_shape_and_dtype": (
            rows_array.shape == (2, config.total_keys)
            and rows_array.dtype == np.uint64
        ),
        "paper_basis_rank_is_eighteen": gf2_rank(
            np.asarray(paper_basis, dtype=np.uint64), width=OUTPUT_BITS
        )
        == 18,
        "all_computed_bases_validate": all_basis_valid,
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for row in summaries
            for key, value in row.items()
            if key.endswith("_rank")
            or key.endswith("_nullity")
            or key.endswith("_fraction")
        ),
    }
    gate = adjudicate_skinny_hwang_readiness(
        config,
        summaries,
        readiness_checks,
        target_span_equalities=target_span_equalities,
        wrong_mapping_equalities=wrong_mapping_equalities,
        paper_masks_valid=paper_masks_valid,
    )
    return {
        "rows": summaries,
        "basis_rows": basis_rows,
        "keys": keys_array,
        "parity_rows": rows_array,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_skinny_r7_hwang_kernel_readiness",
            "cipher": "SKINNY-64/64",
            "rounds": config.rounds,
            "target_active_cell": TARGET_ACTIVE_CELL,
            "control_active_cell": CONTROL_ACTIVE_CELL,
            "state_layout": "4x4 row-major nibbles",
            "output_bit_order": "MSB-first",
            "base_plaintext": f"0x{base_plaintext:016X}",
            "discovery_keys": config.discovery_keys,
            "validation_keys": config.validation_keys,
            "total_keys": config.total_keys,
            "plaintexts_per_key_per_role": 16,
            "seed": config.seed,
            "key_generation_seed": config.seed + 8201,
            "base_plaintext_seed": config.seed + 8202,
            "paper_basis_bits": [list(bits) for bits in PAPER_BASIS_BITS],
            "training_performed": False,
            "author_code_status": "public link returned not_connected",
            "claim_scope": gate["claim_scope"],
        },
    }


def adjudicate_skinny_hwang_readiness(
    config: SkinnyHwangReadinessConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
    *,
    target_span_equalities: dict[str, bool],
    wrong_mapping_equalities: dict[str, bool],
    paper_masks_valid: dict[str, dict[str, bool]],
) -> dict[str, Any]:
    by_role = {str(row["role"]): row for row in rows}
    target = by_role.get("target", {})
    control = by_role.get("control", {})
    signal_checks = {
        "target_discovery_rank_nullity_46_18": (
            target.get("discovery_rank") == 46
            and target.get("discovery_nullity") == 18
        ),
        "target_validation_rank_nullity_46_18": (
            target.get("validation_rank") == 46
            and target.get("validation_nullity") == 18
        ),
        "target_joint_rank_nullity_46_18": (
            target.get("joint_rank") == 46
            and target.get("joint_nullity") == 18
        ),
        "target_spans_equal_hwang_all_splits": all(
            target_span_equalities.values()
        ),
        "all_discovery_directions_survive_validation": (
            target.get("discovery_basis_validation_survivors") == 18
        ),
        "paper_masks_valid_target_all_splits": all(
            paper_masks_valid.get("target", {}).values()
        ),
        "control_joint_full_rank": (
            control.get("joint_rank") == 64
            and control.get("joint_nullity") == 0
        ),
        "control_does_not_validate_paper_span": not paper_masks_valid.get(
            "control", {}
        ).get("joint", True),
        "wrong_bit_orders_do_not_match": not any(wrong_mapping_equalities.values()),
    }
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_skinny_r7_hwang_protocol_invalid"
        next_action = {
            "action": "repair SKINNY vector, key split, parity cache, or GF(2) basis",
            "training": False,
            "remote_scale": False,
        }
    elif all(signal_checks.values()):
        status = "pass"
        decision = "innovation2_skinny_r7_hwang_kernel_reproduced"
        next_action = {
            "action": "audit SKINNY-64/64 r8 two-active-cell exact kernel",
            "next_adjudication": "E21 SKINNY r8 Hwang kernel readiness",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_skinny_r7_hwang_kernel_not_reproduced"
        next_action = {
            "action": "audit active cell, round boundary, state/TK1 order, and data ownership",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "signal_checks": signal_checks,
        "target_span_equalities": target_span_equalities,
        "wrong_mapping_equalities": wrong_mapping_equalities,
        "paper_masks_valid": paper_masks_valid,
        "claim_scope": (
            "local 512-discovery plus 256-validation sampled-key reproduction of "
            "the SKINNY-64/64 r7 raw-bit empirical kernel; not paper-scale, an "
            "all-key proof, neural training, or a new balance-property claim"
        ),
        "next_action": next_action,
    }


def _collect_role_rows(
    config: SkinnyHwangReadinessConfig,
    *,
    keys: tuple[int, ...],
    base_plaintext: int,
    active_cell: int,
    role: str,
    progress_callback: ProgressCallback | None,
) -> np.ndarray:
    rows = np.empty(len(keys), dtype=np.uint64)
    for start in range(0, len(keys), config.key_chunk_size):
        stop = min(start + config.key_chunk_size, len(keys))
        _emit(
            progress_callback,
            "parity_chunk_start",
            {"role": role, "active_cell": active_cell, "start": start, "stop": stop},
        )
        for index in range(start, stop):
            rows[index] = _active_cell_parity_word(
                key=keys[index],
                base_plaintext=base_plaintext,
                active_cell=active_cell,
                rounds=config.rounds,
            )
        _emit(
            progress_callback,
            "parity_chunk_done",
            {"role": role, "active_cell": active_cell, "completed": stop, "total": len(keys)},
        )
    return rows


def _active_cell_parity_word(
    *, key: int, base_plaintext: int, active_cell: int, rounds: int
) -> int:
    if active_cell < 0 or active_cell >= 16:
        raise ValueError("active_cell must be between 0 and 15")
    shift = 4 * (15 - active_cell)
    cleared = base_plaintext & ~(0xF << shift)
    cipher = Skinny64(rounds=rounds, key=key)
    parity = 0
    for value in range(16):
        parity ^= cipher.encrypt(cleared | (value << shift))
    return parity


def _make_unique_keys(count: int, *, seed: int) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    keys: list[int] = []
    used: set[int] = set()
    while len(keys) < count:
        key = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        if key not in used:
            keys.append(key)
            used.add(key)
    return tuple(keys)


def _subspaces_equal(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    left_rows = np.asarray(left, dtype=np.uint64)
    right_rows = np.asarray(right, dtype=np.uint64)
    left_rank = gf2_rank(left_rows, width=OUTPUT_BITS)
    right_rank = gf2_rank(right_rows, width=OUTPUT_BITS)
    if left_rank != len(left) or right_rank != len(right) or left_rank != right_rank:
        return False
    combined = np.asarray(left + right, dtype=np.uint64)
    return gf2_rank(combined, width=OUTPUT_BITS) == left_rank


def _integer_bit_for_paper_bit(bit: int, *, mapping: str) -> int:
    if bit < 0 or bit >= OUTPUT_BITS:
        raise ValueError("paper bit must be between 0 and 63")
    if mapping == "msb_first":
        mapped = bit
    elif mapping == "lsb_first":
        return bit
    elif mapping == "cell_reverse":
        cell, offset = divmod(bit, 4)
        mapped = (15 - cell) * 4 + offset
    elif mapping == "bit_reverse_in_cell":
        cell, offset = divmod(bit, 4)
        mapped = cell * 4 + (3 - offset)
    else:
        raise ValueError(f"unknown paper bit mapping: {mapping}")
    return 63 - mapped


def _emit(
    callback: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event, payload)
