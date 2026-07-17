from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.arx.speck import Speck32_64
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
    gf2_rank,
    kernel_basis_valid,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import (
    SPECK32_ACTIVE_BITS,
    SpeckParityCacheConfig,
    hwang_speck_basis_masks,
    run_cached_speck_parity_rows,
)


OUTPUT_BITS = 32
KEY_GENERATION_OFFSET = 25031
CONTROL_FIXED_BITS = (0, 1)
CONTROL_ACTIVE_BITS = tuple(
    bit for bit in range(OUTPUT_BITS) if bit not in CONTROL_FIXED_BITS
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SpeckHwangPhaseCConfig:
    run_id: str
    seed: int = 0
    discovery_keys: int = 32
    validation_keys: int = 32
    chunk_size: int = 1 << 24
    backend: str = "torch_int32"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.discovery_keys <= 0 or self.validation_keys <= 0:
            raise ValueError("discovery_keys and validation_keys must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.backend != "torch_int32":
            raise ValueError("Phase C requires the torch_int32 backend")
        if not self.device:
            raise ValueError("device must be non-empty")

    @property
    def total_keys(self) -> int:
        return self.discovery_keys + self.validation_keys


def make_phase_c_keys(config: SpeckHwangPhaseCConfig) -> tuple[int, ...]:
    rng = np.random.default_rng(config.seed + KEY_GENERATION_OFFSET)
    keys: list[int] = []
    seen: set[int] = set()
    while len(keys) < config.total_keys:
        key = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return tuple(keys)


def collect_phase_c_parity_rows(
    config: SpeckHwangPhaseCConfig,
    *,
    cache_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = make_phase_c_keys(config)
    roles = {
        "anchor": SpeckParityCacheConfig(
            run_id=f"{config.run_id}:anchor",
            rounds=(6, 7),
            keys=keys,
            active_bits=SPECK32_ACTIVE_BITS,
            fixed_plaintext=0,
            chunk_size=config.chunk_size,
            backend=config.backend,
            device=config.device,
        ),
        "control": SpeckParityCacheConfig(
            run_id=f"{config.run_id}:control",
            rounds=(7,),
            keys=keys,
            active_bits=CONTROL_ACTIVE_BITS,
            fixed_plaintext=0,
            chunk_size=config.chunk_size,
            backend=config.backend,
            device=config.device,
        ),
    }
    first: dict[str, dict[str, Any]] = {}
    resumed: dict[str, dict[str, Any]] = {}
    for role, parity_config in roles.items():
        callback = _role_callback(role, progress_callback)
        first[role] = run_cached_speck_parity_rows(
            parity_config,
            cache_root=cache_root / role,
            progress_callback=callback,
        )
        resumed[role] = run_cached_speck_parity_rows(
            parity_config,
            cache_root=cache_root / role,
            progress_callback=callback,
        )
    return {
        "keys": keys,
        "anchor_parity_rows": first["anchor"]["parity_rows"],
        "control_parity_rows": first["control"]["parity_rows"],
        "completed": {
            role: bool(payload["completed"].all()) for role, payload in first.items()
        },
        "first_rows_generated": {
            role: int(payload["rows_generated"]) for role, payload in first.items()
        },
        "resume_rows_generated": {
            role: int(payload["rows_generated"])
            for role, payload in resumed.items()
        },
        "cache_metadata": {
            role: payload["metadata"] for role, payload in first.items()
        },
    }


def evaluate_phase_c(
    config: SpeckHwangPhaseCConfig,
    *,
    keys: tuple[int, ...],
    anchor_parity_rows: np.ndarray,
    control_parity_rows: np.ndarray,
    completed: dict[str, bool],
    resume_rows_generated: dict[str, int],
    cuda_available: bool,
    device_count: int,
    timing_rows: int,
) -> dict[str, Any]:
    anchor = np.asarray(anchor_parity_rows, dtype=np.uint32)
    control = np.asarray(control_parity_rows, dtype=np.uint32)
    specs = (
        ("anchor", 6, anchor[0], hwang_speck_basis_masks(6)),
        ("anchor", 7, anchor[1], hwang_speck_basis_masks(7)),
        ("control", 7, control[0], hwang_speck_basis_masks(7)),
    )
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    all_basis_valid = True
    for role, rounds, words, paper_basis in specs:
        row, emitted_basis, valid = _summarize_kernel(
            config,
            role=role,
            rounds=rounds,
            words=words,
            paper_basis=paper_basis,
        )
        rows.append(row)
        basis_rows.extend(emitted_basis)
        all_basis_valid = all_basis_valid and valid

    expected_keys = make_phase_c_keys(config)
    readiness_checks = {
        "official_speck32_vector_matches": (
            Speck32_64(rounds=22, key=0x1918111009080100).encrypt(0x6574694C)
            == 0xA86842F2
        ),
        "exact_preregistered_seed_and_key_counts": (
            config.seed == 0
            and config.discovery_keys == 32
            and config.validation_keys == 32
        ),
        "keys_match_deterministic_generation": keys == expected_keys,
        "keys_unique": len(set(keys)) == config.total_keys,
        "key_splits_disjoint": set(keys[: config.discovery_keys]).isdisjoint(
            keys[config.discovery_keys :]
        ),
        "anchor_structure_is_bits_5_6_fixed": (
            SPECK32_ACTIVE_BITS
            == tuple(bit for bit in range(OUTPUT_BITS) if bit not in {5, 6})
        ),
        "control_structure_is_bits_0_1_fixed": (
            CONTROL_ACTIVE_BITS
            == tuple(bit for bit in range(OUTPUT_BITS) if bit not in {0, 1})
        ),
        "exact_torch_cuda_backend": (
            config.backend == "torch_int32" and config.device.startswith("cuda")
        ),
        "parity_shapes_are_exact": (
            anchor.shape == (2, config.total_keys)
            and control.shape == (1, config.total_keys)
        ),
        "all_caches_completed": all(completed.values()),
        "resume_generates_zero_rows": all(
            count == 0 for count in resume_rows_generated.values()
        ),
        "all_computed_bases_validate": all_basis_valid,
        "three_summary_rows_present": len(rows) == 3,
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if key.endswith("_rank")
            or key.endswith("_nullity")
            or key.endswith("_fraction")
        ),
        "cuda_available": cuda_available,
        "cuda_device_count_positive": device_count >= 1,
        "timing_evidence_for_all_192_rows": timing_rows == 192,
    }
    gate = adjudicate_phase_c(config, rows, readiness_checks)
    return {
        "rows": rows,
        "basis_rows": basis_rows,
        "keys": np.asarray(keys, dtype=np.uint64),
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_speck32_hwang_phase_c_exact_kernel",
            "cipher": "SPECK32/64",
            "seed": config.seed,
            "key_generation_seed": config.seed + KEY_GENERATION_OFFSET,
            "discovery_keys": config.discovery_keys,
            "validation_keys": config.validation_keys,
            "total_keys": config.total_keys,
            "anchor_fixed_bits": [5, 6],
            "anchor_rounds": [6, 7],
            "control_fixed_bits": list(CONTROL_FIXED_BITS),
            "control_rounds": [7],
            "fixed_context": "00",
            "assignments_per_key_round_role": 1 << 30,
            "expected_exact_rows": 192,
            "chunk_size": config.chunk_size,
            "backend": config.backend,
            "device": config.device,
            "output_bit_order": "LSB-first",
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def adjudicate_phase_c(
    config: SpeckHwangPhaseCConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
) -> dict[str, Any]:
    by_role_round = {(str(row["role"]), int(row["rounds"])): row for row in rows}
    r6 = by_role_round.get(("anchor", 6), {})
    r7 = by_role_round.get(("anchor", 7), {})
    control = by_role_round.get(("control", 7), {})
    signal_checks = {
        "anchor_r6_joint_rank_nullity_23_9": (
            r6.get("joint_rank") == 23 and r6.get("joint_nullity") == 9
        ),
        "anchor_r6_joint_kernel_equals_hwang_span": bool(
            r6.get("joint_kernel_equals_paper_span")
        ),
        "anchor_r6_hwang_directions_valid_both_key_halves": bool(
            r6.get("paper_basis_in_discovery_kernel")
            and r6.get("paper_basis_in_validation_kernel")
        ),
        "anchor_r7_joint_rank_nullity_31_1": (
            r7.get("joint_rank") == 31 and r7.get("joint_nullity") == 1
        ),
        "anchor_r7_joint_kernel_equals_hwang_span": bool(
            r7.get("joint_kernel_equals_paper_span")
        ),
        "anchor_r7_hwang_direction_valid_both_key_halves": bool(
            r7.get("paper_basis_in_discovery_kernel")
            and r7.get("paper_basis_in_validation_kernel")
        ),
        "control_r7_does_not_contain_hwang_mask": not bool(
            control.get("paper_basis_in_joint_kernel", True)
        ),
        "control_r7_joint_span_differs_from_hwang": not bool(
            control.get("joint_kernel_equals_paper_span", True)
        ),
    }
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    anchor_checks = [
        value for key, value in signal_checks.items() if key.startswith("anchor_")
    ]
    control_checks = [
        value for key, value in signal_checks.items() if key.startswith("control_")
    ]
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_speck_hwang_phase_c_protocol_invalid"
        next_action = {
            "action": "repair key split, exact structures, CUDA cache, timing, or GF(2) basis",
            "training": False,
            "remote_scale": False,
        }
    elif all(anchor_checks) and all(control_checks):
        status = "pass"
        decision = "innovation2_speck_hwang_phase_c_kernel_reproduced"
        next_action = {
            "action": "audit four fixed contexts and structure-mask shortcut baselines",
            "next_adjudication": "E26 SPECK fixed-context kernel diversity and shortcut audit",
            "training": False,
            "remote_scale": False,
        }
    elif all(anchor_checks):
        status = "hold"
        decision = "innovation2_speck_hwang_phase_c_position_control_not_specific"
        next_action = {
            "action": "do not train; redesign the structure family because the paper mask survives the position control",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_speck_hwang_phase_c_kernel_not_reproduced"
        next_action = {
            "action": "stop mechanical key scaling and audit round boundary, key ownership, and paper protocol",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "signal_checks": signal_checks,
        "metrics": {
            f"{row['role']}_r{row['rounds']}": {
                key: value
                for key, value in row.items()
                if key.endswith("_rank")
                or key.endswith("_nullity")
                or key.endswith("_survivors")
                or key in {
                    "paper_span_equal_all_splits",
                    "paper_basis_in_joint_kernel",
                    "joint_kernel_equals_paper_span",
                }
            }
            for row in rows
        },
        "claim_scope": (
            "remote 32-discovery plus 32-fresh-validation sampled-key reproduction "
            "using exact 2^30 plaintext structures and one same-budget position "
            "control; not paper-scale, an all-key proof, neural training, or a new "
            "integral-property claim"
        ),
        "next_action": next_action,
    }


def _summarize_kernel(
    config: SpeckHwangPhaseCConfig,
    *,
    role: str,
    rounds: int,
    words: np.ndarray,
    paper_basis: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    split_words = {
        "discovery": words[: config.discovery_keys],
        "validation": words[config.discovery_keys :],
        "joint": words,
    }
    bases: dict[str, tuple[int, ...]] = {}
    metrics: dict[str, Any] = {}
    basis_rows: list[dict[str, Any]] = []
    all_valid = True
    for split, matrix in split_words.items():
        basis = gf2_kernel_basis(matrix, width=OUTPUT_BITS)
        rank = gf2_rank(matrix, width=OUTPUT_BITS)
        valid = kernel_basis_valid(matrix, basis)
        bases[split] = basis
        metrics[f"{split}_rank"] = rank
        metrics[f"{split}_nullity"] = len(basis)
        metrics[f"{split}_basis_valid"] = valid
        all_valid = all_valid and valid and rank + len(basis) == OUTPUT_BITS
        for basis_index, mask in enumerate(basis):
            basis_rows.append(
                {
                    "run_id": config.run_id,
                    "role": role,
                    "rounds": rounds,
                    "split": split,
                    "basis_index": basis_index,
                    "mask_hex": f"0x{mask:08X}",
                    "mask_weight": mask.bit_count(),
                    "basis_valid": valid,
                }
            )
    discovery_basis = bases["discovery"]
    validation = split_words["validation"]
    survivors = sum(
        kernel_basis_valid(validation, (mask,)) for mask in discovery_basis
    )
    equalities = {
        split: _subspaces_equal(basis, paper_basis)
        for split, basis in bases.items()
    }
    paper_valid = {
        split: kernel_basis_valid(matrix, paper_basis)
        for split, matrix in split_words.items()
    }
    return (
        {
            "run_id": config.run_id,
            "task": "innovation2_speck32_hwang_phase_c_exact_kernel",
            "role": role,
            "rounds": rounds,
            "fixed_bits": "5,6" if role == "anchor" else "0,1",
            "discovery_keys": config.discovery_keys,
            "validation_keys": config.validation_keys,
            **metrics,
            "discovery_basis_validation_survivors": survivors,
            "discovery_basis_validation_survival_fraction": (
                survivors / len(discovery_basis) if discovery_basis else 0.0
            ),
            "paper_span_equal_discovery": equalities["discovery"],
            "paper_span_equal_validation": equalities["validation"],
            "joint_kernel_equals_paper_span": equalities["joint"],
            "paper_span_equal_all_splits": all(equalities.values()),
            "paper_basis_in_discovery_kernel": paper_valid["discovery"],
            "paper_basis_in_validation_kernel": paper_valid["validation"],
            "paper_basis_in_joint_kernel": paper_valid["joint"],
            "joint_kernel_basis": ";".join(
                f"0x{mask:08X}" for mask in bases["joint"]
            ),
        },
        basis_rows,
        all_valid,
    )


def _subspaces_equal(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    left_words = np.asarray(left, dtype=np.uint64)
    right_words = np.asarray(right, dtype=np.uint64)
    left_rank = gf2_rank(left_words, width=OUTPUT_BITS)
    right_rank = gf2_rank(right_words, width=OUTPUT_BITS)
    combined = np.concatenate((left_words, right_words))
    return (
        left_rank == right_rank
        and gf2_rank(combined, width=OUTPUT_BITS) == left_rank
    )


def _role_callback(
    role: str, callback: ProgressCallback | None
) -> ProgressCallback | None:
    if callback is None:
        return None

    def emit(event: str, payload: dict[str, Any]) -> None:
        callback(event, {"role": role, **payload})

    return emit
