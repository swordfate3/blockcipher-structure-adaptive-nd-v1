from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, Present80
from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    present_round_key_matrix,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import make_keys
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
    gf2_rank,
    kernel_basis_valid,
)


OUTPUT_BITS = 64
SUBSPACE_DIMENSION = 16
COORDINATE_ANCHORS = 4
AUDIT_RANDOM_SUBSPACES = 32
AUDIT_KEYS = 128
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class LinearSubspace:
    structure_id: str
    role: str
    basis: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.structure_id:
            raise ValueError("structure_id must be non-empty")
        if self.role not in {"coordinate_anchor", "random_orientation"}:
            raise ValueError("unsupported linear-subspace role")
        canonical = canonical_rref_basis(self.basis)
        if canonical != self.basis:
            raise ValueError("linear-subspace basis must be canonical RREF")

    @property
    def signature(self) -> str:
        return ":".join(f"{vector:016X}" for vector in self.basis)


@dataclass(frozen=True)
class LinearSubspaceAuditConfig:
    run_id: str
    mode: str = "audit"
    rounds: int = 7
    dimension: int = SUBSPACE_DIMENSION
    random_subspaces: int = AUDIT_RANDOM_SUBSPACES
    keys: int = AUDIT_KEYS
    key_seed: int = 13001
    subspace_seed: int = 13002
    key_chunk_size: int = 16

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if self.rounds != 7 or self.dimension != SUBSPACE_DIMENSION:
            raise ValueError("E30 freezes PRESENT round 7 and dimension 16")
        if self.random_subspaces <= 0:
            raise ValueError("random_subspaces must be positive")
        if self.keys <= 0 or self.keys % 2:
            raise ValueError("keys must be positive and even")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")
        if self.mode == "audit" and (
            self.random_subspaces != AUDIT_RANDOM_SUBSPACES
            or self.keys != AUDIT_KEYS
        ):
            raise ValueError("E30 audit freezes 32 random subspaces and 128 keys")


def canonical_rref_basis(vectors: tuple[int, ...]) -> tuple[int, ...]:
    rows = [int(value) & ((1 << OUTPUT_BITS) - 1) for value in vectors]
    rows = [value for value in rows if value]
    pivot_row = 0
    for column in range(OUTPUT_BITS):
        selected = next(
            (index for index in range(pivot_row, len(rows)) if (rows[index] >> column) & 1),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        pivot = rows[pivot_row]
        for index in range(len(rows)):
            if index != pivot_row and ((rows[index] >> column) & 1):
                rows[index] ^= pivot
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return tuple(rows[:pivot_row])


def coordinate_subspaces() -> tuple[LinearSubspace, ...]:
    return tuple(
        LinearSubspace(
            structure_id=f"coordinate_{block}",
            role="coordinate_anchor",
            basis=tuple(1 << bit for bit in range(16 * block, 16 * block + 16)),
        )
        for block in range(COORDINATE_ANCHORS)
    )


def make_linear_subspaces(
    *, random_subspaces: int, seed: int
) -> tuple[LinearSubspace, ...]:
    anchors = coordinate_subspaces()
    signatures = {structure.signature for structure in anchors}
    rng = np.random.default_rng(seed)
    random_structures: list[LinearSubspace] = []
    attempts = 0
    while len(random_structures) < random_subspaces:
        attempts += 1
        if attempts > random_subspaces * 100:
            raise RuntimeError("could not generate enough unique full-rank subspaces")
        raw = tuple(
            int(value)
            for value in rng.integers(
                0,
                np.iinfo(np.uint64).max,
                size=SUBSPACE_DIMENSION,
                dtype=np.uint64,
            )
        )
        basis = canonical_rref_basis(raw)
        if len(basis) != SUBSPACE_DIMENSION:
            continue
        signature = ":".join(f"{vector:016X}" for vector in basis)
        if signature in signatures:
            continue
        signatures.add(signature)
        random_structures.append(
            LinearSubspace(
                structure_id=f"random_{len(random_structures):02d}",
                role="random_orientation",
                basis=basis,
            )
        )
    return anchors + tuple(random_structures)


def enumerate_subspace_points(basis: tuple[int, ...]) -> np.ndarray:
    canonical = canonical_rref_basis(basis)
    if canonical != basis:
        raise ValueError("point enumeration requires a canonical full-rank basis")
    points = np.zeros(1, dtype=np.uint64)
    for vector in basis:
        points = np.concatenate((points, points ^ np.uint64(vector)))
    return points


def run_cached_subspace_parities(
    config: LinearSubspaceAuditConfig,
    structure: LinearSubspace,
    *,
    keys: tuple[int, ...],
    cache_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    cache_root.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_root / "metadata.json"
    basis_path = cache_root / "basis.npy"
    points_path = cache_root / "points.npy"
    parity_path = cache_root / "parity_rows.npy"
    completed_path = cache_root / "completed.npy"
    metadata = {
        "run_id": config.run_id,
        "structure_id": structure.structure_id,
        "role": structure.role,
        "signature": structure.signature,
        "dimension": len(structure.basis),
        "translation": "0x0000000000000000",
        "assignments": 1 << len(structure.basis),
        "rounds": config.rounds,
        "keys": [f"0x{key:020X}" for key in keys],
        "key_chunk_size": config.key_chunk_size,
        "cipher": "PRESENT-80",
        "output_bit_order": "project LSB integer order",
    }
    required = (metadata_path, basis_path, points_path, parity_path, completed_path)
    if any(path.exists() for path in required):
        if not all(path.exists() for path in required):
            raise ValueError("partial linear-subspace cache is missing required files")
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("linear-subspace cache metadata does not match config")
        cached_basis = np.load(basis_path)
        points = np.load(points_path, mmap_mode="r")
        parity = np.load(parity_path, mmap_mode="r+")
        completed = np.load(completed_path, mmap_mode="r+")
        if cached_basis.dtype != np.uint64 or tuple(int(v) for v in cached_basis) != structure.basis:
            raise ValueError("cached basis does not match canonical structure basis")
        if points.shape != (1 << len(structure.basis),) or points.dtype != np.uint64:
            raise ValueError("cached points have wrong shape or dtype")
        if parity.shape != (len(keys),) or parity.dtype != np.uint64:
            raise ValueError("cached parity rows have wrong shape or dtype")
        if completed.shape != (len(keys),) or completed.dtype != np.bool_:
            raise ValueError("cached completion array has wrong shape or dtype")
        cache_status = "resumed"
    else:
        points_array = enumerate_subspace_points(structure.basis)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.save(basis_path, np.asarray(structure.basis, dtype=np.uint64))
        np.save(points_path, points_array)
        points = np.load(points_path, mmap_mode="r")
        parity = np.lib.format.open_memmap(
            parity_path, mode="w+", dtype=np.uint64, shape=(len(keys),)
        )
        completed = np.lib.format.open_memmap(
            completed_path, mode="w+", dtype=np.bool_, shape=(len(keys),)
        )
        parity[:] = 0
        completed[:] = False
        parity.flush()
        completed.flush()
        cache_status = "created"

    round_keys = present_round_key_matrix(keys, rounds=config.rounds)
    rows_generated = 0
    for start in range(0, len(keys), config.key_chunk_size):
        stop = min(start + config.key_chunk_size, len(keys))
        pending = np.flatnonzero(~np.asarray(completed[start:stop])) + start
        if not len(pending):
            continue
        _emit(
            progress_callback,
            "linear_subspace_key_chunk_start",
            {
                "structure_id": structure.structure_id,
                "role": structure.role,
                "key_indices": [int(value) for value in pending],
            },
        )
        parity[pending] = present_output_xor_words(
            np.asarray(points), round_keys[:, pending]
        )
        parity.flush()
        completed[pending] = True
        completed.flush()
        rows_generated += len(pending)
        _emit(
            progress_callback,
            "linear_subspace_key_chunk_done",
            {
                "structure_id": structure.structure_id,
                "role": structure.role,
                "key_indices": [int(value) for value in pending],
            },
        )
    return {
        "parity_rows": np.asarray(parity).copy(),
        "completed": np.asarray(completed).copy(),
        "metadata": metadata,
        "cache_status": cache_status,
        "rows_generated": rows_generated,
    }


def present_output_xor_words(
    plaintexts: np.ndarray, round_keys: np.ndarray
) -> np.ndarray:
    points = np.asarray(plaintexts, dtype=np.uint64).reshape(-1)
    keys = np.asarray(round_keys, dtype=np.uint64)
    if keys.ndim != 2 or keys.shape[0] != 8:
        raise ValueError("round_keys must have shape (8, key_count) for PRESENT r7")
    states = np.broadcast_to(points, (keys.shape[1], points.size)).copy()
    sbox = np.asarray(PRESENT_SBOX, dtype=np.uint64)
    for round_index in range(7):
        states ^= keys[round_index, :, None]
        states = _present_sbox_layer_words(states, sbox)
        states = _present_permutation_layer_words(states)
    states ^= keys[7, :, None]
    return np.bitwise_xor.reduce(states, axis=1).astype(np.uint64, copy=False)


def evaluate_linear_subspaces(
    config: LinearSubspaceAuditConfig,
    *,
    structures: tuple[LinearSubspace, ...],
    keys: tuple[int, ...],
    parity_rows: np.ndarray,
    completed: dict[str, bool],
    resume_rows_generated: dict[str, int],
    scalar_vector_match: bool,
) -> dict[str, Any]:
    matrix = np.asarray(parity_rows)
    half = len(keys) // 2
    rows: list[dict[str, Any]] = []
    all_basis_valid = True
    for index, structure in enumerate(structures):
        discovery = matrix[index, :half]
        validation = matrix[index, half:]
        joint = matrix[index]
        discovery_basis = gf2_kernel_basis(discovery)
        validation_basis = gf2_kernel_basis(validation)
        joint_basis = gf2_kernel_basis(joint)
        discovery_nullity = len(discovery_basis)
        validation_nullity = len(validation_basis)
        joint_nullity = len(joint_basis)
        denominator = min(discovery_nullity, validation_nullity)
        retention = joint_nullity / denominator if denominator else 1.0
        valid = (
            kernel_basis_valid(discovery, discovery_basis)
            and kernel_basis_valid(validation, validation_basis)
            and kernel_basis_valid(joint, joint_basis)
            and gf2_rank(discovery) + discovery_nullity == OUTPUT_BITS
            and gf2_rank(validation) + validation_nullity == OUTPUT_BITS
            and gf2_rank(joint) + joint_nullity == OUTPUT_BITS
        )
        all_basis_valid = all_basis_valid and valid
        rows.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_present_r7_linear_subspace_kernel_diversity",
                "structure_id": structure.structure_id,
                "role": structure.role,
                "signature": structure.signature,
                "dimension": len(structure.basis),
                "discovery_rank": OUTPUT_BITS - discovery_nullity,
                "validation_rank": OUTPUT_BITS - validation_nullity,
                "joint_rank": OUTPUT_BITS - joint_nullity,
                "discovery_nullity": discovery_nullity,
                "validation_nullity": validation_nullity,
                "joint_nullity": joint_nullity,
                "half_intersection_retention": retention,
                "joint_kernel_signature": ":".join(
                    f"{vector:016X}" for vector in joint_basis
                ),
                "basis_validation_pass": valid,
                "training_performed": False,
            }
        )

    random_rows = [row for row in rows if row["role"] == "random_orientation"]
    nontrivial = [row for row in random_rows if int(row["joint_nullity"]) > 0]
    signatures = {str(row["joint_kernel_signature"]) for row in nontrivial}
    mean_retention = (
        float(np.mean([row["half_intersection_retention"] for row in nontrivial]))
        if nontrivial
        else 0.0
    )
    expected_ids = {structure.structure_id for structure in structures}
    readiness = {
        "official_present_vector_and_vectorized_path_match": scalar_vector_match,
        "structure_count_matches_config": len(structures)
        == COORDINATE_ANCHORS + config.random_subspaces,
        "four_coordinate_anchors_present": sum(
            structure.role == "coordinate_anchor" for structure in structures
        )
        == COORDINATE_ANCHORS,
        "all_bases_are_unique_full_rank_rref": (
            len({structure.signature for structure in structures}) == len(structures)
            and all(len(structure.basis) == SUBSPACE_DIMENSION for structure in structures)
            and all(canonical_rref_basis(structure.basis) == structure.basis for structure in structures)
        ),
        "key_halves_are_unique_and_disjoint": (
            len(set(keys)) == len(keys)
            and set(keys[:half]).isdisjoint(keys[half:])
        ),
        "parity_shape_and_dtype": matrix.shape == (len(structures), len(keys))
        and matrix.dtype == np.uint64,
        "cache_roles_match_structures": set(completed) == expected_ids
        and set(resume_rows_generated) == expected_ids,
        "all_caches_completed": bool(completed) and all(completed.values()),
        "resume_generates_zero_rows": bool(resume_rows_generated)
        and all(value == 0 for value in resume_rows_generated.values()),
        "all_kernel_bases_validate": all_basis_valid,
        "all_summary_rows_present": len(rows) == len(structures),
    }
    gate = adjudicate_linear_subspaces(
        config,
        readiness,
        nontrivial_count=len(nontrivial),
        distinct_signatures=len(signatures),
        mean_retention=mean_retention,
    )
    return {
        "rows": rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_present_r7_linear_subspace_kernel_diversity",
            "cipher": "PRESENT-80",
            "rounds": config.rounds,
            "dimension": config.dimension,
            "coordinate_anchors": COORDINATE_ANCHORS,
            "random_subspaces": config.random_subspaces,
            "keys": config.keys,
            "key_half_size": half,
            "key_seed": config.key_seed,
            "subspace_seed": config.subspace_seed,
            "plaintexts_per_structure": 1 << config.dimension,
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def adjudicate_linear_subspaces(
    config: LinearSubspaceAuditConfig,
    readiness: dict[str, bool],
    *,
    nontrivial_count: int,
    distinct_signatures: int,
    mean_retention: float,
) -> dict[str, Any]:
    label_checks = {
        "random_nontrivial_joint_kernels_at_least_8": nontrivial_count >= 8,
        "distinct_nonzero_joint_signatures_at_least_4": distinct_signatures >= 4,
        "mean_half_intersection_retention_at_least_0p50": mean_retention >= 0.50,
    }
    readiness_pass = bool(readiness) and all(readiness.values())
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_present_linear_subspace_protocol_invalid"
        action = "repair RREF ownership, keys, cache, PRESENT vectorization, or GF(2) validation"
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_present_linear_subspace_readiness_passed"
        action = "run the frozen 32-orientation 128-key E30 audit"
    elif all(label_checks.values()):
        status = "pass"
        decision = "innovation2_present_linear_subspace_kernel_family_ready"
        action = "run E31 subspace-by-mask label width and orientation-disjoint shortcut audit"
    else:
        status = "hold"
        decision = "innovation2_present_linear_subspace_kernel_family_too_sparse"
        action = "stop the PRESENT random-orientation benchmark without changing dimension, round, or seed"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness,
        "label_checks": label_checks,
        "metrics": {
            "random_nontrivial_joint_kernel_count": nontrivial_count,
            "distinct_nonzero_joint_kernel_signatures": distinct_signatures,
            "mean_half_intersection_retention": mean_retention,
        },
        "claim_scope": (
            "local exact-2^16 PRESENT-r7 empirical kernel diversity over canonical "
            f"16-dimensional linear subspaces and {config.keys // 2}+{config.keys // 2} keys; not neural training, "
            "an all-key proof, or a claim of inventing affine-space integrals"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "remote_scale": False,
        },
    }


def make_audit_keys(config: LinearSubspaceAuditConfig) -> tuple[int, ...]:
    return make_keys(count=config.keys, seed=config.key_seed)


def scalar_vectorized_fixture_matches() -> bool:
    official_vector_matches = (
        Present80(rounds=31, key=0).encrypt(0) == 0x5579C1387B228445
    )
    key = 0x00000000000000000000
    plaintexts = np.asarray([0, 1, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    round_keys = present_round_key_matrix((key,), rounds=7)
    vectorized = _encrypt_present_words(plaintexts, round_keys) [0]
    scalar = np.asarray(
        [Present80(rounds=7, key=key).encrypt(int(value)) for value in plaintexts],
        dtype=np.uint64,
    )
    return official_vector_matches and bool(np.array_equal(vectorized, scalar))


def _encrypt_present_words(plaintexts: np.ndarray, round_keys: np.ndarray) -> np.ndarray:
    points = np.asarray(plaintexts, dtype=np.uint64).reshape(-1)
    keys = np.asarray(round_keys, dtype=np.uint64)
    states = np.broadcast_to(points, (keys.shape[1], points.size)).copy()
    sbox = np.asarray(PRESENT_SBOX, dtype=np.uint64)
    for round_index in range(keys.shape[0] - 1):
        states ^= keys[round_index, :, None]
        states = _present_sbox_layer_words(states, sbox)
        states = _present_permutation_layer_words(states)
    states ^= keys[-1, :, None]
    return states


def _present_sbox_layer_words(states: np.ndarray, sbox: np.ndarray) -> np.ndarray:
    output = np.zeros_like(states)
    for nibble in range(16):
        shift = np.uint64(4 * nibble)
        values = ((states >> shift) & np.uint64(0xF)).astype(np.uint8)
        output |= sbox[values] << shift
    return output


def _present_permutation_layer_words(states: np.ndarray) -> np.ndarray:
    output = np.zeros_like(states)
    for bit in range(64):
        target = (16 * bit) % 63 if bit < 63 else 63
        output |= ((states >> np.uint64(bit)) & np.uint64(1)) << np.uint64(target)
    return output


def _emit(
    callback: ProgressCallback | None, event: str, payload: dict[str, Any]
) -> None:
    if callback is not None:
        callback(event, payload)
