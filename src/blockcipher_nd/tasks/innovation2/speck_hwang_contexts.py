from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.arx.speck import Speck32_64
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import (
    SPECK32_ACTIVE_BITS,
    SpeckParityCacheConfig,
    chunked_speck_parity_word,
    hwang_speck_basis_masks,
    run_cached_speck_parity_rows,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import (
    KEY_GENERATION_OFFSET,
    SpeckHwangPhaseCConfig,
    make_phase_c_keys,
    summarize_speck_kernel,
)


CONTEXTS = ("00", "01", "10", "11")
ENUMERATED_CONTEXTS = ("01", "10")
PHASE_C_RUN_ID = "i2_speck32_hwang_phase_c_32plus32_gpu0_20260717"
PHASE_C_SOURCE_COMMIT = "700ac88a4c250fb43ff076ce043c79a575faf95d"
PHASE_C_PARITY_SHA256 = (
    "3a6df2692fd428938cf8d30e16521947efd1b3242dfc62f288094d7f5187637f"
)
PHASE_C_METADATA_SHA256 = (
    "67138d81e04240b99f42046d2dd6e64a44a8b1586562947176360339a33afe00"
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SpeckHwangContextConfig:
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
        if self.seed != 0:
            raise ValueError("E26 is paired to the Phase C seed-0 key set")
        if self.discovery_keys != 32 or self.validation_keys != 32:
            raise ValueError("E26 requires the Phase C 32+32 key split")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.backend != "torch_int32":
            raise ValueError("E26 requires torch_int32")
        if not self.device.startswith("cuda"):
            raise ValueError("E26 exact enumeration requires CUDA")

    @property
    def total_keys(self) -> int:
        return self.discovery_keys + self.validation_keys

    def phase_c_config(self) -> SpeckHwangPhaseCConfig:
        return SpeckHwangPhaseCConfig(
            run_id=PHASE_C_RUN_ID,
            seed=self.seed,
            discovery_keys=self.discovery_keys,
            validation_keys=self.validation_keys,
            chunk_size=self.chunk_size,
            backend=self.backend,
            device=self.device,
        )

    def kernel_config(self) -> SpeckHwangPhaseCConfig:
        return SpeckHwangPhaseCConfig(
            run_id=self.run_id,
            seed=self.seed,
            discovery_keys=self.discovery_keys,
            validation_keys=self.validation_keys,
            chunk_size=self.chunk_size,
            backend=self.backend,
            device=self.device,
        )


def fixed_plaintext_for_context(context: str) -> int:
    if context not in CONTEXTS:
        raise ValueError(f"unsupported SPECK fixed context: {context}")
    return int(context, 2) << 5


def load_phase_c_anchor(
    config: SpeckHwangContextConfig, *, phase_c_root: Path
) -> dict[str, Any]:
    parity_path = phase_c_root / "cache/anchor/parity_rows.npy"
    metadata_path = phase_c_root / "cache/anchor/metadata.json"
    completed_path = phase_c_root / "cache/anchor/completed.npy"
    source_path = phase_c_root / "source_expected_commit.txt"
    required = (parity_path, metadata_path, completed_path, source_path)
    if any(not path.is_file() for path in required):
        raise ValueError("Phase C baseline archive is missing required anchor files")
    if _sha256(parity_path) != PHASE_C_PARITY_SHA256:
        raise ValueError("Phase C anchor parity SHA256 mismatch")
    if _sha256(metadata_path) != PHASE_C_METADATA_SHA256:
        raise ValueError("Phase C anchor metadata SHA256 mismatch")
    if source_path.read_text(encoding="utf-8").strip() != PHASE_C_SOURCE_COMMIT:
        raise ValueError("Phase C source commit mismatch")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_keys = make_phase_c_keys(config.phase_c_config())
    expected_fields = {
        "run_id": f"{PHASE_C_RUN_ID}:anchor",
        "rounds": [6, 7],
        "keys": [f"0x{key:016X}" for key in expected_keys],
        "active_bits": list(SPECK32_ACTIVE_BITS),
        "fixed_plaintext": "0x00000000",
        "fixed_mask": "0x00000060",
        "assignments_per_key": 1 << 30,
        "chunk_size": config.chunk_size,
        "backend": config.backend,
        "device": config.device,
    }
    mismatches = [
        field for field, value in expected_fields.items() if metadata.get(field) != value
    ]
    if mismatches:
        raise ValueError(f"Phase C anchor metadata mismatch: {','.join(mismatches)}")
    parity_rows = np.load(parity_path)
    completed = np.load(completed_path)
    if parity_rows.shape != (2, config.total_keys) or parity_rows.dtype != np.uint32:
        raise ValueError("Phase C anchor parity array has wrong shape or dtype")
    if completed.shape != parity_rows.shape or completed.dtype != np.bool_:
        raise ValueError("Phase C anchor completion array has wrong shape or dtype")
    if not bool(completed.all()):
        raise ValueError("Phase C anchor cache is incomplete")
    return {
        "keys": expected_keys,
        "parity_rows": parity_rows.copy(),
        "metadata": metadata,
        "parity_path": parity_path,
        "metadata_path": metadata_path,
        "completed_path": completed_path,
    }


def collect_context_parity_rows(
    config: SpeckHwangContextConfig,
    *,
    phase_c_root: Path,
    cache_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    baseline = load_phase_c_anchor(config, phase_c_root=phase_c_root)
    keys = baseline["keys"]
    cache_configs = {
        context: SpeckParityCacheConfig(
            run_id=f"{config.run_id}:context{context}",
            rounds=(6, 7),
            keys=keys,
            active_bits=SPECK32_ACTIVE_BITS,
            fixed_plaintext=fixed_plaintext_for_context(context),
            chunk_size=config.chunk_size,
            backend=config.backend,
            device=config.device,
        )
        for context in ENUMERATED_CONTEXTS
    }
    cache_configs["11_direct"] = SpeckParityCacheConfig(
        run_id=f"{config.run_id}:context11_direct",
        rounds=(6, 7),
        keys=(keys[0],),
        active_bits=SPECK32_ACTIVE_BITS,
        fixed_plaintext=fixed_plaintext_for_context("11"),
        chunk_size=config.chunk_size,
        backend=config.backend,
        device=config.device,
    )
    first: dict[str, dict[str, Any]] = {}
    resumed: dict[str, dict[str, Any]] = {}
    for context, parity_config in cache_configs.items():
        callback = _context_callback(context, progress_callback)
        first[context] = run_cached_speck_parity_rows(
            parity_config,
            cache_root=cache_root / f"context{context}",
            progress_callback=callback,
        )
        resumed[context] = run_cached_speck_parity_rows(
            parity_config,
            cache_root=cache_root / f"context{context}",
            progress_callback=callback,
        )

    context00 = baseline["parity_rows"]
    context01 = first["01"]["parity_rows"]
    context10 = first["10"]["parity_rows"]
    context11 = context00 ^ context01 ^ context10
    direct11 = first["11_direct"]["parity_rows"]
    direct_checks = [
        int(context11[round_index, 0]) == int(direct11[round_index, 0])
        for round_index in range(2)
    ]
    return {
        "keys": keys,
        "context_parity_rows": np.stack(
            (context00, context01, context10, context11), axis=0
        ),
        "direct_context11_rows": direct11,
        "direct_context11_checks": direct_checks,
        "baseline": baseline,
        "cache_metadata": {
            context: payload["metadata"] for context, payload in first.items()
        },
        "completed": {
            context: bool(payload["completed"].all())
            for context, payload in first.items()
        },
        "first_rows_generated": {
            context: int(payload["rows_generated"])
            for context, payload in first.items()
        },
        "resume_rows_generated": {
            context: int(payload["rows_generated"])
            for context, payload in resumed.items()
        },
    }


def verify_context_partition_fixture() -> bool:
    key = 0x1918111009080100
    active_bits = (0, 1, 2, 3)
    fixed_bits = (4, 5)
    union_bits = tuple(sorted(active_bits + fixed_bits))
    for rounds in (6, 7):
        context_xor = 0
        for context in CONTEXTS:
            context_xor ^= chunked_speck_parity_word(
                rounds=rounds,
                key=key,
                active_bits=active_bits,
                fixed_plaintext=int(context, 2) << 4,
                chunk_size=7,
            )
        union_parity = chunked_speck_parity_word(
            rounds=rounds,
            key=key,
            active_bits=union_bits,
            fixed_plaintext=0,
            chunk_size=11,
        )
        if context_xor != union_parity:
            return False
    return True


def evaluate_context_audit(
    config: SpeckHwangContextConfig,
    *,
    keys: tuple[int, ...],
    context_parity_rows: np.ndarray,
    baseline_valid: bool,
    caches_completed: dict[str, bool],
    resume_rows_generated: dict[str, int],
    direct_context11_checks: list[bool],
    partition_fixture_valid: bool,
    cuda_available: bool,
    device_count: int,
    timing_rows: int,
) -> dict[str, Any]:
    arrays = np.asarray(context_parity_rows, dtype=np.uint32)
    kernel_config = config.kernel_config()
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    all_basis_valid = True
    for context_index, context in enumerate(CONTEXTS):
        for round_index, rounds in enumerate((6, 7)):
            row, emitted, valid = summarize_speck_kernel(
                kernel_config,
                role=f"context{context}",
                rounds=rounds,
                words=arrays[context_index, round_index],
                paper_basis=hwang_speck_basis_masks(rounds),
                fixed_bits="5,6",
            )
            row["context"] = context
            row["context_source"] = (
                "phase_c_verified"
                if context == "00"
                else "exact_enumeration"
                if context in ENUMERATED_CONTEXTS
                else "permutation_partition_derivation"
            )
            rows.append(row)
            for basis_row in emitted:
                basis_row["context"] = context
            basis_rows.extend(emitted)
            all_basis_valid = all_basis_valid and valid

    expected_keys = make_phase_c_keys(config.phase_c_config())
    readiness_checks = {
        "official_speck32_vector_matches": (
            Speck32_64(rounds=22, key=0x1918111009080100).encrypt(0x6574694C)
            == 0xA86842F2
        ),
        "phase_c_verified_baseline_matches": baseline_valid,
        "exact_phase_c_paired_keys": keys == expected_keys,
        "context_matrix_shape_and_dtype": (
            arrays.shape == (4, 2, config.total_keys)
            and arrays.dtype == np.uint32
        ),
        "all_new_caches_completed": all(caches_completed.values()),
        "resume_generates_zero_rows": all(
            value == 0 for value in resume_rows_generated.values()
        ),
        "small_partition_fixture_matches": partition_fixture_valid,
        "context11_direct_crosschecks_pass": len(direct_context11_checks) == 2
        and all(direct_context11_checks),
        "all_computed_bases_validate": all_basis_valid,
        "eight_summary_rows_present": len(rows) == 8,
        "cuda_available": cuda_available,
        "cuda_device_count_positive": device_count >= 1,
        "timing_evidence_for_all_258_new_rows": timing_rows == 258,
    }
    gate = adjudicate_context_audit(config, rows, readiness_checks)
    return {
        "rows": rows,
        "basis_rows": basis_rows,
        "keys": np.asarray(keys, dtype=np.uint64),
        "context_parity_rows": arrays,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_speck32_hwang_fixed_context_audit",
            "cipher": "SPECK32/64",
            "rounds": [6, 7],
            "fixed_bits": [5, 6],
            "contexts": list(CONTEXTS),
            "context00_source_run": PHASE_C_RUN_ID,
            "context00_source_commit": PHASE_C_SOURCE_COMMIT,
            "context00_parity_sha256": PHASE_C_PARITY_SHA256,
            "context00_metadata_sha256": PHASE_C_METADATA_SHA256,
            "context11_derivation": "parity00 xor parity01 xor parity10",
            "discovery_keys": config.discovery_keys,
            "validation_keys": config.validation_keys,
            "total_keys": config.total_keys,
            "key_generation_seed": config.seed + KEY_GENERATION_OFFSET,
            "assignments_per_exact_row": 1 << 30,
            "new_exact_rows": 258,
            "chunk_size": config.chunk_size,
            "backend": config.backend,
            "device": config.device,
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def adjudicate_context_audit(
    config: SpeckHwangContextConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
) -> dict[str, Any]:
    by_context_round = {
        (str(row["context"]), int(row["rounds"])): row for row in rows
    }
    exact_paper = {
        (context, rounds): bool(
            by_context_round.get((context, rounds), {}).get(
                "joint_kernel_equals_paper_span"
            )
        )
        for context in CONTEXTS
        for rounds in (6, 7)
    }
    paper_both_halves = {
        (context, rounds): bool(
            by_context_round.get((context, rounds), {}).get(
                "paper_basis_in_discovery_kernel"
            )
            and by_context_round.get((context, rounds), {}).get(
                "paper_basis_in_validation_kernel"
            )
        )
        for context in CONTEXTS
        for rounds in (6, 7)
    }
    signatures_by_round = {
        rounds: {
            str(by_context_round[(context, rounds)]["joint_kernel_basis"])
            for context in CONTEXTS
        }
        for rounds in (6, 7)
    }
    nontrivial_by_round = {
        rounds: sum(
            int(by_context_round[(context, rounds)]["joint_nullity"]) > 0
            for context in CONTEXTS
        )
        for rounds in (6, 7)
    }
    all_invariant = all(exact_paper.values()) and all(paper_both_halves.values())
    dependent_stable = any(
        len(signatures_by_round[rounds]) >= 2
        and nontrivial_by_round[rounds] >= 2
        for rounds in (6, 7)
    )
    signal_checks = {
        "all_contexts_joint_equal_hwang_spans": all(exact_paper.values()),
        "hwang_directions_valid_in_both_halves_for_all_contexts": all(
            paper_both_halves.values()
        ),
        "context_invariant": all_invariant,
        "context_dependent_stable": dependent_stable,
    }
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_speck_hwang_context_protocol_invalid"
        next_action = {
            "action": "repair Phase C baseline reuse, context cache, derivation, CUDA, or GF(2) evidence",
            "training": False,
            "remote_scale": False,
        }
    elif all_invariant:
        status = "pass"
        decision = "innovation2_speck_hwang_context_invariant"
        next_action = {
            "action": "treat fixed value as nuisance and expand matched fixed-position families",
            "next_adjudication": "E27 SPECK fixed-position kernel family and shortcut audit",
            "training": False,
            "remote_scale": False,
        }
    elif dependent_stable:
        status = "pass"
        decision = "innovation2_speck_hwang_context_dependent_stable"
        next_action = {
            "action": "audit context/mask marginal and group-disjoint shortcut baselines",
            "next_adjudication": "E27 SPECK context-mask shortcut audit",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_speck_hwang_context_family_not_stable"
        next_action = {
            "action": "stop context scaling and audit derived context11 or finite-key instability",
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
            "distinct_joint_signatures_by_round": {
                str(rounds): len(signatures) for rounds, signatures in signatures_by_round.items()
            },
            "nontrivial_contexts_by_round": {
                str(rounds): value for rounds, value in nontrivial_by_round.items()
            },
            "exact_paper_span_context_rounds": sum(exact_paper.values()),
            "paper_valid_both_halves_context_rounds": sum(
                paper_both_halves.values()
            ),
        },
        "claim_scope": (
            "remote paired 32+32 sampled-key audit of four fixed values for the "
            "SPECK32/64 bits-5,6 integral structure using exact 2^30 rows, a "
            "verified Phase C baseline, and a directly checked permutation-partition "
            "derivation; not neural training, paper-scale, or a new integral property"
        ),
        "next_action": next_action,
    }


def _context_callback(
    context: str, callback: ProgressCallback | None
) -> ProgressCallback | None:
    if callback is None:
        return None

    def emit(event: str, payload: dict[str, Any]) -> None:
        callback(event, {"context": context, **payload})

    return emit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
