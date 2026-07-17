from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.arx.speck import Speck32_64
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import (
    SpeckParityCacheConfig,
    assignments_to_plaintexts,
    hwang_speck_basis_masks,
    run_cached_speck_parity_rows,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import (
    KEY_GENERATION_OFFSET,
    SpeckHwangPhaseCConfig,
    make_phase_c_keys,
    summarize_speck_kernel,
)


POSITION_STARTS = tuple(range(15)) + tuple(range(16, 31))
ANCHOR_START = 5
CONTROL_START = 0
SCREEN_KEYS = 8
MAX_VALIDATION_CANDIDATES = 8
PHASE_C_RUN_ID = "i2_speck32_hwang_phase_c_32plus32_gpu0_20260717"
PHASE_C_SOURCE_COMMIT = "700ac88a4c250fb43ff076ce043c79a575faf95d"
PHASE_C_ANCHOR_PARITY_SHA256 = (
    "3a6df2692fd428938cf8d30e16521947efd1b3242dfc62f288094d7f5187637f"
)
PHASE_C_ANCHOR_METADATA_SHA256 = (
    "67138d81e04240b99f42046d2dd6e64a44a8b1586562947176360339a33afe00"
)
PHASE_C_CONTROL_PARITY_SHA256 = (
    "34486c570a630544ce3ca9fccf1297629bc7924fb6ec19adcf939a4b97b485ca"
)
PHASE_C_CONTROL_METADATA_SHA256 = (
    "79d31905668c8121ca8ee2f30fc2fd6bd87d981bdafadf544143e182b5d2b1d3"
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SpeckHwangPositionConfig:
    run_id: str
    seed: int = 0
    discovery_keys: int = 32
    validation_keys: int = 32
    screen_keys: int = SCREEN_KEYS
    max_validation_candidates: int = MAX_VALIDATION_CANDIDATES
    chunk_size: int = 1 << 24
    backend: str = "torch_int32"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.seed != 0:
            raise ValueError("E27 is paired to the Phase C seed-0 key set")
        if self.discovery_keys != 32 or self.validation_keys != 32:
            raise ValueError("E27 requires the Phase C 32+32 key split")
        if self.screen_keys != SCREEN_KEYS:
            raise ValueError("E27 requires the preregistered eight-key screen")
        if self.max_validation_candidates != MAX_VALIDATION_CANDIDATES:
            raise ValueError("E27 validates at most eight screen candidates")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.backend != "torch_int32":
            raise ValueError("E27 requires torch_int32")
        if not self.device.startswith("cuda"):
            raise ValueError("E27 exact enumeration requires CUDA")

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


def active_bits_for_pair(start: int) -> tuple[int, ...]:
    if start not in POSITION_STARTS:
        raise ValueError(f"unsupported adjacent fixed-pair start: {start}")
    return tuple(bit for bit in range(32) if bit not in {start, start + 1})


def select_validation_candidates(candidates: tuple[int, ...]) -> tuple[int, ...]:
    ordered = tuple(sorted(set(candidates)))
    if any(start not in POSITION_STARTS for start in ordered):
        raise ValueError("candidate outside the preregistered position family")
    if ANCHOR_START in ordered or CONTROL_START in ordered:
        raise ValueError("anchor and control must not enter dynamic candidate selection")
    if len(ordered) <= MAX_VALIDATION_CANDIDATES:
        return ordered
    low = [start for start in ordered if start < 16][:4]
    high = [start for start in ordered if start >= 16][:4]
    selected = low + high
    for start in ordered:
        if len(selected) == MAX_VALIDATION_CANDIDATES:
            break
        if start not in selected:
            selected.append(start)
    return tuple(sorted(selected))


def mask_is_balanced(words: np.ndarray, mask: int) -> np.ndarray:
    values = np.asarray(words, dtype=np.uint32).reshape(-1)
    return np.asarray(
        [((int(word) & mask).bit_count() & 1) == 0 for word in values],
        dtype=np.bool_,
    )


def load_phase_c_position_baselines(
    config: SpeckHwangPositionConfig, *, phase_c_root: Path
) -> dict[str, Any]:
    source_path = phase_c_root / "source_expected_commit.txt"
    if not source_path.is_file():
        raise ValueError("Phase C source commit evidence is missing")
    if source_path.read_text(encoding="utf-8").strip() != PHASE_C_SOURCE_COMMIT:
        raise ValueError("Phase C source commit mismatch")

    expected_keys = make_phase_c_keys(config.phase_c_config())
    roles: dict[str, dict[str, Any]] = {}
    frozen = {
        "anchor": (
            PHASE_C_ANCHOR_PARITY_SHA256,
            PHASE_C_ANCHOR_METADATA_SHA256,
            [6, 7],
            ANCHOR_START,
        ),
        "control": (
            PHASE_C_CONTROL_PARITY_SHA256,
            PHASE_C_CONTROL_METADATA_SHA256,
            [7],
            CONTROL_START,
        ),
    }
    for role, (parity_sha, metadata_sha, rounds, start) in frozen.items():
        role_root = phase_c_root / "cache" / role
        parity_path = role_root / "parity_rows.npy"
        metadata_path = role_root / "metadata.json"
        completed_path = role_root / "completed.npy"
        if any(not path.is_file() for path in (parity_path, metadata_path, completed_path)):
            raise ValueError(f"Phase C {role} baseline is incomplete")
        if _sha256(parity_path) != parity_sha or _sha256(metadata_path) != metadata_sha:
            raise ValueError(f"Phase C {role} frozen SHA256 mismatch")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_fields = {
            "run_id": f"{PHASE_C_RUN_ID}:{role}",
            "rounds": rounds,
            "keys": [f"0x{key:016X}" for key in expected_keys],
            "active_bits": list(active_bits_for_pair(start)),
            "fixed_plaintext": "0x00000000",
            "fixed_mask": f"0x{((1 << start) | (1 << (start + 1))):08X}",
            "assignments_per_key": 1 << 30,
            "chunk_size": config.chunk_size,
            "backend": config.backend,
            "device": config.device,
        }
        mismatches = [
            field for field, expected in expected_fields.items()
            if metadata.get(field) != expected
        ]
        if mismatches:
            raise ValueError(f"Phase C {role} metadata mismatch: {','.join(mismatches)}")
        parity_rows = np.load(parity_path)
        completed = np.load(completed_path)
        expected_shape = (len(rounds), config.total_keys)
        if parity_rows.shape != expected_shape or parity_rows.dtype != np.uint32:
            raise ValueError(f"Phase C {role} parity array has wrong shape or dtype")
        if completed.shape != expected_shape or completed.dtype != np.bool_:
            raise ValueError(f"Phase C {role} completion array has wrong shape or dtype")
        if not bool(completed.all()):
            raise ValueError(f"Phase C {role} cache is incomplete")
        roles[role] = {
            "parity_rows": parity_rows.copy(),
            "metadata": metadata,
            "parity_path": parity_path,
            "metadata_path": metadata_path,
            "completed_path": completed_path,
        }
    return {"keys": expected_keys, **roles}


def collect_position_parity_rows(
    config: SpeckHwangPositionConfig,
    *,
    phase_c_root: Path,
    cache_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    baseline = load_phase_c_position_baselines(config, phase_c_root=phase_c_root)
    keys = baseline["keys"]
    paper_mask = hwang_speck_basis_masks(7)[0]
    screen_rows: dict[int, np.ndarray] = {
        ANCHOR_START: baseline["anchor"]["parity_rows"][1, : config.screen_keys],
        CONTROL_START: baseline["control"]["parity_rows"][0, : config.screen_keys],
    }
    first_rows_generated: dict[str, int] = {}
    resume_rows_generated: dict[str, int] = {}
    completed: dict[str, bool] = {}
    cache_metadata: dict[str, dict[str, Any]] = {}

    for start in POSITION_STARTS:
        if start in {ANCHOR_START, CONTROL_START}:
            continue
        role = f"position{start:02d}_screen"
        parity_config = SpeckParityCacheConfig(
            run_id=f"{config.run_id}:{role}",
            rounds=(7,),
            keys=keys[: config.screen_keys],
            active_bits=active_bits_for_pair(start),
            fixed_plaintext=0,
            chunk_size=config.chunk_size,
            backend=config.backend,
            device=config.device,
        )
        callback = _position_callback(start, "screen", progress_callback)
        first = run_cached_speck_parity_rows(
            parity_config,
            cache_root=cache_root / f"position{start:02d}" / "screen",
            progress_callback=callback,
        )
        resumed = run_cached_speck_parity_rows(
            parity_config,
            cache_root=cache_root / f"position{start:02d}" / "screen",
            progress_callback=callback,
        )
        screen_rows[start] = first["parity_rows"][0]
        first_rows_generated[role] = int(first["rows_generated"])
        resume_rows_generated[role] = int(resumed["rows_generated"])
        completed[role] = bool(first["completed"].all())
        cache_metadata[role] = first["metadata"]

    screen_array = np.stack([screen_rows[start] for start in POSITION_STARTS])
    screen_candidates = tuple(
        start
        for start, words in zip(POSITION_STARTS, screen_array)
        if start not in {ANCHOR_START, CONTROL_START}
        and bool(mask_is_balanced(words, paper_mask).all())
    )
    selected = select_validation_candidates(screen_candidates)
    validation_rows: list[np.ndarray] = []
    for start in selected:
        role = f"position{start:02d}_validation"
        parity_config = SpeckParityCacheConfig(
            run_id=f"{config.run_id}:{role}",
            rounds=(7,),
            keys=keys[config.screen_keys :],
            active_bits=active_bits_for_pair(start),
            fixed_plaintext=0,
            chunk_size=config.chunk_size,
            backend=config.backend,
            device=config.device,
        )
        callback = _position_callback(start, "validation", progress_callback)
        first = run_cached_speck_parity_rows(
            parity_config,
            cache_root=cache_root / f"position{start:02d}" / "validation",
            progress_callback=callback,
        )
        resumed = run_cached_speck_parity_rows(
            parity_config,
            cache_root=cache_root / f"position{start:02d}" / "validation",
            progress_callback=callback,
        )
        validation_rows.append(first["parity_rows"][0])
        first_rows_generated[role] = int(first["rows_generated"])
        resume_rows_generated[role] = int(resumed["rows_generated"])
        completed[role] = bool(first["completed"].all())
        cache_metadata[role] = first["metadata"]

    validation_array = (
        np.stack(validation_rows)
        if validation_rows
        else np.empty((0, config.total_keys - config.screen_keys), dtype=np.uint32)
    )
    return {
        "keys": keys,
        "baseline": baseline,
        "screen_parity_rows": screen_array,
        "screen_candidates": screen_candidates,
        "selected_candidates": selected,
        "validation_parity_rows": validation_array,
        "first_rows_generated": first_rows_generated,
        "resume_rows_generated": resume_rows_generated,
        "completed": completed,
        "cache_metadata": cache_metadata,
    }


def evaluate_position_family(
    config: SpeckHwangPositionConfig,
    *,
    keys: tuple[int, ...],
    anchor_words: np.ndarray,
    control_words: np.ndarray,
    screen_parity_rows: np.ndarray,
    screen_candidates: tuple[int, ...],
    selected_candidates: tuple[int, ...],
    validation_parity_rows: np.ndarray,
    baseline_valid: bool,
    caches_completed: dict[str, bool],
    resume_rows_generated: dict[str, int],
    mapping_fixture_valid: bool,
    cuda_available: bool,
    device_count: int,
    timing_rows: int,
) -> dict[str, Any]:
    screen = np.asarray(screen_parity_rows, dtype=np.uint32)
    validation = np.asarray(validation_parity_rows, dtype=np.uint32)
    anchor = np.asarray(anchor_words, dtype=np.uint32).reshape(-1)
    control = np.asarray(control_words, dtype=np.uint32).reshape(-1)
    paper_basis = hwang_speck_basis_masks(7)
    paper_mask = paper_basis[0]
    selected_lookup = {start: index for index, start in enumerate(selected_candidates)}
    kernel_config = config.kernel_config()
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    all_basis_valid = True

    for position_index, start in enumerate(POSITION_STARTS):
        balanced = mask_is_balanced(screen[position_index], paper_mask)
        screen_pass = bool(balanced.all())
        full_words: np.ndarray | None = None
        evidence_source = "screen_only"
        if start == ANCHOR_START:
            full_words = anchor
            evidence_source = "phase_c_anchor"
        elif start == CONTROL_START:
            full_words = control
            evidence_source = "phase_c_control"
        elif start in selected_lookup:
            full_words = np.concatenate(
                (screen[position_index], validation[selected_lookup[start]])
            )
            evidence_source = "screen_plus_validation"

        row: dict[str, Any] = {
            "run_id": config.run_id,
            "task": "innovation2_speck32_hwang_fixed_position_family",
            "rounds": 7,
            "position_start": start,
            "fixed_bits": f"{start},{start + 1}",
            "word": "low" if start < 16 else "high",
            "screen_keys": config.screen_keys,
            "screen_balanced_keys": int(balanced.sum()),
            "screen_pass": screen_pass,
            "validation_selected": start in selected_lookup,
            "evidence_source": evidence_source,
            "evidence_keys": 0 if full_words is None else int(full_words.size),
            "stable_positive": False,
        }
        if full_words is not None:
            summary, emitted, valid = summarize_speck_kernel(
                kernel_config,
                role=f"position{start:02d}",
                rounds=7,
                words=full_words,
                paper_basis=paper_basis,
                fixed_bits=f"{start},{start + 1}",
            )
            stable = bool(
                summary["paper_basis_in_discovery_kernel"]
                and summary["paper_basis_in_validation_kernel"]
                and summary["paper_basis_in_joint_kernel"]
            )
            row.update(
                {
                    key: value
                    for key, value in summary.items()
                    if key.startswith("discovery_")
                    or key.startswith("validation_")
                    or key.startswith("joint_")
                    or key.startswith("paper_")
                }
            )
            row["stable_positive"] = stable
            for basis_row in emitted:
                basis_row["position_start"] = start
            basis_rows.extend(emitted)
            all_basis_valid = all_basis_valid and valid
        rows.append(row)

    expected_keys = make_phase_c_keys(config.phase_c_config())
    expected_candidates = tuple(
        start
        for row, start in zip(rows, POSITION_STARTS)
        if start not in {ANCHOR_START, CONTROL_START} and bool(row["screen_pass"])
    )
    expected_selected = select_validation_candidates(expected_candidates)
    expected_new_rows = 28 * config.screen_keys + len(selected_candidates) * (
        config.total_keys - config.screen_keys
    )
    expected_cache_roles = {
        f"position{start:02d}_screen"
        for start in POSITION_STARTS
        if start not in {ANCHOR_START, CONTROL_START}
    } | {
        f"position{start:02d}_validation" for start in selected_candidates
    }
    by_start = {int(row["position_start"]): row for row in rows}
    anchor_row = by_start[ANCHOR_START]
    control_row = by_start[CONTROL_START]
    stable_positions = tuple(
        int(row["position_start"]) for row in rows if row["stable_positive"]
    )
    sampled_negatives = tuple(
        int(row["position_start"])
        for row in rows
        if not bool(row["screen_pass"])
        or (
            int(row["position_start"]) == CONTROL_START
            and not bool(row["stable_positive"])
        )
    )
    positive_words = {"low" if start < 16 else "high" for start in stable_positions}
    readiness_checks = {
        "official_speck32_vector_matches": (
            Speck32_64(rounds=22, key=0x1918111009080100).encrypt(0x6574694C)
            == 0xA86842F2
        ),
        "phase_c_anchor_and_control_sha_verified": baseline_valid,
        "exact_phase_c_paired_keys": keys == expected_keys,
        "position_family_is_exactly_30_adjacent_pairs": (
            len(POSITION_STARTS) == 30
            and len(set(POSITION_STARTS)) == 30
            and POSITION_STARTS == tuple(range(15)) + tuple(range(16, 31))
        ),
        "screen_shape_and_dtype": (
            screen.shape == (30, config.screen_keys) and screen.dtype == np.uint32
        ),
        "candidate_selection_matches_preregistered_rule": (
            screen_candidates == expected_candidates
            and selected_candidates == expected_selected
        ),
        "validation_shape_and_dtype": (
            validation.shape
            == (len(selected_candidates), config.total_keys - config.screen_keys)
            and validation.dtype == np.uint32
        ),
        "cache_roles_match_dynamic_plan": (
            set(caches_completed) == expected_cache_roles
            and set(resume_rows_generated) == expected_cache_roles
        ),
        "all_new_caches_completed": bool(caches_completed)
        and all(caches_completed.values()),
        "resume_generates_zero_rows": bool(resume_rows_generated)
        and all(value == 0 for value in resume_rows_generated.values()),
        "mapping_fixture_matches": mapping_fixture_valid,
        "all_computed_bases_validate": all_basis_valid,
        "thirty_summary_rows_present": len(rows) == 30,
        "anchor_has_64_key_stable_hwang_mask": bool(anchor_row["stable_positive"]),
        "control_rejects_64_key_hwang_mask": not bool(control_row["stable_positive"]),
        "cuda_available": cuda_available,
        "cuda_device_count_positive": device_count >= 1,
        "timing_evidence_for_all_dynamic_rows": timing_rows == expected_new_rows,
    }
    gate = adjudicate_position_family(
        config,
        rows,
        readiness_checks,
        stable_positions=stable_positions,
        sampled_negatives=sampled_negatives,
        positive_words=positive_words,
        screen_candidates=screen_candidates,
        selected_candidates=selected_candidates,
        expected_new_rows=expected_new_rows,
    )
    return {
        "rows": rows,
        "basis_rows": basis_rows,
        "keys": np.asarray(keys, dtype=np.uint64),
        "screen_parity_rows": screen,
        "validation_parity_rows": validation,
        "selected_candidates": np.asarray(selected_candidates, dtype=np.uint8),
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_speck32_hwang_fixed_position_family",
            "cipher": "SPECK32/64",
            "rounds": [7],
            "position_starts": list(POSITION_STARTS),
            "fixed_context": "00",
            "target_mask": f"0x{paper_mask:08X}",
            "phase_c_source_run": PHASE_C_RUN_ID,
            "phase_c_source_commit": PHASE_C_SOURCE_COMMIT,
            "discovery_keys": config.discovery_keys,
            "validation_keys": config.validation_keys,
            "screen_keys": config.screen_keys,
            "total_keys": config.total_keys,
            "key_generation_seed": config.seed + KEY_GENERATION_OFFSET,
            "assignments_per_exact_row": 1 << 30,
            "expected_new_exact_rows": expected_new_rows,
            "chunk_size": config.chunk_size,
            "backend": config.backend,
            "device": config.device,
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def adjudicate_position_family(
    config: SpeckHwangPositionConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
    *,
    stable_positions: tuple[int, ...],
    sampled_negatives: tuple[int, ...],
    positive_words: set[str],
    screen_candidates: tuple[int, ...],
    selected_candidates: tuple[int, ...],
    expected_new_rows: int,
) -> dict[str, Any]:
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    stable_count = len(stable_positions)
    negative_count = len(sampled_negatives)
    covers_both_words = positive_words == {"low", "high"}
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_speck_hwang_position_family_protocol_invalid"
        next_action = {
            "action": "repair Phase C baselines, position mapping, cache, CUDA timing, or local recomputation",
            "training": False,
            "remote_scale": False,
        }
    elif stable_count >= 4 and negative_count >= 8 and covers_both_words:
        status = "pass"
        decision = "innovation2_speck_hwang_position_family_advance"
        next_action = {
            "action": "freeze position labels and run mask-matched group-disjoint shortcut controls",
            "next_adjudication": "E28 SPECK position/mask group-disjoint shortcut audit",
            "training": False,
            "remote_scale": False,
        }
    elif stable_count in {2, 3} or (stable_count >= 4 and not covers_both_words):
        status = "hold"
        decision = "innovation2_speck_hwang_position_family_narrow"
        next_action = {
            "action": "hold neural training and evaluate preregistered non-adjacent or rotation-equivalent families",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_speck_hwang_position_family_anchor_only"
        next_action = {
            "action": "stop the current SPECK position-label route; do not add keys or train",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "signal_checks": {
            "stable_positive_positions_at_least_four": stable_count >= 4,
            "sampled_negative_positions_at_least_eight": negative_count >= 8,
            "stable_positives_cover_both_words": covers_both_words,
        },
        "metrics": {
            "screen_candidate_count": len(screen_candidates),
            "screen_candidates": list(screen_candidates),
            "candidate_overflow": len(screen_candidates) > MAX_VALIDATION_CANDIDATES,
            "selected_candidate_count": len(selected_candidates),
            "selected_candidates": list(selected_candidates),
            "stable_positive_count": stable_count,
            "stable_positive_positions": list(stable_positions),
            "sampled_negative_count": negative_count,
            "sampled_negative_positions": list(sampled_negatives),
            "positive_words": sorted(positive_words),
            "expected_new_exact_rows": expected_new_rows,
        },
        "claim_scope": (
            "remote eight-key screen and deterministic at-most-eight-candidate 64-key "
            "confirmation over 30 adjacent fixed-bit positions, each using exact 2^30 "
            "plaintext assignments and verified paired Phase C keys; not neural training, "
            "paper-scale, an all-key proof, or a new integral-property claim"
        ),
        "next_action": next_action,
    }


def verify_position_mapping_fixture() -> bool:
    assignments = np.arange(1024, dtype=np.uint32)
    for start in POSITION_STARTS:
        active_bits = active_bits_for_pair(start)
        plaintexts = assignments_to_plaintexts(
            assignments,
            active_bits=active_bits,
            fixed_plaintext=0,
        )
        fixed_mask = (1 << start) | (1 << (start + 1))
        if np.any(plaintexts & np.uint32(fixed_mask)):
            return False
        reconstructed = np.zeros(assignments.shape, dtype=np.uint32)
        for assignment_bit, plaintext_bit in enumerate(active_bits):
            reconstructed |= (
                ((plaintexts >> np.uint32(plaintext_bit)) & np.uint32(1))
                << np.uint32(assignment_bit)
            )
        if not np.array_equal(reconstructed, assignments):
            return False
    return True


def _position_callback(
    start: int, phase: str, callback: ProgressCallback | None
) -> ProgressCallback | None:
    if callback is None:
        return None

    def emit(event: str, payload: dict[str, Any]) -> None:
        callback(event, {"position_start": start, "phase": phase, **payload})

    return emit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
