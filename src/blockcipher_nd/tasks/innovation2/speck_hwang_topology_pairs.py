from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.arx.speck import Speck32_64
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import (
    SpeckParityCacheConfig,
    hwang_speck_basis_masks,
    run_cached_speck_parity_rows,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import (
    SpeckHwangPhaseCConfig,
    make_phase_c_keys,
    summarize_speck_kernel,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_positions import (
    PHASE_C_RUN_ID,
    load_phase_c_position_baselines,
    mask_is_balanced,
)


FAMILIES = ("ror7_add_aligned", "offset_minus_one")
LANES = tuple(range(16))
SCREEN_KEYS = 8
MAX_PER_FAMILY = 4
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SpeckHwangTopologyPairConfig:
    run_id: str
    seed: int = 0
    discovery_keys: int = 32
    validation_keys: int = 32
    screen_keys: int = SCREEN_KEYS
    max_per_family: int = MAX_PER_FAMILY
    chunk_size: int = 1 << 24
    backend: str = "torch_int32"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.seed != 0:
            raise ValueError("E27-N is paired to the Phase C seed-0 keys")
        if self.discovery_keys != 32 or self.validation_keys != 32:
            raise ValueError("E27-N requires the Phase C 32+32 key split")
        if self.screen_keys != SCREEN_KEYS or self.max_per_family != MAX_PER_FAMILY:
            raise ValueError("E27-N requires the frozen 8-key and 4-per-family budget")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.backend != "torch_int32" or not self.device.startswith("cuda"):
            raise ValueError("E27-N exact enumeration requires torch_int32 CUDA")

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


def topology_pair_bits(family: str, lane: int) -> tuple[int, int]:
    if family not in FAMILIES:
        raise ValueError(f"unsupported topology family: {family}")
    if lane not in LANES:
        raise ValueError(f"unsupported SPECK lane: {lane}")
    offset = 7 if family == "ror7_add_aligned" else 6
    return lane, 16 + ((lane + offset) % 16)


def active_bits_for_topology_pair(family: str, lane: int) -> tuple[int, ...]:
    fixed = set(topology_pair_bits(family, lane))
    return tuple(bit for bit in range(32) if bit not in fixed)


def select_family_candidates(
    candidates: dict[str, tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    if set(candidates) != set(FAMILIES):
        raise ValueError("candidate map must contain both frozen families")
    selected: dict[str, tuple[int, ...]] = {}
    for family in FAMILIES:
        ordered = tuple(sorted(set(candidates[family])))
        if any(lane not in LANES for lane in ordered):
            raise ValueError("candidate lane outside 0..15")
        selected[family] = ordered[:MAX_PER_FAMILY]
    return selected


def collect_topology_pair_rows(
    config: SpeckHwangTopologyPairConfig,
    *,
    phase_c_root: Path,
    cache_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    baseline = load_phase_c_position_baselines(config, phase_c_root=phase_c_root)
    keys = baseline["keys"]
    paper_mask = hwang_speck_basis_masks(7)[0]
    screen = np.zeros((2, 16, config.screen_keys), dtype=np.uint32)
    metadata: dict[str, dict[str, Any]] = {}
    first_rows: dict[str, int] = {}
    resumed_rows: dict[str, int] = {}
    completed: dict[str, bool] = {}
    for family_index, family in enumerate(FAMILIES):
        for lane in LANES:
            role = f"{family}_lane{lane:02d}_screen"
            parity_config = SpeckParityCacheConfig(
                run_id=f"{config.run_id}:{role}",
                rounds=(7,),
                keys=keys[: config.screen_keys],
                active_bits=active_bits_for_topology_pair(family, lane),
                fixed_plaintext=0,
                chunk_size=config.chunk_size,
                backend=config.backend,
                device=config.device,
            )
            callback = _pair_callback(family, lane, "screen", progress_callback)
            first = run_cached_speck_parity_rows(
                parity_config,
                cache_root=cache_root / family / f"lane{lane:02d}" / "screen",
                progress_callback=callback,
            )
            resumed = run_cached_speck_parity_rows(
                parity_config,
                cache_root=cache_root / family / f"lane{lane:02d}" / "screen",
                progress_callback=callback,
            )
            screen[family_index, lane] = first["parity_rows"][0]
            metadata[role] = first["metadata"]
            first_rows[role] = int(first["rows_generated"])
            resumed_rows[role] = int(resumed["rows_generated"])
            completed[role] = bool(first["completed"].all())

    candidates = {
        family: tuple(
            lane
            for lane in LANES
            if bool(mask_is_balanced(screen[family_index, lane], paper_mask).all())
        )
        for family_index, family in enumerate(FAMILIES)
    }
    selected = select_family_candidates(candidates)
    selected_specs = tuple(
        (family, lane) for family in FAMILIES for lane in selected[family]
    )
    validation_rows: list[np.ndarray] = []
    for family, lane in selected_specs:
        role = f"{family}_lane{lane:02d}_validation"
        parity_config = SpeckParityCacheConfig(
            run_id=f"{config.run_id}:{role}",
            rounds=(7,),
            keys=keys[config.screen_keys :],
            active_bits=active_bits_for_topology_pair(family, lane),
            fixed_plaintext=0,
            chunk_size=config.chunk_size,
            backend=config.backend,
            device=config.device,
        )
        callback = _pair_callback(family, lane, "validation", progress_callback)
        first = run_cached_speck_parity_rows(
            parity_config,
            cache_root=cache_root / family / f"lane{lane:02d}" / "validation",
            progress_callback=callback,
        )
        resumed = run_cached_speck_parity_rows(
            parity_config,
            cache_root=cache_root / family / f"lane{lane:02d}" / "validation",
            progress_callback=callback,
        )
        validation_rows.append(first["parity_rows"][0])
        metadata[role] = first["metadata"]
        first_rows[role] = int(first["rows_generated"])
        resumed_rows[role] = int(resumed["rows_generated"])
        completed[role] = bool(first["completed"].all())
    validation = (
        np.stack(validation_rows)
        if validation_rows
        else np.empty((0, config.total_keys - config.screen_keys), dtype=np.uint32)
    )
    return {
        "keys": keys,
        "baseline": baseline,
        "screen_parity_rows": screen,
        "candidates": candidates,
        "selected": selected,
        "selected_specs": selected_specs,
        "validation_parity_rows": validation,
        "cache_metadata": metadata,
        "first_rows_generated": first_rows,
        "resume_rows_generated": resumed_rows,
        "completed": completed,
    }


def evaluate_topology_pairs(
    config: SpeckHwangTopologyPairConfig,
    *,
    keys: tuple[int, ...],
    screen_parity_rows: np.ndarray,
    candidates: dict[str, tuple[int, ...]],
    selected: dict[str, tuple[int, ...]],
    selected_specs: tuple[tuple[str, int], ...],
    validation_parity_rows: np.ndarray,
    baseline_valid: bool,
    caches_completed: dict[str, bool],
    resume_rows_generated: dict[str, int],
    cuda_available: bool,
    device_count: int,
    timing_rows: int,
) -> dict[str, Any]:
    screen = np.asarray(screen_parity_rows, dtype=np.uint32)
    validation = np.asarray(validation_parity_rows, dtype=np.uint32)
    paper_basis = hwang_speck_basis_masks(7)
    paper_mask = paper_basis[0]
    selected_lookup = {spec: index for index, spec in enumerate(selected_specs)}
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    all_basis_valid = True
    for family_index, family in enumerate(FAMILIES):
        for lane in LANES:
            fixed_bits = topology_pair_bits(family, lane)
            balanced = mask_is_balanced(screen[family_index, lane], paper_mask)
            spec = (family, lane)
            row: dict[str, Any] = {
                "run_id": config.run_id,
                "task": "innovation2_speck32_hwang_topology_pair_control",
                "family": family,
                "lane": lane,
                "fixed_bits": f"{fixed_bits[0]},{fixed_bits[1]}",
                "screen_balanced_keys": int(balanced.sum()),
                "screen_pass": bool(balanced.all()),
                "validation_selected": spec in selected_lookup,
                "evidence_keys": 64 if spec in selected_lookup else 8,
                "stable_positive": False,
            }
            if spec in selected_lookup:
                full_words = np.concatenate(
                    (screen[family_index, lane], validation[selected_lookup[spec]])
                )
                summary, emitted, valid = summarize_speck_kernel(
                    config.kernel_config(),
                    role=f"{family}_lane{lane:02d}",
                    rounds=7,
                    words=full_words,
                    paper_basis=paper_basis,
                    fixed_bits=row["fixed_bits"],
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
                row["stable_positive"] = bool(
                    summary["paper_basis_in_discovery_kernel"]
                    and summary["paper_basis_in_validation_kernel"]
                    and summary["paper_basis_in_joint_kernel"]
                )
                for basis_row in emitted:
                    basis_row["family"] = family
                    basis_row["lane"] = lane
                basis_rows.extend(emitted)
                all_basis_valid = all_basis_valid and valid
            rows.append(row)

    expected_candidates = {
        family: tuple(
            lane
            for lane in LANES
            if bool(mask_is_balanced(screen[family_index, lane], paper_mask).all())
        )
        for family_index, family in enumerate(FAMILIES)
    }
    expected_selected = select_family_candidates(expected_candidates)
    expected_specs = tuple(
        (family, lane) for family in FAMILIES for lane in expected_selected[family]
    )
    expected_roles = {
        f"{family}_lane{lane:02d}_screen"
        for family in FAMILIES
        for lane in LANES
    } | {
        f"{family}_lane{lane:02d}_validation" for family, lane in selected_specs
    }
    expected_new_rows = 32 * config.screen_keys + len(selected_specs) * (
        config.total_keys - config.screen_keys
    )
    stable = {
        family: tuple(
            int(row["lane"])
            for row in rows
            if row["family"] == family and row["stable_positive"]
        )
        for family in FAMILIES
    }
    readiness = {
        "official_speck32_vector_matches": (
            Speck32_64(rounds=22, key=0x1918111009080100).encrypt(0x6574694C)
            == 0xA86842F2
        ),
        "phase_c_anchor_and_control_sha_verified": baseline_valid,
        "exact_phase_c_paired_keys": keys == make_phase_c_keys(config.phase_c_config()),
        "two_families_have_16_unique_pairs_each": _families_are_exact(),
        "screen_shape_and_dtype": screen.shape == (2, 16, 8) and screen.dtype == np.uint32,
        "candidate_selection_matches_frozen_rule": (
            candidates == expected_candidates
            and selected == expected_selected
            and selected_specs == expected_specs
        ),
        "validation_shape_and_dtype": (
            validation.shape == (len(selected_specs), 56)
            and validation.dtype == np.uint32
        ),
        "cache_roles_match_dynamic_plan": (
            set(caches_completed) == expected_roles
            and set(resume_rows_generated) == expected_roles
        ),
        "all_new_caches_completed": bool(caches_completed)
        and all(caches_completed.values()),
        "resume_generates_zero_rows": bool(resume_rows_generated)
        and all(value == 0 for value in resume_rows_generated.values()),
        "all_computed_bases_validate": all_basis_valid,
        "thirty_two_summary_rows_present": len(rows) == 32,
        "cuda_available": cuda_available,
        "cuda_device_count_positive": device_count >= 1,
        "timing_evidence_for_all_dynamic_rows": timing_rows == expected_new_rows,
    }
    gate = adjudicate_topology_pairs(
        config,
        readiness,
        candidates=candidates,
        selected=selected,
        stable=stable,
        expected_new_rows=expected_new_rows,
    )
    return {
        "rows": rows,
        "basis_rows": basis_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_speck32_hwang_topology_pair_control",
            "cipher": "SPECK32/64",
            "rounds": [7],
            "families": list(FAMILIES),
            "lanes": list(LANES),
            "target_mask": f"0x{paper_mask:08X}",
            "screen_keys": config.screen_keys,
            "total_keys": config.total_keys,
            "assignments_per_exact_row": 1 << 30,
            "expected_new_exact_rows": expected_new_rows,
            "chunk_size": config.chunk_size,
            "backend": config.backend,
            "device": config.device,
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def adjudicate_topology_pairs(
    config: SpeckHwangTopologyPairConfig,
    readiness: dict[str, bool],
    *,
    candidates: dict[str, tuple[int, ...]],
    selected: dict[str, tuple[int, ...]],
    stable: dict[str, tuple[int, ...]],
    expected_new_rows: int,
) -> dict[str, Any]:
    true_hits = len(candidates["ror7_add_aligned"])
    control_hits = len(candidates["offset_minus_one"])
    delta = true_hits - control_hits
    true_stable = len(stable["ror7_add_aligned"])
    control_stable = len(stable["offset_minus_one"])
    if not all(readiness.values()):
        status = "fail"
        decision = "innovation2_speck_topology_pair_protocol_invalid"
        action = "repair pair mapping, Phase C keys, CUDA cache, timing, or GF(2) validation"
    elif true_stable >= 2 and control_stable == 0 and delta >= 2:
        status = "pass"
        decision = "innovation2_speck_topology_aligned_family"
        action = "construct a topology-pair by output-mask label-width audit"
    elif true_stable == 0 and control_stable == 0:
        status = "hold"
        decision = "innovation2_speck_topology_pair_no_signal"
        action = "stop the current SPECK fixed-pair route"
    elif control_stable > 0:
        status = "hold"
        decision = "innovation2_speck_topology_pair_not_specific"
        action = "stop offset scanning; the true alignment did not beat its matched control"
    elif true_stable == 1:
        status = "hold"
        decision = "innovation2_speck_topology_pair_too_narrow"
        action = "stop the current topology-pair family because one stable lane is insufficient"
    elif delta < 2:
        status = "hold"
        decision = "innovation2_speck_topology_pair_not_specific"
        action = "stop offset scanning; the true alignment did not beat its matched control"
    else:
        raise AssertionError("unreachable E27-N adjudication state")
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness,
        "metrics": {
            "screen_candidates": {key: list(value) for key, value in candidates.items()},
            "selected_candidates": {key: list(value) for key, value in selected.items()},
            "stable_lanes": {key: list(value) for key, value in stable.items()},
            "true_screen_hits": true_hits,
            "control_screen_hits": control_hits,
            "screen_hit_delta": delta,
            "true_stable_count": true_stable,
            "control_stable_count": control_stable,
            "expected_new_exact_rows": expected_new_rows,
        },
        "claim_scope": (
            "remote exact-2^30 paired-key screen and at-most-four-per-family 64-key "
            "confirmation of 16 ROR7-to-addition aligned cross-word pairs versus 16 "
            "offset-minus-one controls; not neural training or an all-key proof"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "remote_scale": False,
        },
    }


def _families_are_exact() -> bool:
    true_pairs = {topology_pair_bits("ror7_add_aligned", lane) for lane in LANES}
    controls = {topology_pair_bits("offset_minus_one", lane) for lane in LANES}
    return len(true_pairs) == 16 and len(controls) == 16 and true_pairs.isdisjoint(controls)


def _pair_callback(
    family: str,
    lane: int,
    phase: str,
    callback: ProgressCallback | None,
) -> ProgressCallback | None:
    if callback is None:
        return None

    def emit(event: str, payload: dict[str, Any]) -> None:
        callback(
            event,
            {"family": family, "lane": lane, "phase": phase, **payload},
        )

    return emit
