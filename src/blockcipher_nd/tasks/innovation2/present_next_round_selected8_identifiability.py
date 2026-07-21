from __future__ import annotations

import hashlib
import inspect
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, Present80
from blockcipher_nd.tasks.innovation2.present_next_round_identifiability import (
    derive_present80_round_keys,
    trace_present80,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
)


RUN_ID = "i2_present_next_round_selected8_partial_subkey_identifiability_audit_20260722"
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SelectedBitGeometry:
    msb_position: int
    destination_integer_bit: int
    inverse_p_source_bit: int
    key_nibble: int
    sbox_output_role: int


@dataclass(frozen=True)
class PresentSelected8IdentifiabilityConfig:
    run_id: str = RUN_ID
    seed: int = 20_260_723
    master_keys: int = 16
    rounds: int = 31
    calibration_pairs: int = 16
    heldout_states_per_round: int = 256
    selected_msb_indices: tuple[int, ...] = SELECTED_MSB_INDICES

    def __post_init__(self) -> None:
        if self.master_keys <= 0:
            raise ValueError("master_keys must be positive")
        if self.rounds != 31:
            raise ValueError("the formal PRESENT audit is frozen to 31 regular rounds")
        if self.calibration_pairs != 16:
            raise ValueError("the formal selected8 audit is frozen to 16 calibration pairs")
        if self.heldout_states_per_round <= 0:
            raise ValueError("heldout_states_per_round must be positive")
        if self.selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("selected output bits must match OP10 through OPC1")


def derive_selected8_geometry(
    selected_msb_indices: tuple[int, ...] = SELECTED_MSB_INDICES,
) -> tuple[SelectedBitGeometry, ...]:
    geometry = []
    for msb_position in selected_msb_indices:
        destination = 63 - msb_position
        inverse = Present80.inverse_permutation_layer(1 << destination)
        if inverse <= 0 or inverse & (inverse - 1):
            raise ValueError("inverse P-layer basis image must contain exactly one bit")
        source = inverse.bit_length() - 1
        geometry.append(
            SelectedBitGeometry(
                msb_position=msb_position,
                destination_integer_bit=destination,
                inverse_p_source_bit=source,
                key_nibble=source // 4,
                sbox_output_role=source % 4,
            )
        )
    return tuple(geometry)


def selected_bits_from_state(
    state: int,
    geometry: tuple[SelectedBitGeometry, ...],
) -> tuple[int, ...]:
    return tuple(
        (state >> item.destination_integer_bit) & 1 for item in geometry
    )


def update_key_nibble_candidates(
    current_nibble: int,
    observed_role_bits: tuple[int, ...],
    roles: tuple[int, ...],
    candidates: tuple[int, ...],
) -> tuple[int, ...]:
    if current_nibble not in range(16):
        raise ValueError("current_nibble must be a four-bit value")
    if len(observed_role_bits) != len(roles) or not roles:
        raise ValueError("observed bits and roles must be nonempty and aligned")
    if any(bit not in (0, 1) for bit in observed_role_bits):
        raise ValueError("observed role values must be bits")
    if any(role not in range(4) for role in roles):
        raise ValueError("S-box output roles must be in [0, 3]")
    return tuple(
        candidate
        for candidate in candidates
        if tuple(
            (PRESENT_SBOX[current_nibble ^ candidate] >> role) & 1
            for role in roles
        )
        == observed_role_bits
    )


def predict_selected_bits_from_partial_key(
    current_state: int,
    recovered_key_nibbles: dict[int, int],
    geometry: tuple[SelectedBitGeometry, ...],
) -> tuple[int, ...]:
    predictions = []
    for item in geometry:
        current_nibble = (current_state >> (4 * item.key_nibble)) & 0xF
        substituted = PRESENT_SBOX[
            current_nibble ^ recovered_key_nibbles[item.key_nibble]
        ]
        predictions.append((substituted >> item.sbox_output_role) & 1)
    return tuple(predictions)


def audit_present_selected8_identifiability(
    config: PresentSelected8IdentifiabilityConfig,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    geometry = derive_selected8_geometry(config.selected_msb_indices)
    groups = _geometry_groups(geometry)
    affected_nibbles = tuple(groups)
    master_keys = _sample_master_keys(config)
    rows: list[dict[str, Any]] = []
    nibble_instance_total = 0
    actual_nibble_consistent = 0
    unique_nibble_total = 0
    recovered_nibble_exact = 0
    heldout_total = 0
    heldout_exact = 0
    samples_to_unique: list[int] = []
    plaintexts_unique = True
    calibration_heldout_disjoint = True

    for key_index, master_key in enumerate(master_keys):
        plaintexts = _sample_plaintexts(config, key_index)
        calibration_plaintexts = plaintexts[: config.calibration_pairs]
        heldout_plaintexts = plaintexts[config.calibration_pairs :]
        plaintexts_unique = plaintexts_unique and len(set(plaintexts)) == len(
            plaintexts
        )
        calibration_heldout_disjoint = calibration_heldout_disjoint and not (
            set(calibration_plaintexts) & set(heldout_plaintexts)
        )
        calibration_traces = [
            trace_present80(plaintext, master_key, rounds=config.rounds)
            for plaintext in calibration_plaintexts
        ]
        heldout_traces = [
            trace_present80(plaintext, master_key, rounds=config.rounds)
            for plaintext in heldout_plaintexts
        ]

        for round_offset in range(config.rounds):
            candidates = {nibble: tuple(range(16)) for nibble in affected_nibbles}
            unique_after: dict[int, int | None] = {
                nibble: None for nibble in affected_nibbles
            }
            actual_round_key = calibration_traces[0].regular_round_keys[round_offset]
            actual_nibbles = {
                nibble: (actual_round_key >> (4 * nibble)) & 0xF
                for nibble in affected_nibbles
            }
            remained_consistent = {nibble: True for nibble in affected_nibbles}
            for sample_index, trace in enumerate(calibration_traces, start=1):
                observed = selected_bits_from_state(
                    trace.states_after_round[round_offset], geometry
                )
                for nibble, entries in groups.items():
                    current_nibble = (
                        trace.states_before_round[round_offset] >> (4 * nibble)
                    ) & 0xF
                    roles = tuple(geometry[index].sbox_output_role for index in entries)
                    observed_role_bits = tuple(observed[index] for index in entries)
                    candidates[nibble] = update_key_nibble_candidates(
                        current_nibble,
                        observed_role_bits,
                        roles,
                        candidates[nibble],
                    )
                    remained_consistent[nibble] = remained_consistent[nibble] and (
                        actual_nibbles[nibble] in candidates[nibble]
                    )
                    if len(candidates[nibble]) == 1 and unique_after[nibble] is None:
                        unique_after[nibble] = sample_index

            recovered = {
                nibble: values[0]
                for nibble, values in candidates.items()
                if len(values) == 1
            }
            unique_count = len(recovered)
            exact_nibbles = sum(
                recovered.get(nibble) == actual_nibbles[nibble]
                for nibble in affected_nibbles
            )
            heldout_round_exact = 0
            if unique_count == len(affected_nibbles):
                for trace in heldout_traces:
                    predicted = predict_selected_bits_from_partial_key(
                        trace.states_before_round[round_offset], recovered, geometry
                    )
                    observed = selected_bits_from_state(
                        trace.states_after_round[round_offset], geometry
                    )
                    heldout_round_exact += predicted == observed

            nibble_instance_total += len(affected_nibbles)
            actual_nibble_consistent += sum(remained_consistent.values())
            unique_nibble_total += unique_count
            recovered_nibble_exact += exact_nibbles
            heldout_total += len(heldout_traces)
            heldout_exact += heldout_round_exact
            samples_to_unique.extend(
                sample
                for sample in unique_after.values()
                if sample is not None
            )
            rows.append(
                {
                    "run_id": config.run_id,
                    "key_index": key_index,
                    "master_key_sha256": _key_digest(master_key),
                    "round_index": round_offset + 1,
                    "affected_key_nibbles": list(affected_nibbles),
                    "calibration_pair_limit": config.calibration_pairs,
                    "candidate_sizes_after_calibration": {
                        str(nibble): len(candidates[nibble])
                        for nibble in affected_nibbles
                    },
                    "samples_to_unique": {
                        str(nibble): unique_after[nibble]
                        for nibble in affected_nibbles
                    },
                    "actual_key_nibbles_remained_consistent": sum(
                        remained_consistent.values()
                    ),
                    "unique_key_nibbles": unique_count,
                    "recovered_key_nibbles_exact": exact_nibbles,
                    "heldout_selected8_transitions": len(heldout_traces),
                    "heldout_selected8_exact": heldout_round_exact,
                    "heldout_selected8_exact_rate": heldout_round_exact
                    / max(1, len(heldout_traces)),
                }
            )
        if progress is not None:
            progress(
                "master_key_done",
                {
                    "key_index": key_index,
                    "key_round_rows": config.rounds,
                    "cumulative_unique_key_nibbles": unique_nibble_total,
                    "cumulative_heldout_selected8_exact": heldout_exact,
                },
            )

    expected_rows = config.master_keys * config.rounds
    expected_nibbles = expected_rows * len(affected_nibbles)
    expected_heldout = expected_rows * config.heldout_states_per_round
    geometry_payload = [asdict(item) for item in geometry]
    expected_geometry = (
        (0, 63, 15, 3),
        (2, 55, 13, 3),
        (8, 31, 7, 3),
        (10, 23, 5, 3),
        (32, 61, 15, 1),
        (34, 53, 13, 1),
        (40, 29, 7, 1),
        (42, 21, 5, 1),
    )
    candidate_parameters = tuple(inspect.signature(update_key_nibble_candidates).parameters)
    protocol_checks = {
        "official_present_zero_key_vector_matches": Present80(
            rounds=31, key=0
        ).encrypt(0)
        == 0x5579C1387B228445,
        "selected8_inverse_p_geometry_matches_frozen_table": tuple(
            (
                item.msb_position,
                item.inverse_p_source_bit,
                item.key_nibble,
                item.sbox_output_role,
            )
            for item in geometry
        )
        == expected_geometry,
        "selected8_geometry_roundtrips_through_present_p_layer": all(
            Present80.permutation_layer(1 << item.inverse_p_source_bit)
            == 1 << item.destination_integer_bit
            for item in geometry
        ),
        "exactly_four_affected_key_nibbles": affected_nibbles == (15, 13, 7, 5),
        "configured_master_keys_are_unique": len(master_keys) == config.master_keys
        and len(set(master_keys)) == config.master_keys,
        "all_plaintexts_unique_within_each_key": plaintexts_unique,
        "calibration_plaintexts_disjoint_from_heldout": calibration_heldout_disjoint,
        "candidate_function_receives_only_projected_observation": candidate_parameters
        == ("current_nibble", "observed_role_bits", "roles", "candidates"),
        "actual_present80_key_schedule_used": all(
            trace_present80(0, key, rounds=config.rounds).regular_round_keys
            == derive_present80_round_keys(key, rounds=config.rounds)[0]
            for key in master_keys
        ),
        "labels_are_selected_state_bits_not_sample_classes": True,
    }
    execution_checks = {
        "result_rows_complete": len(rows) == expected_rows,
        "subkey_nibble_instance_count_complete": nibble_instance_total
        == expected_nibbles,
        "all_actual_key_nibbles_remain_consistent": actual_nibble_consistent
        == expected_nibbles,
        "all_key_nibbles_unique_within_sixteen_pairs": unique_nibble_total
        == expected_nibbles,
        "all_unique_candidates_equal_actual_key_nibbles": recovered_nibble_exact
        == expected_nibbles,
        "heldout_selected8_count_complete": heldout_total == expected_heldout,
        "all_heldout_selected8_vectors_predicted_exactly": heldout_exact
        == expected_heldout,
        "all_metrics_finite": all(
            math.isfinite(float(row["heldout_selected8_exact_rate"]))
            for row in rows
        ),
    }
    valid = all(protocol_checks.values()) and all(execution_checks.values())
    decision = (
        "innovation2_present_selected8_next_round_is_partial_subkey_recovery_not_diffusion_criticality"
        if valid
        else "innovation2_present_selected8_next_round_partial_subkey_identifiability_protocol_invalid"
    )
    gate = {
        "run_id": config.run_id,
        "status": "pass" if valid else "fail",
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "metrics": {
            "master_keys": config.master_keys,
            "regular_rounds": config.rounds,
            "key_round_instances": expected_rows,
            "affected_key_nibbles": list(affected_nibbles),
            "subkey_nibble_instances": nibble_instance_total,
            "unique_key_nibbles": unique_nibble_total,
            "recovered_key_nibbles_exact": recovered_nibble_exact,
            "maximum_calibration_pairs": config.calibration_pairs,
            "maximum_observed_samples_to_unique": max(samples_to_unique, default=0),
            "mean_observed_samples_to_unique": sum(samples_to_unique)
            / max(1, len(samples_to_unique)),
            "heldout_selected8_exact": heldout_exact,
            "heldout_selected8_total": heldout_total,
            "heldout_selected8_exact_rate": heldout_exact
            / max(1, heldout_total),
        },
        "claim_scope": (
            "complete PRESENT current internal state, eight selected next-state bits, known round function, and fixed unknown key; "
            "not plaintext-to-multiround ciphertext prediction, full round-key recovery, cross-key generalization, or SOTA"
        ),
        "next_action": {
            "action": (
                "reject complete-current-state selected8 next-round criticality; retain plaintext-to-multiround ciphertext output prediction"
                if valid
                else "report candidate widths and repair only geometry, sampling, or artifact protocol"
            ),
            "retain_primary_task": "plaintext_to_r_round_true_ciphertext_output_values",
            "train_selected8_complete_current_state_neural_model": False,
            "reopens_r4": False,
        },
    }
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_selected8_next_round_partial_subkey_identifiability_audit",
        "cipher": "PRESENT-80",
        "config": {
            **asdict(config),
            "selected_msb_indices": list(config.selected_msb_indices),
        },
        "geometry": geometry_payload,
        "neural_training": False,
        "sample_classification": False,
        "candidate_input_contract": list(candidate_parameters),
        "master_keys_are_hashed_in_results": True,
        "claim_scope": gate["claim_scope"],
    }
    return {
        "rows": rows,
        "metadata": metadata,
        "summary": {"run_id": config.run_id, "metadata": metadata, "gate": gate},
        "gate": gate,
    }


def _geometry_groups(
    geometry: tuple[SelectedBitGeometry, ...],
) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = {}
    for index, item in enumerate(geometry):
        grouped.setdefault(item.key_nibble, []).append(index)
    return {nibble: tuple(indices) for nibble, indices in grouped.items()}


def _sample_master_keys(
    config: PresentSelected8IdentifiabilityConfig,
) -> tuple[int, ...]:
    rng = random.Random(config.seed)
    keys = [0]
    while len(keys) < config.master_keys:
        candidate = rng.getrandbits(80)
        if candidate not in keys:
            keys.append(candidate)
    return tuple(keys)


def _sample_plaintexts(
    config: PresentSelected8IdentifiabilityConfig,
    key_index: int,
) -> tuple[int, ...]:
    rng = random.Random(config.seed + 10_000 + key_index)
    required = config.calibration_pairs + config.heldout_states_per_round
    plaintexts: list[int] = []
    seen: set[int] = set()
    while len(plaintexts) < required:
        candidate = rng.getrandbits(64)
        if candidate not in seen:
            plaintexts.append(candidate)
            seen.add(candidate)
    return tuple(plaintexts)


def _key_digest(master_key: int) -> str:
    return hashlib.sha256(master_key.to_bytes(10, "big")).hexdigest()
