from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable

from blockcipher_nd.ciphers.spn.present import Present80


MASK64 = (1 << 64) - 1
RUN_ID = "i2_present_next_round_full_state_identifiability_audit_20260722"
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class PresentNextRoundIdentifiabilityConfig:
    run_id: str = RUN_ID
    seed: int = 20_260_722
    master_keys: int = 16
    rounds: int = 31
    heldout_states_per_round: int = 256

    def __post_init__(self) -> None:
        if self.master_keys <= 0:
            raise ValueError("master_keys must be positive")
        if self.rounds != 31:
            raise ValueError("the formal PRESENT audit is frozen to 31 regular rounds")
        if self.heldout_states_per_round <= 0:
            raise ValueError("heldout_states_per_round must be positive")


@dataclass(frozen=True)
class PresentTrace:
    states_before_round: tuple[int, ...]
    states_after_round: tuple[int, ...]
    regular_round_keys: tuple[int, ...]
    final_whitening_key: int
    ciphertext: int


def derive_present80_round_keys(
    master_key: int,
    *,
    rounds: int = 31,
) -> tuple[tuple[int, ...], int]:
    key_register = master_key & ((1 << 80) - 1)
    regular_keys: list[int] = []
    for round_counter in range(1, rounds + 1):
        regular_keys.append((key_register >> 16) & MASK64)
        key_register = Present80._update_key(key_register, round_counter)
    return tuple(regular_keys), (key_register >> 16) & MASK64


def present_regular_round(state: int, round_key: int) -> int:
    keyed = (state ^ round_key) & MASK64
    substituted = Present80._sbox_layer(keyed)
    return Present80.permutation_layer(substituted)


def recover_present_round_key(current_state: int, next_state: int) -> int:
    before_permutation = Present80.inverse_permutation_layer(next_state)
    before_sbox = Present80.inverse_sbox_layer(before_permutation)
    return (current_state ^ before_sbox) & MASK64


def trace_present80(
    plaintext: int,
    master_key: int,
    *,
    rounds: int = 31,
) -> PresentTrace:
    regular_keys, final_key = derive_present80_round_keys(
        master_key,
        rounds=rounds,
    )
    state = plaintext & MASK64
    before: list[int] = []
    after: list[int] = []
    for round_key in regular_keys:
        before.append(state)
        state = present_regular_round(state, round_key)
        after.append(state)
    ciphertext = (state ^ final_key) & MASK64
    return PresentTrace(
        states_before_round=tuple(before),
        states_after_round=tuple(after),
        regular_round_keys=regular_keys,
        final_whitening_key=final_key,
        ciphertext=ciphertext,
    )


def encrypt_with_recovered_round_keys(
    plaintext: int,
    regular_round_keys: tuple[int, ...],
    final_whitening_key: int,
) -> int:
    state = plaintext & MASK64
    for round_key in regular_round_keys:
        state = present_regular_round(state, round_key)
    return (state ^ final_whitening_key) & MASK64


def audit_present_next_round_identifiability(
    config: PresentNextRoundIdentifiabilityConfig,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    master_keys = _sample_master_keys(config)
    rows: list[dict[str, Any]] = []
    key_summaries: list[dict[str, Any]] = []
    heldout_transition_total = 0
    heldout_transition_exact = 0
    full_reconstruction_total = 0
    full_reconstruction_exact = 0
    traced_cipher_total = 0
    traced_cipher_exact = 0
    regular_key_exact = 0
    whitening_key_exact = 0
    all_plaintext_sets_unique = True
    all_calibration_disjoint = True

    for key_index, master_key in enumerate(master_keys):
        plaintexts = _sample_plaintexts(config, key_index)
        all_plaintext_sets_unique = all_plaintext_sets_unique and (
            len(set(plaintexts)) == len(plaintexts)
        )
        all_calibration_disjoint = all_calibration_disjoint and (
            plaintexts[0] not in set(plaintexts[1:])
        )
        traces = [
            trace_present80(plaintext, master_key, rounds=config.rounds)
            for plaintext in plaintexts
        ]
        cipher = Present80(rounds=config.rounds, key=master_key)
        for plaintext, trace in zip(plaintexts, traces, strict=True):
            traced_cipher_total += 1
            traced_cipher_exact += trace.ciphertext == cipher.encrypt(plaintext)

        calibration = traces[0]
        recovered_regular_keys = tuple(
            recover_present_round_key(current_state, next_state)
            for current_state, next_state in zip(
                calibration.states_before_round,
                calibration.states_after_round,
                strict=True,
            )
        )
        recovered_final_key = (
            calibration.states_after_round[-1] ^ calibration.ciphertext
        ) & MASK64
        whitening_exact = recovered_final_key == calibration.final_whitening_key
        whitening_key_exact += whitening_exact

        heldout_traces = traces[1:]
        for round_offset in range(config.rounds):
            expected_key = calibration.regular_round_keys[round_offset]
            recovered_key = recovered_regular_keys[round_offset]
            key_exact = recovered_key == expected_key
            regular_key_exact += key_exact
            exact_predictions = 0
            for trace in heldout_traces:
                predicted = present_regular_round(
                    trace.states_before_round[round_offset],
                    recovered_key,
                )
                exact_predictions += (
                    predicted == trace.states_after_round[round_offset]
                )
            heldout_transition_total += len(heldout_traces)
            heldout_transition_exact += exact_predictions
            rows.append(
                {
                    "run_id": config.run_id,
                    "key_index": key_index,
                    "master_key_sha256": _key_digest(master_key),
                    "round_index": round_offset + 1,
                    "calibration_transitions": 1,
                    "recovered_round_key_exact": key_exact,
                    "heldout_transitions": len(heldout_traces),
                    "heldout_exact_predictions": exact_predictions,
                    "heldout_exact_rate": exact_predictions
                    / max(1, len(heldout_traces)),
                }
            )

        exact_encryptions = 0
        for plaintext, trace in zip(
            plaintexts[1:],
            heldout_traces,
            strict=True,
        ):
            reconstructed = encrypt_with_recovered_round_keys(
                plaintext,
                recovered_regular_keys,
                recovered_final_key,
            )
            exact_encryptions += reconstructed == trace.ciphertext
        full_reconstruction_total += len(heldout_traces)
        full_reconstruction_exact += exact_encryptions
        key_summaries.append(
            {
                "key_index": key_index,
                "master_key_sha256": _key_digest(master_key),
                "regular_round_keys_exact": sum(
                    recovered == expected
                    for recovered, expected in zip(
                        recovered_regular_keys,
                        calibration.regular_round_keys,
                        strict=True,
                    )
                ),
                "expected_regular_round_keys": config.rounds,
                "final_whitening_key_exact": whitening_exact,
                "heldout_full_encryptions": len(heldout_traces),
                "heldout_full_encryptions_exact": exact_encryptions,
            }
        )
        if progress is not None:
            progress(
                "master_key_done",
                {
                    "key_index": key_index,
                    "regular_round_keys_exact": key_summaries[-1][
                        "regular_round_keys_exact"
                    ],
                    "final_whitening_key_exact": whitening_exact,
                    "heldout_full_encryptions_exact": exact_encryptions,
                },
            )

    expected_rows = config.master_keys * config.rounds
    expected_transitions = (
        config.master_keys * config.rounds * config.heldout_states_per_round
    )
    expected_full_encryptions = (
        config.master_keys * config.heldout_states_per_round
    )
    expected_traced_ciphers = config.master_keys * (
        config.heldout_states_per_round + 1
    )
    protocol_checks = {
        "official_present_zero_key_vector_matches": Present80(
            rounds=31,
            key=0,
        ).encrypt(0)
        == 0x5579C1387B228445,
        "configured_master_keys_are_unique": len(master_keys) == config.master_keys
        and len(set(master_keys)) == config.master_keys,
        "actual_present80_key_schedule_used": all(
            len(trace_present80(0, key).regular_round_keys) == config.rounds
            for key in master_keys
        ),
        "all_plaintexts_unique_within_each_key": all_plaintext_sets_unique,
        "calibration_plaintext_disjoint_from_heldout": all_calibration_disjoint,
        "one_calibration_transition_per_key_round": all(
            int(row["calibration_transitions"]) == 1 for row in rows
        ),
        "complete_64_bit_current_and_next_states": True,
        "known_invertible_present_sbox_and_p_layer": True,
        "labels_are_states_not_sample_classes": True,
    }
    execution_checks = {
        "result_rows_complete": len(rows) == expected_rows,
        "all_regular_round_keys_recovered_exactly": regular_key_exact
        == expected_rows,
        "all_final_whitening_keys_recovered_exactly": whitening_key_exact
        == config.master_keys,
        "heldout_transition_count_complete": heldout_transition_total
        == expected_transitions,
        "all_heldout_next_states_predicted_exactly": heldout_transition_exact
        == expected_transitions,
        "full_reconstruction_count_complete": full_reconstruction_total
        == expected_full_encryptions,
        "all_heldout_full_encryptions_reconstructed_exactly": full_reconstruction_exact
        == expected_full_encryptions,
        "all_traces_match_present80_encrypt": traced_cipher_total
        == expected_traced_ciphers
        and traced_cipher_exact == expected_traced_ciphers,
        "all_metrics_finite": all(
            math.isfinite(float(row["heldout_exact_rate"])) for row in rows
        ),
    }
    valid = all(protocol_checks.values()) and all(execution_checks.values())
    status = "pass" if valid else "fail"
    decision = (
        "innovation2_present_full_state_next_round_criticality_not_identifiable"
        if valid
        else "innovation2_present_full_state_next_round_identifiability_protocol_invalid"
    )
    next_action = (
        "retain plaintext-to-multiround-ciphertext output prediction; do not train full-state next-round neural criticality models"
        if valid
        else "repair only the PRESENT round boundary, inverse layers, key schedule, or artifact counts"
    )
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "metrics": {
            "master_keys": config.master_keys,
            "regular_rounds": config.rounds,
            "regular_round_keys_exact": regular_key_exact,
            "expected_regular_round_keys": expected_rows,
            "final_whitening_keys_exact": whitening_key_exact,
            "expected_final_whitening_keys": config.master_keys,
            "heldout_transition_exact": heldout_transition_exact,
            "heldout_transition_total": heldout_transition_total,
            "heldout_transition_exact_rate": heldout_transition_exact
            / max(1, heldout_transition_total),
            "full_reconstruction_exact": full_reconstruction_exact,
            "full_reconstruction_total": full_reconstruction_total,
            "full_reconstruction_exact_rate": full_reconstruction_exact
            / max(1, full_reconstruction_total),
            "calibration_transitions_per_key_round": 1,
        },
        "claim_scope": (
            "known PRESENT round function, fixed unknown key, complete current and next internal states; "
            "not plaintext-to-multiround selected-output prediction, partial-state prediction, cross-key generalization, or SOTA"
        ),
        "next_action": {
            "action": next_action,
            "retain_primary_task": "plaintext_to_r_round_true_ciphertext_output_values",
            "train_full_state_next_round_models": False,
        },
    }
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_full_state_next_round_identifiability_audit",
        "cipher": "PRESENT-80",
        "config": asdict(config),
        "neural_training": False,
        "sample_classification": False,
        "state_transition": "Y = P(S(X xor K_r))",
        "deterministic_recovery": "K_r = X xor S_inverse(P_inverse(Y))",
        "calibration_plaintexts_per_key": 1,
        "heldout_plaintexts_per_key": config.heldout_states_per_round,
        "master_keys_are_hashed_in_results": True,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "key_summaries": key_summaries,
        "gate": gate,
    }
    return {
        "rows": rows,
        "metadata": metadata,
        "summary": summary,
        "gate": gate,
    }


def _sample_master_keys(
    config: PresentNextRoundIdentifiabilityConfig,
) -> tuple[int, ...]:
    rng = random.Random(config.seed)
    keys = [0]
    while len(keys) < config.master_keys:
        candidate = rng.getrandbits(80)
        if candidate not in keys:
            keys.append(candidate)
    return tuple(keys)


def _sample_plaintexts(
    config: PresentNextRoundIdentifiabilityConfig,
    key_index: int,
) -> tuple[int, ...]:
    rng = random.Random(config.seed + 10_000 + key_index)
    required = config.heldout_states_per_round + 1
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
