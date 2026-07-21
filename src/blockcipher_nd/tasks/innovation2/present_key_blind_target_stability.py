from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, Present80
from blockcipher_nd.tasks.innovation2.present_next_round_identifiability import (
    derive_present80_round_keys,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i2_output_prediction_opk1_present_r3_key_blind_target_stability_audit_20260722"
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class PresentKeyBlindStabilityConfig:
    run_id: str = RUN_ID
    seed: int = 20_260_724
    rounds: int = 3
    plaintexts: int = 1024
    reference_keys: int = 256
    evaluation_keys: int = 256
    selected_msb_indices: tuple[int, ...] = SELECTED_MSB_INDICES
    maximum_mean_directional_auc: float = 0.510
    maximum_bit_directional_auc: float = 0.515
    maximum_mean_accuracy_gain: float = 0.005
    minimum_evaluation_prevalence: float = 0.48
    maximum_evaluation_prevalence: float = 0.52

    def __post_init__(self) -> None:
        if self.rounds != 3:
            raise ValueError("OPK1 is frozen to PRESENT round three")
        if min(self.plaintexts, self.reference_keys, self.evaluation_keys) <= 0:
            raise ValueError("plaintext and key counts must be positive")
        if self.selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("OPK1 positions must match OP10 through OPC1")
        if not 0.5 <= self.maximum_mean_directional_auc <= 1.0:
            raise ValueError("invalid mean directional AUC gate")
        if not 0.5 <= self.maximum_bit_directional_auc <= 1.0:
            raise ValueError("invalid bit directional AUC gate")
        if not 0.0 <= self.minimum_evaluation_prevalence < 0.5:
            raise ValueError("invalid minimum prevalence gate")
        if not 0.5 < self.maximum_evaluation_prevalence <= 1.0:
            raise ValueError("invalid maximum prevalence gate")


def audit_present_key_blind_target_stability(
    config: PresentKeyBlindStabilityConfig,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    reference_keys, evaluation_keys = _sample_keys(config)
    plaintexts = _sample_plaintexts(config)
    output_bits = len(config.selected_msb_indices)
    reference_sums = np.zeros((config.plaintexts, output_bits), dtype=np.uint16)

    for start in range(0, config.reference_keys, 32):
        stop = min(config.reference_keys, start + 32)
        reference_sums += np.sum(
            encrypt_selected_bits_batch(
                reference_keys[start:stop],
                plaintexts,
                rounds=config.rounds,
                selected_msb_indices=config.selected_msb_indices,
            ),
            axis=0,
            dtype=np.uint16,
        )
        if progress is not None:
            progress(
                "reference_keys_done",
                {"completed": stop, "expected": config.reference_keys},
            )

    reference_scores = reference_sums.astype(np.float64) / config.reference_keys
    evaluation_labels = np.empty(
        (config.evaluation_keys, config.plaintexts, output_bits), dtype=np.uint8
    )
    for start in range(0, config.evaluation_keys, 32):
        stop = min(config.evaluation_keys, start + 32)
        evaluation_labels[start:stop] = encrypt_selected_bits_batch(
            evaluation_keys[start:stop],
            plaintexts,
            rounds=config.rounds,
            selected_msb_indices=config.selected_msb_indices,
        )
        if progress is not None:
            progress(
                "evaluation_keys_done",
                {"completed": stop, "expected": config.evaluation_keys},
            )

    rows: list[dict[str, Any]] = []
    for column, msb_position in enumerate(config.selected_msb_indices):
        labels = evaluation_labels[:, :, column].reshape(-1).astype(np.float64)
        scores = np.broadcast_to(
            reference_scores[:, column],
            (config.evaluation_keys, config.plaintexts),
        ).reshape(-1)
        prevalence = float(np.mean(labels))
        reference_prevalence = float(np.mean(reference_scores[:, column]))
        majority_accuracy = max(prevalence, 1.0 - prevalence)
        accuracy = float(np.mean((scores >= 0.5) == labels))
        auc = float(binary_auc(labels, scores))
        per_plaintext_prevalence = np.mean(
            evaluation_labels[:, :, column], axis=0
        )
        rows.append(
            {
                "run_id": config.run_id,
                "msb_index": msb_position,
                "target": "true_present_r3_ciphertext_output_bit",
                "sample_classification": False,
                "reference_key_prevalence": reference_prevalence,
                "evaluation_key_prevalence": prevalence,
                "auc": auc,
                "directional_auc": max(auc, 1.0 - auc),
                "threshold_accuracy": accuracy,
                "majority_accuracy": majority_accuracy,
                "accuracy_minus_majority": accuracy - majority_accuracy,
                "brier": float(np.mean((scores - labels) ** 2)),
                "mean_plaintext_prevalence_abs_deviation_from_half": float(
                    np.mean(np.abs(per_plaintext_prevalence - 0.5))
                ),
                "maximum_plaintext_prevalence_abs_deviation_from_half": float(
                    np.max(np.abs(per_plaintext_prevalence - 0.5))
                ),
            }
        )

    directional_aucs = [float(row["directional_auc"]) for row in rows]
    accuracy_gains = [float(row["accuracy_minus_majority"]) for row in rows]
    evaluation_prevalences = [
        float(row["evaluation_key_prevalence"]) for row in rows
    ]
    mean_directional_auc = float(np.mean(directional_aucs))
    mean_accuracy_gain = float(np.mean(accuracy_gains))
    exact_vector_majority = _mean_exact_vector_majority(evaluation_labels)
    protocol_checks = {
        "official_present_zero_key_vector_matches": Present80(
            rounds=31, key=0
        ).encrypt(0)
        == 0x5579C1387B228445,
        "reference_keys_unique": len(set(reference_keys)) == config.reference_keys,
        "evaluation_keys_unique": len(set(evaluation_keys))
        == config.evaluation_keys,
        "reference_and_evaluation_keys_disjoint": not (
            set(reference_keys) & set(evaluation_keys)
        ),
        "shared_plaintexts_unique": len(set(plaintexts)) == config.plaintexts,
        "same_plaintexts_used_for_both_key_sets": True,
        "scores_use_reference_keys_only": reference_scores.shape
        == (config.plaintexts, output_bits),
        "evaluation_labels_are_true_ciphertext_bits": _spot_check_labels(
            config, plaintexts, evaluation_keys, evaluation_labels
        ),
        "labels_are_outputs_not_sample_classes": True,
    }
    execution_checks = {
        "eight_result_rows_complete": len(rows) == output_bits,
        "reference_observation_count_complete": len(reference_keys)
        * len(plaintexts)
        == config.reference_keys * config.plaintexts,
        "evaluation_observation_count_complete": evaluation_labels.shape
        == (config.evaluation_keys, config.plaintexts, output_bits),
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in rows
            for field in (
                "reference_key_prevalence",
                "evaluation_key_prevalence",
                "auc",
                "directional_auc",
                "threshold_accuracy",
                "majority_accuracy",
                "accuracy_minus_majority",
                "brier",
            )
        ),
    }
    stability_checks = {
        "mean_directional_auc_at_most_0_510": mean_directional_auc
        <= config.maximum_mean_directional_auc,
        "every_bit_directional_auc_at_most_0_515": max(directional_aucs)
        <= config.maximum_bit_directional_auc,
        "mean_accuracy_gain_at_most_0_005": mean_accuracy_gain
        <= config.maximum_mean_accuracy_gain,
        "every_evaluation_prevalence_between_0_48_and_0_52": all(
            config.minimum_evaluation_prevalence
            <= prevalence
            <= config.maximum_evaluation_prevalence
            for prevalence in evaluation_prevalences
        ),
    }
    valid = all(protocol_checks.values()) and all(execution_checks.values())
    if not valid:
        status = "fail"
        decision = "innovation2_present_r3_key_blind_target_stability_protocol_invalid"
        action = "repair only the key split, plaintext reuse, true-output replay, metrics, or artifacts"
    elif all(stability_checks.values()):
        status = "pass"
        decision = "innovation2_present_r3_key_blind_zero_shot_target_not_stable"
        action = "report per-key retraining as cross-key repeatability; require support-set conditioning for stronger generalization"
    else:
        status = "hold"
        decision = "innovation2_present_r3_key_blind_cross_key_bias_requires_replication"
        action = "repeat the frozen lookup audit on fresh disjoint key sets before any neural training"
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "stability_checks": stability_checks,
        "metrics": {
            "reference_keys": config.reference_keys,
            "evaluation_keys": config.evaluation_keys,
            "shared_plaintexts": config.plaintexts,
            "reference_observations": config.reference_keys * config.plaintexts,
            "evaluation_observations": config.evaluation_keys * config.plaintexts,
            "mean_auc": float(np.mean([float(row["auc"]) for row in rows])),
            "mean_directional_auc": mean_directional_auc,
            "maximum_directional_auc": max(directional_aucs),
            "mean_accuracy_minus_majority": mean_accuracy_gain,
            "minimum_evaluation_prevalence": min(evaluation_prevalences),
            "maximum_evaluation_prevalence": max(evaluation_prevalences),
            "mean_exact_selected8_vector_majority_rate_per_plaintext": exact_vector_majority,
        },
        "claim_scope": (
            "PRESENT r3, same plaintexts across disjoint reference/evaluation keys, no key identity or calibration input; "
            "not per-key retraining, support-set conditioning, key-input prediction, r4 evidence, or SOTA"
        ),
        "next_action": {
            "action": action,
            "retain_primary_task": "fixed_unknown_key_plaintext_to_true_ciphertext_output_values",
            "key_blind_zero_shot_model_authorized": False,
            "support_set_conditioning_requires_new_plan": True,
            "opb1_and_opc1_unchanged": True,
        },
    }
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_key_blind_cross_key_target_stability_audit",
        "cipher": "PRESENT-80",
        "config": {
            **asdict(config),
            "selected_msb_indices": list(config.selected_msb_indices),
        },
        "reference_key_set_sha256": _key_set_digest(reference_keys),
        "evaluation_key_set_sha256": _key_set_digest(evaluation_keys),
        "plaintext_set_sha256": _plaintext_set_digest(plaintexts),
        "neural_training": False,
        "sample_classification": False,
        "score_contract": "per-plaintext per-bit frequency estimated only from reference keys",
        "claim_scope": gate["claim_scope"],
    }
    return {
        "rows": rows,
        "metadata": metadata,
        "summary": {"run_id": config.run_id, "metadata": metadata, "rows": rows, "gate": gate},
        "gate": gate,
    }


def _sample_keys(
    config: PresentKeyBlindStabilityConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rng = random.Random(config.seed)
    required = config.reference_keys + config.evaluation_keys
    keys: list[int] = []
    seen: set[int] = set()
    while len(keys) < required:
        candidate = rng.getrandbits(80)
        if candidate not in seen:
            seen.add(candidate)
            keys.append(candidate)
    return (
        tuple(keys[: config.reference_keys]),
        tuple(keys[config.reference_keys :]),
    )


def _sample_plaintexts(config: PresentKeyBlindStabilityConfig) -> tuple[int, ...]:
    rng = random.Random(config.seed + 1)
    plaintexts: list[int] = []
    seen: set[int] = set()
    while len(plaintexts) < config.plaintexts:
        candidate = rng.getrandbits(64)
        if candidate not in seen:
            seen.add(candidate)
            plaintexts.append(candidate)
    return tuple(plaintexts)


def _selected_bits(
    ciphertext: int,
    selected_msb_indices: tuple[int, ...],
) -> np.ndarray:
    return np.asarray(
        [(ciphertext >> (63 - position)) & 1 for position in selected_msb_indices],
        dtype=np.uint8,
    )


def encrypt_selected_bits_batch(
    keys: tuple[int, ...],
    plaintexts: tuple[int, ...],
    *,
    rounds: int,
    selected_msb_indices: tuple[int, ...],
) -> np.ndarray:
    if not keys or not plaintexts:
        raise ValueError("keys and plaintexts must be nonempty")
    regular_keys = []
    whitening_keys = []
    for key in keys:
        regular, whitening = derive_present80_round_keys(key, rounds=rounds)
        regular_keys.append(regular)
        whitening_keys.append(whitening)
    round_key_array = np.asarray(regular_keys, dtype=np.uint64)
    states = np.broadcast_to(
        np.asarray(plaintexts, dtype=np.uint64),
        (len(keys), len(plaintexts)),
    ).copy()
    for round_index in range(rounds):
        states ^= round_key_array[:, round_index, None]
        states = _present_sbox_layer_batch(states)
        states = _present_permutation_layer_batch(states)
    states ^= np.asarray(whitening_keys, dtype=np.uint64)[:, None]
    return np.stack(
        [
            ((states >> np.uint64(63 - position)) & np.uint64(1)).astype(np.uint8)
            for position in selected_msb_indices
        ],
        axis=2,
    )


def _present_sbox_layer_batch(states: np.ndarray) -> np.ndarray:
    out = np.zeros_like(states, dtype=np.uint64)
    sbox = np.asarray(PRESENT_SBOX, dtype=np.uint64)
    for nibble_index in range(16):
        shift = np.uint64(4 * nibble_index)
        nibble = ((states >> shift) & np.uint64(0xF)).astype(np.uint8)
        out |= sbox[nibble] << shift
    return out


def _present_permutation_layer_batch(states: np.ndarray) -> np.ndarray:
    out = np.zeros_like(states, dtype=np.uint64)
    for source in range(63):
        destination = (16 * source) % 63
        out |= (
            (states >> np.uint64(source)) & np.uint64(1)
        ) << np.uint64(destination)
    out |= ((states >> np.uint64(63)) & np.uint64(1)) << np.uint64(63)
    return out


def _spot_check_labels(
    config: PresentKeyBlindStabilityConfig,
    plaintexts: tuple[int, ...],
    evaluation_keys: tuple[int, ...],
    labels: np.ndarray,
) -> bool:
    key_indices = sorted({0, len(evaluation_keys) // 2, len(evaluation_keys) - 1})
    plaintext_indices = sorted({0, len(plaintexts) // 2, len(plaintexts) - 1})
    for key_index in key_indices:
        cipher = Present80(rounds=config.rounds, key=evaluation_keys[key_index])
        for plaintext_index in plaintext_indices:
            expected = _selected_bits(
                cipher.encrypt(plaintexts[plaintext_index]),
                config.selected_msb_indices,
            )
            if not np.array_equal(labels[key_index, plaintext_index], expected):
                return False
    return True


def _mean_exact_vector_majority(labels: np.ndarray) -> float:
    weights = (1 << np.arange(labels.shape[2], dtype=np.uint16)).reshape(1, 1, -1)
    packed = np.sum(labels.astype(np.uint16) * weights, axis=2)
    rates = [
        np.max(np.bincount(packed[:, plaintext_index], minlength=256))
        / labels.shape[0]
        for plaintext_index in range(labels.shape[1])
    ]
    return float(np.mean(rates))


def _key_set_digest(keys: tuple[int, ...]) -> str:
    return hashlib.sha256(b"".join(key.to_bytes(10, "big") for key in keys)).hexdigest()


def _plaintext_set_digest(plaintexts: tuple[int, ...]) -> str:
    return hashlib.sha256(
        b"".join(plaintext.to_bytes(8, "big") for plaintext in plaintexts)
    ).hexdigest()
