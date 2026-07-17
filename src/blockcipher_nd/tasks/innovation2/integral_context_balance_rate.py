from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from blockcipher_nd.tasks.innovation2.integral_bit_transition_audit import (
    BitIntegralStructure,
    scalar_bit_integral_output_xor,
)
from blockcipher_nd.tasks.innovation2.integral_context_diversity import ACTIVE_BITS
from blockcipher_nd.tasks.innovation2.integral_hwang_readiness import _collect_xor_words
from blockcipher_nd.tasks.innovation2.integral_property_prediction import make_keys
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import gf2_kernel_basis


EXPECTED_SOURCE_DECISION = "innovation2_context_kernel_fresh_key_unstable"
EXPECTED_SOURCE_TASK = "innovation2_present_r7_fresh_expanded_context_kernel_diversity"
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ContextBalanceRateConfig:
    run_id: str
    seed: int = 0
    rounds: int = 7
    contexts: int = 64
    keys: int = 128
    key_chunk_size: int = 4

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 7:
            raise ValueError("the frozen E19 audit requires PRESENT r7")
        if self.contexts != 64:
            raise ValueError("the frozen E19 audit requires exactly 64 contexts")
        if self.keys != 128:
            raise ValueError("the frozen E19 audit requires exactly 128 keys")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")


def output_nibble_masks() -> tuple[tuple[int, int, int], ...]:
    masks = tuple(
        (output_nibble, local_mask, local_mask << (4 * output_nibble))
        for output_nibble in range(16)
        for local_mask in range(1, 16)
    )
    if len(masks) != 240 or len({mask for _, _, mask in masks}) != 240:
        raise AssertionError("output mask family must contain 240 unique masks")
    return masks


def run_context_balance_rate_audit(
    config: ContextBalanceRateConfig,
    *,
    source_gate: dict[str, Any],
    source_metadata: dict[str, Any],
    source_result_rows: list[dict[str, Any]],
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    source_checks = validate_source(source_gate, source_metadata, source_result_rows)
    contexts = tuple(int(value, 16) for value in source_metadata["contexts"])
    keys = make_keys(
        count=config.keys,
        seed=int(source_metadata["fresh_key_generation_seed"]),
    )
    structures = tuple(
        BitIntegralStructure(
            structure_id=f"present-r7-rate-context-{context_id:02d}",
            active_bits=ACTIVE_BITS,
            output_nibble=0,
            output_mask=1,
            fixed_plaintext=context,
        )
        for context_id, context in enumerate(contexts)
    )
    xor_words = np.zeros((config.contexts, config.keys), dtype=np.uint64)
    for context_id, structure in enumerate(structures):
        xor_words[context_id] = _collect_xor_words(
            structure,
            keys,
            rounds=config.rounds,
            key_chunk_size=config.key_chunk_size,
            progress_callback=progress_callback,
        )
    scalar_matches = all(
        int(xor_words[context_id, 0])
        == scalar_bit_integral_output_xor(
            structures[context_id],
            rounds=config.rounds,
            key=keys[0],
        )
        for context_id in (0, 16)
    )
    return evaluate_context_balance_rates(
        config,
        contexts=contexts,
        xor_words=xor_words,
        source_result_rows=source_result_rows,
        source_checks=source_checks,
        scalar_matches=scalar_matches,
    )


def validate_source(
    source_gate: dict[str, Any],
    source_metadata: dict[str, Any],
    source_result_rows: list[dict[str, Any]],
) -> dict[str, bool]:
    contexts = source_metadata.get("contexts")
    return {
        "source_gate_is_e18_fresh_key_hold": (
            source_gate.get("status") == "hold"
            and source_gate.get("decision") == EXPECTED_SOURCE_DECISION
        ),
        "source_metadata_is_e18_fresh_context": (
            source_metadata.get("task") == EXPECTED_SOURCE_TASK
            and source_metadata.get("training_performed") is False
            and int(source_metadata.get("fresh_keys", -1)) == 128
        ),
        "source_has_sixty_four_contexts": isinstance(contexts, list)
        and len(contexts) == 64,
        "source_has_sixty_four_result_rows": len(source_result_rows) == 64,
    }


def evaluate_context_balance_rates(
    config: ContextBalanceRateConfig,
    *,
    contexts: tuple[int, ...],
    xor_words: np.ndarray,
    source_result_rows: list[dict[str, Any]],
    source_checks: dict[str, bool],
    scalar_matches: bool,
) -> dict[str, Any]:
    if len(contexts) != 64 or len(set(contexts)) != 64:
        raise ValueError("contexts must contain 64 unique values")
    if xor_words.shape != (64, 128) or xor_words.dtype != np.uint64:
        raise ValueError("xor_words must have shape (64, 128) and dtype uint64")
    source_signatures = {
        int(row["context_id"]): str(row["joint_basis_signature"])
        for row in source_result_rows
    }
    reproduced_signatures = sum(
        ":".join(f"{vector:016X}" for vector in gf2_kernel_basis(xor_words[index]))
        == source_signatures[index]
        for index in range(64)
    )

    masks = output_nibble_masks()
    discovery_rates = np.zeros((64, 240), dtype=np.float64)
    validation_rates = np.zeros((64, 240), dtype=np.float64)
    parity = np.asarray([value.bit_count() & 1 for value in range(16)], dtype=np.uint8)
    mask_index = 0
    for output_nibble in range(16):
        discovery_nibbles = (
            xor_words[:, :64] >> np.uint64(4 * output_nibble)
        ) & np.uint64(0xF)
        validation_nibbles = (
            xor_words[:, 64:] >> np.uint64(4 * output_nibble)
        ) & np.uint64(0xF)
        for local_mask in range(1, 16):
            discovery_balanced = parity[
                (discovery_nibbles & np.uint64(local_mask)).astype(np.uint8)
            ] == 0
            validation_balanced = parity[
                (validation_nibbles & np.uint64(local_mask)).astype(np.uint8)
            ] == 0
            discovery_rates[:, mask_index] = discovery_balanced.mean(axis=1)
            validation_rates[:, mask_index] = validation_balanced.mean(axis=1)
            mask_index += 1

    discovery_residuals = additive_residuals(discovery_rates)
    validation_residuals = additive_residuals(validation_rates)
    rate_correlation = pearson_correlation(discovery_rates, validation_rates)
    residual_correlation = pearson_correlation(
        discovery_residuals, validation_residuals
    )
    mean_absolute_half_difference = float(
        np.mean(np.abs(discovery_rates - validation_rates))
    )
    validation_residual_std = float(np.std(validation_residuals, ddof=1))
    pooled_rates = (discovery_rates + validation_rates) / 2.0
    binomial_noise_variance = float(
        np.mean(pooled_rates * (1.0 - pooled_rates) / 64.0)
    )
    validation_residual_variance = float(np.var(validation_residuals, ddof=1))
    interaction_excess_variance = max(
        0.0, validation_residual_variance - binomial_noise_variance
    )
    rng_context = np.random.default_rng(config.seed + 9101)
    context_shuffle_correlation = pearson_correlation(
        discovery_residuals,
        validation_residuals[rng_context.permutation(64)],
    )
    rng_labels = np.random.default_rng(config.seed + 9201)
    label_shuffle_correlation = pearson_correlation(
        discovery_residuals,
        rng_labels.permutation(validation_residuals.reshape(-1)).reshape(
            validation_residuals.shape
        ),
    )

    cell_rows: list[dict[str, Any]] = []
    for context_id, context in enumerate(contexts):
        for index, (output_nibble, local_mask, mask_value) in enumerate(masks):
            cell_rows.append(
                {
                    "run_id": config.run_id,
                    "context_id": context_id,
                    "fixed_plaintext": f"0x{context:016X}",
                    "mask_index": index,
                    "output_nibble": output_nibble,
                    "local_mask": local_mask,
                    "mask_hex": f"0x{mask_value:016X}",
                    "mask_weight": local_mask.bit_count(),
                    "discovery_balance_rate": float(discovery_rates[context_id, index]),
                    "validation_balance_rate": float(validation_rates[context_id, index]),
                    "discovery_interaction_residual": float(
                        discovery_residuals[context_id, index]
                    ),
                    "validation_interaction_residual": float(
                        validation_residuals[context_id, index]
                    ),
                }
            )
    metrics = {
        "rate_half_correlation": rate_correlation,
        "interaction_residual_half_correlation": residual_correlation,
        "mean_absolute_half_rate_difference": mean_absolute_half_difference,
        "validation_residual_standard_deviation": validation_residual_std,
        "validation_residual_variance": validation_residual_variance,
        "binomial_noise_variance_estimate": binomial_noise_variance,
        "interaction_excess_variance": interaction_excess_variance,
        "context_shuffle_residual_correlation": context_shuffle_correlation,
        "label_shuffle_residual_correlation": label_shuffle_correlation,
    }
    readiness = {
        **source_checks,
        "xor_words_shape_and_dtype_valid": xor_words.shape == (64, 128)
        and xor_words.dtype == np.uint64,
        "zero_and_first_new_context_match_scalar": scalar_matches,
        "all_e18_joint_signatures_reproduced": reproduced_signatures == 64,
        "two_hundred_forty_unique_output_masks": len(masks) == 240,
        "complete_context_mask_grid": len(cell_rows) == 64 * 240,
        "all_metrics_finite": all(math.isfinite(value) for value in metrics.values()),
    }
    gate = adjudicate_context_balance_rates(
        config,
        readiness,
        metrics,
        reproduced_signatures=reproduced_signatures,
    )
    summary_rows = [
        {
            "run_id": config.run_id,
            "metric": key,
            "value": value,
        }
        for key, value in metrics.items()
    ]
    return {
        "rows": summary_rows,
        "cell_rows": cell_rows,
        "xor_words": xor_words,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_present_r7_context_mask_balance_rate_readiness",
            "source_run_id": source_result_rows[0].get("run_id"),
            "cipher": "PRESENT-80",
            "rounds": config.rounds,
            "contexts": 64,
            "keys": 128,
            "key_half_size": 64,
            "key_generation_seed": 8801 + config.seed,
            "candidate_masks": 240,
            "cells": len(cell_rows),
            "plaintexts_per_context_per_key": 1 << 16,
            "xor_words_shape": [64, 128],
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def additive_residuals(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional context by mask matrix")
    return values - values.mean(axis=1, keepdims=True) - values.mean(
        axis=0, keepdims=True
    ) + values.mean()


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_flat.shape != right_flat.shape:
        raise ValueError("correlation arrays must have equal size")
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = float(
        np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    )
    if denominator <= 1e-15:
        return 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def adjudicate_context_balance_rates(
    config: ContextBalanceRateConfig,
    readiness_checks: dict[str, bool],
    metrics: dict[str, float],
    *,
    reproduced_signatures: int,
) -> dict[str, Any]:
    signal_checks = {
        "rate_half_correlation_at_least_0p25": (
            metrics["rate_half_correlation"] >= 0.25
        ),
        "interaction_residual_half_correlation_at_least_0p20": (
            metrics["interaction_residual_half_correlation"] >= 0.20
        ),
        "mean_absolute_half_rate_difference_at_most_0p15": (
            metrics["mean_absolute_half_rate_difference"] <= 0.15
        ),
        "validation_residual_std_at_least_0p05": (
            metrics["validation_residual_standard_deviation"] >= 0.05
        ),
        "interaction_excess_variance_positive": (
            metrics["interaction_excess_variance"] > 0.0
        ),
        "context_shuffle_abs_correlation_below_0p10": (
            abs(metrics["context_shuffle_residual_correlation"]) < 0.10
        ),
        "label_shuffle_abs_correlation_below_0p10": (
            abs(metrics["label_shuffle_residual_correlation"]) < 0.10
        ),
    }
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_balance_rate_protocol_invalid"
        next_action = {
            "action": "repair E18 replay, XOR cache, mask grid, or scalar validation",
            "training": False,
            "remote_scale": False,
        }
    elif all(signal_checks.values()):
        status = "pass"
        decision = "innovation2_balance_rate_interaction_ready"
        next_action = {
            "action": "design continuous balance-rate prediction with group-disjoint splits",
            "next_adjudication": "E20 continuous output-property model readiness",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_balance_rate_interaction_not_reproducible"
        next_action = {
            "action": "stop PRESENT r7 context balance-rate branch",
            "reason": "cross-key interaction residual is weak, noisy, or shortcut-like",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "signal_checks": signal_checks,
        "reproduced_e18_joint_signatures": reproduced_signatures,
        "metrics": metrics,
        "claim_scope": (
            "64-context by 240-mask balance-rate readiness over the two frozen E18 "
            "64-key halves; derived from sampled keys, not a neural result"
        ),
        "next_action": next_action,
    }
