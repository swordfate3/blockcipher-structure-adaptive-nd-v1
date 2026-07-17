from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, Present80
from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    _present_permutation_layer_words,
    _present_sbox_layer_words,
    present_round_key_matrix,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import make_keys


ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class MultiNibbleIntegralStructure:
    structure_id: str
    active_nibbles: tuple[int, ...]
    output_nibble: int
    output_mask: int
    fixed_plaintext: int

    def __post_init__(self) -> None:
        if not self.structure_id:
            raise ValueError("structure_id must be non-empty")
        if not 1 <= len(self.active_nibbles) <= 2:
            raise ValueError("active_nibbles must contain one or two positions")
        if tuple(sorted(set(self.active_nibbles))) != self.active_nibbles:
            raise ValueError("active_nibbles must be sorted and unique")
        if any(not 0 <= nibble < 16 for nibble in self.active_nibbles):
            raise ValueError("active nibble positions must be in [0, 15]")
        if not 0 <= self.output_nibble < 16:
            raise ValueError("output_nibble must be in [0, 15]")
        if not 1 <= self.output_mask < 16:
            raise ValueError("output_mask must be a nonzero 4-bit mask")
        if self.fixed_plaintext < 0 or self.fixed_plaintext >> 64:
            raise ValueError("fixed_plaintext must fit in 64 bits")
        if self.fixed_plaintext & self.active_word_mask:
            raise ValueError("fixed_plaintext must clear every active nibble")

    @property
    def active_word_mask(self) -> int:
        mask = 0
        for nibble in self.active_nibbles:
            mask |= 0xF << (4 * nibble)
        return mask

    @property
    def set_size(self) -> int:
        return 16 ** len(self.active_nibbles)

    @property
    def signature(self) -> str:
        active = "-".join(f"{nibble:02d}" for nibble in self.active_nibbles)
        return (
            f"a{active}-o{self.output_nibble:02d}-m{self.output_mask:X}-"
            f"p{self.fixed_plaintext:016X}"
        )

    def plaintexts(self) -> np.ndarray:
        values = np.arange(self.set_size, dtype=np.uint64)
        plaintexts = np.full(self.set_size, self.fixed_plaintext, dtype=np.uint64)
        for digit_index, nibble in enumerate(self.active_nibbles):
            active_values = (values >> np.uint64(4 * digit_index)) & np.uint64(0xF)
            plaintexts |= active_values << np.uint64(4 * nibble)
        return plaintexts


@dataclass(frozen=True)
class TransitionAuditConfig:
    run_id: str
    rounds: int = 6
    seed: int = 0
    structures_per_width: int = 64
    keys_per_structure: int = 32
    structure_chunk_size: int = 8

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 6:
            raise ValueError("the frozen transition audit requires PRESENT r6")
        if self.structures_per_width < 16:
            raise ValueError("structures_per_width must be at least 16")
        if self.keys_per_structure < 2:
            raise ValueError("keys_per_structure must be at least 2")
        if self.structure_chunk_size <= 0:
            raise ValueError("structure_chunk_size must be positive")


def make_transition_structures(
    *,
    active_nibble_count: int,
    count: int,
    seed: int,
) -> tuple[MultiNibbleIntegralStructure, ...]:
    if active_nibble_count not in {1, 2}:
        raise ValueError("active_nibble_count must be one or two")
    if count < 16:
        raise ValueError("count must be at least 16")
    rng = np.random.default_rng(seed)
    output_positions = np.resize(np.arange(16, dtype=np.int64), count)
    rng.shuffle(output_positions)
    structures: list[MultiNibbleIntegralStructure] = []
    signatures: set[tuple[tuple[int, ...], int, int, int]] = set()
    while len(structures) < count:
        index = len(structures)
        active_nibbles = tuple(
            sorted(
                int(value)
                for value in rng.choice(16, size=active_nibble_count, replace=False)
            )
        )
        output_nibble = int(output_positions[index])
        output_mask = int(rng.integers(1, 16))
        fixed_plaintext = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        for nibble in active_nibbles:
            fixed_plaintext &= ~(0xF << (4 * nibble))
        signature = (
            active_nibbles,
            output_nibble,
            output_mask,
            fixed_plaintext,
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        structures.append(
            MultiNibbleIntegralStructure(
                structure_id=f"w{active_nibble_count}-{index:04d}",
                active_nibbles=active_nibbles,
                output_nibble=output_nibble,
                output_mask=output_mask,
                fixed_plaintext=fixed_plaintext,
            )
        )
    return tuple(structures)


def multi_nibble_parity_matrix(
    structures: tuple[MultiNibbleIntegralStructure, ...],
    round_keys: np.ndarray,
    *,
    structure_chunk_size: int = 8,
) -> np.ndarray:
    if not structures:
        raise ValueError("structures must be non-empty")
    widths = {len(structure.active_nibbles) for structure in structures}
    if len(widths) != 1:
        raise ValueError("all structures in one matrix must have the same active width")
    if round_keys.ndim != 2 or round_keys.shape[0] < 2:
        raise ValueError("round_keys must have shape (rounds + 1, key_count)")
    if structure_chunk_size <= 0:
        raise ValueError("structure_chunk_size must be positive")

    rounds = round_keys.shape[0] - 1
    result = np.zeros((len(structures), round_keys.shape[1]), dtype=np.uint8)
    sbox = np.asarray(PRESENT_SBOX, dtype=np.uint64)
    parity_lookup = np.asarray(
        [value.bit_count() & 1 for value in range(16)],
        dtype=np.uint8,
    )
    for start in range(0, len(structures), structure_chunk_size):
        stop = min(start + structure_chunk_size, len(structures))
        chunk = structures[start:stop]
        plaintexts = np.asarray(
            [structure.plaintexts() for structure in chunk],
            dtype=np.uint64,
        )
        states = np.broadcast_to(
            plaintexts,
            (round_keys.shape[1], *plaintexts.shape),
        ).copy()
        for round_index in range(rounds):
            states ^= round_keys[round_index, :, None, None]
            states = _present_sbox_layer_words(states, sbox)
            states = _present_permutation_layer_words(states)
        states ^= round_keys[rounds, :, None, None]
        xor_words = np.bitwise_xor.reduce(states, axis=2)
        shifts = np.asarray(
            [4 * structure.output_nibble for structure in chunk],
            dtype=np.uint64,
        )
        masks = np.asarray(
            [structure.output_mask for structure in chunk],
            dtype=np.uint64,
        )
        output_nibbles = (xor_words >> shifts[None, :]) & np.uint64(0xF)
        masked = (output_nibbles & masks[None, :]).astype(np.uint8)
        result[start:stop] = parity_lookup[masked].T
    return result


def scalar_integral_parity(
    structure: MultiNibbleIntegralStructure,
    *,
    rounds: int,
    key: int,
) -> int:
    cipher = Present80(rounds=rounds, key=key)
    xor_word = 0
    for plaintext in structure.plaintexts():
        xor_word ^= cipher.encrypt(int(plaintext))
    output_value = (xor_word >> (4 * structure.output_nibble)) & 0xF
    return (output_value & structure.output_mask).bit_count() & 1


def run_transition_audit(
    config: TransitionAuditConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = make_keys(count=config.keys_per_structure, seed=config.seed + 401)
    round_keys = present_round_key_matrix(keys, rounds=config.rounds)
    structure_sets = {
        width: make_transition_structures(
            active_nibble_count=width,
            count=config.structures_per_width,
            seed=config.seed + 100 * width + 701,
        )
        for width in (1, 2)
    }
    rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    parity_by_width: dict[int, np.ndarray] = {}
    for width in (1, 2):
        _emit(
            progress_callback,
            "parity_matrix_start",
            {"active_nibble_count": width, "set_size": 16**width},
        )
        parity = multi_nibble_parity_matrix(
            structure_sets[width],
            round_keys,
            structure_chunk_size=config.structure_chunk_size,
        )
        parity_by_width[width] = parity
        summary, rates, priors = _summarize_width(
            config,
            width=width,
            structures=structure_sets[width],
            parity=parity,
        )
        rows.append(summary)
        structure_rows.extend(rates)
        position_rows.extend(priors)
        _emit(
            progress_callback,
            "parity_matrix_done",
            {
                "active_nibble_count": width,
                "q1_rate": summary["q1_rate"],
                "balance_rate_std": summary["balance_rate_std"],
            },
        )

    scalar_matches = _crosscheck_scalar(
        structure_sets,
        parity_by_width,
        keys,
        rounds=config.rounds,
    )
    readiness = {
        "two_width_rows_present": len(rows) == 2,
        "same_keys_used_for_both_widths": True,
        "all_output_positions_covered": all(
            {structure.output_nibble for structure in structures} == set(range(16))
            for structures in structure_sets.values()
        ),
        "all_active_context_nibbles_cleared": all(
            structure.fixed_plaintext & structure.active_word_mask == 0
            for structures in structure_sets.values()
            for structure in structures
        ),
        "vectorized_parity_matches_scalar": scalar_matches,
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "q1_rate",
                "mean_balance_rate",
                "balance_rate_std",
                "output_position_residual_std",
                "mixed_structure_fraction",
            )
        ),
    }
    gate = adjudicate_transition_audit(config, rows, readiness)
    return {
        "rows": rows,
        "structure_rows": structure_rows,
        "position_rows": position_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "cipher": "PRESENT-80",
            "rounds": config.rounds,
            "seed": config.seed,
            "structures_per_width": config.structures_per_width,
            "keys_per_structure": config.keys_per_structure,
            "active_nibble_counts": [1, 2],
            "training_performed": False,
            "task": "innovation2_integral_output_property_transition_audit",
            "label": "within-structure masked-output XOR parity per key",
            "keys": [f"{key:020X}" for key in keys],
        },
    }


def adjudicate_transition_audit(
    config: TransitionAuditConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
) -> dict[str, Any]:
    by_width = {int(row["active_nibble_count"]): row for row in rows}
    candidate = by_width.get(2, {})
    q1_rate = _metric(candidate, "q1_rate")
    rate_std = _metric(candidate, "excess_balance_rate_std")
    residual_std = _metric(
        candidate,
        "excess_output_position_residual_std",
    )
    mixed_fraction = _metric(candidate, "mixed_structure_fraction")
    signal_checks = {
        "candidate_q1_rate_not_constant": 0.05 <= q1_rate <= 0.95,
        "candidate_excess_balance_std_at_least_0p03": rate_std >= 0.03,
        "candidate_excess_output_position_residual_std_at_least_0p02": (
            residual_std >= 0.02
        ),
        "candidate_mixed_structure_fraction_at_least_0p10": (
            mixed_fraction >= 0.10
        ),
    }
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_output_property_transition_audit_invalid"
        next_action = {
            "action": "repair parity generation or audit invariants before research use",
            "remote_scale": False,
        }
    elif all(signal_checks.values()):
        status = "pass"
        decision = "innovation2_r6_two_nibble_output_prediction_benchmark_ready"
        next_action = {
            "action": "freeze a local geometry-disjoint r6 output-property training matrix",
            "models": ["structure_mlp", "linear_same_input", "shuffled_label_mlp"],
            "required_controls": [
                "output_position_prior",
                "active_position_prior",
                "output_mask_prior",
                "marginal_distribution_matched_linear",
            ],
            "remote_scale": False,
        }
    elif q1_rate < 0.05:
        status = "hold"
        decision = "innovation2_r6_two_nibble_almost_always_balanced"
        next_action = {
            "action": "repeat the same two-nibble transition audit at PRESENT r7",
            "one_variable": "rounds 6 to 7",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_r6_two_nibble_output_prediction_benchmark_not_ready"
        next_action = {
            "action": "audit fine-grained 5-to-7 active-bit widths before any training",
            "reason": "r6 two-nibble labels lack usable residual structure",
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
            "candidate_q1_rate": q1_rate,
            "candidate_excess_balance_rate_std": rate_std,
            "candidate_excess_output_position_residual_std": residual_std,
            "candidate_mixed_structure_fraction": mixed_fraction,
        },
        "claim_scope": (
            "local label-distribution and marginal-residual audit only; no neural "
            "training, deterministic integral proof, or remote-scale evidence"
        ),
        "next_action": next_action,
    }


def _summarize_width(
    config: TransitionAuditConfig,
    *,
    width: int,
    structures: tuple[MultiNibbleIntegralStructure, ...],
    parity: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    q1_rates = parity.mean(axis=1)
    balance_rates = 1.0 - q1_rates
    position_priors = {
        position: float(
            balance_rates[
                [
                    index
                    for index, structure in enumerate(structures)
                    if structure.output_nibble == position
                ]
            ].mean()
        )
        for position in range(16)
    }
    residuals = np.asarray(
        [
            balance_rates[index] - position_priors[structure.output_nibble]
            for index, structure in enumerate(structures)
        ],
        dtype=np.float64,
    )
    observation_noise_variances = (
        balance_rates * (1.0 - balance_rates) / (parity.shape[1] - 1)
    )
    overall_noise_variance = float(
        observation_noise_variances.mean() * (1.0 - 1.0 / len(structures))
    )
    observed_balance_variance = float(balance_rates.var())
    excess_balance_variance = max(
        0.0,
        observed_balance_variance - overall_noise_variance,
    )
    position_residual_noise_variances = np.empty(len(structures), dtype=np.float64)
    for position in range(16):
        indices = np.asarray(
            [
                index
                for index, structure in enumerate(structures)
                if structure.output_nibble == position
            ],
            dtype=np.int64,
        )
        group_noise = observation_noise_variances[indices]
        group_size = len(indices)
        position_residual_noise_variances[indices] = (
            group_noise * (1.0 - 2.0 / group_size)
            + group_noise.sum() / (group_size * group_size)
        )
    observed_residual_variance = float(residuals.var())
    excess_residual_variance = max(
        0.0,
        observed_residual_variance
        - float(position_residual_noise_variances.mean()),
    )
    structure_rows = [
        {
            "run_id": config.run_id,
            "active_nibble_count": width,
            "set_size": structure.set_size,
            "structure_id": structure.structure_id,
            "signature": structure.signature,
            "active_nibbles": ";".join(str(value) for value in structure.active_nibbles),
            "output_nibble": structure.output_nibble,
            "output_mask": structure.output_mask,
            "fixed_plaintext": f"{structure.fixed_plaintext:016X}",
            "q1_count": int(parity[index].sum()),
            "keys": parity.shape[1],
            "q1_rate": float(q1_rates[index]),
            "balance_rate": float(balance_rates[index]),
            "output_position_balance_prior": position_priors[structure.output_nibble],
            "output_position_residual": float(residuals[index]),
        }
        for index, structure in enumerate(structures)
    ]
    position_rows = [
        {
            "run_id": config.run_id,
            "active_nibble_count": width,
            "output_nibble": position,
            "structures": sum(
                structure.output_nibble == position for structure in structures
            ),
            "mean_balance_rate": position_priors[position],
        }
        for position in range(16)
    ]
    summary = {
        "run_id": config.run_id,
        "task": "innovation2_integral_output_property_transition_audit",
        "role": "anchor" if width == 1 else "candidate",
        "active_nibble_count": width,
        "set_size": 16**width,
        "rounds": config.rounds,
        "seed": config.seed,
        "structures": len(structures),
        "keys_per_structure": parity.shape[1],
        "q1_rate": float(parity.mean()),
        "mean_balance_rate": float(balance_rates.mean()),
        "balance_rate_std": float(balance_rates.std()),
        "output_position_residual_std": float(residuals.std()),
        "estimated_binomial_noise_std": math.sqrt(overall_noise_variance),
        "excess_balance_rate_std": math.sqrt(excess_balance_variance),
        "estimated_position_residual_noise_std": math.sqrt(
            float(position_residual_noise_variances.mean())
        ),
        "excess_output_position_residual_std": math.sqrt(
            excess_residual_variance
        ),
        "mixed_structure_fraction": float(
            np.mean((balance_rates > 0.05) & (balance_rates < 0.95))
        ),
        "near_random_structure_fraction": float(
            np.mean((balance_rates >= 0.40) & (balance_rates <= 0.60))
        ),
        "fully_balanced_structure_fraction": float(np.mean(balance_rates == 1.0)),
        "top_quartile_mean_balance_rate": float(
            np.sort(balance_rates)[-max(1, len(balance_rates) // 4) :].mean()
        ),
        "training_performed": False,
    }
    return summary, structure_rows, position_rows


def _crosscheck_scalar(
    structure_sets: dict[int, tuple[MultiNibbleIntegralStructure, ...]],
    parity_by_width: dict[int, np.ndarray],
    keys: tuple[int, ...],
    *,
    rounds: int,
) -> bool:
    for width in (1, 2):
        for structure_index in range(min(2, len(structure_sets[width]))):
            for key_index in range(min(2, len(keys))):
                expected = scalar_integral_parity(
                    structure_sets[width][structure_index],
                    rounds=rounds,
                    key=keys[key_index],
                )
                if expected != int(parity_by_width[width][structure_index, key_index]):
                    return False
    return True


def _metric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _emit(
    callback: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event, payload)
