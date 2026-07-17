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


ACTIVE_BIT_WIDTHS = (5, 6, 7)
ALLOWED_ACTIVE_BIT_WIDTHS = (4, *ACTIVE_BIT_WIDTHS, 16)
MARGINAL_INPUT_BITS = 64 + 16 + 15
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class BitIntegralStructure:
    structure_id: str
    active_bits: tuple[int, ...]
    output_nibble: int
    output_mask: int
    fixed_plaintext: int

    def __post_init__(self) -> None:
        if not self.structure_id:
            raise ValueError("structure_id must be non-empty")
        if len(self.active_bits) not in ALLOWED_ACTIVE_BIT_WIDTHS:
            raise ValueError(
                "active_bits must contain four, five, six, seven, or sixteen positions"
            )
        if tuple(sorted(set(self.active_bits))) != self.active_bits:
            raise ValueError("active_bits must be sorted and unique")
        if any(not 0 <= bit < 64 for bit in self.active_bits):
            raise ValueError("active bit positions must be in [0, 63]")
        if not 0 <= self.output_nibble < 16:
            raise ValueError("output_nibble must be in [0, 15]")
        if not 1 <= self.output_mask < 16:
            raise ValueError("output_mask must be a nonzero 4-bit mask")
        if self.fixed_plaintext < 0 or self.fixed_plaintext >> 64:
            raise ValueError("fixed_plaintext must fit in 64 bits")
        if self.fixed_plaintext & self.active_word_mask:
            raise ValueError("fixed_plaintext must clear every active bit")

    @property
    def active_word_mask(self) -> int:
        mask = 0
        for bit in self.active_bits:
            mask |= 1 << bit
        return mask

    @property
    def set_size(self) -> int:
        return 1 << len(self.active_bits)

    @property
    def signature(self) -> str:
        active = "-".join(f"{bit:02d}" for bit in self.active_bits)
        return (
            f"b{active}-o{self.output_nibble:02d}-m{self.output_mask:X}-"
            f"p{self.fixed_plaintext:016X}"
        )

    def plaintexts(self) -> np.ndarray:
        assignments = np.arange(self.set_size, dtype=np.uint64)
        plaintexts = np.full(self.set_size, self.fixed_plaintext, dtype=np.uint64)
        for source_bit, target_bit in enumerate(self.active_bits):
            values = (assignments >> np.uint64(source_bit)) & np.uint64(1)
            plaintexts |= values << np.uint64(target_bit)
        return plaintexts

    def marginal_feature_vector(self) -> np.ndarray:
        features = np.zeros(MARGINAL_INPUT_BITS, dtype=np.float64)
        features[list(self.active_bits)] = 1.0
        features[64 + self.output_nibble] = 1.0
        features[64 + 16 + self.output_mask - 1] = 1.0
        return features


@dataclass(frozen=True)
class BitTransitionAuditConfig:
    run_id: str
    rounds: int = 6
    seed: int = 0
    structures_per_width: int = 64
    keys_per_structure: int = 256
    structure_chunk_size: int = 4
    ridge_alpha: float = 1.0
    crossfit_folds: int = 4

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 6:
            raise ValueError("the frozen bit transition audit requires PRESENT r6")
        if self.structures_per_width < 64 or self.structures_per_width % 16:
            raise ValueError("structures_per_width must be a multiple of 16 and at least 64")
        if self.keys_per_structure < 4 or self.keys_per_structure % 2:
            raise ValueError("keys_per_structure must be even and at least four")
        if self.structure_chunk_size <= 0:
            raise ValueError("structure_chunk_size must be positive")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if self.crossfit_folds != 4:
            raise ValueError("the frozen audit requires four cross-fitting folds")


def make_bit_transition_structures(
    *,
    active_bit_width: int,
    count: int,
    seed: int,
) -> tuple[BitIntegralStructure, ...]:
    if active_bit_width not in ACTIVE_BIT_WIDTHS:
        raise ValueError("active_bit_width must be five, six, or seven")
    if count < 64 or count % 16:
        raise ValueError("count must be a multiple of 16 and at least 64")
    rng = np.random.default_rng(seed)
    output_positions = np.resize(np.arange(16, dtype=np.int64), count)
    output_masks = np.resize(np.arange(1, 16, dtype=np.int64), count)
    rng.shuffle(output_positions)
    rng.shuffle(output_masks)
    structures: list[BitIntegralStructure] = []
    signatures: set[tuple[tuple[int, ...], int, int, int]] = set()
    while len(structures) < count:
        index = len(structures)
        active_bits = tuple(
            sorted(
                int(value)
                for value in rng.choice(64, size=active_bit_width, replace=False)
            )
        )
        output_nibble = int(output_positions[index])
        output_mask = int(output_masks[index])
        fixed_plaintext = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        for bit in active_bits:
            fixed_plaintext &= ~(1 << bit)
        signature = (active_bits, output_nibble, output_mask, fixed_plaintext)
        if signature in signatures:
            continue
        signatures.add(signature)
        structures.append(
            BitIntegralStructure(
                structure_id=f"w{active_bit_width}-{index:04d}",
                active_bits=active_bits,
                output_nibble=output_nibble,
                output_mask=output_mask,
                fixed_plaintext=fixed_plaintext,
            )
        )
    return tuple(structures)


def bit_integral_parity_matrix(
    structures: tuple[BitIntegralStructure, ...],
    round_keys: np.ndarray,
    *,
    structure_chunk_size: int = 4,
) -> np.ndarray:
    xor_words = bit_integral_output_xor_matrix(
        structures,
        round_keys,
        structure_chunk_size=structure_chunk_size,
    )
    result = np.zeros(xor_words.shape, dtype=np.uint8)
    parity_lookup = np.asarray(
        [value.bit_count() & 1 for value in range(16)],
        dtype=np.uint8,
    )
    for index, structure in enumerate(structures):
        output_nibbles = (
            xor_words[index] >> np.uint64(4 * structure.output_nibble)
        ) & np.uint64(0xF)
        masked = (output_nibbles & np.uint64(structure.output_mask)).astype(
            np.uint8
        )
        result[index] = parity_lookup[masked]
    return result


def bit_integral_output_xor_matrix(
    structures: tuple[BitIntegralStructure, ...],
    round_keys: np.ndarray,
    *,
    structure_chunk_size: int = 4,
) -> np.ndarray:
    if not structures:
        raise ValueError("structures must be non-empty")
    widths = {len(structure.active_bits) for structure in structures}
    if len(widths) != 1:
        raise ValueError("all structures in one matrix must have the same active width")
    if round_keys.ndim != 2 or round_keys.shape[0] < 2:
        raise ValueError("round_keys must have shape (rounds + 1, key_count)")
    if structure_chunk_size <= 0:
        raise ValueError("structure_chunk_size must be positive")

    rounds = round_keys.shape[0] - 1
    result = np.zeros((len(structures), round_keys.shape[1]), dtype=np.uint64)
    sbox = np.asarray(PRESENT_SBOX, dtype=np.uint64)
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
        result[start:stop] = xor_words.T
    return result


def scalar_bit_integral_parity(
    structure: BitIntegralStructure,
    *,
    rounds: int,
    key: int,
) -> int:
    xor_word = scalar_bit_integral_output_xor(
        structure,
        rounds=rounds,
        key=key,
    )
    output_value = (xor_word >> (4 * structure.output_nibble)) & 0xF
    return (output_value & structure.output_mask).bit_count() & 1


def scalar_bit_integral_output_xor(
    structure: BitIntegralStructure,
    *,
    rounds: int,
    key: int,
) -> int:
    cipher = Present80(rounds=rounds, key=key)
    xor_word = 0
    for plaintext in structure.plaintexts():
        xor_word ^= cipher.encrypt(int(plaintext))
    return xor_word


def run_bit_transition_audit(
    config: BitTransitionAuditConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    keys = make_keys(count=config.keys_per_structure, seed=config.seed + 1401)
    round_keys = present_round_key_matrix(keys, rounds=config.rounds)
    structure_sets = {
        width: make_bit_transition_structures(
            active_bit_width=width,
            count=config.structures_per_width,
            seed=config.seed + width * 100 + 1701,
        )
        for width in ACTIVE_BIT_WIDTHS
    }
    rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    parity_by_width: dict[int, np.ndarray] = {}
    for width in ACTIVE_BIT_WIDTHS:
        _emit(
            progress_callback,
            "parity_matrix_start",
            {
                "active_bit_width": width,
                "set_size": 1 << width,
                "structures": config.structures_per_width,
                "keys": config.keys_per_structure,
            },
        )
        parity = bit_integral_parity_matrix(
            structure_sets[width],
            round_keys,
            structure_chunk_size=config.structure_chunk_size,
        )
        parity_by_width[width] = parity
        summary, rates, marginals = _summarize_width(
            config,
            width=width,
            structures=structure_sets[width],
            parity=parity,
        )
        rows.append(summary)
        structure_rows.extend(rates)
        marginal_rows.extend(marginals)
        _emit(
            progress_callback,
            "parity_matrix_done",
            {
                "active_bit_width": width,
                "q1_rate": summary["q1_rate"],
                "cross_half_combined_marginal_residual_std": summary[
                    "cross_half_combined_marginal_residual_std"
                ],
                "cross_half_combined_marginal_residual_correlation": summary[
                    "cross_half_combined_marginal_residual_correlation"
                ],
            },
        )

    scalar_matches = _crosscheck_scalar(
        structure_sets,
        parity_by_width,
        keys,
        rounds=config.rounds,
    )
    half = config.keys_per_structure // 2
    readiness = {
        "three_width_rows_present": len(rows) == len(ACTIVE_BIT_WIDTHS),
        "same_keys_used_for_all_widths": True,
        "key_halves_nonempty_and_disjoint": (
            half > 0 and set(keys[:half]).isdisjoint(keys[half:])
        ),
        "all_output_positions_covered": all(
            {structure.output_nibble for structure in structures} == set(range(16))
            for structures in structure_sets.values()
        ),
        "all_output_masks_covered": all(
            {structure.output_mask for structure in structures} == set(range(1, 16))
            for structures in structure_sets.values()
        ),
        "all_active_context_bits_cleared": all(
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
                "cross_half_structure_std",
                "cross_half_output_position_residual_std",
                "cross_half_combined_marginal_residual_std",
                "cross_half_combined_marginal_residual_correlation",
                "mixed_structure_fraction",
            )
        ),
    }
    gate = adjudicate_bit_transition_audit(config, rows, readiness)
    return {
        "rows": rows,
        "structure_rows": structure_rows,
        "marginal_rows": marginal_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_integral_output_property_active_bit_transition",
            "cipher": "PRESENT-80",
            "rounds": config.rounds,
            "seed": config.seed,
            "active_bit_widths": list(ACTIVE_BIT_WIDTHS),
            "structures_per_width": config.structures_per_width,
            "keys_per_structure": config.keys_per_structure,
            "key_half_size": half,
            "ridge_alpha": config.ridge_alpha,
            "crossfit_folds": config.crossfit_folds,
            "marginal_input_bits": MARGINAL_INPUT_BITS,
            "training_performed": False,
            "label": "within-structure masked-output XOR parity per key",
            "keys": [f"{key:020X}" for key in keys],
        },
    }


def adjudicate_bit_transition_audit(
    config: BitTransitionAuditConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
) -> dict[str, Any]:
    by_width = {int(row["active_bit_width"]): row for row in rows}
    width_checks: dict[str, dict[str, bool]] = {}
    passing_widths: list[int] = []
    for width in ACTIVE_BIT_WIDTHS:
        row = by_width.get(width, {})
        checks = {
            "q1_rate_not_constant": 0.05 <= _metric(row, "q1_rate") <= 0.95,
            "cross_half_structure_std_at_least_0p03": (
                _metric(row, "cross_half_structure_std") >= 0.03
            ),
            "cross_half_combined_residual_std_at_least_0p02": (
                _metric(row, "cross_half_combined_marginal_residual_std") >= 0.02
            ),
            "cross_half_combined_residual_correlation_at_least_0p20": (
                _metric(
                    row,
                    "cross_half_combined_marginal_residual_correlation",
                )
                >= 0.20
            ),
            "mixed_structure_fraction_at_least_0p10": (
                _metric(row, "mixed_structure_fraction") >= 0.10
            ),
        }
        width_checks[str(width)] = checks
        if all(checks.values()):
            passing_widths.append(width)

    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    selected_width: int | None = None
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_r6_active_bit_transition_audit_invalid"
        next_action = {
            "action": "repair data, key halves, cross-fitting, or scalar parity checks",
            "training": False,
            "remote_scale": False,
        }
    elif passing_widths:
        best_residual = max(
            _metric(by_width[width], "cross_half_combined_marginal_residual_std")
            for width in passing_widths
        )
        selected_width = min(
            width
            for width in passing_widths
            if best_residual
            - _metric(
                by_width[width],
                "cross_half_combined_marginal_residual_std",
            )
            <= 0.002
        )
        status = "pass"
        decision = "innovation2_r6_active_bit_transition_benchmark_ready"
        next_action = {
            "action": "run the frozen local geometry-disjoint output-property matrix",
            "selected_active_bit_width": selected_width,
            "plaintexts_per_structure": 1 << selected_width,
            "models": ["structure_mlp", "linear_same_input", "shuffled_label_mlp"],
            "required_controls": [
                "cross_fitted_combined_marginal_ridge",
                "field_distribution_matched_linear",
            ],
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_r6_active_bit_transition_benchmark_not_ready"
        next_action = {
            "action": "stop PRESENT r6 current structure-description training route",
            "reason": "no 5-to-7 active-bit width has reproducible marginal residual",
            "training": False,
            "seed1": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "width_checks": width_checks,
        "passing_active_bit_widths": passing_widths,
        "selected_active_bit_width": selected_width,
        "claim_scope": (
            "local cross-key-half label and marginal-residual audit only; no neural "
            "training, deterministic integral proof, or remote-scale evidence"
        ),
        "next_action": next_action,
    }


def _summarize_width(
    config: BitTransitionAuditConfig,
    *,
    width: int,
    structures: tuple[BitIntegralStructure, ...],
    parity: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    half = parity.shape[1] // 2
    balance_all = 1.0 - parity.mean(axis=1)
    balance_half0 = 1.0 - parity[:, :half].mean(axis=1)
    balance_half1 = 1.0 - parity[:, half:].mean(axis=1)
    design = np.asarray(
        [structure.marginal_feature_vector() for structure in structures],
        dtype=np.float64,
    )
    fold_ids = _stratified_fold_ids(
        structures,
        fold_count=config.crossfit_folds,
        seed=config.seed + width * 1000 + 1901,
    )
    output_design = design[:, 64:80]
    output_pred0 = _cross_fitted_ridge(
        output_design,
        balance_half0,
        fold_ids,
        alpha=config.ridge_alpha,
    )
    output_pred1 = _cross_fitted_ridge(
        output_design,
        balance_half1,
        fold_ids,
        alpha=config.ridge_alpha,
    )
    combined_pred0 = _cross_fitted_ridge(
        design,
        balance_half0,
        fold_ids,
        alpha=config.ridge_alpha,
    )
    combined_pred1 = _cross_fitted_ridge(
        design,
        balance_half1,
        fold_ids,
        alpha=config.ridge_alpha,
    )
    output_residual0 = balance_half0 - output_pred0
    output_residual1 = balance_half1 - output_pred1
    combined_residual0 = balance_half0 - combined_pred0
    combined_residual1 = balance_half1 - combined_pred1

    structure_rows = [
        {
            "run_id": config.run_id,
            "active_bit_width": width,
            "set_size": structure.set_size,
            "structure_id": structure.structure_id,
            "signature": structure.signature,
            "active_bits": ";".join(str(bit) for bit in structure.active_bits),
            "active_word_mask": f"{structure.active_word_mask:016X}",
            "output_nibble": structure.output_nibble,
            "output_mask": structure.output_mask,
            "fixed_plaintext": f"{structure.fixed_plaintext:016X}",
            "q1_count": int(parity[index].sum()),
            "keys": parity.shape[1],
            "balance_rate": float(balance_all[index]),
            "balance_rate_half0": float(balance_half0[index]),
            "balance_rate_half1": float(balance_half1[index]),
        }
        for index, structure in enumerate(structures)
    ]
    marginal_rows = [
        {
            "run_id": config.run_id,
            "active_bit_width": width,
            "structure_id": structure.structure_id,
            "fold": int(fold_ids[index]),
            "balance_rate_half0": float(balance_half0[index]),
            "balance_rate_half1": float(balance_half1[index]),
            "output_position_prediction_half0": float(output_pred0[index]),
            "output_position_prediction_half1": float(output_pred1[index]),
            "output_position_residual_half0": float(output_residual0[index]),
            "output_position_residual_half1": float(output_residual1[index]),
            "combined_marginal_prediction_half0": float(combined_pred0[index]),
            "combined_marginal_prediction_half1": float(combined_pred1[index]),
            "combined_marginal_residual_half0": float(combined_residual0[index]),
            "combined_marginal_residual_half1": float(combined_residual1[index]),
        }
        for index, structure in enumerate(structures)
    ]
    summary = {
        "run_id": config.run_id,
        "task": "innovation2_integral_output_property_active_bit_transition",
        "active_bit_width": width,
        "set_size": 1 << width,
        "rounds": config.rounds,
        "seed": config.seed,
        "structures": len(structures),
        "keys_per_structure": parity.shape[1],
        "key_half_size": half,
        "q1_rate": float(parity.mean()),
        "mean_balance_rate": float(balance_all.mean()),
        "observed_balance_rate_std": float(balance_all.std()),
        "cross_half_structure_std": _cross_half_std(
            balance_half0,
            balance_half1,
        ),
        "cross_half_structure_correlation": _correlation(
            balance_half0,
            balance_half1,
        ),
        "cross_half_output_position_residual_std": _cross_half_std(
            output_residual0,
            output_residual1,
        ),
        "cross_half_output_position_residual_correlation": _correlation(
            output_residual0,
            output_residual1,
        ),
        "cross_half_combined_marginal_residual_std": _cross_half_std(
            combined_residual0,
            combined_residual1,
        ),
        "cross_half_combined_marginal_residual_correlation": _correlation(
            combined_residual0,
            combined_residual1,
        ),
        "mixed_structure_fraction": float(
            np.mean((balance_all > 0.05) & (balance_all < 0.95))
        ),
        "near_random_structure_fraction": float(
            np.mean((balance_all >= 0.40) & (balance_all <= 0.60))
        ),
        "top_quartile_mean_balance_rate": float(
            np.sort(balance_all)[-max(1, len(balance_all) // 4) :].mean()
        ),
        "ridge_alpha": config.ridge_alpha,
        "crossfit_folds": config.crossfit_folds,
        "training_performed": False,
    }
    return summary, structure_rows, marginal_rows


def _stratified_fold_ids(
    structures: tuple[BitIntegralStructure, ...],
    *,
    fold_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(len(structures), dtype=np.int64)
    for output_nibble in range(16):
        indices = np.asarray(
            [
                index
                for index, structure in enumerate(structures)
                if structure.output_nibble == output_nibble
            ],
            dtype=np.int64,
        )
        rng.shuffle(indices)
        fold_ids[indices] = np.arange(len(indices), dtype=np.int64) % fold_count
    return fold_ids


def _cross_fitted_ridge(
    design: np.ndarray,
    targets: np.ndarray,
    fold_ids: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    predictions = np.empty(len(targets), dtype=np.float64)
    identity = np.eye(design.shape[1], dtype=np.float64)
    for fold in sorted(int(value) for value in np.unique(fold_ids)):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        train_x = design[train_mask]
        train_y = targets[train_mask]
        x_mean = train_x.mean(axis=0)
        y_mean = float(train_y.mean())
        centered_x = train_x - x_mean
        coefficients = np.linalg.solve(
            centered_x.T @ centered_x + alpha * identity,
            centered_x.T @ (train_y - y_mean),
        )
        predictions[test_mask] = (
            design[test_mask] - x_mean
        ) @ coefficients + y_mean
    return np.clip(predictions, 0.0, 1.0)


def _cross_half_std(left: np.ndarray, right: np.ndarray) -> float:
    covariance = float(
        np.mean((left - left.mean()) * (right - right.mean()))
    )
    return math.sqrt(max(0.0, covariance))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std == 0.0 or right_std == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _crosscheck_scalar(
    structure_sets: dict[int, tuple[BitIntegralStructure, ...]],
    parity_by_width: dict[int, np.ndarray],
    keys: tuple[int, ...],
    *,
    rounds: int,
) -> bool:
    key_indices = (0, len(keys) // 2)
    for width in ACTIVE_BIT_WIDTHS:
        structure = structure_sets[width][0]
        for key_index in key_indices:
            expected = scalar_bit_integral_parity(
                structure,
                rounds=rounds,
                key=keys[key_index],
            )
            if expected != int(parity_by_width[width][0, key_index]):
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
