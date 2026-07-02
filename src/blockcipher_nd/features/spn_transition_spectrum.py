from __future__ import annotations

from math import log2

import numpy as np
from numpy.typing import NDArray

from blockcipher_nd.ciphers import ReducedRoundCipher
from blockcipher_nd.features.spn_aligned import inverse_permutation_difference


def present_bit_transition_spectrum_features(
    pairs: list[tuple[int, int]],
    *,
    width: int,
    cipher: ReducedRoundCipher,
    shuffled: bool = False,
    shuffle_seed: int = 20260702,
) -> NDArray[np.float32]:
    """Return pairset-level bit/cell transition statistics for PRESENT-like SPNs."""

    if not pairs:
        raise ValueError("pairs must not be empty")
    pair_features = np.stack(
        [
            present_pair_bit_transition_spectrum_features(
                int(left),
                int(right),
                width=width,
                cipher=cipher,
                shuffled=shuffled,
                shuffle_seed=shuffle_seed,
            )
            for left, right in pairs
        ],
        axis=0,
    ).astype(np.float32)
    means = pair_features.mean(axis=0)
    stds = pair_features.std(axis=0)
    spans = pair_features.max(axis=0) - pair_features.min(axis=0)
    global_stats = np.array(
        [
            float(pair_features.mean()),
            float(pair_features.std()),
            float(np.mean(stds)),
            float(np.max(stds)),
            float(np.mean(spans)),
            float(np.max(spans)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([means, stds, spans, global_stats]).astype(np.float32)


def present_pair_bit_transition_spectrum_features(
    left: int,
    right: int,
    *,
    width: int,
    cipher: ReducedRoundCipher,
    shuffled: bool = False,
    shuffle_seed: int = 20260702,
) -> NDArray[np.float32]:
    if width % 4 != 0:
        raise ValueError("PRESENT transition spectrum requires nibble-aligned width")
    cells = width // 4
    difference = (left ^ right) & ((1 << width) - 1)
    aligned_difference = _aligned_difference(
        difference,
        width=width,
        cipher=cipher,
        shuffled=shuffled,
        shuffle_seed=shuffle_seed,
    )
    source_counts = _cell_active_bit_counts(aligned_difference, width)
    target_counts = _cell_active_bit_counts(difference, width)
    transition_matrix = _transition_matrix(aligned_difference, width=width, shuffled=shuffled, shuffle_seed=shuffle_seed)
    row_sums = transition_matrix.sum(axis=1)
    col_sums = transition_matrix.sum(axis=0)
    overlap = _bit_overlap(difference, aligned_difference, width)
    total_active = float(max(1, int(source_counts.sum())))
    scale = float(max(1, 4 * cells))
    features = [
        float(source_counts.sum() / scale),
        float(target_counts.sum() / scale),
        float(np.count_nonzero(source_counts) / cells),
        float(np.count_nonzero(target_counts) / cells),
        float(source_counts.mean() / 4.0),
        float(target_counts.mean() / 4.0),
        float(source_counts.std() / 4.0),
        float(target_counts.std() / 4.0),
        float(row_sums.max() / 4.0),
        float(col_sums.max() / 4.0),
        _entropy01(row_sums),
        _entropy01(col_sums),
        float(np.count_nonzero(transition_matrix) / (cells * cells)),
        float(transition_matrix.max() / 4.0),
        float(overlap / max(1, width)),
        float(overlap / total_active),
        float(np.abs(source_counts - target_counts).mean() / 4.0),
        float(np.abs(row_sums - col_sums).mean() / 4.0),
    ]
    features.extend((source_counts / 4.0).astype(np.float32).tolist())
    features.extend((target_counts / 4.0).astype(np.float32).tolist())
    features.extend((row_sums / 4.0).astype(np.float32).tolist())
    features.extend((col_sums / 4.0).astype(np.float32).tolist())
    features.extend((transition_matrix.reshape(-1) / 4.0).astype(np.float32).tolist())
    return np.array(features, dtype=np.float32)


def _aligned_difference(
    difference: int,
    *,
    width: int,
    cipher: ReducedRoundCipher,
    shuffled: bool,
    shuffle_seed: int,
) -> int:
    if not shuffled:
        return inverse_permutation_difference(difference, width, cipher)
    permutation = _shuffled_bit_permutation(width, shuffle_seed)
    return _apply_inverse_mapping(difference, permutation)


def _transition_matrix(difference: int, *, width: int, shuffled: bool, shuffle_seed: int) -> NDArray[np.float32]:
    cells = width // 4
    matrix = np.zeros((cells, cells), dtype=np.float32)
    permutation = _shuffled_bit_permutation(width, shuffle_seed) if shuffled else _present_p_layer_permutation(width)
    for source_bit in range(width):
        if ((difference >> source_bit) & 1) == 0:
            continue
        target_bit = permutation[source_bit]
        matrix[source_bit // 4, target_bit // 4] += 1.0
    return matrix


def _cell_active_bit_counts(word: int, width: int) -> NDArray[np.float32]:
    return np.array(
        [int(((word >> (4 * cell)) & 0xF).bit_count()) for cell in range(width // 4)],
        dtype=np.float32,
    )


def _present_p_layer_permutation(width: int) -> list[int]:
    if width != 64:
        raise ValueError("PRESENT p-layer transition spectrum currently supports width=64")
    return [((16 * bit_index) % 63) if bit_index < 63 else 63 for bit_index in range(width)]


def _shuffled_bit_permutation(width: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.permutation(width).astype(int).tolist()


def _apply_inverse_mapping(word: int, permutation: list[int]) -> int:
    out = 0
    for source_bit, target_bit in enumerate(permutation):
        bit = (word >> target_bit) & 1
        out |= bit << source_bit
    return out


def _bit_overlap(left: int, right: int, width: int) -> int:
    return int(((left & right) & ((1 << width) - 1)).bit_count())


def _entropy01(values: NDArray[np.float32]) -> float:
    total = float(np.sum(np.maximum(values, 0.0)))
    if total <= 0.0:
        return 0.0
    probabilities = [float(value / total) for value in values if value > 0]
    entropy = -sum(probability * log2(probability) for probability in probabilities)
    return float(entropy / log2(len(values))) if len(values) > 1 else 0.0


__all__ = [
    "present_bit_transition_spectrum_features",
    "present_pair_bit_transition_spectrum_features",
]
