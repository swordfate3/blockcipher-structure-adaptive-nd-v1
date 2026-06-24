from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class ActivePatternSummary(TypedDict):
    active_masks: NDArray[np.uint8]
    active_count: NDArray[np.uint8]
    position_frequency: NDArray[np.float32]
    density_mean: NDArray[np.float32]
    density_std: NDArray[np.float32]
    density_span: NDArray[np.float32]


def active_mask16_from_word(word: int) -> NDArray[np.uint8]:
    mask = np.zeros(16, dtype=np.uint8)
    for nibble_index in range(16):
        mask[nibble_index] = 1 if ((int(word) >> (4 * nibble_index)) & 0xF) != 0 else 0
    return mask


def active_masks16_from_words(words: NDArray[np.uint64]) -> NDArray[np.uint8]:
    word_array = np.asarray(words, dtype=np.uint64)
    flat = word_array.reshape(-1)
    masks = np.stack([active_mask16_from_word(int(word)) for word in flat], axis=0)
    return masks.reshape(*word_array.shape, 16)


def active_pattern_summary_from_words(words: NDArray[np.uint64]) -> ActivePatternSummary:
    word_array = np.asarray(words, dtype=np.uint64)
    if word_array.ndim != 2:
        raise ValueError("words must have shape (rows, words_per_row)")
    active_masks = active_masks16_from_words(word_array)
    active_count = active_masks.sum(axis=-1).astype(np.uint8)
    density = active_count.astype(np.float32) / 16.0
    return {
        "active_masks": active_masks,
        "active_count": active_count,
        "position_frequency": active_masks.mean(axis=1, dtype=np.float32),
        "density_mean": density.mean(axis=1, dtype=np.float32),
        "density_std": density.std(axis=1, dtype=np.float32),
        "density_span": (density.max(axis=1) - density.min(axis=1)).astype(np.float32),
    }


def uint64_words_from_bit_rows(bit_rows: NDArray[np.uint8], *, words_per_row: int) -> NDArray[np.uint64]:
    rows = np.asarray(bit_rows, dtype=np.uint8)
    if rows.ndim != 2:
        raise ValueError("bit_rows must have shape (rows, bits)")
    expected_bits = words_per_row * 64
    if rows.shape[1] != expected_bits:
        raise ValueError(f"expected {expected_bits} bits, got {rows.shape[1]}")
    reshaped = rows.reshape(rows.shape[0], words_per_row, 64)
    powers = (1 << np.arange(63, -1, -1, dtype=np.uint64)).reshape(1, 1, 64)
    return (reshaped.astype(np.uint64) * powers).sum(axis=2, dtype=np.uint64)


def extract_active_pattern_features(bit_rows: NDArray[np.uint8], *, words_per_row: int) -> NDArray[np.float32]:
    words = uint64_words_from_bit_rows(bit_rows, words_per_row=words_per_row)
    summary = active_pattern_summary_from_words(words)
    active_count = summary["active_count"].astype(np.float32)
    count_mean = active_count.mean(axis=1, keepdims=True) / 16.0
    count_std = active_count.std(axis=1, keepdims=True) / 16.0
    count_min = active_count.min(axis=1, keepdims=True) / 16.0
    count_max = active_count.max(axis=1, keepdims=True) / 16.0
    density_stats = np.stack(
        [
            summary["density_mean"],
            summary["density_std"],
            summary["density_span"],
            summary["position_frequency"].std(axis=1, dtype=np.float32),
        ],
        axis=1,
    )
    return np.concatenate(
        [
            summary["position_frequency"].astype(np.float32),
            np.concatenate([count_mean, count_std, count_min, count_max], axis=1).astype(np.float32),
            density_stats.astype(np.float32),
        ],
        axis=1,
    )


def active_label_diagnostics(labels: NDArray[np.uint8]) -> dict[str, float | int]:
    label_array = np.asarray(labels, dtype=np.uint8)
    if label_array.ndim != 2:
        raise ValueError("labels must have shape (rows, positions)")
    total = int(label_array.size)
    positives = int(label_array.sum())
    negatives = total - positives
    return {
        "rows": int(label_array.shape[0]),
        "positions": int(label_array.shape[1]),
        "active_positive_rate": positives / total if total else 0.0,
        "inactive_negative_rate": negatives / total if total else 0.0,
        "all_inactive_accuracy": negatives / total if total else 0.0,
        "mean_active_per_row": float(label_array.sum(axis=1).mean()) if label_array.shape[0] else 0.0,
    }
