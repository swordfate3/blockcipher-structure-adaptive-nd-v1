from __future__ import annotations

from dataclasses import dataclass
from math import log2

import numpy as np
from numpy.typing import NDArray

from blockcipher_nd.ciphers import ReducedRoundCipher
from blockcipher_nd.features.encoders.present import (
    present_active_nibble_count,
    present_sbox_ddt_beam_statistics_words,
    present_sbox_ddt_score_nibble,
    present_structural_inverse_sbox_difference,
)


@dataclass(frozen=True)
class PresentPairCandidateEvidence:
    top_word: int
    confidence_word: int
    margin_word: int
    disagreement_word: int
    confidence_union_word: int
    margin_union_word: int
    score_word: int
    cumulative_word: int
    active_word: int


def present_pair_candidate_evidence_layers(
    left: int,
    right: int,
    *,
    width: int,
    cipher: ReducedRoundCipher,
    beam_width: int = 4,
    depth: int = 3,
    source: str = "structural_inverse",
) -> list[PresentPairCandidateEvidence]:
    """Return layer-wise S-box-DDT beam evidence for one PRESENT ciphertext pair."""

    if source == "structural_inverse":
        source_difference = present_structural_inverse_sbox_difference(left, right, width, cipher)
    elif source == "xor":
        source_difference = left ^ right
    else:
        raise ValueError(f"unsupported candidate evidence source: {source}")

    words = present_sbox_ddt_beam_statistics_words(
        source_difference,
        width,
        cipher,
        beam_width=beam_width,
        depth=depth,
    )
    if len(words) != depth * 9:
        raise ValueError("unexpected PRESENT beam statistics word count")
    layers = []
    for offset in range(0, len(words), 9):
        layers.append(PresentPairCandidateEvidence(*words[offset : offset + 9]))
    return layers


def present_pair_candidate_evidence_features(
    left: int,
    right: int,
    *,
    width: int,
    cipher: ReducedRoundCipher,
    beam_width: int = 4,
    depth: int = 3,
    source: str = "structural_inverse",
) -> NDArray[np.float32]:
    """Compress one pair's candidate-trail evidence into small numeric features."""

    layers = present_pair_candidate_evidence_layers(
        left,
        right,
        width=width,
        cipher=cipher,
        beam_width=beam_width,
        depth=depth,
        source=source,
    )
    features: list[float] = []
    for layer in layers:
        confidence_values = _nibbles(layer.confidence_word, width)
        margin_values = _nibbles(layer.margin_word, width)
        disagreement_values = _nibbles(layer.disagreement_word, width)
        confidence_union_values = _nibbles(layer.confidence_union_word, width)
        margin_union_values = _nibbles(layer.margin_union_word, width)
        score_values = _nibbles(layer.score_word, 4 * beam_width)
        cumulative_values = _nibbles(layer.cumulative_word, 4 * beam_width)
        active_values = _nibbles(layer.active_word, 4 * beam_width)
        top_active = present_active_nibble_count(layer.top_word, width) / 15.0
        top_score = present_sbox_ddt_score_nibble(sum(confidence_values), width) / 15.0
        features.extend(
            [
                _mean(confidence_values) / 15.0,
                _std(confidence_values) / 15.0,
                _mean(margin_values) / 15.0,
                _std(margin_values) / 15.0,
                _positive_rate(margin_values),
                _positive_rate(disagreement_values),
                _mean(confidence_union_values) / 15.0,
                _mean(margin_union_values) / 15.0,
                _mean(score_values) / 15.0,
                _std(score_values) / 15.0,
                _max(score_values) / 15.0,
                _top2_margin(score_values) / 15.0,
                _entropy01(score_values),
                _mean(cumulative_values) / 15.0,
                _max(cumulative_values) / 15.0,
                _mean(active_values) / 15.0,
                _std(active_values) / 15.0,
                _top2_margin(active_values) / 15.0,
                top_active,
                top_score,
            ]
        )
    return np.array(features, dtype=np.float32)


def present_pairset_candidate_evidence_features(
    pairs: list[tuple[int, int]],
    *,
    width: int,
    cipher: ReducedRoundCipher,
    beam_width: int = 4,
    depth: int = 3,
    source: str = "structural_inverse",
) -> NDArray[np.float32]:
    """Return pair-level means/stds plus pair-consistency statistics for one sample."""

    if not pairs:
        raise ValueError("pairs must not be empty")
    pair_features = np.stack(
        [
            present_pair_candidate_evidence_features(
                left,
                right,
                width=width,
                cipher=cipher,
                beam_width=beam_width,
                depth=depth,
                source=source,
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


def _nibbles(word: int, width: int) -> list[int]:
    return [int((word >> (4 * index)) & 0xF) for index in range(width // 4)]


def _mean(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[int]) -> float:
    return float(np.std(values)) if values else 0.0


def _max(values: list[int]) -> float:
    return float(max(values)) if values else 0.0


def _positive_rate(values: list[int]) -> float:
    return float(sum(1 for value in values if value > 0) / len(values)) if values else 0.0


def _top2_margin(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0
    return float(max(0, top - second))


def _entropy01(values: list[int]) -> float:
    total = float(sum(max(0, value) for value in values))
    if total <= 0.0:
        return 0.0
    probabilities = [value / total for value in values if value > 0]
    entropy = -sum(probability * log2(probability) for probability in probabilities)
    return float(entropy / log2(len(values))) if len(values) > 1 else 0.0


__all__ = [
    "PresentPairCandidateEvidence",
    "present_pair_candidate_evidence_features",
    "present_pair_candidate_evidence_layers",
    "present_pairset_candidate_evidence_features",
]
