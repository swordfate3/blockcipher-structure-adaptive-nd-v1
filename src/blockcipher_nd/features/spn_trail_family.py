from __future__ import annotations

from dataclasses import dataclass
from math import log2

import numpy as np
from numpy.typing import NDArray

from blockcipher_nd.ciphers import ReducedRoundCipher
from blockcipher_nd.features.spn_candidate_evidence import (
    PresentPairCandidateEvidence,
    present_pair_candidate_evidence_layers,
)


DEFAULT_TRAIL_FAMILY_SEED = 20260702


@dataclass(frozen=True)
class PresentTrailFamilyTemplate:
    """Compact deterministic trail-family view for one PRESENT ciphertext pair."""

    active_masks: NDArray[np.float32]
    confidence: NDArray[np.float32]
    margin: NDArray[np.float32]
    disagreement: NDArray[np.float32]
    score: NDArray[np.float32]
    cumulative_score: NDArray[np.float32]
    active_count: NDArray[np.float32]


def present_pair_trail_family_template(
    left: int,
    right: int,
    *,
    width: int,
    cipher: ReducedRoundCipher,
    beam_width: int = 4,
    depth: int = 3,
    source: str = "structural_inverse",
) -> PresentTrailFamilyTemplate:
    """Return a label-free candidate trail-family template for one pair."""

    if width % 4 != 0:
        raise ValueError("PRESENT trail-family features require nibble-aligned width")
    layers = present_pair_candidate_evidence_layers(
        left,
        right,
        width=width,
        cipher=cipher,
        beam_width=beam_width,
        depth=depth,
        source=source,
    )
    cells = width // 4
    return PresentTrailFamilyTemplate(
        active_masks=np.stack([_active_mask(layer.top_word, cells) for layer in layers], axis=0),
        confidence=np.stack([_normalized_nibbles(layer.confidence_word, cells) for layer in layers], axis=0),
        margin=np.stack([_normalized_nibbles(layer.margin_word, cells) for layer in layers], axis=0),
        disagreement=np.stack([_active_mask(layer.disagreement_word, cells) for layer in layers], axis=0),
        score=np.stack([_beam_values(layer.score_word, beam_width) for layer in layers], axis=0),
        cumulative_score=np.stack([_beam_values(layer.cumulative_word, beam_width) for layer in layers], axis=0),
        active_count=np.stack([_beam_values(layer.active_word, beam_width) for layer in layers], axis=0),
    )


def present_pair_trail_family_features(
    left: int,
    right: int,
    *,
    width: int,
    cipher: ReducedRoundCipher,
    beam_width: int = 4,
    depth: int = 3,
    source: str = "structural_inverse",
) -> NDArray[np.float32]:
    """Return per-pair trail-family features for smoke diagnostics."""

    template = present_pair_trail_family_template(
        left,
        right,
        width=width,
        cipher=cipher,
        beam_width=beam_width,
        depth=depth,
        source=source,
    )
    features: list[float] = []
    for layer_index in range(depth):
        active = template.active_masks[layer_index]
        confidence = template.confidence[layer_index]
        margin = template.margin[layer_index]
        disagreement = template.disagreement[layer_index]
        score = template.score[layer_index]
        cumulative = template.cumulative_score[layer_index]
        active_count = template.active_count[layer_index]
        features.extend(
            [
                float(active.mean()),
                float(active.std()),
                float(confidence.mean()),
                float(confidence.std()),
                float(margin.mean()),
                float(margin.std()),
                float(disagreement.mean()),
                _entropy01(score),
                float(score.max()) if score.size else 0.0,
                _top2_margin(score),
                float(cumulative.max()) if cumulative.size else 0.0,
                float(active_count.mean()) if active_count.size else 0.0,
            ]
        )
        features.extend(active.astype(np.float32).tolist())
    return np.array(features, dtype=np.float32)


def present_pairset_trail_family_features(
    pairs: list[tuple[int, int]],
    *,
    width: int,
    cipher: ReducedRoundCipher,
    beam_width: int = 4,
    depth: int = 3,
    source: str = "structural_inverse",
    false_family: bool = False,
    false_family_seed: int = DEFAULT_TRAIL_FAMILY_SEED,
) -> NDArray[np.float32]:
    """Return pair-set trail-family agreement features for one 16-pair sample."""

    if not pairs:
        raise ValueError("pairs must not be empty")
    templates = [
        present_pair_trail_family_template(
            int(left),
            int(right),
            width=width,
            cipher=cipher,
            beam_width=beam_width,
            depth=depth,
            source=source,
        )
        for left, right in pairs
    ]
    pair_features = np.stack(
        [
            present_pair_trail_family_features(
                int(left),
                int(right),
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
    masks = np.stack([template.active_masks for template in templates], axis=0).astype(np.float32)
    if false_family:
        masks = _false_family_masks(masks, seed=false_family_seed)
    confidence = np.stack([template.confidence for template in templates], axis=0).astype(np.float32)
    margin = np.stack([template.margin for template in templates], axis=0).astype(np.float32)
    score = np.stack([template.score for template in templates], axis=0).astype(np.float32)

    family_features: list[float] = []
    for layer_index in range(depth):
        layer_masks = masks[:, layer_index, :]
        active_rates = layer_masks.mean(axis=0)
        consensus = (active_rates >= 0.5).astype(np.float32)
        agreements = np.array([_mask_jaccard(mask, consensus) for mask in layer_masks], dtype=np.float32)
        family_features.extend(
            [
                float(active_rates.mean()),
                float(active_rates.std()),
                float(active_rates.max()),
                float(np.mean([_binary_entropy01(rate) for rate in active_rates])),
                float(consensus.mean()),
                float(agreements.mean()),
                float(agreements.std()),
                float(agreements.min()),
                float(agreements.max()),
                _top2_margin(active_rates),
                float(confidence[:, layer_index, :].mean()),
                float(margin[:, layer_index, :].mean()),
                float(score[:, layer_index, :].mean()),
            ]
        )
        family_features.extend(active_rates.astype(np.float32).tolist())

    pair_means = pair_features.mean(axis=0)
    pair_stds = pair_features.std(axis=0)
    pair_spans = pair_features.max(axis=0) - pair_features.min(axis=0)
    global_stats = np.array(
        [
            float(pair_features.mean()),
            float(pair_features.std()),
            float(pair_stds.mean()),
            float(pair_stds.max()),
            float(pair_spans.mean()),
            float(pair_spans.max()),
            float(masks.mean()),
            float(masks.std()),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            np.array(family_features, dtype=np.float32),
            pair_means,
            pair_stds,
            pair_spans,
            global_stats,
        ]
    ).astype(np.float32)


def _active_mask(word: int, cells: int) -> NDArray[np.float32]:
    return np.array([1.0 if ((word >> (4 * index)) & 0xF) else 0.0 for index in range(cells)], dtype=np.float32)


def _normalized_nibbles(word: int, cells: int) -> NDArray[np.float32]:
    return np.array([((word >> (4 * index)) & 0xF) / 15.0 for index in range(cells)], dtype=np.float32)


def _beam_values(word: int, beam_width: int) -> NDArray[np.float32]:
    return np.array([((word >> (4 * index)) & 0xF) / 15.0 for index in range(beam_width)], dtype=np.float32)


def _false_family_masks(masks: NDArray[np.float32], *, seed: int) -> NDArray[np.float32]:
    shifted = masks.copy()
    cells = int(masks.shape[-1])
    for pair_index in range(int(masks.shape[0])):
        for layer_index in range(int(masks.shape[1])):
            offset = ((pair_index + 1) * 5 + (layer_index + 1) * 3 + seed) % cells
            if offset == 0:
                offset = 1
            shifted[pair_index, layer_index, :] = np.roll(masks[pair_index, layer_index, :], offset)
    return shifted


def _mask_jaccard(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    left_bool = left > 0.5
    right_bool = right > 0.5
    union = int(np.logical_or(left_bool, right_bool).sum())
    if union == 0:
        return 1.0
    intersection = int(np.logical_and(left_bool, right_bool).sum())
    return float(intersection / union)


def _binary_entropy01(probability: float) -> float:
    p = min(1.0, max(0.0, float(probability)))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * log2(p) + (1.0 - p) * log2(1.0 - p)))


def _entropy01(values: NDArray[np.float32]) -> float:
    total = float(np.sum(np.maximum(values, 0.0)))
    if total <= 0.0:
        return 0.0
    probabilities = [float(value / total) for value in values if value > 0]
    entropy = -sum(probability * log2(probability) for probability in probabilities)
    return float(entropy / log2(len(values))) if len(values) > 1 else 0.0


def _top2_margin(values: NDArray[np.float32]) -> float:
    if values.size == 0:
        return 0.0
    ordered = np.sort(values.astype(np.float32))[::-1]
    top = float(ordered[0])
    second = float(ordered[1]) if ordered.size > 1 else 0.0
    return max(0.0, top - second)


__all__ = [
    "DEFAULT_TRAIL_FAMILY_SEED",
    "PresentTrailFamilyTemplate",
    "present_pair_trail_family_features",
    "present_pair_trail_family_template",
    "present_pairset_trail_family_features",
]
