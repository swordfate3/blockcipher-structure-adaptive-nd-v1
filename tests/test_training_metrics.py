from __future__ import annotations

import numpy as np

from blockcipher_nd.training.metrics import best_threshold_accuracy_and_threshold


def brute_force_best_threshold_accuracy(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    best = 0.0
    best_threshold = 0.5
    for threshold in np.unique(scores):
        predictions = (scores >= threshold).astype(np.float32)
        accuracy = float((predictions == labels).mean())
        if accuracy > best:
            best = accuracy
            best_threshold = float(threshold)
    return best, best_threshold


def test_best_threshold_matches_bruteforce_with_ties() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.float32)
    scores = np.array([0.1, 0.4, 0.4, 0.3, 0.9, 0.3, 0.8, 0.9], dtype=np.float32)

    assert best_threshold_accuracy_and_threshold(labels, scores) == brute_force_best_threshold_accuracy(
        labels,
        scores,
    )


def test_best_threshold_matches_bruteforce_for_many_unique_scores() -> None:
    rng = np.random.default_rng(20260625)
    labels = rng.integers(0, 2, size=4096).astype(np.float32)
    scores = rng.random(size=4096, dtype=np.float32)

    assert best_threshold_accuracy_and_threshold(labels, scores) == brute_force_best_threshold_accuracy(
        labels,
        scores,
    )
