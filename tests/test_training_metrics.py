from __future__ import annotations

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.training.metrics import (
    best_threshold_accuracy_and_threshold,
    evaluate_binary_classifier,
)
from blockcipher_nd.training.trainer import train_binary_classifier
from blockcipher_nd.training.types import TrainingConfig


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


class CountingLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.forward_calls = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.linear(features)


def test_evaluate_binary_classifier_uses_one_forward_pass_per_batch() -> None:
    dataset = DifferentialDataset(
        features=np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=np.uint8,
        ),
        labels=np.array([0, 0, 1, 1], dtype=np.uint8),
        metadata={"feature_encoding": "test"},
    )
    model = CountingLinear()

    metrics = evaluate_binary_classifier(model, dataset, batch_size=2, device="cpu")

    assert model.forward_calls == 2
    assert set(metrics) >= {"loss", "accuracy", "auc", "calibrated_accuracy"}


def test_train_binary_classifier_writes_selected_checkpoint(tmp_path) -> None:
    dataset = DifferentialDataset(
        features=np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=np.uint8,
        ),
        labels=np.array([0, 0, 1, 1], dtype=np.uint8),
        metadata={"feature_encoding": "test"},
    )
    checkpoint_path = tmp_path / "selected.pt"
    model = torch.nn.Linear(2, 1)

    result = train_binary_classifier(
        model,
        dataset,
        dataset,
        TrainingConfig(
            epochs=1,
            batch_size=2,
            device="cpu",
            checkpoint_output=checkpoint_path,
        ),
    )

    assert result.metadata["checkpoint_output"] == str(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(payload) == {"state_dict", "history", "final_metrics", "metadata"}
    assert payload["metadata"]["checkpoint_output"] == str(checkpoint_path)
    assert set(payload["state_dict"]) == set(model.state_dict())
