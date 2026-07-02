from __future__ import annotations

import numpy as np
import torch

from blockcipher_nd.training.metrics import best_threshold_accuracy_and_threshold


def build_baseline(input_dim: int, model: str) -> torch.nn.Module:
    if model == "linear":
        return torch.nn.Linear(input_dim, 1)
    return torch.nn.Sequential(
        torch.nn.LayerNorm(input_dim),
        torch.nn.Linear(input_dim, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 1),
    )


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    model_name: str,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> torch.nn.Module:
    torch.manual_seed(0)
    x = torch.from_numpy(features.astype(np.float32)).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1).to(device)
    model = build_baseline(features.shape[1], model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return model


def binary_accuracy(labels: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5) -> float:
    predictions = (probabilities >= threshold).astype(np.uint8)
    return float((predictions == labels.astype(np.uint8)).mean())


def calibrated_binary_accuracy(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    return best_threshold_accuracy_and_threshold(labels.astype(np.float32), probabilities.astype(np.float32))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    label_array = labels.astype(np.uint8)
    positive_count = int(label_array.sum())
    negative_count = int(label_array.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = float(original_ranks[label_array == 1].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )
