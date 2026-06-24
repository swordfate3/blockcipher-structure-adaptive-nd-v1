from __future__ import annotations

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.training.data import make_loader, select_device


def predict_binary_probabilities(
    model: nn.Module,
    dataset: DifferentialDataset,
    batch_size: int = 256,
    device: str = "auto",
) -> np.ndarray:
    selected_device = select_device(device)
    model = model.to(selected_device)
    model.eval()
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)

    probabilities: list[float] = []
    with torch.no_grad():
        for features, _batch_labels in loader:
            features = features.to(selected_device)
            logits = model(features).squeeze(1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probabilities.extend(float(item) for item in probs)
    return np.array(probabilities, dtype=np.float32)


def evaluate_binary_classifier(
    model: nn.Module,
    dataset: DifferentialDataset,
    batch_size: int = 256,
    device: str = "auto",
) -> dict[str, float]:
    selected_device = select_device(device)
    model = model.to(selected_device)
    model.eval()
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    labels: list[float] = []
    with torch.no_grad():
        for features, batch_labels in loader:
            features = features.to(selected_device)
            batch_labels = batch_labels.to(selected_device)
            logits = model(features).squeeze(1)
            total_loss += float(loss_fn(logits, batch_labels).cpu())
            labels.extend(float(item) for item in batch_labels.cpu().numpy())

    label_array = np.array(labels, dtype=np.float32)
    prob_array = predict_binary_probabilities(
        model,
        dataset,
        batch_size=batch_size,
        device=str(selected_device),
    )
    predictions = (prob_array >= 0.5).astype(np.float32)
    accuracy = float((predictions == label_array).mean()) if len(label_array) else 0.0
    calibrated_accuracy, calibrated_threshold = best_threshold_accuracy_and_threshold(
        label_array,
        prob_array,
    )
    return {
        "loss": total_loss / max(1, len(label_array)),
        "accuracy": accuracy,
        "advantage": 2.0 * accuracy - 1.0,
        "auc": binary_auc(label_array, prob_array),
        "best_accuracy": calibrated_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_advantage": 2.0 * calibrated_accuracy - 1.0,
        "calibrated_threshold": calibrated_threshold,
    }


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_mask = labels == 1
    positive_count = int(positive_mask.sum())
    negative_count = int((labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[positive_mask].sum())
    u_statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(u_statistic / (positive_count * negative_count))


def best_threshold_accuracy_and_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    if len(labels) == 0:
        return 0.0, 0.5
    thresholds = np.unique(scores)
    best = 0.0
    best_threshold = 0.5
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(np.float32)
        accuracy = float((predictions == labels).mean())
        if accuracy > best:
            best = accuracy
            best_threshold = float(threshold)
    return best, best_threshold
