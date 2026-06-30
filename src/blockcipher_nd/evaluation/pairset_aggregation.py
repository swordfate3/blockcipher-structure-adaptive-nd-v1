from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.training.data import make_loader, select_device
from blockcipher_nd.training.metrics import (
    best_threshold_accuracy_and_threshold,
    binary_auc,
)


@dataclass(frozen=True)
class PairSetAggregationConfig:
    pair_bits: int = 128
    pairs_per_sample: int = 16
    mode: str = "sum_logodds"
    top_k: int = 4
    lse_temperature: float = 1.0


def pairset_aggregation_metrics(
    scorer: nn.Module,
    dataset: DifferentialDataset,
    config: PairSetAggregationConfig | None = None,
    *,
    batch_size: int = 256,
    device: str = "auto",
) -> dict[str, float | str]:
    labels, scores = pairset_aggregation_scores(
        scorer,
        dataset,
        config or PairSetAggregationConfig(),
        batch_size=batch_size,
        device=device,
    )
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -80.0, 80.0)))
    predictions = (probabilities >= 0.5).astype(np.float32)
    accuracy = float((predictions == labels).mean()) if len(labels) else 0.0
    calibrated_accuracy, calibrated_threshold = best_threshold_accuracy_and_threshold(
        labels,
        probabilities,
    )
    return {
        "accuracy": accuracy,
        "advantage": 2.0 * accuracy - 1.0,
        "auc": binary_auc(labels, probabilities),
        "best_accuracy": calibrated_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_advantage": 2.0 * calibrated_accuracy - 1.0,
        "calibrated_threshold": calibrated_threshold,
        "aggregation_mode": config.mode if config else PairSetAggregationConfig().mode,
    }


def pairset_aggregation_scores(
    scorer: nn.Module,
    dataset: DifferentialDataset,
    config: PairSetAggregationConfig,
    *,
    batch_size: int = 256,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    _validate_pairset_config(config)
    selected_device = select_device(device)
    scorer = scorer.to(selected_device)
    scorer.eval()
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)

    labels: list[float] = []
    scores: list[float] = []
    with torch.no_grad():
        for features, batch_labels in loader:
            features = features.to(selected_device)
            pair_features = split_pairset_features(
                features,
                pair_bits=config.pair_bits,
                pairs_per_sample=config.pairs_per_sample,
            )
            flat_pair_features = pair_features.reshape(-1, config.pair_bits)
            pair_logits = scorer(flat_pair_features).reshape(
                features.shape[0],
                config.pairs_per_sample,
            )
            sample_scores = aggregate_pair_logits(
                pair_logits,
                mode=config.mode,
                top_k=config.top_k,
                lse_temperature=config.lse_temperature,
            )
            labels.extend(float(item) for item in batch_labels.cpu().numpy())
            scores.extend(float(item) for item in sample_scores.detach().cpu().numpy())

    return (
        np.array(labels, dtype=np.float32),
        np.array(scores, dtype=np.float32),
    )


def split_pairset_features(
    features: torch.Tensor,
    *,
    pair_bits: int,
    pairs_per_sample: int,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"expected [batch, bits], got {tuple(features.shape)}")
    expected_bits = pair_bits * pairs_per_sample
    if features.shape[1] != expected_bits:
        raise ValueError(f"expected {expected_bits} bits, got {features.shape[1]}")
    return features.reshape(features.shape[0], pairs_per_sample, pair_bits)


def aggregate_pair_logits(
    pair_logits: torch.Tensor,
    *,
    mode: str = "sum_logodds",
    top_k: int = 4,
    lse_temperature: float = 1.0,
) -> torch.Tensor:
    if pair_logits.ndim != 2:
        raise ValueError(f"expected [batch, pairs], got {tuple(pair_logits.shape)}")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if lse_temperature <= 0.0:
        raise ValueError("lse_temperature must be > 0")

    if mode == "mean_logit":
        return pair_logits.mean(dim=1)
    if mode == "sum_logodds":
        return pair_logits.sum(dim=1)
    if mode == "topk_mean_logit":
        k = min(top_k, pair_logits.shape[1])
        return torch.topk(pair_logits, k=k, dim=1).values.mean(dim=1)
    if mode == "topk_logsumexp":
        k = min(top_k, pair_logits.shape[1])
        top_values = torch.topk(pair_logits, k=k, dim=1).values
        return torch.logsumexp(top_values / lse_temperature, dim=1) * lse_temperature
    raise ValueError(f"unsupported aggregation mode: {mode}")


def _validate_pairset_config(config: PairSetAggregationConfig) -> None:
    if config.pair_bits < 1:
        raise ValueError("pair_bits must be >= 1")
    if config.pairs_per_sample < 1:
        raise ValueError("pairs_per_sample must be >= 1")
    if config.top_k < 1:
        raise ValueError("top_k must be >= 1")
    if config.lse_temperature <= 0.0:
        raise ValueError("lse_temperature must be > 0")
