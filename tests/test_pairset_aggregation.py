from __future__ import annotations

import numpy as np
import pytest
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.evaluation.pairset_aggregation import (
    PairSetAggregationConfig,
    aggregate_pair_logits,
    pairset_aggregation_metrics,
    pairset_aggregation_scores,
    split_pairset_features,
)


class FirstBitScorer(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, :1].float() * 2.0 - 1.0


def test_split_pairset_features_returns_pair_axis_view() -> None:
    features = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    pair_features = split_pairset_features(
        features,
        pair_bits=3,
        pairs_per_sample=4,
    )

    assert pair_features.shape == (2, 4, 3)
    assert torch.equal(pair_features[0, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(pair_features[1, 3], torch.tensor([21.0, 22.0, 23.0]))


def test_split_pairset_features_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="expected 12 bits"):
        split_pairset_features(
            torch.zeros(2, 11),
            pair_bits=3,
            pairs_per_sample=4,
        )


def test_aggregate_pair_logits_modes() -> None:
    logits = torch.tensor(
        [
            [1.0, -1.0, 2.0, 0.0],
            [-2.0, -1.0, 3.0, 4.0],
        ]
    )

    assert torch.allclose(
        aggregate_pair_logits(logits, mode="mean_logit"),
        torch.tensor([0.5, 1.0]),
    )
    assert torch.allclose(
        aggregate_pair_logits(logits, mode="sum_logodds"),
        torch.tensor([2.0, 4.0]),
    )
    assert torch.allclose(
        aggregate_pair_logits(logits, mode="topk_mean_logit", top_k=2),
        torch.tensor([1.5, 3.5]),
    )
    expected = torch.logsumexp(torch.tensor([[2.0, 1.0], [4.0, 3.0]]), dim=1)
    assert torch.allclose(
        aggregate_pair_logits(logits, mode="topk_logsumexp", top_k=2),
        expected,
    )


def test_pairset_aggregation_scores_use_frozen_scorer_per_pair() -> None:
    dataset = DifferentialDataset(
        features=np.array(
            [
                [1, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        labels=np.array([1, 0, 1], dtype=np.uint8),
        metadata={"feature_encoding": "toy_pairset"},
    )

    labels, scores = pairset_aggregation_scores(
        FirstBitScorer(),
        dataset,
        PairSetAggregationConfig(pair_bits=2, pairs_per_sample=3, mode="sum_logodds"),
        batch_size=2,
        device="cpu",
    )

    assert labels.tolist() == [1.0, 0.0, 1.0]
    assert scores.tolist() == [1.0, -3.0, 1.0]


def test_pairset_aggregation_metrics_reports_binary_metrics() -> None:
    dataset = DifferentialDataset(
        features=np.array(
            [
                [1, 0, 1, 1],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.uint8,
        ),
        labels=np.array([1, 0, 1, 0], dtype=np.uint8),
        metadata={"feature_encoding": "toy_pairset"},
    )

    metrics = pairset_aggregation_metrics(
        FirstBitScorer(),
        dataset,
        PairSetAggregationConfig(pair_bits=2, pairs_per_sample=2, mode="sum_logodds"),
        batch_size=2,
        device="cpu",
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["calibrated_accuracy"] == 1.0
    assert metrics["aggregation_mode"] == "sum_logodds"
