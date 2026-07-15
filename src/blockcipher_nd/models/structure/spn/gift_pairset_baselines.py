from __future__ import annotations

import torch
from torch import nn


def _pair_sequences(
    features: torch.Tensor,
    *,
    input_bits: int,
    pair_bits: int,
) -> tuple[torch.Tensor, int, int]:
    if features.ndim != 2 or features.shape[1] != input_bits:
        raise ValueError(f"expected {input_bits} input bits, got {tuple(features.shape)}")
    batch = features.shape[0]
    pairs_per_sample = input_bits // pair_bits
    sequences = (
        features.float()
        .reshape(batch, pairs_per_sample, 2, pair_bits // 2)
        .permute(0, 1, 3, 2)
        .reshape(batch * pairs_per_sample, pair_bits // 2, 2)
    )
    return sequences, batch, pairs_per_sample


def _pool_pair_embeddings(
    embeddings: torch.Tensor,
    *,
    batch: int,
    pairs_per_sample: int,
) -> torch.Tensor:
    pair_embeddings = embeddings.reshape(batch, pairs_per_sample, -1)
    return torch.cat(
        [pair_embeddings.mean(dim=1), pair_embeddings.max(dim=1).values],
        dim=1,
    )


class Gift64SunStyleLstmPairSetDistinguisher(nn.Module):
    """Capacity-matched LSTM-family baseline on raw GIFT-64 pair sets."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        hidden_bits: int = 128,
        classifier_bits: int = 128,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("GIFT-64 LSTM baseline requires 128-bit ciphertext pairs")
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of pair_bits")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.encoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden_bits,
            batch_first=True,
            bidirectional=True,
        )
        pooled_bits = 4 * hidden_bits
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_bits),
            nn.Linear(pooled_bits, classifier_bits),
            nn.ReLU(),
            nn.Linear(classifier_bits, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequences, batch, pairs_per_sample = _pair_sequences(
            features,
            input_bits=self.input_bits,
            pair_bits=self.pair_bits,
        )
        _, (hidden, _) = self.encoder(sequences)
        pair_embeddings = torch.cat([hidden[-2], hidden[-1]], dim=1)
        pooled = _pool_pair_embeddings(
            pair_embeddings,
            batch=batch,
            pairs_per_sample=pairs_per_sample,
        )
        return self.classifier(pooled)


class _Gift64ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


class Gift64GohrStyleResNetPairSetDistinguisher(nn.Module):
    """Capacity-matched residual-CNN baseline on raw GIFT-64 pair sets."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        channels: int = 64,
        blocks: int = 7,
        classifier_bits: int = 128,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("GIFT-64 ResNet baseline requires 128-bit ciphertext pairs")
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of pair_bits")
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.stem = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *(_Gift64ResidualBlock(channels) for _ in range(blocks))
        )
        self.pair_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * channels),
            nn.Linear(2 * channels, classifier_bits),
            nn.ReLU(),
            nn.Linear(classifier_bits, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequences, batch, pairs_per_sample = _pair_sequences(
            features,
            input_bits=self.input_bits,
            pair_bits=self.pair_bits,
        )
        hidden = self.stem(sequences.transpose(1, 2))
        hidden = self.residual_blocks(hidden)
        pair_embeddings = self.pair_pool(hidden).squeeze(2)
        pooled = _pool_pair_embeddings(
            pair_embeddings,
            batch=batch,
            pairs_per_sample=pairs_per_sample,
        )
        return self.classifier(pooled)
