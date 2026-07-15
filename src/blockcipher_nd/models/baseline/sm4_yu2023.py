from __future__ import annotations

import torch
from torch import nn


class _PositionResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


class Sm4Yu2023PositionResNetDistinguisher(nn.Module):
    """Position-preserving SM4 Conv-ResNet aligned to Yu/Wu/Zhang's description."""

    def __init__(
        self,
        input_bits: int,
        *,
        channels: int = 32,
        blocks: int = 5,
        classifier_bits: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if input_bits != 256:
            raise ValueError("SM4 Yu2023 position ResNet requires one 256-bit pair")
        if blocks < 1:
            raise ValueError("SM4 Yu2023 position ResNet requires at least one block")
        self.input_bits = input_bits
        self.sequence_bits = input_bits // 2
        self.flattened_width = channels * self.sequence_bits
        self.stem = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.residual_tower = nn.Sequential(
            *[_PositionResidualBlock(channels) for _ in range(blocks)]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_width, classifier_bits),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_bits, classifier_bits),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_bits, 1),
        )

    def position_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pair = features.float().reshape(features.shape[0], 2, self.sequence_bits)
        return self.residual_tower(self.stem(pair))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.position_features(features))


__all__ = ["Sm4Yu2023PositionResNetDistinguisher"]
