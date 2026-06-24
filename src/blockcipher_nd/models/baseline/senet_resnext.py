from __future__ import annotations

import torch
from torch import nn


class SqueezeExcitation1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, squeezed),
            nn.ReLU(),
            nn.Linear(squeezed, channels),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = self.gate(features).unsqueeze(-1)
        return features * weights


class SeResNeXtBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 4) -> None:
        super().__init__()
        group_count = max(1, min(groups, channels))
        while channels % group_count != 0:
            group_count -= 1
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=group_count,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            SqueezeExcitation1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


class SeResNeXtDistinguisher(nn.Module):
    """SE-ResNeXt style 1D neural distinguisher for bit-pair features."""

    def __init__(self, input_bits: int, channels: int = 32, blocks: int = 3) -> None:
        super().__init__()
        if input_bits % 2 != 0:
            raise ValueError(
                "SeResNeXtDistinguisher requires an even number of input bits"
            )
        self.input_bits = input_bits
        self.sequence_bits = input_bits // 2
        self.stem = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *(SeResNeXtBlock(channels) for _ in range(blocks))
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        batch = features.float().reshape(features.shape[0], 2, self.sequence_bits)
        hidden = self.stem(batch)
        hidden = self.blocks(hidden)
        return self.head(hidden)
