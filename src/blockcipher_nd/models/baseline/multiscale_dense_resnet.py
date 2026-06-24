from __future__ import annotations

import torch
from torch import nn


class MultiScaleDenseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branch3 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.branch5 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.branch7 = nn.Conv1d(channels, channels, kernel_size=7, padding=3)
        self.merge = nn.Sequential(
            nn.BatchNorm1d(channels * 3),
            nn.ReLU(),
            nn.Conv1d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat(
            [
                self.branch3(features),
                self.branch5(features),
                self.branch7(features),
            ],
            dim=1,
        )
        return self.activation(features + self.merge(multi_scale))


class DenseResidualStack(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MultiScaleDenseBlock(channels) for _ in range(blocks)]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * blocks, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = []
        hidden = features
        for block in self.blocks:
            hidden = block(hidden)
            outputs.append(hidden)
        return self.fuse(torch.cat(outputs, dim=1))


class MultiScaleDenseResNetDistinguisher(nn.Module):
    """Multi-scale convolutional distinguisher with dense residual reuse."""

    def __init__(self, input_bits: int, channels: int = 32, blocks: int = 3) -> None:
        super().__init__()
        if input_bits % 2 != 0:
            raise ValueError(
                "MultiScaleDenseResNetDistinguisher requires an even number of input bits"
            )
        if blocks < 1:
            raise ValueError("MultiScaleDenseResNetDistinguisher requires at least one block")
        self.input_bits = input_bits
        self.sequence_bits = input_bits // 2
        self.stem = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.features = DenseResidualStack(channels=channels, blocks=blocks)
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
        hidden = self.features(hidden)
        return self.head(hidden)
