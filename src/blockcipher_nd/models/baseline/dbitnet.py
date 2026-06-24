from __future__ import annotations

import torch
from torch import nn


class DBitNetDistinguisher(nn.Module):
    """Dilated convolution baseline for cipher-agnostic bit-pair features."""

    def __init__(self, input_bits: int, channels: int = 32) -> None:
        super().__init__()
        if input_bits % 2 != 0:
            raise ValueError("DBitNetDistinguisher requires an even number of input bits")
        self.input_bits = input_bits
        self.sequence_bits = input_bits // 2
        self.features = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        batch = features.float().reshape(features.shape[0], 2, self.sequence_bits)
        hidden = self.features(batch).squeeze(-1)
        return self.classifier(hidden)
