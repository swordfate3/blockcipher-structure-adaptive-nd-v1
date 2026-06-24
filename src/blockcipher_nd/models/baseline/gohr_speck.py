from __future__ import annotations

import torch
from torch import nn


def reshape_speck_ciphertext_pair(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != 64:
        raise ValueError("expected 64-bit SPECK32/64 ciphertext-pair input")
    return features.float().reshape(features.shape[0], 4, 16)


class GohrSpeckResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


class GohrSpeckDistinguisher(nn.Module):
    """Gohr-style SPECK32/64 word-aware residual neural distinguisher."""

    def __init__(self, input_bits: int, filters: int = 32, blocks: int = 1) -> None:
        super().__init__()
        if input_bits != 64:
            raise ValueError("GohrSpeckDistinguisher requires 64-bit SPECK32/64 ciphertext-pair input")
        self.input_bits = input_bits
        self.word_bits = 16
        self.filters = filters
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv1d(4, filters, kernel_size=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *(GohrSpeckResidualBlock(filters) for _ in range(blocks))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * self.word_bits, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = reshape_speck_ciphertext_pair(features)
        hidden = self.stem(hidden)
        hidden = self.residual_blocks(hidden)
        return self.head(hidden)
