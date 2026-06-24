from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PresentZhangWangKerasMCNDDistinguisher(nn.Module):
    """PyTorch port of Zhang/Wang 2022 PRESENT MCND Keras ResNet."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        blocks: int = 5,
        activation: str = "relu",
        dropout: float = 0.0,
        initial_kernel_sizes: tuple[int, ...] = (1, 2, 4),
        residual_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("Zhang/Wang PRESENT Keras MCND expects 128 bits per ciphertext pair")
        if input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a multiple of pair_bits")
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.word_size = 32
        self.num_filters = base_channels
        self.channels = base_channels * len(initial_kernel_sizes)
        self.blocks = blocks
        self.activation = activation
        self.dropout = dropout

        self.initial_branches = nn.ModuleList(
            [_GroupedSameLengthConv1d(8, base_channels, kernel_size) for kernel_size in initial_kernel_sizes]
        )
        self.initial_norm = _GroupedBatchNorm1d(self.channels)
        self.initial_activation = _activation(activation)
        self.residual_blocks = nn.ModuleList(
            [
                _ZhangWangResidualBlock(
                    channels=self.channels,
                    kernel_size=residual_kernel_size + 2 * index,
                    activation=activation,
                    dropout=dropout,
                )
                for index in range(blocks)
            ]
        )
        self.classifier = nn.Linear(self.channels, 1)

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        hidden = features.float().reshape(features.shape[0], self.pairs_per_sample, 8, 16)
        hidden = hidden.permute(0, 1, 3, 2)
        hidden = torch.cat([branch(hidden) for branch in self.initial_branches], dim=-1)
        hidden = self.initial_activation(self.initial_norm(hidden))
        for block in self.residual_blocks:
            hidden = block(hidden)
        pooled = hidden.mean(dim=(1, 2))
        return self.classifier(pooled)


class _ZhangWangResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.conv1 = _GroupedSameLengthConv1d(channels, channels, kernel_size)
        self.norm1 = _GroupedBatchNorm1d(channels)
        self.activation1 = _activation(activation)
        self.conv2 = _GroupedSameLengthConv1d(channels, channels, kernel_size)
        self.norm2 = _GroupedBatchNorm1d(channels)
        self.activation2 = _activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.activation1(self.norm1(self.conv1(features)))
        hidden = self.activation2(self.norm2(self.conv2(hidden)))
        hidden = self.dropout(hidden)
        return features + hidden


class _GroupedSameLengthConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"expected grouped Conv1D input with 4 dims, got {tuple(features.shape)}")
        batch, groups, steps, channels = features.shape
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        flattened = features.reshape(batch * groups, steps, channels).transpose(1, 2)
        convolved = self.conv(F.pad(flattened, (left, right)))
        return convolved.transpose(1, 2).reshape(batch, groups, steps, -1)


class _GroupedBatchNorm1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, groups, steps, channels = features.shape
        flattened = features.reshape(batch * groups, steps, channels).transpose(1, 2)
        normalized = self.norm(flattened)
        return normalized.transpose(1, 2).reshape(batch, groups, steps, channels)


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported Zhang/Wang activation: {name}")
