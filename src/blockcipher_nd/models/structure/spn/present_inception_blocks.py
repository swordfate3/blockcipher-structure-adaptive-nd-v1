from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation


def conv1d_norm(name: str, channels: int) -> nn.Module:
    key = name.lower()
    if key in {"batchnorm1d", "batchnorm"}:
        return nn.BatchNorm1d(channels)
    if key in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"unsupported convolution norm: {name}")


def conv2d_norm(name: str, channels: int) -> nn.Module:
    key = name.lower()
    if key in {"batchnorm2d", "batchnorm"}:
        return nn.BatchNorm2d(channels)
    if key in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"unsupported convolution norm: {name}")


class PresentInceptionMCNDBlock(nn.Module):
    """Inception-style residual block over a set of PRESENT ciphertext pairs."""

    def __init__(
        self,
        channels: int,
        branch_channels: int,
        activation: str = "gelu",
        norm: str = "batchnorm1d",
        dropout: float = 0.0,
        kernel_sizes: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        if branch_channels < 1:
            raise ValueError("branch_channels must be >= 1")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")
        conv_branches = [
            nn.Sequential(
                nn.Conv1d(channels, branch_channels, kernel_size=kernel_size, padding="same"),
                conv1d_norm(norm, branch_channels),
                build_activation(activation),
            )
            for kernel_size in kernel_sizes
        ]
        self.branches = nn.ModuleList(
            [
                *conv_branches,
                nn.Sequential(
                    nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                    nn.Conv1d(channels, branch_channels, kernel_size=1),
                    conv1d_norm(norm, branch_channels),
                    build_activation(activation),
                ),
            ]
        )
        out_channels = branch_channels * len(self.branches)
        self.projection = nn.Conv1d(out_channels, channels, kernel_size=1)
        self.norm = conv1d_norm(norm, channels)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat([branch(features) for branch in self.branches], dim=1)
        hidden = self.projection(hidden)
        hidden = self.dropout(hidden)
        return self.activation(self.norm(features + hidden))


class PresentInceptionMCNDMatrixBlock(nn.Module):
    """2D Inception residual block over PRESENT MCND cell matrices."""

    def __init__(
        self,
        channels: int,
        branch_channels: int,
        activation: str = "gelu",
        norm: str = "batchnorm2d",
        dropout: float = 0.0,
        kernel_sizes: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (2, 4)),
    ) -> None:
        super().__init__()
        if branch_channels < 1:
            raise ValueError("branch_channels must be >= 1")
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, kernel_size=kernel_size, padding=padding),
                    conv2d_norm(norm, branch_channels),
                    build_activation(activation),
                )
            )
        self.branches.append(
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(channels, branch_channels, kernel_size=1),
                conv2d_norm(norm, branch_channels),
                build_activation(activation),
            )
        )
        self.projection = nn.Conv2d(branch_channels * len(self.branches), channels, kernel_size=1)
        self.norm = conv2d_norm(norm, channels)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        target_height, target_width = features.shape[-2:]
        branch_outputs = []
        for branch in self.branches:
            branch_output = branch(features)
            branch_outputs.append(branch_output[..., :target_height, :target_width])
        hidden = torch.cat(branch_outputs, dim=1)
        hidden = self.projection(hidden)
        hidden = self.dropout(hidden)
        return self.activation(self.norm(features + hidden))
