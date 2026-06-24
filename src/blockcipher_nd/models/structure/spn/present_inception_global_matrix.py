from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation
from blockcipher_nd.models.structure.spn.present_inception_blocks import (
    PresentInceptionMCNDMatrixBlock,
    conv2d_norm,
)


class PresentInceptionMCNDGlobalMatrixDistinguisher(nn.Module):
    """Protocol-reproduction PRESENT MCND model over the full m x cell matrix."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        branches: int | None = None,
        blocks: int = 3,
        activation: str = "gelu",
        norm: str = "batchnorm2d",
        dropout: float = 0.0,
        kernel_sizes: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (2, 4)),
        cell_bits: int = 4,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("PresentInceptionMCNDGlobalMatrix input_bits must be a multiple of pair_bits")
        if pair_bits % cell_bits != 0:
            raise ValueError("pair_bits must be divisible by cell_bits")
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.cell_bits = cell_bits
        self.cell_width = pair_bits // cell_bits
        self.structure = "SPN"
        self.base_channels = base_channels
        self.branch_channels = branches or max(4, base_channels // 4)
        self.blocks = blocks
        self.activation = activation
        self.norm = norm
        self.dropout = dropout
        self.kernel_sizes = tuple(kernel_sizes)

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=(1, 3), padding=(0, 1)),
            conv2d_norm(norm, base_channels),
            build_activation(activation),
            nn.Dropout2d(dropout),
        )
        self.blocks_layer = nn.Sequential(
            *[
                PresentInceptionMCNDMatrixBlock(
                    base_channels,
                    branch_channels=self.branch_channels,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                    kernel_sizes=self.kernel_sizes,
                )
                for _ in range(blocks)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 3, max(64, base_channels * 4)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 4), max(32, base_channels * 2)),
            build_activation(activation),
            nn.Linear(max(32, base_channels * 2), 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        matrices = features.float().reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.cell_bits,
            self.cell_width,
        )
        global_matrix = matrices.permute(0, 2, 1, 3).reshape(
            features.shape[0],
            1,
            self.cell_bits,
            self.pairs_per_sample * self.cell_width,
        )
        hidden = self.stem(global_matrix)
        hidden = self.blocks_layer(hidden)
        mean_embedding = hidden.mean(dim=(2, 3))
        max_embedding = hidden.amax(dim=(2, 3))
        width_edge = hidden[:, :, :, -1].mean(dim=2) - hidden[:, :, :, 0].mean(dim=2)
        return self.classifier(torch.cat([mean_embedding, max_embedding, width_edge], dim=1))
