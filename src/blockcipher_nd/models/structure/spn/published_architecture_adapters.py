from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_inception_blocks import conv2d_norm
from blockcipher_nd.models.structure.spn.present_invp_state_matrix_conv2d import (
    PresentStateMatrixResidualBlock,
)


class SpnLiuCase3Conv2DAdapterDistinguisher(nn.Module):
    """Liu-style Case-3 state-matrix Conv2D backbone on raw pair observables."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        cell_bits: int = 4,
        conv_depth: int = 3,
        kernel_size: int = 3,
        activation: str = "relu",
        norm: str = "batchnorm2d",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of pair_bits")
        if pair_bits <= 0 or pair_bits % 2 != 0:
            raise ValueError("pair_bits must contain two equal-width ciphertexts")
        self.state_bits = pair_bits // 2
        if cell_bits <= 0 or self.state_bits % cell_bits != 0:
            raise ValueError("each ciphertext must contain an integer number of cells")
        if conv_depth < 1:
            raise ValueError("conv_depth must be >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.cell_bits = cell_bits
        self.cells_per_state = self.state_bits // cell_bits
        self.embedding_bits = base_channels * 4

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=1),
            conv2d_norm(norm, base_channels),
            build_activation(activation),
        )
        self.residual_blocks = nn.ModuleList(
            [
                PresentStateMatrixResidualBlock(
                    channels=base_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for _ in range(conv_depth)
            ]
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(base_channels * 2, self.embedding_bits),
            build_activation(activation),
        )
        classifier_hidden = max(64, base_channels * 8)
        self.classifier = nn.Sequential(
            build_norm("layernorm", self.embedding_bits * 2),
            nn.Linear(self.embedding_bits * 2, classifier_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def case3_view(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pairs = features.float().reshape(
            features.shape[0], self.pairs_per_sample, 2, self.state_bits
        )
        difference = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
        channels = torch.stack(
            [pairs[:, :, 0], pairs[:, :, 1], difference], dim=2
        )
        return channels.reshape(
            features.shape[0],
            self.pairs_per_sample,
            3,
            self.cells_per_state,
            self.cell_bits,
        ).permute(0, 1, 2, 4, 3)

    def encode_pairs(self, case3_matrices: torch.Tensor) -> torch.Tensor:
        expected = (self.pairs_per_sample, 3, self.cell_bits, self.cells_per_state)
        if case3_matrices.ndim != 5 or tuple(case3_matrices.shape[1:]) != expected:
            raise ValueError(
                "expected Case-3 matrices with shape "
                f"[batch, {', '.join(str(item) for item in expected)}], "
                f"got {tuple(case3_matrices.shape)}"
            )
        batch = case3_matrices.shape[0]
        hidden = case3_matrices.reshape(
            batch * self.pairs_per_sample,
            3,
            self.cell_bits,
            self.cells_per_state,
        )
        hidden = self.stem(hidden)
        for block in self.residual_blocks:
            hidden = block(hidden)
        pooled = torch.cat(
            [hidden.mean(dim=(2, 3)), hidden.amax(dim=(2, 3))], dim=1
        )
        return self.pair_projection(pooled).reshape(
            batch, self.pairs_per_sample, self.embedding_bits
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pair_embeddings = self.encode_pairs(self.case3_view(features.float()))
        pooled = torch.cat(
            [pair_embeddings.mean(dim=1), pair_embeddings.max(dim=1).values], dim=1
        )
        return self.classifier(pooled)


__all__ = ["SpnLiuCase3Conv2DAdapterDistinguisher"]
