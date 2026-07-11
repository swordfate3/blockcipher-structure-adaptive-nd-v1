from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_inception_blocks import conv2d_norm
from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
    present_inverse_p_indices,
)


class PresentStateMatrixResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation: str = "relu",
        norm: str = "batchnorm2d",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm1 = conv2d_norm(norm, channels)
        self.activation1 = build_activation(activation)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm2 = conv2d_norm(norm, channels)
        self.activation2 = build_activation(activation)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(features)
        hidden = self.norm1(hidden)
        hidden = self.activation1(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        return self.activation2(features + hidden)


class PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher(nn.Module):
    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        conv_depth: int = 3,
        kernel_size: int = 3,
        activation: str = "relu",
        norm: str = "batchnorm2d",
        dropout: float = 0.0,
        mapping_mode: str = "true",
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError(
                "PresentNibbleStateMatrixConv2D expects raw 128-bit ciphertext pairs"
            )
        if input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a multiple of pair_bits")
        if conv_depth < 1:
            raise ValueError("conv_depth must be >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if mapping_mode not in {"true", "shuffled", "delta"}:
            raise ValueError(f"unsupported mapping_mode: {mapping_mode}")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.embedding_bits = base_channels * 4

        mapping_indices = (
            torch.arange(64, dtype=torch.long)
            if mapping_mode == "delta"
            else present_inverse_p_indices(mapping_mode)
        )
        self.register_buffer("mapping_indices", mapping_indices, persistent=False)

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=1),
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

    def state_matrix_view(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        raw_pairs = features.reshape(features.shape[0], self.pairs_per_sample, 2, 64)
        difference = (raw_pairs[:, :, 0, :] - raw_pairs[:, :, 1, :]).abs()
        mapped_difference = difference.index_select(dim=2, index=self.mapping_indices)
        return mapped_difference.reshape(
            features.shape[0], self.pairs_per_sample, 16, 4
        ).permute(0, 1, 3, 2)

    def encode_pairs(self, state_matrices: torch.Tensor) -> torch.Tensor:
        hidden = state_matrices.reshape(
            state_matrices.shape[0] * self.pairs_per_sample, 1, 4, 16
        )
        hidden = self.stem(hidden)
        for block in self.residual_blocks:
            hidden = block(hidden)
        pooled = torch.cat([hidden.mean(dim=(2, 3)), hidden.amax(dim=(2, 3))], dim=1)
        return self.pair_projection(pooled).reshape(
            state_matrices.shape[0], self.pairs_per_sample, self.embedding_bits
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pair_embeddings = self.encode_pairs(self.state_matrix_view(features.float()))
        pooled = torch.cat(
            [pair_embeddings.mean(dim=1), pair_embeddings.max(dim=1).values], dim=1
        )
        return self.classifier(pooled)


class PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher(
    PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["mapping_mode"] = "true"
        super().__init__(*args, **kwargs)


class PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher(
    PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["mapping_mode"] = "shuffled"
        super().__init__(*args, **kwargs)


class PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher(
    PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["mapping_mode"] = "delta"
        super().__init__(*args, **kwargs)


__all__ = [
    "PresentStateMatrixResidualBlock",
    "PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher",
    "PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher",
    "PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher",
    "PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher",
]
