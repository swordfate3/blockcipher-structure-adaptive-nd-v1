from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_inception_blocks import (
    PresentInceptionMCNDMatrixBlock,
    conv2d_norm,
)


class PresentInceptionMCNDMatrixDistinguisher(nn.Module):
    """Zhang/Wang-style PRESENT MCND model over m x 4 x 32 cell matrices."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        branches: int | None = None,
        blocks: int = 3,
        activation: str = "gelu",
        norm: str = "batchnorm2d",
        pooling: str = "attention_mean_max",
        dropout: float = 0.0,
        kernel_sizes: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (2, 4)),
        cell_bits: int = 4,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("PresentInceptionMCNDMatrix input_bits must be a multiple of pair_bits")
        if pair_bits % cell_bits != 0:
            raise ValueError("pair_bits must be divisible by cell_bits")
        if pooling not in {"attention", "attention_mean_max", "mean_max"}:
            raise ValueError(f"unsupported pooling: {pooling}")
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
        self.pooling = pooling
        self.dropout = dropout
        self.kernel_sizes = tuple(kernel_sizes)

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=(2, 3), padding=(1, 1)),
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
        self.pair_embedding_bits = max(32, base_channels * 2)
        self.pair_projection = nn.Sequential(
            nn.Linear(base_channels * 3, max(64, base_channels * 4)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 4), self.pair_embedding_bits),
            build_activation(activation),
        )
        self.attention = nn.Sequential(
            build_norm("layernorm", self.pair_embedding_bits),
            nn.Linear(self.pair_embedding_bits, max(16, base_channels)),
            build_activation(activation),
            nn.Linear(max(16, base_channels), 1),
        )
        pooling_multiplier = 3 if pooling == "attention_mean_max" else 2 if pooling == "mean_max" else 1
        self.classifier = nn.Sequential(
            build_norm("layernorm", self.pair_embedding_bits * pooling_multiplier),
            nn.Linear(self.pair_embedding_bits * pooling_multiplier, max(64, base_channels * 4)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 4), max(32, base_channels * 2)),
            build_activation(activation),
            nn.Linear(max(32, base_channels * 2), 1),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def _encode_pair_matrices(self, pair_features: torch.Tensor) -> torch.Tensor:
        matrices = pair_features.float().reshape(pair_features.shape[0], 1, self.cell_bits, self.cell_width)
        hidden = self.stem(matrices)
        hidden = self.blocks_layer(hidden)
        mean_embedding = hidden.mean(dim=(2, 3))
        max_embedding = hidden.amax(dim=(2, 3))
        width_edge = hidden[:, :, :, -1].mean(dim=2) - hidden[:, :, :, 0].mean(dim=2)
        return self.pair_projection(torch.cat([mean_embedding, max_embedding, width_edge], dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        pair_features = features.reshape(features.shape[0] * self.pairs_per_sample, self.pair_bits)
        pair_embeddings = self._encode_pair_matrices(pair_features).reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.pair_embedding_bits,
        )
        attention_logits = self.attention(pair_embeddings).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        self.last_attention_weights = attention_weights.detach()
        attention_embedding = torch.sum(pair_embeddings * attention_weights.unsqueeze(-1), dim=1)
        if self.pooling == "attention":
            pooled = attention_embedding
        elif self.pooling == "mean_max":
            pooled = torch.cat([pair_embeddings.mean(dim=1), pair_embeddings.max(dim=1).values], dim=1)
        else:
            pooled = torch.cat(
                [
                    attention_embedding,
                    pair_embeddings.mean(dim=1),
                    pair_embeddings.max(dim=1).values,
                ],
                dim=1,
            )
        return self.classifier(pooled)
