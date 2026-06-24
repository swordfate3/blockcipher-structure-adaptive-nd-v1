from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_inception_blocks import (
    PresentInceptionMCNDBlock,
    conv1d_norm,
)


class PresentInceptionMCNDDistinguisher(nn.Module):
    """PRESENT-oriented multi-ciphertext neural distinguisher baseline."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        branches: int | None = None,
        blocks: int = 3,
        activation: str = "gelu",
        norm: str = "batchnorm1d",
        pooling: str = "attention_mean_max",
        dropout: float = 0.0,
        kernel_sizes: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("PresentInceptionMCND input_bits must be a multiple of pair_bits")
        if pair_bits < 1:
            raise ValueError("pair_bits must be >= 1")
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        if pooling not in {"attention", "attention_mean_max", "mean_max"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "SPN"
        self.base_channels = base_channels
        self.branch_channels = branches or max(4, base_channels // 4)
        self.blocks = blocks
        self.activation = activation
        self.norm = norm
        self.pooling = pooling
        self.dropout = dropout
        self.kernel_sizes = tuple(kernel_sizes)

        self.pair_bit_encoder = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, padding=3),
            conv1d_norm(norm, base_channels),
            build_activation(activation),
            nn.Dropout(dropout),
        )
        self.inception_blocks = nn.Sequential(
            *[
                PresentInceptionMCNDBlock(
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

    def _encode_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        hidden = pair_features.unsqueeze(1).float()
        hidden = self.pair_bit_encoder(hidden)
        hidden = self.inception_blocks(hidden)
        mean_embedding = hidden.mean(dim=2)
        max_embedding = hidden.max(dim=2).values
        edge_embedding = hidden[:, :, -1] - hidden[:, :, 0]
        return self.pair_projection(torch.cat([mean_embedding, max_embedding, edge_embedding], dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        pair_features = features.reshape(features.shape[0] * self.pairs_per_sample, self.pair_bits)
        pair_embeddings = self._encode_pairs(pair_features).reshape(
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
