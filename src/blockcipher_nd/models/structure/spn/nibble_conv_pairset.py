from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import (
    AttentionPooling,
    EvidencePooling,
    GatedAttentionPooling,
    build_activation,
    build_norm,
)


class SpnNibbleConvPairSetDistinguisher(nn.Module):
    """SPN-focused pair-set model that preserves nibble position before pooling.

    Unlike SpnCellPairSetDBitNetDistinguisher, this model keeps the 4-bit cell
    sequence inside each pair and applies residual 1D convolutions across cells
    before aggregating multiple pairs.  It is intended as the SPN expert for
    innovation-one structure-adaptive experiments.
    """

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 192,
        base_channels: int = 32,
        nibble_bits: int = 4,
        nibble_embed_dim: int | None = None,
        conv_depth: int = 3,
        kernel_size: int = 3,
        activation: str = "gelu",
        norm: str = "layernorm",
        pooling: str = "attention_mean_max",
        dropout: float = 0.0,
        top_k: int = 4,
        lse_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("SpnNibbleConvPairSet input_bits must be a multiple of pair_bits")
        if pair_bits % nibble_bits != 0:
            raise ValueError("SpnNibbleConvPairSet pair_bits must be a multiple of nibble_bits")
        if conv_depth < 1:
            raise ValueError("SpnNibbleConvPairSet conv_depth must be >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("SpnNibbleConvPairSet kernel_size must be a positive odd integer")
        if pooling not in {"attention", "attention_mean_max", "mean_max", "gated_attention", "topk_mean", "logsumexp", "topk_logsumexp"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "SPN"
        self.nibble_bits = nibble_bits
        self.nibbles_per_pair = pair_bits // nibble_bits
        self.nibble_embed_dim = nibble_embed_dim or max(16, base_channels * 2)
        self.conv_depth = conv_depth
        self.kernel_size = kernel_size
        self.activation = activation
        self.norm = norm
        self.pooling = pooling
        self.dropout = dropout
        self.top_k = top_k
        self.lse_temperature = lse_temperature

        self.nibble_encoder = nn.Sequential(
            nn.Linear(nibble_bits, self.nibble_embed_dim),
            build_activation(activation),
            build_norm(norm, self.nibble_embed_dim),
        )
        conv_blocks: list[nn.Module] = []
        for _ in range(conv_depth):
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.nibble_embed_dim,
                        self.nibble_embed_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    build_activation(activation),
                    nn.Dropout(dropout),
                    nn.Conv1d(
                        self.nibble_embed_dim,
                        self.nibble_embed_dim,
                        kernel_size=1,
                    ),
                )
            )
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.sequence_norm = build_norm(norm, self.nibble_embed_dim)
        self.pair_embedding_bits = self.nibble_embed_dim * 3
        self.pair_projection = nn.Sequential(
            nn.Linear(self.pair_embedding_bits, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), max(32, base_channels * 4)),
            build_activation(activation),
        )
        self.projected_pair_embedding_bits = max(32, base_channels * 4)
        if pooling == "gated_attention":
            self.attention = GatedAttentionPooling(
                self.projected_pair_embedding_bits,
                hidden_bits=max(32, base_channels * 4),
            )
        elif pooling in {"topk_mean", "logsumexp", "topk_logsumexp"}:
            self.attention = EvidencePooling(
                self.projected_pair_embedding_bits,
                hidden_bits=max(32, base_channels * 4),
                mode=pooling,
                top_k=top_k,
                lse_temperature=lse_temperature,
                activation=activation,
                norm=norm,
            )
        else:
            self.attention = AttentionPooling(
                self.projected_pair_embedding_bits,
                hidden_bits=max(32, base_channels * 4),
                activation=activation,
                norm=norm,
            )
        pooling_multiplier = 3 if pooling == "attention_mean_max" else 2 if pooling == "mean_max" else 1
        self.classifier = nn.Sequential(
            build_norm(norm, self.projected_pair_embedding_bits * pooling_multiplier),
            nn.Linear(self.projected_pair_embedding_bits * pooling_multiplier, 256),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            build_activation(activation),
            nn.Linear(128, 1),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def _encode_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        nibbles = pair_features.reshape(
            pair_features.shape[0],
            self.nibbles_per_pair,
            self.nibble_bits,
        )
        hidden = self.nibble_encoder(nibbles)
        for block in self.conv_blocks:
            residual = hidden
            conv_hidden = block(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = self.sequence_norm(residual + conv_hidden)
        mean_embedding = hidden.mean(dim=1)
        max_embedding = hidden.max(dim=1).values
        first_last_delta = hidden[:, -1, :] - hidden[:, 0, :]
        pair_embedding = torch.cat([mean_embedding, max_embedding, first_last_delta], dim=1)
        return self.pair_projection(pair_embedding)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        pair_features = features.float().reshape(
            features.shape[0] * self.pairs_per_sample,
            self.pair_bits,
        )
        pair_embeddings = self._encode_pairs(pair_features).reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.projected_pair_embedding_bits,
        )
        attention_embedding, attention_weights = self.attention(pair_embeddings)
        self.last_attention_weights = attention_weights.detach()
        mean_embedding = pair_embeddings.mean(dim=1)
        max_embedding = pair_embeddings.max(dim=1).values
        if self.pooling in {"attention", "gated_attention", "topk_mean", "logsumexp", "topk_logsumexp"}:
            pooled = attention_embedding
        elif self.pooling == "mean_max":
            pooled = torch.cat([mean_embedding, max_embedding], dim=1)
        else:
            pooled = torch.cat([attention_embedding, mean_embedding, max_embedding], dim=1)
        return self.classifier(pooled)



__all__ = ["SpnNibbleConvPairSetDistinguisher"]
