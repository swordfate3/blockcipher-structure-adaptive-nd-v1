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
from blockcipher_nd.models.structure.spn.present_trail_mixer import PresentTrailMixerPairSetDistinguisher


class PresentMatrixTrailHybridPairSetDistinguisher(nn.Module):
    """PRESENT pair-set model that fuses cell-matrix and public trail evidence."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 768,
        base_channels: int = 32,
        nibble_bits: int = 4,
        token_dim: int | None = None,
        mixer_depth: int = 3,
        role_mixer_depth: int = 2,
        matrix_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "gelu",
        norm: str = "layernorm",
        pooling: str = "topk_logsumexp",
        dropout: float = 0.0,
        top_k: int = 4,
        lse_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("PresentMatrixTrailHybridPairSet input_bits must be a multiple of pair_bits")
        if pair_bits % 64 != 0:
            raise ValueError("PresentMatrixTrailHybridPairSet pair_bits must be a multiple of 64-bit PRESENT words")
        if pair_bits % nibble_bits != 0:
            raise ValueError("PresentMatrixTrailHybridPairSet pair_bits must be a multiple of nibble_bits")
        if matrix_depth < 1:
            raise ValueError("PresentMatrixTrailHybridPairSet matrix_depth must be >= 1")
        if pooling not in {
            "attention",
            "attention_mean_max",
            "mean_max",
            "gated_attention",
            "topk_mean",
            "logsumexp",
            "topk_logsumexp",
        }:
            raise ValueError(f"unsupported pooling: {pooling}")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "SPN"
        self.nibble_bits = nibble_bits
        self.words_per_pair = pair_bits // 64
        self.nibbles_per_pair = pair_bits // nibble_bits
        self.matrix_depth = matrix_depth
        self.pooling = pooling
        self.top_k = top_k
        self.lse_temperature = lse_temperature

        self.trail_branch = PresentTrailMixerPairSetDistinguisher(
            input_bits=pair_bits,
            pair_bits=pair_bits,
            base_channels=base_channels,
            nibble_bits=nibble_bits,
            token_dim=token_dim,
            mixer_depth=mixer_depth,
            role_mixer_depth=role_mixer_depth,
            token_mlp_ratio=token_mlp_ratio,
            activation=activation,
            norm=norm,
            pooling="attention",
            dropout=dropout,
            top_k=top_k,
            lse_temperature=lse_temperature,
        )
        self.trail_pair_embedding_bits = self.trail_branch.projected_pair_embedding_bits

        matrix_channels = max(8, base_channels)
        matrix_blocks: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(1, matrix_channels, kernel_size=(1, 3), padding=(0, 1)),
                nn.BatchNorm2d(matrix_channels),
                build_activation(activation),
            )
        ]
        for _ in range(matrix_depth - 1):
            matrix_blocks.append(
                nn.Sequential(
                    nn.Conv2d(matrix_channels, matrix_channels, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(matrix_channels),
                    build_activation(activation),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(matrix_channels, matrix_channels, kernel_size=1),
                    nn.BatchNorm2d(matrix_channels),
                    build_activation(activation),
                )
            )
        self.matrix_encoder = nn.Sequential(*matrix_blocks)
        self.matrix_pair_embedding_bits = max(32, base_channels * 4)
        self.matrix_projection = nn.Sequential(
            nn.Linear(matrix_channels * 3, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), self.matrix_pair_embedding_bits),
            build_activation(activation),
        )

        self.fused_pair_embedding_bits = self.trail_pair_embedding_bits + self.matrix_pair_embedding_bits
        if pooling == "gated_attention":
            self.attention = GatedAttentionPooling(
                self.fused_pair_embedding_bits,
                hidden_bits=max(32, base_channels * 4),
            )
        elif pooling in {"topk_mean", "logsumexp", "topk_logsumexp"}:
            self.attention = EvidencePooling(
                self.fused_pair_embedding_bits,
                hidden_bits=max(32, base_channels * 4),
                mode=pooling,
                top_k=top_k,
                lse_temperature=lse_temperature,
                activation=activation,
                norm=norm,
            )
        else:
            self.attention = AttentionPooling(
                self.fused_pair_embedding_bits,
                hidden_bits=max(32, base_channels * 4),
                activation=activation,
                norm=norm,
            )
        pooling_multiplier = 3 if pooling == "attention_mean_max" else 2 if pooling == "mean_max" else 1
        self.classifier = nn.Sequential(
            build_norm(norm, self.fused_pair_embedding_bits * pooling_multiplier),
            nn.Linear(self.fused_pair_embedding_bits * pooling_multiplier, 256),
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

    def _encode_matrix_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        matrix = pair_features.float().reshape(
            pair_features.shape[0],
            1,
            self.nibble_bits,
            self.nibbles_per_pair,
        )
        hidden = self.matrix_encoder(matrix)
        mean_embedding = hidden.mean(dim=(2, 3))
        max_embedding = hidden.amax(dim=(2, 3))
        edge_embedding = hidden[:, :, :, -1].mean(dim=2) - hidden[:, :, :, 0].mean(dim=2)
        return self.matrix_projection(torch.cat([mean_embedding, max_embedding, edge_embedding], dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        pair_features = features.float().reshape(features.shape[0] * self.pairs_per_sample, self.pair_bits)
        trail_embeddings = self.trail_branch._encode_pairs(pair_features)
        matrix_embeddings = self._encode_matrix_pairs(pair_features)
        fused_pair_embeddings = torch.cat([trail_embeddings, matrix_embeddings], dim=1).reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.fused_pair_embedding_bits,
        )
        attention_embedding, attention_weights = self.attention(fused_pair_embeddings)
        self.last_attention_weights = attention_weights.detach()
        mean_embedding = fused_pair_embeddings.mean(dim=1)
        max_embedding = fused_pair_embeddings.max(dim=1).values
        if self.pooling in {"attention", "gated_attention", "topk_mean", "logsumexp", "topk_logsumexp"}:
            pooled = attention_embedding
        elif self.pooling == "mean_max":
            pooled = torch.cat([mean_embedding, max_embedding], dim=1)
        else:
            pooled = torch.cat([attention_embedding, mean_embedding, max_embedding], dim=1)
        return self.classifier(pooled)


__all__ = ["PresentMatrixTrailHybridPairSetDistinguisher"]
