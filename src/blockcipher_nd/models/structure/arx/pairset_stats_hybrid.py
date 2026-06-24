from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm
from blockcipher_nd.models.structure.arx.word_mixer_pairset import (
    ArxWordMixerPairSetDistinguisher,
)


class ArxPairSetStatsHybridDistinguisher(nn.Module):
    """SPECK pair-set model that fuses ARX word mixing with cross-pair statistics."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 224,
        base_channels: int = 32,
        word_bits: int = 16,
        token_dim: int | None = None,
        mixer_depth: int = 3,
        token_mlp_ratio: int = 2,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        stats_hidden_bits: int | None = None,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("ArxPairSetStatsHybrid input_bits must be a multiple of pair_bits")
        if pair_bits % 32 != 0:
            raise ValueError("ArxPairSetStatsHybrid pair_bits must be a multiple of 32-bit ARX feature words")
        if word_bits != 16:
            raise ValueError("ArxPairSetStatsHybrid currently supports 16-bit SPECK32 words")
        if input_bits // pair_bits < 2:
            raise ValueError("ArxPairSetStatsHybrid needs at least two pairs per sample")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "ARX"
        self.word_bits = word_bits
        self.feature_words_per_pair = pair_bits // 32
        self.half_words_per_feature = 2
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.stats_hidden_bits = stats_hidden_bits or max(64, base_channels * 8)
        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        self.word_branch = ArxWordMixerPairSetDistinguisher(
            input_bits=pair_bits,
            pair_bits=pair_bits,
            base_channels=base_channels,
            word_bits=word_bits,
            token_dim=self.token_dim,
            mixer_depth=mixer_depth,
            token_mlp_ratio=token_mlp_ratio,
            activation=activation,
            norm=norm,
            pooling="attention",
            dropout=dropout,
            top_k=1,
            lse_temperature=1.0,
        )
        self.word_pair_embedding_bits = self.word_branch.projected_pair_embedding_bits
        self.stats_feature_bits = (
            self.feature_words_per_pair * self.half_words_per_feature * 5
            + self.feature_words_per_pair * 7
        )
        self.stats_projection = nn.Sequential(
            build_norm(norm, self.stats_feature_bits),
            nn.Linear(self.stats_feature_bits, self.stats_hidden_bits),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.stats_hidden_bits, self.stats_hidden_bits),
            build_activation(activation),
        )
        self.fused_embedding_bits = self.word_pair_embedding_bits * 3 + self.stats_hidden_bits
        self.classifier = nn.Sequential(
            build_norm(norm, self.fused_embedding_bits),
            nn.Linear(self.fused_embedding_bits, max(128, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(128, base_channels * 8), max(64, base_channels * 4)),
            build_activation(activation),
            nn.Linear(max(64, base_channels * 4), 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def _pairset_statistics(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        halves = features.float().reshape(
            batch_size,
            self.pairs_per_sample,
            self.feature_words_per_pair,
            self.half_words_per_feature,
            self.word_bits,
        )
        half_active = halves.mean(dim=-1)
        half_mean = half_active.mean(dim=1)
        centered = half_active - half_mean.unsqueeze(1)
        half_var = (centered * centered).mean(dim=1)
        half_max = half_active.max(dim=1).values
        half_min = half_active.min(dim=1).values
        first_last_delta = half_active[:, 0] - half_active[:, -1]

        word_active = half_active.mean(dim=-1)
        word_mean = word_active.mean(dim=1)
        word_var = word_active.var(dim=1, unbiased=False)
        word_max = word_active.amax(dim=1)
        word_min = word_active.amin(dim=1)
        word_edge = half_active[:, :, :, 0].mean(dim=1) - half_active[:, :, :, 1].mean(dim=1)
        rotation_density = word_active[:, :, 3].mean(dim=1) if self.feature_words_per_pair > 3 else word_mean[:, 0]
        partial_inverse_density = (
            word_active[:, :, 4:].mean(dim=(1, 2))
            if self.feature_words_per_pair > 4
            else word_mean.mean(dim=1)
        )
        global_delta = (rotation_density - partial_inverse_density).unsqueeze(1).expand(
            -1,
            self.feature_words_per_pair,
        )
        word_density_delta = (word_mean[:, -1:] - word_mean[:, :1]).expand(-1, self.feature_words_per_pair)

        return torch.cat(
            [
                half_mean.flatten(1),
                half_var.flatten(1),
                half_max.flatten(1),
                half_min.flatten(1),
                first_last_delta.flatten(1),
                word_mean,
                word_var,
                word_max,
                word_min,
                word_edge,
                global_delta,
                word_density_delta,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        pair_features = features.float().reshape(
            features.shape[0] * self.pairs_per_sample,
            self.pair_bits,
        )
        word_embeddings = self.word_branch._encode_pairs(pair_features).reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.word_pair_embedding_bits,
        )
        word_mean = word_embeddings.mean(dim=1)
        word_max = word_embeddings.max(dim=1).values
        word_std = word_embeddings.std(dim=1, unbiased=False)
        stats_embedding = self.stats_projection(self._pairset_statistics(features))
        return self.classifier(torch.cat([word_mean, word_max, word_std, stats_embedding], dim=1))


__all__ = ["ArxPairSetStatsHybridDistinguisher"]
