from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_trail_mixer import (
    PresentTrailMixerPairSetDistinguisher,
)


class PresentPairSetStatsHybridDistinguisher(nn.Module):
    """PRESENT pair-set model that exposes cross-pair nibble statistics."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 2496,
        base_channels: int = 32,
        nibble_bits: int = 4,
        token_dim: int | None = None,
        mixer_depth: int = 2,
        role_mixer_depth: int = 1,
        token_mlp_ratio: int = 2,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        stats_hidden_bits: int | None = None,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("PresentPairSetStatsHybrid input_bits must be a multiple of pair_bits")
        if pair_bits % 64 != 0:
            raise ValueError("PresentPairSetStatsHybrid pair_bits must be a multiple of 64-bit PRESENT words")
        if pair_bits % nibble_bits != 0:
            raise ValueError("PresentPairSetStatsHybrid pair_bits must be a multiple of nibble_bits")
        if input_bits // pair_bits < 2:
            raise ValueError("PresentPairSetStatsHybrid needs at least two pairs per sample")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "SPN"
        self.nibble_bits = nibble_bits
        self.words_per_pair = pair_bits // 64
        self.nibbles_per_pair = pair_bits // nibble_bits
        self.cells_per_word = 64 // nibble_bits
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.stats_hidden_bits = stats_hidden_bits or max(64, base_channels * 8)
        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        self.trail_branch = PresentTrailMixerPairSetDistinguisher(
            input_bits=pair_bits,
            pair_bits=pair_bits,
            base_channels=base_channels,
            nibble_bits=nibble_bits,
            token_dim=self.token_dim,
            mixer_depth=mixer_depth,
            role_mixer_depth=role_mixer_depth,
            token_mlp_ratio=token_mlp_ratio,
            activation=activation,
            norm=norm,
            pooling="attention",
            dropout=dropout,
            top_k=1,
            lse_temperature=1.0,
        )
        self.trail_pair_embedding_bits = self.trail_branch.projected_pair_embedding_bits
        self.stats_feature_bits = self.words_per_pair * self.cells_per_word * 5 + self.words_per_pair * 8
        self.stats_projection = nn.Sequential(
            build_norm(norm, self.stats_feature_bits),
            nn.Linear(self.stats_feature_bits, self.stats_hidden_bits),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.stats_hidden_bits, self.stats_hidden_bits),
            build_activation(activation),
        )
        self.fused_embedding_bits = self.trail_pair_embedding_bits * 3 + self.stats_hidden_bits
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
        cells = features.float().reshape(
            batch_size,
            self.pairs_per_sample,
            self.words_per_pair,
            self.cells_per_word,
            self.nibble_bits,
        )
        cell_active = cells.mean(dim=-1)
        cell_mean = cell_active.mean(dim=1)
        centered = cell_active - cell_mean.unsqueeze(1)
        cell_var = (centered * centered).mean(dim=1)
        cell_max = cell_active.max(dim=1).values
        cell_min = cell_active.min(dim=1).values
        first_pair_delta = cell_active[:, 0] - cell_active[:, -1]
        even_mean = cell_active[:, :, :, ::2].mean(dim=(1, 3))
        odd_mean = cell_active[:, :, :, 1::2].mean(dim=(1, 3))
        word_mean = cell_active.mean(dim=(1, 3))
        word_var = cell_active.var(dim=(1, 3), unbiased=False)
        word_max = cell_active.amax(dim=(1, 3))
        word_min = cell_active.amin(dim=(1, 3))
        word_edge = cell_active[:, :, :, -1].mean(dim=1) - cell_active[:, :, :, 0].mean(dim=1)
        word_density_delta = (word_mean[:, -1:] - word_mean[:, :1]).expand(-1, self.words_per_pair)

        return torch.cat(
            [
                cell_mean.flatten(1),
                cell_var.flatten(1),
                cell_max.flatten(1),
                cell_min.flatten(1),
                first_pair_delta.flatten(1),
                even_mean,
                odd_mean,
                word_mean,
                word_var,
                word_max,
                word_min,
                word_edge,
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
        trail_embeddings = self.trail_branch._encode_pairs(pair_features).reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.trail_pair_embedding_bits,
        )
        trail_mean = trail_embeddings.mean(dim=1)
        trail_max = trail_embeddings.max(dim=1).values
        trail_std = trail_embeddings.std(dim=1, unbiased=False)
        stats_embedding = self.stats_projection(self._pairset_statistics(features))
        return self.classifier(torch.cat([trail_mean, trail_max, trail_std, stats_embedding], dim=1))


__all__ = ["PresentPairSetStatsHybridDistinguisher"]
