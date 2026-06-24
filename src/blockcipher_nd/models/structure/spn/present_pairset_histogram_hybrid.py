from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_trail_mixer import (
    PresentTrailMixerPairSetDistinguisher,
)


class PresentPairSetHistogramHybridDistinguisher(nn.Module):
    """PRESENT pair-set model with cross-pair 4-bit cell value histograms."""

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
        histogram_hidden_bits: int | None = None,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("PresentPairSetHistogramHybrid input_bits must be a multiple of pair_bits")
        if pair_bits % 64 != 0:
            raise ValueError("PresentPairSetHistogramHybrid pair_bits must be a multiple of 64-bit PRESENT words")
        if nibble_bits != 4:
            raise ValueError("PresentPairSetHistogramHybrid currently expects 4-bit PRESENT cells")
        if input_bits // pair_bits < 2:
            raise ValueError("PresentPairSetHistogramHybrid needs at least two pairs per sample")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "SPN"
        self.nibble_bits = nibble_bits
        self.nibble_values = 1 << nibble_bits
        self.words_per_pair = pair_bits // 64
        self.cells_per_word = 64 // nibble_bits
        self.nibbles_per_pair = self.words_per_pair * self.cells_per_word
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.histogram_hidden_bits = histogram_hidden_bits or max(64, base_channels * 8)
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
        self.register_buffer(
            "nibble_basis",
            torch.arange(self.nibble_values, dtype=torch.long),
            persistent=False,
        )
        self.histogram_feature_bits = (
            self.words_per_pair * self.nibble_values * 4
            + self.cells_per_word * self.nibble_values * 2
            + self.nibble_values * 4
        )
        self.histogram_projection = nn.Sequential(
            build_norm(norm, self.histogram_feature_bits),
            nn.Linear(self.histogram_feature_bits, self.histogram_hidden_bits),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.histogram_hidden_bits, self.histogram_hidden_bits),
            build_activation(activation),
        )
        self.fused_embedding_bits = self.trail_pair_embedding_bits * 3 + self.histogram_hidden_bits
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

    def _nibble_values(self, features: torch.Tensor) -> torch.Tensor:
        cells = features.float().reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.words_per_pair,
            self.cells_per_word,
            self.nibble_bits,
        )
        weights = torch.tensor([8.0, 4.0, 2.0, 1.0], device=features.device, dtype=features.dtype)
        values = torch.sum(cells * weights, dim=-1).round().long().clamp(0, self.nibble_values - 1)
        return values

    def _histogram_statistics(self, features: torch.Tensor) -> torch.Tensor:
        values = self._nibble_values(features)
        one_hot = (values.unsqueeze(-1) == self.nibble_basis.to(values.device)).float()

        word_hist = one_hot.mean(dim=(1, 3))
        word_hist_var = one_hot.mean(dim=3).var(dim=1, unbiased=False)
        first_last_word_delta = one_hot[:, 0].mean(dim=2) - one_hot[:, -1].mean(dim=2)
        word_edge = one_hot[:, :, :, -1].mean(dim=1) - one_hot[:, :, :, 0].mean(dim=1)

        position_hist = one_hot.mean(dim=(1, 2))
        position_hist_var = one_hot.mean(dim=2).var(dim=1, unbiased=False)
        global_hist = one_hot.mean(dim=(1, 2, 3))
        first_last_global_delta = one_hot[:, 0].mean(dim=(1, 2)) - one_hot[:, -1].mean(dim=(1, 2))
        even_odd_delta = one_hot[:, :, :, ::2].mean(dim=(1, 2, 3)) - one_hot[:, :, :, 1::2].mean(dim=(1, 2, 3))
        word_density_delta = word_hist[:, -1] - word_hist[:, 0]

        return torch.cat(
            [
                word_hist.flatten(1),
                word_hist_var.flatten(1),
                first_last_word_delta.flatten(1),
                word_edge.flatten(1),
                position_hist.flatten(1),
                position_hist_var.flatten(1),
                global_hist,
                first_last_global_delta,
                even_odd_delta,
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
        histogram_embedding = self.histogram_projection(self._histogram_statistics(features))
        return self.classifier(torch.cat([trail_mean, trail_max, trail_std, histogram_embedding], dim=1))


__all__ = ["PresentPairSetHistogramHybridDistinguisher"]
