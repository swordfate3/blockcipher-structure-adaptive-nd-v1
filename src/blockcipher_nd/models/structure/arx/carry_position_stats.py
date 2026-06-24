from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm


class ArxCarryPositionStatsPairSetDistinguisher(nn.Module):
    """SPECK carry-position statistics over carrychain-plus pair sets.

    This model is intentionally narrow: it reads the public
    ``ciphertext_pair_xor_arx_partial_inverse_rx_carrychain_plus_bits`` layout
    and keeps bit-position evidence that density-only statistics wash out.
    """

    carry_role_indices = (17, 18, 19, 20, 21, 22)
    band_slices = ((0, 5), (5, 11), (11, 16))

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 736,
        base_channels: int = 32,
        word_bits: int = 16,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        stats_hidden_bits: int | None = None,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("ArxCarryPositionStats input_bits must be a multiple of pair_bits")
        if pair_bits % 32 != 0:
            raise ValueError("ArxCarryPositionStats pair_bits must be a multiple of 32-bit ARX feature words")
        if word_bits != 16:
            raise ValueError("ArxCarryPositionStats currently supports 16-bit SPECK32 words")
        if pair_bits // 32 < 23:
            raise ValueError("ArxCarryPositionStats requires carrychain-plus SPECK32 feature words")
        if input_bits // pair_bits < 2:
            raise ValueError("ArxCarryPositionStats needs at least two pairs per sample")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "ARX"
        self.word_bits = word_bits
        self.feature_words_per_pair = pair_bits // 32
        self.half_words_per_feature = 2
        self.role_group_count = len(self.carry_role_indices)
        self.band_count = len(self.band_slices)
        self.stats_hidden_bits = stats_hidden_bits or max(64, base_channels * 8)
        self.activation = activation
        self.norm = norm
        self.dropout = dropout
        self.stats_feature_bits = self.position_stats_feature_bits(
            self.feature_words_per_pair,
            self.role_group_count,
            self.band_count,
            word_bits=self.word_bits,
        )
        classifier_hidden = max(64, base_channels * 4)
        self.classifier = nn.Sequential(
            build_norm(norm, self.stats_feature_bits),
            nn.Linear(self.stats_feature_bits, self.stats_hidden_bits),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.stats_hidden_bits, classifier_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    @staticmethod
    def position_stats_feature_bits(
        feature_words_per_pair: int,
        carry_role_count: int,
        band_count: int,
        *,
        word_bits: int = 16,
    ) -> int:
        return (
            feature_words_per_pair * 2 * word_bits * 2
            + carry_role_count * 2 * word_bits * 3
            + carry_role_count * 2 * band_count * 2
            + carry_role_count * 2 * 4
            + 8
        )

    def _position_statistics(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        halves = features.float().reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.feature_words_per_pair,
            self.half_words_per_feature,
            self.word_bits,
        )
        all_position_mean = halves.mean(dim=1)
        all_position_var = halves.var(dim=1, unbiased=False)

        carry = halves[:, :, self.carry_role_indices]
        carry_mean = carry.mean(dim=1)
        carry_var = carry.var(dim=1, unbiased=False)
        carry_max = carry.amax(dim=1)

        band_values = []
        for start, stop in self.band_slices:
            band_values.append(carry[:, :, :, :, start:stop].mean(dim=-1))
        bands = torch.stack(band_values, dim=-1)
        band_mean = bands.mean(dim=1)
        band_var = bands.var(dim=1, unbiased=False)

        run_histogram = self._carry_run_histogram(carry)
        rotation_stats = self._rotation_correlation_stats(halves)

        return torch.cat(
            [
                all_position_mean.flatten(1),
                all_position_var.flatten(1),
                carry_mean.flatten(1),
                carry_var.flatten(1),
                carry_max.flatten(1),
                band_mean.flatten(1),
                band_var.flatten(1),
                run_histogram.flatten(1),
                rotation_stats,
            ],
            dim=1,
        )

    def _carry_run_histogram(self, carry: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(carry, (1, 4), value=0.0)
        starts = (padded[..., 1 : self.word_bits + 1] > 0.5) & (
            padded[..., : self.word_bits] <= 0.5
        )
        next_1 = padded[..., 2 : self.word_bits + 2] > 0.5
        next_2 = padded[..., 3 : self.word_bits + 3] > 0.5
        next_3 = padded[..., 4 : self.word_bits + 4] > 0.5
        buckets = [
            starts & ~next_1,
            starts & next_1 & ~next_2,
            starts & next_1 & next_2 & ~next_3,
            starts & next_1 & next_2 & next_3,
        ]
        return torch.stack([bucket.float().mean(dim=(1, -1)) for bucket in buckets], dim=-1)

    def _rotation_correlation_stats(self, halves: torch.Tensor) -> torch.Tensor:
        role_17 = halves[:, :, 17]
        role_19 = halves[:, :, 19]
        role_21 = halves[:, :, 21]
        role_22 = halves[:, :, 22]
        shifted_19 = torch.roll(role_19, shifts=7, dims=-1)
        shifted_22 = torch.roll(role_22, shifts=2, dims=-1)
        same_17_19 = (role_17 == shifted_19).float().mean(dim=(1, 2, 3))
        both_17_19 = (role_17 * shifted_19).mean(dim=(1, 2, 3))
        same_21_22 = (role_21 == shifted_22).float().mean(dim=(1, 2, 3))
        both_21_22 = (role_21 * shifted_22).mean(dim=(1, 2, 3))
        edge_17 = (role_17[..., 1:] - role_17[..., :-1]).abs().mean(dim=(1, 2, 3))
        edge_19 = (role_19[..., 1:] - role_19[..., :-1]).abs().mean(dim=(1, 2, 3))
        edge_21 = (role_21[..., 1:] - role_21[..., :-1]).abs().mean(dim=(1, 2, 3))
        edge_22 = (role_22[..., 1:] - role_22[..., :-1]).abs().mean(dim=(1, 2, 3))
        return torch.stack(
            [
                same_17_19,
                both_17_19,
                same_21_22,
                both_21_22,
                edge_17,
                edge_19,
                edge_21,
                edge_22,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._position_statistics(features))


__all__ = ["ArxCarryPositionStatsPairSetDistinguisher"]
