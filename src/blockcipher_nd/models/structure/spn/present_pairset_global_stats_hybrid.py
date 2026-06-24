from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_trail_mixer import (
    PresentTrailMixerPairSetDistinguisher,
)


class PresentPairSetGlobalStatsHybridDistinguisher(nn.Module):
    """PRESENT pair-set model with an explicit r7-oriented global-statistics branch."""

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
        global_hidden_bits: int | None = None,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("PresentPairSetGlobalStatsHybrid input_bits must be a multiple of pair_bits")
        if pair_bits % 64 != 0:
            raise ValueError("PresentPairSetGlobalStatsHybrid pair_bits must be a multiple of 64-bit PRESENT words")
        if pair_bits % nibble_bits != 0:
            raise ValueError("PresentPairSetGlobalStatsHybrid pair_bits must be a multiple of nibble_bits")
        if input_bits // pair_bits < 2:
            raise ValueError("PresentPairSetGlobalStatsHybrid needs at least two pairs per sample")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "SPN"
        self.nibble_bits = nibble_bits
        self.words_per_pair = pair_bits // 64
        self.cells_per_word = 64 // nibble_bits
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.global_hidden_bits = global_hidden_bits or max(64, base_channels * 8)
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
        self.global_stats_bits = (
            self.words_per_pair * 4
            + self.cells_per_word * 4
            + self.pairs_per_sample * 4
            + 12
        )
        self.global_projection = nn.Sequential(
            build_norm(norm, self.global_stats_bits),
            nn.Linear(self.global_stats_bits, self.global_hidden_bits),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.global_hidden_bits, self.global_hidden_bits),
            build_activation(activation),
        )
        self.fused_embedding_bits = self.trail_pair_embedding_bits * 4 + self.global_hidden_bits
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

    def _global_statistics(self, features: torch.Tensor) -> torch.Tensor:
        return present_global_pairset_statistics(
            features.float(),
            pairs_per_sample=self.pairs_per_sample,
            words_per_pair=self.words_per_pair,
            cells_per_word=self.cells_per_word,
            nibble_bits=self.nibble_bits,
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
        trail_delta = trail_embeddings[:, -1] - trail_embeddings[:, 0]
        global_embedding = self.global_projection(self._global_statistics(features))
        return self.classifier(
            torch.cat([trail_mean, trail_max, trail_std, trail_delta, global_embedding], dim=1)
        )


class PresentPairSetGlobalStatsDistinguisher(nn.Module):
    """PRESENT pair-set classifier using only r7-oriented global statistics."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 2496,
        base_channels: int = 32,
        nibble_bits: int = 4,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        global_hidden_bits: int | None = None,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("PresentPairSetGlobalStats input_bits must be a multiple of pair_bits")
        if pair_bits % 64 != 0:
            raise ValueError("PresentPairSetGlobalStats pair_bits must be a multiple of 64-bit PRESENT words")
        if pair_bits % nibble_bits != 0:
            raise ValueError("PresentPairSetGlobalStats pair_bits must be a multiple of nibble_bits")
        if input_bits // pair_bits < 2:
            raise ValueError("PresentPairSetGlobalStats needs at least two pairs per sample")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "SPN"
        self.nibble_bits = nibble_bits
        self.words_per_pair = pair_bits // 64
        self.cells_per_word = 64 // nibble_bits
        self.global_hidden_bits = global_hidden_bits or max(64, base_channels * 8)
        self.global_stats_bits = present_global_stats_feature_bits(
            self.words_per_pair,
            self.cells_per_word,
            self.pairs_per_sample,
        )
        classifier_hidden = max(64, base_channels * 4)
        self.classifier = nn.Sequential(
            build_norm(norm, self.global_stats_bits),
            nn.Linear(self.global_stats_bits, self.global_hidden_bits),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.global_hidden_bits, classifier_hidden),
            build_activation(activation),
            nn.Linear(classifier_hidden, 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        stats = present_global_pairset_statistics(
            features.float(),
            pairs_per_sample=self.pairs_per_sample,
            words_per_pair=self.words_per_pair,
            cells_per_word=self.cells_per_word,
            nibble_bits=self.nibble_bits,
        )
        return self.classifier(stats)


def present_global_stats_feature_bits(
    words_per_pair: int,
    cells_per_word: int,
    pairs_per_sample: int,
) -> int:
    return words_per_pair * 4 + cells_per_word * 4 + pairs_per_sample * 4 + 12


def present_global_pairset_statistics(
    features: torch.Tensor,
    *,
    pairs_per_sample: int,
    words_per_pair: int,
    cells_per_word: int,
    nibble_bits: int,
) -> torch.Tensor:
    batch_size = features.shape[0]
    cells = features.reshape(
        batch_size,
        pairs_per_sample,
        words_per_pair,
        cells_per_word,
        nibble_bits,
    )
    cell_activity = cells.mean(dim=-1)
    word_activity = cell_activity.mean(dim=-1)
    pair_activity = cell_activity.mean(dim=(2, 3))

    word_mean = word_activity.mean(dim=1)
    word_std = word_activity.std(dim=1, unbiased=False)
    word_delta = word_activity[:, -1] - word_activity[:, 0]
    word_span = word_activity.amax(dim=1) - word_activity.amin(dim=1)

    cell_mean = cell_activity.mean(dim=(1, 2))
    cell_std = cell_activity.std(dim=(1, 2), unbiased=False)
    first_last_cell_delta = cell_activity[:, -1].mean(dim=1) - cell_activity[:, 0].mean(dim=1)
    even_odd_cell_delta = (
        cell_activity[:, :, :, ::2].mean(dim=(1, 2))
        - cell_activity[:, :, :, 1::2].mean(dim=(1, 2))
    )

    pair_centered = pair_activity - pair_activity.mean(dim=1, keepdim=True)
    pair_stats = torch.cat(
        [
            pair_activity,
            pair_centered,
            pair_activity - pair_activity[:, :1],
            pair_activity[:, -1:] - pair_activity,
        ],
        dim=1,
    )

    global_stats = torch.stack(
        [
            cell_activity.mean(dim=(1, 2, 3)),
            cell_activity.std(dim=(1, 2, 3), unbiased=False),
            word_activity.mean(dim=(1, 2)),
            word_activity.std(dim=(1, 2), unbiased=False),
            pair_activity.mean(dim=1),
            pair_activity.std(dim=1, unbiased=False),
            pair_activity[:, -1] - pair_activity[:, 0],
            word_activity[:, :, -1].mean(dim=1) - word_activity[:, :, 0].mean(dim=1),
            cell_activity[:, :, :, -1].mean(dim=(1, 2)) - cell_activity[:, :, :, 0].mean(dim=(1, 2)),
            cell_activity[:, :, ::2].mean(dim=(1, 2, 3)) - cell_activity[:, :, 1::2].mean(dim=(1, 2, 3)),
        ],
        dim=1,
    )
    global_abs_stats = torch.abs(global_stats)

    return torch.cat(
        [
            word_mean,
            word_std,
            word_delta,
            word_span,
            cell_mean,
            cell_std,
            first_last_cell_delta,
            even_odd_cell_delta,
            pair_stats,
            global_stats,
            global_abs_stats,
        ],
        dim=1,
    )


__all__ = [
    "PresentPairSetGlobalStatsDistinguisher",
    "PresentPairSetGlobalStatsHybridDistinguisher",
    "present_global_pairset_statistics",
    "present_global_stats_feature_bits",
]
