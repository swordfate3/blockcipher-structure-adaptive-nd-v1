from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_zhang_wang_keras import (
    PresentZhangWangKerasMCNDDistinguisher,
)
from blockcipher_nd.models.structure.spn.token_mixer_pairset import SpnTokenMixerBlock


class PresentNibblePAlignedMCNDDistinguisher(nn.Module):
    """Zhang/Wang MCND backbone fused with a minimal PRESENT SPN cell view."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        blocks: int = 5,
        spn_token_dim: int | None = None,
        spn_mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        initial_kernel_sizes: tuple[int, ...] = (1, 2, 4),
        residual_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("PresentNibblePAlignedMCND expects raw 128-bit ciphertext pairs")
        if input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a multiple of pair_bits")
        if spn_mixer_depth < 1:
            raise ValueError("spn_mixer_depth must be >= 1")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.spn_pair_bits = 128
        self.spn_nibbles_per_pair = 32
        self.spn_token_dim = spn_token_dim or max(16, base_channels * 2)

        self.raw_branch = PresentZhangWangKerasMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits,
            base_channels=base_channels,
            blocks=blocks,
            activation=activation,
            dropout=dropout,
            initial_kernel_sizes=initial_kernel_sizes,
            residual_kernel_size=residual_kernel_size,
        )
        self.spn_cell_encoder = nn.Sequential(
            nn.Linear(4, self.spn_token_dim),
            build_activation(activation),
            build_norm(norm, self.spn_token_dim),
        )
        self.spn_position_embedding = nn.Parameter(
            torch.zeros(1, self.spn_nibbles_per_pair, self.spn_token_dim)
        )
        nn.init.trunc_normal_(self.spn_position_embedding, std=0.02)
        self.spn_mixers = nn.ModuleList(
            [
                SpnTokenMixerBlock(
                    nibbles_per_pair=self.spn_nibbles_per_pair,
                    token_dim=self.spn_token_dim,
                    token_mlp_ratio=token_mlp_ratio,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for _ in range(spn_mixer_depth)
            ]
        )
        self.spn_norm = build_norm(norm, self.spn_token_dim)
        self.spn_pair_projection = nn.Sequential(
            nn.Linear(self.spn_token_dim * 3, max(32, base_channels * 4)),
            build_activation(activation),
            nn.Dropout(dropout),
        )
        self.spn_embedding_dim = max(32, base_channels * 4)
        self.classifier = nn.Sequential(
            build_norm(norm, self.raw_branch.embedding_bits + self.spn_embedding_dim * 2),
            nn.Linear(self.raw_branch.embedding_bits + self.spn_embedding_dim * 2, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), 1),
        )

        inverse_p = [_present_inverse_p_index(index) for index in range(64)]
        self.register_buffer("inverse_p_indices", torch.tensor(inverse_p, dtype=torch.long), persistent=False)

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        raw_embedding = self.raw_branch.encode(features)
        spn_pair_features = self._present_nibble_paligned_view(features.float())
        spn_pair_embeddings = self._encode_spn_pairs(spn_pair_features)
        spn_mean = spn_pair_embeddings.mean(dim=1)
        spn_max = spn_pair_embeddings.max(dim=1).values
        return self.classifier(torch.cat([raw_embedding, spn_mean, spn_max], dim=1))

    def _present_nibble_paligned_view(self, features: torch.Tensor) -> torch.Tensor:
        raw_pairs = features.reshape(features.shape[0], self.pairs_per_sample, 2, 64)
        difference = (raw_pairs[:, :, 0, :] - raw_pairs[:, :, 1, :]).abs()
        aligned_difference = difference.index_select(dim=2, index=self.inverse_p_indices)
        cells = torch.cat([difference, aligned_difference], dim=2).reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.spn_nibbles_per_pair,
            4,
        )
        return cells.permute(0, 1, 3, 2).reshape(features.shape[0], self.pairs_per_sample, self.spn_pair_bits)

    def _encode_spn_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        nibbles = pair_features.reshape(
            pair_features.shape[0] * self.pairs_per_sample,
            4,
            self.spn_nibbles_per_pair,
        ).transpose(1, 2).reshape(
            pair_features.shape[0] * self.pairs_per_sample,
            self.spn_nibbles_per_pair,
            4,
        )
        hidden = self.spn_cell_encoder(nibbles) + self.spn_position_embedding
        for mixer in self.spn_mixers:
            hidden = mixer(hidden)
        hidden = self.spn_norm(hidden)
        mean_embedding = hidden.mean(dim=1)
        max_embedding = hidden.max(dim=1).values
        active_embedding = torch.sum(hidden * nibbles.mean(dim=2, keepdim=True), dim=1) / (
            nibbles.mean(dim=2, keepdim=True).sum(dim=1).clamp_min(1.0)
        )
        projected = self.spn_pair_projection(
            torch.cat([mean_embedding, max_embedding, active_embedding], dim=1)
        )
        return projected.reshape(pair_features.shape[0], self.pairs_per_sample, self.spn_embedding_dim)


def _present_inverse_p_index(target_bit_index: int) -> int:
    source_lsb_index = _present_inverse_p_lsb_index(63 - target_bit_index)
    return 63 - source_lsb_index


def _present_inverse_p_lsb_index(target_lsb_index: int) -> int:
    if target_lsb_index == 63:
        return 63
    return (16 * target_lsb_index) % 63


__all__ = ["PresentNibblePAlignedMCNDDistinguisher"]
