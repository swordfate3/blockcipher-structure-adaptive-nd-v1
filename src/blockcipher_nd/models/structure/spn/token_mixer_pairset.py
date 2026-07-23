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


def _linear_parameter_count(input_dim: int, output_dim: int) -> int:
    return input_dim * output_dim + output_dim


class SpnTokenMixerBlock(nn.Module):
    def __init__(
        self,
        nibbles_per_pair: int,
        token_dim: int,
        token_mlp_ratio: int = 2,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if token_mlp_ratio < 1:
            raise ValueError("SpnTokenMixerBlock token_mlp_ratio must be >= 1")
        token_hidden = max(nibbles_per_pair, nibbles_per_pair * token_mlp_ratio)
        channel_hidden = max(token_dim, token_dim * token_mlp_ratio)
        self.token_norm = build_norm(norm, token_dim)
        self.token_mixer = nn.Sequential(
            nn.Linear(nibbles_per_pair, token_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, nibbles_per_pair),
        )
        self.channel_norm = build_norm(norm, token_dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(token_dim, channel_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, token_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        token_hidden = self.token_norm(features).transpose(1, 2)
        features = features + self.token_mixer(token_hidden).transpose(1, 2)
        features = features + self.channel_mixer(self.channel_norm(features))
        return features


class EquivariantSpnTokenMixerBlock(nn.Module):
    """Parameter-matched cell-permutation-equivariant token mixer."""

    def __init__(
        self,
        nibbles_per_pair: int,
        token_dim: int,
        token_mlp_ratio: int = 2,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if token_mlp_ratio < 1:
            raise ValueError(
                "EquivariantSpnTokenMixerBlock token_mlp_ratio must be >= 1"
            )
        token_hidden = max(nibbles_per_pair, nibbles_per_pair * token_mlp_ratio)
        target_token_parameters = _linear_parameter_count(
            nibbles_per_pair, token_hidden
        ) + _linear_parameter_count(token_hidden, nibbles_per_pair)
        denominator = 3 * token_dim + 1
        equivariant_hidden = max(
            1, (target_token_parameters - token_dim) // denominator
        )
        active_mlp_parameters = _linear_parameter_count(
            token_dim * 2, equivariant_hidden
        ) + _linear_parameter_count(equivariant_hidden, token_dim)
        residual_parameters = target_token_parameters - active_mlp_parameters
        if residual_parameters < 0:
            raise ValueError(
                "cannot parameter-match equivariant token mixer for the requested dimensions"
            )

        self.token_norm = build_norm(norm, token_dim)
        self.equivariant_mixer = nn.Sequential(
            nn.Linear(token_dim * 2, equivariant_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(equivariant_hidden, token_dim),
        )
        self.equivariant_residual = nn.Parameter(
            torch.zeros(residual_parameters)
        )
        channel_hidden = max(token_dim, token_dim * token_mlp_ratio)
        self.channel_norm = build_norm(norm, token_dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(token_dim, channel_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, token_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.token_norm(features)
        global_mean = normalized.mean(dim=1, keepdim=True)
        mixed = self.equivariant_mixer(
            torch.cat([normalized, global_mean.expand_as(normalized)], dim=2)
        )
        mixed = mixed + self._residual_update(normalized, global_mean)
        features = features + mixed
        return features + self.channel_mixer(self.channel_norm(features))

    def _residual_update(
        self,
        normalized: torch.Tensor,
        global_mean: torch.Tensor,
    ) -> torch.Tensor:
        token_dim = normalized.shape[2]
        expanded_mean = global_mean.expand_as(normalized)
        bases = (
            normalized,
            expanded_mean,
            normalized.square(),
            expanded_mean.square(),
            normalized * expanded_mean,
            normalized.abs(),
        )
        result = torch.zeros_like(normalized)
        offset = 0
        group = 0
        while offset < self.equivariant_residual.numel():
            count = min(token_dim, self.equivariant_residual.numel() - offset)
            update = (
                bases[group % len(bases)][..., :count]
                * self.equivariant_residual[offset : offset + count]
            )
            result = result + torch.nn.functional.pad(
                update, (0, token_dim - count)
            )
            offset += count
            group += 1
        return result


class SpnTokenMixerPairSetDistinguisher(nn.Module):
    """SPN pair-set expert with position-preserving nibble token mixing.

    PRESENT-like SPN ciphers combine local 4-bit S-box substitution with a
    position permutation layer.  This expert therefore encodes each nibble as a
    token, adds a learned position embedding, mixes across token positions, and
    only then aggregates multiple ciphertext pairs.
    """

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 192,
        base_channels: int = 32,
        nibble_bits: int = 4,
        token_dim: int | None = None,
        mixer_depth: int = 3,
        token_mlp_ratio: int = 2,
        activation: str = "gelu",
        norm: str = "layernorm",
        pooling: str = "attention_mean_max",
        dropout: float = 0.0,
        top_k: int = 4,
        lse_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("SpnTokenMixerPairSet input_bits must be a multiple of pair_bits")
        if pair_bits % nibble_bits != 0:
            raise ValueError("SpnTokenMixerPairSet pair_bits must be a multiple of nibble_bits")
        if mixer_depth < 1:
            raise ValueError("SpnTokenMixerPairSet mixer_depth must be >= 1")
        if token_mlp_ratio < 1:
            raise ValueError("SpnTokenMixerPairSet token_mlp_ratio must be >= 1")
        if pooling not in {"attention", "attention_mean_max", "mean_max", "gated_attention", "topk_mean", "logsumexp", "topk_logsumexp"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "SPN"
        self.nibble_bits = nibble_bits
        self.nibbles_per_pair = pair_bits // nibble_bits
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.mixer_depth = mixer_depth
        self.token_mlp_ratio = token_mlp_ratio
        self.activation = activation
        self.norm = norm
        self.pooling = pooling
        self.dropout = dropout
        self.top_k = top_k
        self.lse_temperature = lse_temperature

        self.nibble_encoder = nn.Sequential(
            nn.Linear(nibble_bits, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.nibbles_per_pair, self.token_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.mixer_blocks = nn.ModuleList(
            [
                SpnTokenMixerBlock(
                    nibbles_per_pair=self.nibbles_per_pair,
                    token_dim=self.token_dim,
                    token_mlp_ratio=token_mlp_ratio,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for _ in range(mixer_depth)
            ]
        )
        self.sequence_norm = build_norm(norm, self.token_dim)
        self.pair_embedding_bits = self.token_dim * 4
        projected_bits = max(32, base_channels * 4)
        self.pair_projection = nn.Sequential(
            nn.Linear(self.pair_embedding_bits, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), projected_bits),
            build_activation(activation),
        )
        self.projected_pair_embedding_bits = projected_bits
        if pooling == "gated_attention":
            self.attention = GatedAttentionPooling(
                projected_bits,
                hidden_bits=max(32, base_channels * 4),
            )
        elif pooling in {"topk_mean", "logsumexp", "topk_logsumexp"}:
            self.attention = EvidencePooling(
                projected_bits,
                hidden_bits=max(32, base_channels * 4),
                mode=pooling,
                top_k=top_k,
                lse_temperature=lse_temperature,
                activation=activation,
                norm=norm,
            )
        else:
            self.attention = AttentionPooling(
                projected_bits,
                hidden_bits=max(32, base_channels * 4),
                activation=activation,
                norm=norm,
            )
        pooling_multiplier = 3 if pooling == "attention_mean_max" else 2 if pooling == "mean_max" else 1
        self.classifier = nn.Sequential(
            build_norm(norm, projected_bits * pooling_multiplier),
            nn.Linear(projected_bits * pooling_multiplier, 256),
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
        hidden = self.nibble_encoder(nibbles) + self.position_embedding
        for block in self.mixer_blocks:
            hidden = block(hidden)
        hidden = self.sequence_norm(hidden)
        mean_embedding = hidden.mean(dim=1)
        max_embedding = hidden.max(dim=1).values
        first_last_delta = hidden[:, -1, :] - hidden[:, 0, :]
        active_embedding = torch.sum(hidden * nibbles.mean(dim=2, keepdim=True), dim=1) / (
            nibbles.mean(dim=2, keepdim=True).sum(dim=1).clamp_min(1.0)
        )
        pair_embedding = torch.cat(
            [mean_embedding, max_embedding, first_last_delta, active_embedding],
            dim=1,
        )
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




__all__ = [
    "EquivariantSpnTokenMixerBlock",
    "SpnTokenMixerBlock",
    "SpnTokenMixerPairSetDistinguisher",
]
