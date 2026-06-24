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


class CarryRunMixerBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        kernel_size: int = 3,
        token_mlp_ratio: int = 2,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = build_norm(norm, token_dim)
        self.conv = nn.Conv1d(token_dim, token_dim, kernel_size=kernel_size, padding=padding)
        self.mlp = nn.Sequential(
            build_norm(norm, token_dim),
            nn.Linear(token_dim, max(token_dim, token_dim * token_mlp_ratio)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(token_dim, token_dim * token_mlp_ratio), token_dim),
            nn.Dropout(dropout),
        )
        self.activation = build_activation(activation)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mixed = self.conv(self.norm(tokens).transpose(1, 2)).transpose(1, 2)
        tokens = tokens + self.activation(mixed)
        return tokens + self.mlp(tokens)


class ArxCarryRunMixerPairSetDistinguisher(nn.Module):
    """SPECK carry-chain sequence model over carrychain-plus pair sets."""

    carry_role_indices = tuple(range(17, 23))
    support_role_indices = tuple(range(11, 17))

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 736,
        base_channels: int = 32,
        word_bits: int = 16,
        token_dim: int | None = None,
        mixer_depth: int = 3,
        token_mlp_ratio: int = 2,
        kernel_size: int = 3,
        activation: str = "gelu",
        norm: str = "layernorm",
        pooling: str = "topk_logsumexp",
        dropout: float = 0.0,
        top_k: int = 4,
        lse_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("ArxCarryRunMixer input_bits must be a multiple of pair_bits")
        if pair_bits % 32 != 0:
            raise ValueError("ArxCarryRunMixer pair_bits must be a multiple of 32-bit ARX feature words")
        if word_bits != 16:
            raise ValueError("ArxCarryRunMixer currently supports 16-bit SPECK32 words")
        if pair_bits // 32 < 23:
            raise ValueError("ArxCarryRunMixer requires carrychain-plus SPECK32 feature words")
        if input_bits // pair_bits < 2:
            raise ValueError("ArxCarryRunMixer needs at least two pairs per sample")
        if mixer_depth < 1:
            raise ValueError("ArxCarryRunMixer mixer_depth must be >= 1")
        if kernel_size % 2 != 1:
            raise ValueError("ArxCarryRunMixer kernel_size must be odd")
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
        self.structure = "ARX"
        self.word_bits = word_bits
        self.feature_words_per_pair = pair_bits // 32
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.mixer_depth = mixer_depth
        self.token_mlp_ratio = token_mlp_ratio
        self.kernel_size = kernel_size
        self.activation = activation
        self.norm = norm
        self.pooling = pooling
        self.dropout = dropout
        self.top_k = top_k
        self.lse_temperature = lse_temperature
        self.sequence_feature_channels = (
            (len(self.carry_role_indices) + len(self.support_role_indices)) * 2
            + len(self.carry_role_indices) * 2 * 4
            + 12
        )

        self.token_encoder = nn.Sequential(
            nn.Linear(self.sequence_feature_channels, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, word_bits, self.token_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            [
                CarryRunMixerBlock(
                    token_dim=self.token_dim,
                    kernel_size=kernel_size,
                    token_mlp_ratio=token_mlp_ratio,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for _ in range(mixer_depth)
            ]
        )
        self.sequence_norm = build_norm(norm, self.token_dim)

        self.pair_embedding_bits = self.token_dim * 7 + 8
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
            self.attention = GatedAttentionPooling(projected_bits, hidden_bits=max(32, base_channels * 4))
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
            nn.Linear(projected_bits * pooling_multiplier, max(128, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(128, base_channels * 8), max(64, base_channels * 4)),
            build_activation(activation),
            nn.Linear(max(64, base_channels * 4), 1),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def _sequence_features(self, pair_features: torch.Tensor) -> torch.Tensor:
        words = pair_features.float().reshape(
            pair_features.shape[0],
            self.feature_words_per_pair,
            2,
            self.word_bits,
        )
        carry = words[:, self.carry_role_indices].permute(0, 3, 1, 2)
        support = words[:, self.support_role_indices].permute(0, 3, 1, 2)

        padded = torch.nn.functional.pad(carry, (0, 0, 0, 0, 1, 1), value=0.0)
        prev_bits = padded[:, : self.word_bits]
        cur_bits = padded[:, 1 : self.word_bits + 1]
        next_bits = padded[:, 2 : self.word_bits + 2]
        run_start = (cur_bits > 0.5).float() * (prev_bits <= 0.5).float()
        run_end = (cur_bits > 0.5).float() * (next_bits <= 0.5).float()
        run_continue = (cur_bits > 0.5).float() * (prev_bits > 0.5).float()
        run_edge = (next_bits - cur_bits).abs()

        role_17 = words[:, 17]
        role_19 = torch.roll(words[:, 19], shifts=7, dims=-1)
        role_21 = words[:, 21]
        role_22 = torch.roll(words[:, 22], shifts=2, dims=-1)
        rotation = torch.stack(
            [
                (role_17 == role_19).float(),
                role_17 * role_19,
                (role_21 == role_22).float(),
                role_21 * role_22,
            ],
            dim=2,
        ).permute(0, 3, 2, 1)

        bit_pos = torch.linspace(0.0, 1.0, self.word_bits, device=pair_features.device)
        bit_pos = bit_pos.reshape(1, self.word_bits, 1).expand(pair_features.shape[0], -1, 1)
        bands = torch.stack(
            [
                (bit_pos.squeeze(-1) < 5 / 16).float(),
                ((bit_pos.squeeze(-1) >= 5 / 16) & (bit_pos.squeeze(-1) < 11 / 16)).float(),
                (bit_pos.squeeze(-1) >= 11 / 16).float(),
                torch.sin(bit_pos.squeeze(-1) * torch.pi),
            ],
            dim=-1,
        )

        return torch.cat(
            [
                carry.flatten(2),
                support.flatten(2),
                run_start.flatten(2),
                run_continue.flatten(2),
                run_end.flatten(2),
                run_edge.flatten(2),
                rotation.flatten(2),
                bands,
            ],
            dim=2,
        )

    def _run_summary(self, sequence_features: torch.Tensor) -> torch.Tensor:
        carry_channels = len(self.carry_role_indices) * 2
        carry = sequence_features[:, :, :carry_channels]
        run_channels = len(self.carry_role_indices) * 2
        run_start = sequence_features[:, :, carry_channels * 2 : carry_channels * 2 + run_channels]
        run_continue = sequence_features[
            :,
            :,
            carry_channels * 2 + run_channels : carry_channels * 2 + run_channels * 2,
        ]
        return torch.cat(
            [
                carry.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                carry.std(dim=(1, 2), unbiased=False).unsqueeze(1),
                run_start.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                run_continue.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                sequence_features[:, :5].mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                sequence_features[:, 5:11].mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                sequence_features[:, 11:].mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                sequence_features[:, -1] .mean(dim=1, keepdim=True)
                - sequence_features[:, 0].mean(dim=1, keepdim=True),
            ],
            dim=1,
        )

    def _encode_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        sequence_features = self._sequence_features(pair_features)
        hidden = self.token_encoder(sequence_features) + self.position_embedding
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.sequence_norm(hidden)

        token_mean = hidden.mean(dim=1)
        token_max = hidden.max(dim=1).values
        token_std = hidden.std(dim=1, unbiased=False)
        first_last = hidden[:, -1] - hidden[:, 0]
        low_band = hidden[:, :5].mean(dim=1)
        mid_band = hidden[:, 5:11].mean(dim=1)
        high_band = hidden[:, 11:].mean(dim=1)
        run_summary = self._run_summary(sequence_features)
        pair_embedding = torch.cat(
            [token_mean, token_max, token_std, first_last, low_band, mid_band, high_band, run_summary],
            dim=1,
        )
        return self.pair_projection(pair_embedding)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        pair_features = features.float().reshape(features.shape[0] * self.pairs_per_sample, self.pair_bits)
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


__all__ = ["ArxCarryRunMixerPairSetDistinguisher", "CarryRunMixerBlock"]
