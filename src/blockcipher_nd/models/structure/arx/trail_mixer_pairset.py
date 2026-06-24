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
from blockcipher_nd.models.structure.arx.word_mixer_pairset import ArxWordMixerBlock


class ArxTrailMixerPairSetDistinguisher(nn.Module):
    """SPECK pair-set model for public RX/partial-inverse trail-word features."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 352,
        base_channels: int = 32,
        word_bits: int = 16,
        token_dim: int | None = None,
        mixer_depth: int = 3,
        role_mixer_depth: int = 2,
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
            raise ValueError("ArxTrailMixerPairSet input_bits must be a multiple of pair_bits")
        if pair_bits % 32 != 0:
            raise ValueError("ArxTrailMixerPairSet pair_bits must be a multiple of 32-bit ARX feature words")
        if word_bits != 16:
            raise ValueError("ArxTrailMixerPairSet currently supports 16-bit SPECK32 words")
        if mixer_depth < 1:
            raise ValueError("ArxTrailMixerPairSet mixer_depth must be >= 1")
        if role_mixer_depth < 1:
            raise ValueError("ArxTrailMixerPairSet role_mixer_depth must be >= 1")
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
        self.tokens_per_pair = self.feature_words_per_pair * 2
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.mixer_depth = mixer_depth
        self.role_mixer_depth = role_mixer_depth
        self.token_mlp_ratio = token_mlp_ratio
        self.activation = activation
        self.norm = norm
        self.pooling = pooling
        self.dropout = dropout
        self.top_k = top_k
        self.lse_temperature = lse_temperature

        self.word_encoder = nn.Sequential(
            nn.Linear(word_bits, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.side_embedding = nn.Parameter(torch.zeros(1, 2, 1, self.token_dim))
        self.feature_word_embedding = nn.Parameter(torch.zeros(1, 1, self.feature_words_per_pair, self.token_dim))
        nn.init.trunc_normal_(self.side_embedding, std=0.02)
        nn.init.trunc_normal_(self.feature_word_embedding, std=0.02)

        self.arx_blocks = nn.ModuleList(
            [
                ArxWordMixerBlock(
                    token_dim=self.token_dim,
                    token_mlp_ratio=token_mlp_ratio,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for _ in range(mixer_depth)
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=max(1, min(8, self.token_dim // 8)),
            dim_feedforward=max(self.token_dim * token_mlp_ratio, self.token_dim),
            dropout=dropout,
            activation="gelu" if activation == "gelu" else "relu",
            batch_first=True,
            norm_first=True,
        )
        self.role_mixer = nn.TransformerEncoder(encoder_layer, num_layers=role_mixer_depth)
        self.sequence_norm = build_norm(norm, self.token_dim)

        self.pair_embedding_bits = self.token_dim * 7
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

    def _carry_proxy(self, words: torch.Tensor) -> torch.Tensor:
        left = words[:, :, 0, :]
        right = words[:, :, 1, :]
        carry_bits = left * right
        return carry_bits.unsqueeze(1).expand(-1, 2, -1, -1).reshape(
            words.shape[0],
            self.tokens_per_pair,
            self.word_bits,
        ).mean(dim=-1, keepdim=True)

    def _encode_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        words = pair_features.reshape(pair_features.shape[0], self.feature_words_per_pair, 2, self.word_bits)
        tokens = words.transpose(1, 2)
        hidden = self.word_encoder(tokens) + self.side_embedding + self.feature_word_embedding
        ror7_message = self.word_encoder(torch.roll(tokens, shifts=7, dims=-1)).reshape(
            pair_features.shape[0],
            self.tokens_per_pair,
            self.token_dim,
        )
        rol2_message = self.word_encoder(torch.roll(tokens, shifts=-2, dims=-1)).reshape(
            pair_features.shape[0],
            self.tokens_per_pair,
            self.token_dim,
        )
        hidden = hidden.reshape(pair_features.shape[0], self.tokens_per_pair, self.token_dim)
        carry_proxy = self._carry_proxy(words)
        for block in self.arx_blocks:
            hidden = block(hidden, ror7_message, rol2_message, carry_proxy)
        hidden = self.sequence_norm(hidden)

        by_side = hidden.reshape(pair_features.shape[0], 2, self.feature_words_per_pair, self.token_dim)
        role_tokens = by_side.mean(dim=1)
        role_tokens = self.role_mixer(role_tokens)

        token_mean = hidden.mean(dim=1)
        token_max = hidden.max(dim=1).values
        role_mean = role_tokens.mean(dim=1)
        role_max = role_tokens.max(dim=1).values
        side_delta = by_side[:, 0].mean(dim=1) - by_side[:, 1].mean(dim=1)
        trail_delta = role_tokens[:, -1, :] - role_tokens[:, 0, :]
        active_weights = words.reshape(
            pair_features.shape[0],
            self.tokens_per_pair,
            self.word_bits,
        ).mean(dim=2, keepdim=True)
        active_embedding = torch.sum(hidden * active_weights, dim=1) / active_weights.sum(dim=1).clamp_min(1.0)
        pair_embedding = torch.cat(
            [token_mean, token_max, active_embedding, role_mean, role_max, side_delta, trail_delta],
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


__all__ = ["ArxTrailMixerPairSetDistinguisher"]
