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


class ArxWordMixerBlock(nn.Module):
    """SPECK-style ARX word mixer with public rotation and carry-proxy messages."""

    def __init__(
        self,
        token_dim: int,
        token_mlp_ratio: int = 2,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if token_mlp_ratio < 1:
            raise ValueError("ArxWordMixerBlock token_mlp_ratio must be >= 1")
        channel_hidden = max(token_dim, token_dim * token_mlp_ratio)
        self.local_norm = build_norm(norm, token_dim)
        self.local_mlp = nn.Sequential(
            nn.Linear(token_dim, channel_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, token_dim),
        )
        self.message_norm = build_norm(norm, token_dim * 5)
        self.message_mlp = nn.Sequential(
            nn.Linear(token_dim * 5, channel_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, token_dim),
        )
        self.channel_norm = build_norm(norm, token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, channel_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, token_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        ror7_message: torch.Tensor,
        rol2_message: torch.Tensor,
        carry_proxy: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.local_mlp(self.local_norm(hidden))
        batch, tokens, channels = hidden.shape
        words_per_role = tokens // 2
        peer = hidden.reshape(batch, 2, words_per_role, channels).flip(dims=[1]).reshape(
            batch,
            tokens,
            channels,
        )
        carry_message = carry_proxy * hidden
        message_input = torch.cat([hidden, ror7_message, rol2_message, peer, carry_message], dim=-1)
        hidden = hidden + self.message_mlp(self.message_norm(message_input))
        hidden = hidden + self.channel_mlp(self.channel_norm(hidden))
        return hidden


class ArxWordMixerPairSetDistinguisher(nn.Module):
    """ARX pair-set model preserving 16-bit word, rotation, and carry-proxy structure."""

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
        pooling: str = "topk_logsumexp",
        dropout: float = 0.0,
        top_k: int = 4,
        lse_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("ArxWordMixerPairSet input_bits must be a multiple of pair_bits")
        if pair_bits % 32 != 0:
            raise ValueError("ArxWordMixerPairSet pair_bits must be a multiple of 32-bit ARX words")
        if word_bits != 16:
            raise ValueError("ArxWordMixerPairSet currently supports 16-bit SPECK32 words")
        if mixer_depth < 1:
            raise ValueError("ArxWordMixerPairSet mixer_depth must be >= 1")
        if pooling not in {"attention", "attention_mean_max", "mean_max", "gated_attention", "topk_mean", "logsumexp", "topk_logsumexp"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "ARX"
        self.word_bits = word_bits
        self.arx_words_per_pair = pair_bits // 32
        self.tokens_per_pair = self.arx_words_per_pair * 2
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.mixer_depth = mixer_depth
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
        self.word_role_embedding = nn.Parameter(torch.zeros(1, 2, 1, self.token_dim))
        self.feature_word_embedding = nn.Parameter(torch.zeros(1, 1, self.arx_words_per_pair, self.token_dim))
        nn.init.trunc_normal_(self.word_role_embedding, std=0.02)
        nn.init.trunc_normal_(self.feature_word_embedding, std=0.02)
        self.mixer_blocks = nn.ModuleList(
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
        self.sequence_norm = build_norm(norm, self.token_dim)
        self.pair_embedding_bits = self.token_dim * 5
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
        carry_tokens = carry_bits.unsqueeze(1).expand(-1, 2, -1, -1).reshape(
            words.shape[0],
            self.tokens_per_pair,
            self.word_bits,
        )
        return carry_tokens.mean(dim=-1, keepdim=True)

    def _encode_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        words = pair_features.reshape(pair_features.shape[0], self.arx_words_per_pair, 2, self.word_bits)
        tokens = words.transpose(1, 2).reshape(pair_features.shape[0], 2, self.arx_words_per_pair, self.word_bits)
        ror7_tokens = torch.roll(tokens, shifts=7, dims=-1)
        rol2_tokens = torch.roll(tokens, shifts=-2, dims=-1)
        hidden = self.word_encoder(tokens) + self.word_role_embedding + self.feature_word_embedding
        ror7_message = self.word_encoder(ror7_tokens).reshape(
            pair_features.shape[0],
            self.tokens_per_pair,
            self.token_dim,
        )
        rol2_message = self.word_encoder(rol2_tokens).reshape(
            pair_features.shape[0],
            self.tokens_per_pair,
            self.token_dim,
        )
        hidden = hidden.reshape(pair_features.shape[0], self.tokens_per_pair, self.token_dim)
        carry_proxy = self._carry_proxy(words)
        for block in self.mixer_blocks:
            hidden = block(hidden, ror7_message, rol2_message, carry_proxy)
        hidden = self.sequence_norm(hidden)
        hidden_by_role = hidden.reshape(pair_features.shape[0], 2, self.arx_words_per_pair, self.token_dim)
        mean_embedding = hidden.mean(dim=1)
        max_embedding = hidden.max(dim=1).values
        left_right_delta = hidden_by_role[:, 0].mean(dim=1) - hidden_by_role[:, 1].mean(dim=1)
        carry_embedding = torch.sum(hidden * carry_proxy, dim=1) / carry_proxy.sum(dim=1).clamp_min(1.0)
        first_last_delta = hidden[:, -1, :] - hidden[:, 0, :]
        pair_embedding = torch.cat([mean_embedding, max_embedding, left_right_delta, carry_embedding, first_last_delta], dim=1)
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


__all__ = ["ArxWordMixerBlock", "ArxWordMixerPairSetDistinguisher"]
