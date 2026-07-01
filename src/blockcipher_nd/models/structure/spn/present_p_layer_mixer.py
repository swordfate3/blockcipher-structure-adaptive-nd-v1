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


def _present_nibble_adjacency_indices() -> list[list[int]]:
    groups: list[list[int]] = []
    inverse_multiplier = pow(16, -1, 63)
    for token_index in range(16):
        output_nibble = 15 - token_index
        source_tokens = set()
        for output_bit in range(output_nibble * 4, output_nibble * 4 + 4):
            source_bit = 63 if output_bit == 63 else inverse_multiplier * output_bit % 63
            source_tokens.add(15 - (source_bit // 4))
        groups.append(sorted(source_tokens))
    return groups


class PresentPLayerMixerBlock(nn.Module):
    """PRESENT-specific token mixer using public P-layer nibble adjacency."""

    def __init__(
        self,
        words_per_pair: int,
        token_dim: int,
        token_mlp_ratio: int = 2,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        p_topology: str = "true",
    ) -> None:
        super().__init__()
        if token_mlp_ratio < 1:
            raise ValueError("PresentPLayerMixerBlock token_mlp_ratio must be >= 1")
        if p_topology not in {"true", "shuffled"}:
            raise ValueError(f"unsupported p_topology: {p_topology}")
        self.words_per_pair = words_per_pair
        self.token_dim = token_dim
        adjacency = _present_nibble_adjacency_indices()
        if p_topology == "shuffled":
            generator = torch.Generator().manual_seed(20260701)
            permutation = torch.randperm(16, generator=generator).tolist()
            adjacency = [
                sorted({permutation[source] for source in sources})
                for sources in adjacency
            ]
        self.register_buffer(
            "p_sources",
            torch.tensor(adjacency, dtype=torch.long),
            persistent=False,
        )
        inverse: list[list[int]] = [[] for _ in range(16)]
        for target, sources in enumerate(adjacency):
            for source in sources:
                inverse[source].append(target)
        max_inverse = max(len(items) for items in inverse)
        inverse_padded = [items + [items[-1]] * (max_inverse - len(items)) for items in inverse]
        self.register_buffer(
            "p_targets",
            torch.tensor(inverse_padded, dtype=torch.long),
            persistent=False,
        )
        channel_hidden = max(token_dim, token_dim * token_mlp_ratio)
        self.local_norm = build_norm(norm, token_dim)
        self.local_mlp = nn.Sequential(
            nn.Linear(token_dim, channel_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, token_dim),
        )
        self.message_norm = build_norm(norm, token_dim * 3)
        self.message_mlp = nn.Sequential(
            nn.Linear(token_dim * 3, channel_hidden),
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

    def _gather_by_nibble(self, hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = hidden.shape
        by_word = hidden.reshape(batch, self.words_per_pair, 16, channels)
        flat_indices = indices.reshape(-1)
        gathered = by_word.index_select(dim=2, index=flat_indices).reshape(
            batch,
            self.words_per_pair,
            indices.shape[0],
            indices.shape[1],
            channels,
        )
        return gathered.mean(dim=3).reshape(batch, tokens, channels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.local_mlp(self.local_norm(hidden))
        p_message = self._gather_by_nibble(hidden, self.p_sources)
        invp_message = self._gather_by_nibble(hidden, self.p_targets)
        message_input = torch.cat([hidden, p_message, invp_message], dim=-1)
        hidden = hidden + self.message_mlp(self.message_norm(message_input))
        hidden = hidden + self.channel_mlp(self.channel_norm(hidden))
        return hidden


class PresentPLayerMixerPairSetDistinguisher(nn.Module):
    """PRESENT-specific pair-set model with public P-layer message passing."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        nibble_bits: int = 4,
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
            raise ValueError("PresentPLayerMixerPairSet input_bits must be a multiple of pair_bits")
        if pair_bits % 64 != 0:
            raise ValueError("PresentPLayerMixerPairSet currently requires 64-bit PRESENT word groups")
        if pair_bits % nibble_bits != 0:
            raise ValueError("PresentPLayerMixerPairSet pair_bits must be a multiple of nibble_bits")
        if mixer_depth < 1:
            raise ValueError("PresentPLayerMixerPairSet mixer_depth must be >= 1")
        if pooling not in {"attention", "attention_mean_max", "mean_max", "gated_attention", "topk_mean", "logsumexp", "topk_logsumexp"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = "SPN"
        self.nibble_bits = nibble_bits
        self.words_per_pair = pair_bits // 64
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
        self.word_embedding = nn.Parameter(torch.zeros(1, self.words_per_pair, 1, self.token_dim))
        self.nibble_embedding = nn.Parameter(torch.zeros(1, 1, 16, self.token_dim))
        nn.init.trunc_normal_(self.word_embedding, std=0.02)
        nn.init.trunc_normal_(self.nibble_embedding, std=0.02)
        self.mixer_blocks = nn.ModuleList(
            [
                PresentPLayerMixerBlock(
                    words_per_pair=self.words_per_pair,
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

    @staticmethod
    def present_bit_to_input_position(bit_index: int, width: int = 64) -> int:
        if bit_index < 0 or bit_index >= width:
            raise ValueError("bit_index out of PRESENT block range")
        return width - 1 - bit_index

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def _encode_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        nibbles = pair_features.reshape(pair_features.shape[0], self.words_per_pair, 16, self.nibble_bits)
        hidden = self.nibble_encoder(nibbles) + self.word_embedding + self.nibble_embedding
        hidden = hidden.reshape(pair_features.shape[0], self.nibbles_per_pair, self.token_dim)
        for block in self.mixer_blocks:
            hidden = block(hidden)
        hidden = self.sequence_norm(hidden)
        mean_embedding = hidden.mean(dim=1)
        max_embedding = hidden.max(dim=1).values
        first_last_delta = hidden[:, -1, :] - hidden[:, 0, :]
        active_weights = nibbles.reshape(pair_features.shape[0], self.nibbles_per_pair, self.nibble_bits).mean(dim=2, keepdim=True)
        active_embedding = torch.sum(hidden * active_weights, dim=1) / active_weights.sum(dim=1).clamp_min(1.0)
        pair_embedding = torch.cat([mean_embedding, max_embedding, first_last_delta, active_embedding], dim=1)
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


__all__ = ["PresentPLayerMixerBlock", "PresentPLayerMixerPairSetDistinguisher"]
