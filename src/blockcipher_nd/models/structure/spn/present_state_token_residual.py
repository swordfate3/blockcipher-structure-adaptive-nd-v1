from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import AttentionPooling, build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_trail_position_stats import (
    PresentTrailPositionStatsPairSetDistinguisher,
)


class PresentStateTokenResidualDistinguisher(nn.Module):
    """Tokenized PRESENT span-stat expert for residual-correction screens."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 2496,
        base_channels: int = 32,
        nibble_bits: int = 4,
        trail_depth: int = 4,
        trail_words_per_depth: int = 9,
        token_dim: int | None = None,
        hidden_bits: int | None = None,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pair_bits % 64 != 0:
            raise ValueError("PresentStateTokenResidual pair_bits must be a multiple of 64")
        if 64 % nibble_bits != 0:
            raise ValueError("PresentStateTokenResidual nibble_bits must divide 64")
        words_per_pair = pair_bits // 64
        cells_per_word = 64 // nibble_bits
        prefix_words = words_per_pair - trail_depth * trail_words_per_depth
        expected_bits = PresentTrailPositionStatsPairSetDistinguisher.position_stats_feature_bits(
            words_per_pair,
            cells_per_word,
            trail_depth,
            prefix_words,
            trail_words_per_depth,
        )
        if input_bits != expected_bits:
            raise ValueError(
                "PresentStateTokenResidual expects trail_position_stats feature layout "
                f"with {expected_bits} features, got {input_bits}"
            )
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.structure = "SPN"
        self.nibble_bits = nibble_bits
        self.words_per_pair = words_per_pair
        self.cells_per_word = cells_per_word
        self.trail_depth = trail_depth
        self.trail_words_per_depth = trail_words_per_depth
        self.prefix_words = prefix_words
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.hidden_bits = hidden_bits or max(64, base_channels * 4)
        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        spec = _span_token_spec(
            words_per_pair=words_per_pair,
            cells_per_word=cells_per_word,
            trail_depth=trail_depth,
            trail_words_per_depth=trail_words_per_depth,
        )
        self.register_buffer("span_feature_indices", spec["feature_indices"], persistent=False)
        self.register_buffer("span_family_ids", spec["family_ids"], persistent=False)
        self.register_buffer("span_depth_ids", spec["depth_ids"], persistent=False)
        self.register_buffer("span_word_ids", spec["word_ids"], persistent=False)
        self.register_buffer("span_cell_ids", spec["cell_ids"], persistent=False)
        self.selected_span_feature_bits = int(spec["feature_indices"].numel())

        self.value_encoder = nn.Sequential(
            nn.Linear(1, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.family_embedding = nn.Embedding(5, self.token_dim)
        self.depth_embedding = nn.Embedding(trail_depth + 1, self.token_dim)
        self.word_embedding = nn.Embedding(words_per_pair + 1, self.token_dim)
        self.cell_embedding = nn.Embedding(cells_per_word + 1, self.token_dim)
        self.token_projection = nn.Sequential(
            build_norm(norm, self.token_dim),
            nn.Linear(self.token_dim, self.hidden_bits),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_bits, self.token_dim),
            build_activation(activation),
        )
        self.attention = AttentionPooling(
            self.token_dim,
            hidden_bits=self.hidden_bits,
            activation=activation,
            norm=norm,
        )
        self.classifier = nn.Sequential(
            build_norm(norm, self.token_dim * 3),
            nn.Linear(self.token_dim * 3, self.hidden_bits),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_bits, 1),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def _span_tokens(self, features: torch.Tensor) -> torch.Tensor:
        selected = features.index_select(1, self.span_feature_indices).float().unsqueeze(-1)
        tokens = self.value_encoder(selected)
        tokens = tokens + self.family_embedding(self.span_family_ids)
        tokens = tokens + self.depth_embedding(self.span_depth_ids)
        tokens = tokens + self.word_embedding(self.span_word_ids)
        tokens = tokens + self.cell_embedding(self.span_cell_ids)
        return self.token_projection(tokens)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} features, got {tuple(features.shape)}")
        tokens = self._span_tokens(features)
        attention_embedding, attention_weights = self.attention(tokens)
        self.last_attention_weights = attention_weights.detach()
        mean_embedding = tokens.mean(dim=1)
        max_embedding = tokens.max(dim=1).values
        return self.classifier(torch.cat([attention_embedding, mean_embedding, max_embedding], dim=1))


def _span_token_spec(
    *,
    words_per_pair: int,
    cells_per_word: int,
    trail_depth: int,
    trail_words_per_depth: int,
) -> dict[str, torch.Tensor]:
    offsets = _position_stat_offsets(
        words_per_pair=words_per_pair,
        cells_per_word=cells_per_word,
        trail_depth=trail_depth,
        trail_words_per_depth=trail_words_per_depth,
    )
    feature_indices: list[int] = []
    family_ids: list[int] = []
    depth_ids: list[int] = []
    word_ids: list[int] = []
    cell_ids: list[int] = []

    def add(index: int, *, family: int, depth: int = 0, word: int = 0, cell: int = 0) -> None:
        feature_indices.append(index)
        family_ids.append(family)
        depth_ids.append(depth)
        word_ids.append(word)
        cell_ids.append(cell)

    for word in range(words_per_pair):
        add(offsets["word_span"] + word, family=0, word=word + 1)
    for cell in range(cells_per_word):
        add(offsets["cell_span"] + cell, family=1, cell=cell + 1)
    for depth in range(trail_depth):
        for word in range(trail_words_per_depth):
            for cell in range(cells_per_word):
                index = offsets["depth_word_cell_span"] + (
                    (depth * trail_words_per_depth + word) * cells_per_word + cell
                )
                add(index, family=2, depth=depth + 1, word=word + 1, cell=cell + 1)
    for depth in range(trail_depth):
        for cell in range(cells_per_word):
            index = offsets["depth_cell_span"] + depth * cells_per_word + cell
            add(index, family=3, depth=depth + 1, cell=cell + 1)
    for depth in range(trail_depth):
        for word in range(trail_words_per_depth):
            index = offsets["depth_word_span"] + depth * trail_words_per_depth + word
            add(index, family=4, depth=depth + 1, word=word + 1)

    return {
        "feature_indices": torch.tensor(feature_indices, dtype=torch.long),
        "family_ids": torch.tensor(family_ids, dtype=torch.long),
        "depth_ids": torch.tensor(depth_ids, dtype=torch.long),
        "word_ids": torch.tensor(word_ids, dtype=torch.long),
        "cell_ids": torch.tensor(cell_ids, dtype=torch.long),
    }


def _position_stat_offsets(
    *,
    words_per_pair: int,
    cells_per_word: int,
    trail_depth: int,
    trail_words_per_depth: int,
) -> dict[str, int]:
    offset = 0
    offset += words_per_pair * cells_per_word * 2
    offset += words_per_pair
    offset += words_per_pair
    offset += words_per_pair
    word_span = offset
    offset += words_per_pair
    offset += cells_per_word
    offset += cells_per_word
    offset += cells_per_word
    cell_span = offset
    offset += cells_per_word
    offset += trail_depth * trail_words_per_depth * cells_per_word
    offset += trail_depth * trail_words_per_depth * cells_per_word
    depth_word_cell_span = offset
    offset += trail_depth * trail_words_per_depth * cells_per_word
    offset += trail_depth * cells_per_word
    offset += trail_depth * cells_per_word
    offset += trail_depth * cells_per_word
    depth_cell_span = offset
    offset += trail_depth * cells_per_word
    offset += trail_depth * trail_words_per_depth
    offset += trail_depth * trail_words_per_depth
    offset += trail_depth * trail_words_per_depth
    depth_word_span = offset
    return {
        "word_span": word_span,
        "cell_span": cell_span,
        "depth_word_cell_span": depth_word_cell_span,
        "depth_cell_span": depth_cell_span,
        "depth_word_span": depth_word_span,
    }


__all__ = ["PresentStateTokenResidualDistinguisher"]
