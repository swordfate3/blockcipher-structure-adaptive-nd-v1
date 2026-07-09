from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import (
    AttentionPooling,
    EvidencePooling,
    build_activation,
    build_norm,
)
from blockcipher_nd.models.structure.spn.present_p_layer_mixer import (
    _present_p_layer_target_cells,
)


def _active_source_cells() -> list[int]:
    return [15 - active_nibble for active_nibble in range(16)]


def _active_target_mask(graph_mode: str) -> list[list[int]]:
    if graph_mode not in {"true", "shuffled"}:
        raise ValueError("active target mask graph_mode must be true or shuffled")
    permutation = list(range(16))
    if graph_mode == "shuffled":
        generator = torch.Generator().manual_seed(20260709)
        permutation = torch.randperm(16, generator=generator).tolist()
    rows: list[list[int]] = []
    for active_nibble in range(16):
        row = [0 for _ in range(16)]
        for target in _present_p_layer_target_cells(active_nibble):
            row[permutation[target]] = 1
        rows.append(row)
    return rows


def _active_role_ids(graph_mode: str) -> list[list[int]]:
    target_masks = _active_target_mask(graph_mode)
    source_cells = _active_source_cells()
    rows: list[list[int]] = []
    for active_nibble, source_cell in enumerate(source_cells):
        row = [1 if is_target else 0 for is_target in target_masks[active_nibble]]
        row[source_cell] = 2
        rows.append(row)
    return rows


class PresentActiveCellGraphLayer(nn.Module):
    """Per-sample source-target message layer selected by active PRESENT nibble."""

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
            raise ValueError("PresentActiveCellGraphLayer token_mlp_ratio must be >= 1")
        hidden_dim = max(token_dim, token_dim * token_mlp_ratio)
        self.message_norm = build_norm(norm, token_dim * 3)
        self.message_mlp = nn.Sequential(
            nn.Linear(token_dim * 3, hidden_dim),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.local_norm = build_norm(norm, token_dim)
        self.local_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        source_cells: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, cells, channels = hidden.shape
        source_hidden = hidden.gather(
            dim=1,
            index=source_cells[:, None, None].expand(batch, 1, channels),
        ).squeeze(1)
        source_mask = torch.zeros(batch, cells, 1, dtype=hidden.dtype, device=hidden.device)
        source_mask.scatter_(dim=1, index=source_cells[:, None, None], value=1.0)
        mask = target_mask.to(hidden.dtype).unsqueeze(-1)
        target_sum = (hidden * mask).sum(dim=1)
        target_count = mask.sum(dim=1).clamp_min(1.0)
        target_mean = target_sum / target_count
        source_message = source_hidden[:, None, :] * mask
        target_message = target_mean[:, None, :] * source_mask
        hidden = hidden + self.message_mlp(
            self.message_norm(torch.cat([hidden, source_message, target_message], dim=-1))
        )
        hidden = hidden + self.local_mlp(self.local_norm(hidden))
        return hidden


class PresentActiveCellGraphPairSetDistinguisher(nn.Module):
    """Raw-prefix PRESENT cell graph with per-sample active-nibble edges."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 320,
        base_channels: int = 32,
        token_dim: int | None = None,
        graph_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "gelu",
        norm: str = "layernorm",
        pooling: str = "topk_logsumexp",
        dropout: float = 0.0,
        top_k: int = 4,
        lse_temperature: float = 1.0,
        metadata_bits: int = 0,
        graph_mode: str = "true",
    ) -> None:
        super().__init__()
        if graph_mode not in {"true", "shuffled", "metadata_only"}:
            raise ValueError("PresentActiveCellGraph graph_mode must be true, shuffled, or metadata_only")
        if metadata_bits != 16:
            raise ValueError("PresentActiveCellGraph requires 16 active-nibble metadata bits")
        if graph_depth < 1:
            raise ValueError("PresentActiveCellGraph graph_depth must be >= 1")
        if pair_bits % 64 != 0:
            raise ValueError("PresentActiveCellGraph expects whole 64-bit PRESENT words")
        if input_bits <= metadata_bits or (input_bits - metadata_bits) % pair_bits != 0:
            raise ValueError("PresentActiveCellGraph input bits must be pairs plus metadata")
        if pooling not in {"attention", "topk_mean", "logsumexp", "topk_logsumexp"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.input_bits = input_bits
        self.base_input_bits = input_bits - metadata_bits
        self.metadata_bits = metadata_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = self.base_input_bits // pair_bits
        self.words_per_pair = pair_bits // 64
        self.cells_per_word = 16
        self.cell_feature_bits = self.words_per_pair * 4
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.graph_depth = graph_depth
        self.graph_mode = graph_mode
        self.structure = "SPN"
        self.pooling = pooling

        self.cell_encoder = nn.Sequential(
            nn.Linear(self.cell_feature_bits, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.cell_embedding = nn.Parameter(torch.zeros(1, self.cells_per_word, self.token_dim))
        nn.init.trunc_normal_(self.cell_embedding, std=0.02)
        self.active_metadata_projection = nn.Sequential(
            nn.Linear(metadata_bits, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.role_embedding = nn.Embedding(3, self.token_dim)
        nn.init.trunc_normal_(self.role_embedding.weight, std=0.02)
        self.layers = nn.ModuleList(
            [
                PresentActiveCellGraphLayer(
                    token_dim=self.token_dim,
                    token_mlp_ratio=token_mlp_ratio,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for _ in range(graph_depth)
            ]
        )
        self.sequence_norm = build_norm(norm, self.token_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.token_dim * 3, max(self.token_dim, self.token_dim * token_mlp_ratio)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(self.token_dim, self.token_dim * token_mlp_ratio), self.token_dim),
            build_activation(activation),
        )
        self.register_buffer(
            "source_cells",
            torch.tensor(_active_source_cells(), dtype=torch.long),
            persistent=False,
        )
        if graph_mode in {"true", "shuffled"}:
            self.register_buffer(
                "target_masks",
                torch.tensor(_active_target_mask(graph_mode), dtype=torch.bool),
                persistent=False,
            )
            self.register_buffer(
                "role_ids",
                torch.tensor(_active_role_ids(graph_mode), dtype=torch.long),
                persistent=False,
            )
        else:
            self.register_buffer(
                "target_masks",
                torch.zeros(16, self.cells_per_word, dtype=torch.bool),
                persistent=False,
            )
            self.register_buffer(
                "role_ids",
                torch.zeros(16, self.cells_per_word, dtype=torch.long),
                persistent=False,
            )
        self.pair_embedding_bits = self.token_dim * 6
        projected_bits = max(32, base_channels * 4)
        self.pair_projection = nn.Sequential(
            nn.Linear(self.pair_embedding_bits, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), projected_bits),
            build_activation(activation),
        )
        self.projected_pair_embedding_bits = projected_bits
        if pooling in {"topk_mean", "logsumexp", "topk_logsumexp"}:
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
        self.classifier = nn.Sequential(
            build_norm(norm, projected_bits + self.token_dim),
            nn.Linear(projected_bits + self.token_dim, 256),
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

    def _cell_tokens(self, pair_features: torch.Tensor) -> torch.Tensor:
        words = pair_features.reshape(
            pair_features.shape[0],
            self.words_per_pair,
            self.cells_per_word,
            4,
        )
        cells = words.permute(0, 2, 1, 3).reshape(
            pair_features.shape[0],
            self.cells_per_word,
            self.cell_feature_bits,
        )
        return self.cell_encoder(cells) + self.cell_embedding

    def _encode_pairs(
        self,
        pair_features: torch.Tensor,
        active_metadata: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._cell_tokens(pair_features)
        active_indices = active_metadata.float().argmax(dim=1)
        metadata_embedding = self.active_metadata_projection(active_metadata.float())
        source_cells = self.source_cells.to(hidden.device).index_select(dim=0, index=active_indices)
        target_mask = self.target_masks.to(hidden.device).index_select(dim=0, index=active_indices)
        role_ids = self.role_ids.to(hidden.device).index_select(dim=0, index=active_indices)
        if self.graph_mode in {"true", "shuffled"}:
            hidden = hidden + self.role_embedding(role_ids)
            for layer in self.layers:
                hidden = layer(hidden, source_cells, target_mask)
        else:
            for layer in self.layers:
                hidden = hidden + layer.local_mlp(layer.local_norm(hidden))
        hidden = self.sequence_norm(hidden)
        mean_embedding = hidden.mean(dim=1)
        max_embedding = hidden.max(dim=1).values
        source_embedding = hidden.gather(
            dim=1,
            index=source_cells[:, None, None].expand(hidden.shape[0], 1, self.token_dim),
        ).squeeze(1)
        target_weights = target_mask.to(hidden.dtype).unsqueeze(-1)
        target_embedding = (hidden * target_weights).sum(dim=1) / target_weights.sum(dim=1).clamp_min(1.0)
        if self.graph_mode in {"true", "shuffled"}:
            source_for_edges = source_embedding[:, None, :].expand_as(hidden)
            edge_inputs = torch.cat(
                [
                    source_for_edges,
                    hidden,
                    hidden - source_for_edges,
                ],
                dim=-1,
            )
            edge_values = self.edge_encoder(edge_inputs) * target_weights
            edge_embedding = edge_values.sum(dim=1) / target_weights.sum(dim=1).clamp_min(1.0)
        else:
            edge_embedding = torch.zeros_like(source_embedding)
        pair_embedding = torch.cat(
            [
                mean_embedding,
                max_embedding,
                source_embedding,
                target_embedding,
                edge_embedding,
                metadata_embedding,
            ],
            dim=1,
        )
        return self.pair_projection(pair_embedding)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        base_features = features[:, : self.base_input_bits].float()
        active_metadata = features[:, -self.metadata_bits :].float()
        pair_features = base_features.reshape(features.shape[0] * self.pairs_per_sample, self.pair_bits)
        pair_metadata = active_metadata[:, None, :].expand(
            features.shape[0],
            self.pairs_per_sample,
            self.metadata_bits,
        ).reshape(features.shape[0] * self.pairs_per_sample, self.metadata_bits)
        pair_embeddings = self._encode_pairs(pair_features, pair_metadata).reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.projected_pair_embedding_bits,
        )
        attention_embedding, attention_weights = self.attention(pair_embeddings)
        self.last_attention_weights = attention_weights.detach()
        metadata_embedding = self.active_metadata_projection(active_metadata)
        return self.classifier(torch.cat([attention_embedding, metadata_embedding], dim=1))


__all__ = [
    "PresentActiveCellGraphLayer",
    "PresentActiveCellGraphPairSetDistinguisher",
]
