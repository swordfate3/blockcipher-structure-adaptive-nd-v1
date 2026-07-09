from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

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


def _persistent_edge_indices(graph_mode: str) -> tuple[list[int], list[int]]:
    if graph_mode not in {"true", "shuffled"}:
        raise ValueError("persistent edge graph_mode must be true or shuffled")
    permutation = list(range(16))
    if graph_mode == "shuffled":
        generator = torch.Generator().manual_seed(20260709)
        permutation = torch.randperm(16, generator=generator).tolist()
    sources: list[int] = []
    targets: list[int] = []
    for source_nibble in range(16):
        source_cell = 15 - source_nibble
        for target in _present_p_layer_target_cells(source_nibble):
            sources.append(source_cell)
            targets.append(permutation[target])
    return sources, targets


def _persistent_edge_role_ids(graph_mode: str) -> list[list[int]]:
    sources, targets = _persistent_edge_indices(graph_mode)
    source_cells = _active_source_cells()
    target_masks = _active_target_mask(graph_mode)
    rows: list[list[int]] = []
    for active_nibble, source_cell in enumerate(source_cells):
        row: list[int] = []
        for edge_source, edge_target in zip(sources, targets):
            if edge_source == source_cell:
                row.append(2)
            elif edge_target == source_cell:
                row.append(3)
            elif target_masks[active_nibble][edge_target]:
                row.append(1)
            else:
                row.append(0)
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
        edge_mode: str = "active_only",
        cross_pair_consistency: str = "none",
        active_metadata_fusion: str = "direct",
        topology_auxiliary_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if graph_mode not in {"true", "shuffled", "metadata_only"}:
            raise ValueError("PresentActiveCellGraph graph_mode must be true, shuffled, or metadata_only")
        if edge_mode not in {"active_only", "persistent"}:
            raise ValueError("PresentActiveCellGraph edge_mode must be active_only or persistent")
        if cross_pair_consistency not in {"none", "edge_mean_absdev"}:
            raise ValueError("PresentActiveCellGraph cross_pair_consistency must be none or edge_mean_absdev")
        if cross_pair_consistency != "none" and edge_mode != "persistent":
            raise ValueError("PresentActiveCellGraph cross-pair consistency requires persistent edge_mode")
        if active_metadata_fusion not in {"direct", "coordinate_only"}:
            raise ValueError("PresentActiveCellGraph active_metadata_fusion must be direct or coordinate_only")
        if topology_auxiliary_scale < 0.0:
            raise ValueError("PresentActiveCellGraph topology_auxiliary_scale must be non-negative")
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
        self.edge_mode = edge_mode
        self.cross_pair_consistency = cross_pair_consistency
        self.active_metadata_fusion = active_metadata_fusion
        self.topology_auxiliary_scale = topology_auxiliary_scale
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
        self.persistent_edge_role_embedding = nn.Embedding(4, self.token_dim)
        nn.init.trunc_normal_(self.persistent_edge_role_embedding.weight, std=0.02)
        self.persistent_edge_encoder = nn.Sequential(
            nn.Linear(self.token_dim * 4, max(self.token_dim, self.token_dim * token_mlp_ratio)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(self.token_dim, self.token_dim * token_mlp_ratio), self.token_dim),
            build_activation(activation),
        )
        self.persistent_edge_norm = build_norm(norm, self.token_dim)
        self.cross_pair_projection = nn.Sequential(
            nn.Linear(self.token_dim * 3, max(self.token_dim, self.token_dim * token_mlp_ratio)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(self.token_dim, self.token_dim * token_mlp_ratio), self.token_dim),
            build_activation(activation),
        )
        self.topology_auxiliary_head = nn.Sequential(
            nn.Linear(self.token_dim, max(self.token_dim, self.token_dim * token_mlp_ratio)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(self.token_dim, self.token_dim * token_mlp_ratio), 1),
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
            edge_sources, edge_targets = _persistent_edge_indices(graph_mode)
            self.register_buffer(
                "persistent_edge_sources",
                torch.tensor(edge_sources, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "persistent_edge_targets",
                torch.tensor(edge_targets, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "persistent_edge_role_ids",
                torch.tensor(_persistent_edge_role_ids(graph_mode), dtype=torch.long),
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
            self.register_buffer(
                "persistent_edge_sources",
                torch.zeros(1, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "persistent_edge_targets",
                torch.zeros(1, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "persistent_edge_role_ids",
                torch.zeros(16, 1, dtype=torch.long),
                persistent=False,
            )
        true_sources, true_targets = _persistent_edge_indices("true")
        shuffled_sources, shuffled_targets = _persistent_edge_indices("shuffled")
        self.register_buffer(
            "topology_auxiliary_true_sources",
            torch.tensor(true_sources, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "topology_auxiliary_true_targets",
            torch.tensor(true_targets, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "topology_auxiliary_true_role_ids",
            torch.tensor(_persistent_edge_role_ids("true"), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "topology_auxiliary_shuffled_sources",
            torch.tensor(shuffled_sources, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "topology_auxiliary_shuffled_targets",
            torch.tensor(shuffled_targets, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "topology_auxiliary_shuffled_role_ids",
            torch.tensor(_persistent_edge_role_ids("shuffled"), dtype=torch.long),
            persistent=False,
        )
        self.pair_embedding_bits = self.token_dim * (6 if self.active_metadata_fusion == "direct" else 5)
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
        classifier_input_bits = projected_bits
        if self.active_metadata_fusion == "direct":
            classifier_input_bits += self.token_dim
        if self.cross_pair_consistency != "none":
            classifier_input_bits += self.token_dim
        self.classifier = nn.Sequential(
            build_norm(norm, classifier_input_bits),
            nn.Linear(classifier_input_bits, 256),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            build_activation(activation),
            nn.Linear(128, 1),
        )
        self.last_attention_weights: torch.Tensor | None = None
        self.last_auxiliary_loss: torch.Tensor | None = None

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

    def _persistent_edge_tokens(
        self,
        hidden: torch.Tensor,
        active_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.graph_mode not in {"true", "shuffled"}:
            edge_count = self.persistent_edge_sources.numel()
            return hidden.new_zeros(hidden.shape[0], edge_count, self.token_dim)
        return self._edge_tokens_from_buffers(
            hidden=hidden,
            active_indices=active_indices,
            edge_sources=self.persistent_edge_sources,
            edge_targets=self.persistent_edge_targets,
            edge_role_ids=self.persistent_edge_role_ids,
        )

    def _edge_tokens_from_buffers(
        self,
        hidden: torch.Tensor,
        active_indices: torch.Tensor,
        edge_sources: torch.Tensor,
        edge_targets: torch.Tensor,
        edge_role_ids: torch.Tensor,
    ) -> torch.Tensor:
        edge_sources = edge_sources.to(hidden.device)
        edge_targets = edge_targets.to(hidden.device)
        source_tokens = hidden.index_select(dim=1, index=edge_sources)
        target_tokens = hidden.index_select(dim=1, index=edge_targets)
        role_ids = edge_role_ids.to(hidden.device).index_select(dim=0, index=active_indices)
        role_tokens = self.persistent_edge_role_embedding(role_ids)
        edge_inputs = torch.cat(
            [
                source_tokens,
                target_tokens,
                target_tokens - source_tokens,
                source_tokens * target_tokens,
            ],
            dim=-1,
        )
        return self.persistent_edge_norm(self.persistent_edge_encoder(edge_inputs) + role_tokens)

    def _topology_auxiliary_loss(
        self,
        hidden: torch.Tensor,
        active_indices: torch.Tensor,
    ) -> torch.Tensor:
        true_tokens = self._edge_tokens_from_buffers(
            hidden=hidden,
            active_indices=active_indices,
            edge_sources=self.topology_auxiliary_true_sources,
            edge_targets=self.topology_auxiliary_true_targets,
            edge_role_ids=self.topology_auxiliary_true_role_ids,
        ).mean(dim=1)
        shuffled_tokens = self._edge_tokens_from_buffers(
            hidden=hidden,
            active_indices=active_indices,
            edge_sources=self.topology_auxiliary_shuffled_sources,
            edge_targets=self.topology_auxiliary_shuffled_targets,
            edge_role_ids=self.topology_auxiliary_shuffled_role_ids,
        ).mean(dim=1)
        true_logits = self.topology_auxiliary_head(true_tokens).squeeze(1)
        shuffled_logits = self.topology_auxiliary_head(shuffled_tokens).squeeze(1)
        true_targets = torch.ones_like(true_logits)
        shuffled_targets = torch.zeros_like(shuffled_logits)
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(true_logits, true_targets)
            + F.binary_cross_entropy_with_logits(shuffled_logits, shuffled_targets)
        )
        return loss * self.topology_auxiliary_scale

    def _persistent_edge_embedding(
        self,
        hidden: torch.Tensor,
        active_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_tokens = self._persistent_edge_tokens(hidden, active_indices)
        if self.graph_mode not in {"true", "shuffled"}:
            return edge_tokens.mean(dim=1), edge_tokens
        role_ids = self.persistent_edge_role_ids.to(hidden.device).index_select(dim=0, index=active_indices)
        active_edge_weights = (role_ids == 2).to(hidden.dtype).unsqueeze(-1)
        active_edge_embedding = (edge_tokens * active_edge_weights).sum(dim=1) / active_edge_weights.sum(
            dim=1
        ).clamp_min(1.0)
        return 0.5 * (edge_tokens.mean(dim=1) + active_edge_embedding), edge_tokens

    def _encode_pairs(
        self,
        pair_features: torch.Tensor,
        active_metadata: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.last_auxiliary_loss = (
            self._topology_auxiliary_loss(hidden, active_indices)
            if self.topology_auxiliary_scale > 0.0
            else None
        )
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
            consistency_edge_tokens = hidden.new_zeros(
                hidden.shape[0],
                self.persistent_edge_sources.numel(),
                self.token_dim,
            )
            if self.edge_mode == "persistent":
                edge_embedding, consistency_edge_tokens = self._persistent_edge_embedding(hidden, active_indices)
        else:
            edge_embedding = torch.zeros_like(source_embedding)
            consistency_edge_tokens = self._persistent_edge_tokens(hidden, active_indices)
        pair_parts = [mean_embedding, max_embedding, source_embedding, target_embedding, edge_embedding]
        if self.active_metadata_fusion == "direct":
            pair_parts.append(metadata_embedding)
        pair_embedding = torch.cat(pair_parts, dim=1)
        return self.pair_projection(pair_embedding), consistency_edge_tokens

    def _cross_pair_consistency_embedding(self, edge_tokens: torch.Tensor) -> torch.Tensor:
        edge_mean = edge_tokens.mean(dim=1)
        edge_absdev = (edge_tokens - edge_mean[:, None, :, :]).abs().mean(dim=1)
        edge_max = edge_tokens.max(dim=1).values
        consistency_tokens = self.cross_pair_projection(torch.cat([edge_mean, edge_absdev, edge_max], dim=-1))
        return consistency_tokens.mean(dim=1)

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
        flat_pair_embeddings, flat_edge_tokens = self._encode_pairs(pair_features, pair_metadata)
        pair_embeddings = flat_pair_embeddings.reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.projected_pair_embedding_bits,
        )
        edge_tokens = flat_edge_tokens.reshape(
            features.shape[0],
            self.pairs_per_sample,
            flat_edge_tokens.shape[1],
            self.token_dim,
        )
        attention_embedding, attention_weights = self.attention(pair_embeddings)
        self.last_attention_weights = attention_weights.detach()
        classifier_inputs = [attention_embedding]
        if self.active_metadata_fusion == "direct":
            classifier_inputs.append(self.active_metadata_projection(active_metadata))
        if self.cross_pair_consistency != "none":
            classifier_inputs.append(self._cross_pair_consistency_embedding(edge_tokens))
        return self.classifier(torch.cat(classifier_inputs, dim=1))


__all__ = [
    "PresentActiveCellGraphLayer",
    "PresentActiveCellGraphPairSetDistinguisher",
]
