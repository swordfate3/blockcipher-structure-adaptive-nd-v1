from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.small_spn_graph_models import (
    SmallSpnGpsBlock,
    _truth_table_bits,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)


@dataclass(frozen=True)
class SmallSpnRelationModelSpec:
    model_name: str
    topology_mode: str = "true"
    hidden_dim: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.model_name not in {"deepsets", "rcca"}:
            raise ValueError("model_name must be deepsets or rcca")
        if self.topology_mode not in {"true", "corrupted"}:
            raise ValueError("topology_mode must be true or corrupted")
        if self.hidden_dim <= 0 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if self.layers <= 0:
            raise ValueError("layers must be positive")


class SmallSpnRelationPredictor(nn.Module):
    def __init__(
        self,
        spec: SmallSpnRelationModelSpec,
        *,
        sboxes: np.ndarray,
        players: np.ndarray,
        structure_active_bits: np.ndarray,
        output_mask_bits: np.ndarray,
        relation_pairs: np.ndarray,
        relation_round_indices: np.ndarray,
    ) -> None:
        super().__init__()
        self.spec = spec
        hidden = spec.hidden_dim
        player_array = topology_players(players, spec.topology_mode)
        inverse = np.argsort(player_array, axis=1)
        self.register_buffer(
            "sbox_truth_bits", torch.from_numpy(_truth_table_bits(sboxes))
        )
        self.register_buffer("players", torch.from_numpy(player_array.copy()))
        self.register_buffer("inverse_players", torch.from_numpy(inverse.copy()))
        self.register_buffer(
            "structure_active_bits",
            torch.from_numpy(np.asarray(structure_active_bits, dtype=np.float32)),
        )
        self.register_buffer(
            "output_mask_bits",
            torch.from_numpy(np.asarray(output_mask_bits, dtype=np.float32)),
        )
        self.register_buffer(
            "relation_pairs",
            torch.from_numpy(np.asarray(relation_pairs, dtype=np.int64)),
        )
        self.register_buffer(
            "relation_round_indices",
            torch.from_numpy(np.asarray(relation_round_indices, dtype=np.int64)),
        )
        self.lane_embedding = nn.Embedding(4, hidden)
        self.round_embedding = nn.Embedding(4, hidden)
        self.binary_input = nn.Linear(2, hidden)
        self.sbox_encoder = nn.Sequential(
            nn.Linear(128, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.graph_input_norm = nn.LayerNorm(hidden)
        self.graph_blocks = nn.ModuleList(
            SmallSpnGpsBlock(hidden, spec.heads, spec.dropout)
            for _ in range(spec.layers)
        )
        self.coordinate_pool = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden * 2),
            nn.SiLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.coordinate_phi = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden * 2, hidden),
        )
        if spec.model_name == "rcca":
            self.cross_attention = nn.MultiheadAttention(
                hidden, spec.heads, dropout=spec.dropout, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(hidden)
            self.cross_ffn = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 2),
                nn.SiLU(),
                nn.Dropout(spec.dropout),
                nn.Linear(hidden * 2, hidden),
            )
            self.cross_output_norm = nn.LayerNorm(hidden)
        else:
            self.cross_attention = None
            self.cross_norm = None
            self.cross_ffn = None
            self.cross_output_norm = None
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.SiLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(
        self, variant_index: torch.Tensor, relation_index: torch.Tensor
    ) -> torch.Tensor:
        graph, sbox_hidden, round_hidden = self._encode_cipher_graph(
            variant_index, relation_index
        )
        query_nodes, active, output_mask = self._coordinate_query_nodes(
            relation_index, round_hidden
        )
        if self.spec.model_name == "rcca":
            query_nodes = self._cross_attend(query_nodes, graph)
        coordinate_tokens = self._pool_coordinate_nodes(
            query_nodes, active, output_mask
        )
        relation_pool = self.coordinate_phi(coordinate_tokens).mean(dim=1)
        graph_pool = graph.mean(dim=1)
        return self.head(
            torch.cat((relation_pool, graph_pool, sbox_hidden, round_hidden), dim=-1)
        ).squeeze(-1)

    def _encode_cipher_graph(
        self, variant_index: torch.Tensor, relation_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = variant_index.device
        batch = len(variant_index)
        bit_index = torch.arange(16, device=device)
        round_index = self.relation_round_indices[relation_index]
        round_hidden = self.round_embedding(round_index)
        sbox_hidden = self.sbox_encoder(
            self.sbox_truth_bits[variant_index].reshape(batch, -1)
        )
        graph = (
            self.lane_embedding(bit_index % 4).unsqueeze(0)
            + round_hidden[:, None, :]
            + sbox_hidden[:, None, :]
        )
        graph = self.graph_input_norm(graph)
        incoming_index = self.inverse_players[variant_index]
        outgoing_index = self.players[variant_index]
        gather_shape = (-1, -1, self.spec.hidden_dim)
        for block in self.graph_blocks:
            incoming = torch.gather(
                graph,
                1,
                incoming_index.unsqueeze(-1).expand(*gather_shape),
            )
            outgoing = torch.gather(
                graph,
                1,
                outgoing_index.unsqueeze(-1).expand(*gather_shape),
            )
            graph = block(graph, incoming, outgoing)
        return graph, sbox_hidden, round_hidden

    def _coordinate_query_nodes(
        self, relation_index: torch.Tensor, round_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.relation_pairs[relation_index]
        structure_index = pair // 64
        mask_index = pair % 64
        active = self.structure_active_bits[structure_index]
        output_mask = self.output_mask_bits[mask_index]
        lane = self.lane_embedding(
            torch.arange(16, device=relation_index.device) % 4
        )
        query = (
            lane.view(1, 1, 16, -1)
            + self.binary_input(torch.stack((active, output_mask), dim=-1))
            + round_hidden[:, None, None, :]
        )
        return query, active, output_mask

    def _cross_attend(
        self, query_nodes: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        if (
            self.cross_attention is None
            or self.cross_norm is None
            or self.cross_ffn is None
            or self.cross_output_norm is None
        ):
            raise RuntimeError("RCCA cross-attention modules are missing")
        batch, relation_size, nodes, hidden = query_nodes.shape
        aligned = query_nodes + graph[:, None, :, :]
        flattened = aligned.reshape(batch * relation_size, nodes, hidden)
        graph_repeated = graph[:, None, :, :].expand(-1, relation_size, -1, -1)
        graph_repeated = graph_repeated.reshape(batch * relation_size, nodes, hidden)
        attended, _ = self.cross_attention(
            flattened, graph_repeated, graph_repeated, need_weights=False
        )
        flattened = self.cross_norm(flattened + attended)
        flattened = self.cross_output_norm(flattened + self.cross_ffn(flattened))
        return flattened.reshape(batch, relation_size, nodes, hidden)

    def _pool_coordinate_nodes(
        self,
        query_nodes: torch.Tensor,
        active: torch.Tensor,
        output_mask: torch.Tensor,
    ) -> torch.Tensor:
        active_weight = active.unsqueeze(-1)
        mask_weight = output_mask.unsqueeze(-1)
        mean_pool = query_nodes.mean(dim=2)
        active_pool = (query_nodes * active_weight).sum(dim=2) / active_weight.sum(
            dim=2
        ).clamp_min(1.0)
        mask_pool = (query_nodes * mask_weight).sum(dim=2) / mask_weight.sum(
            dim=2
        ).clamp_min(1.0)
        return self.coordinate_pool(
            torch.cat((mean_pool, active_pool, mask_pool), dim=-1)
        )
