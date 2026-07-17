from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)


@dataclass(frozen=True)
class SmallSpnEdgeTokenSpec:
    topology_mode: str = "true"
    hidden_dim: int = 64
    layers: int = 3
    heads: int = 4
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.topology_mode not in {"true", "shuffled", "corrupted"}:
            raise ValueError("topology_mode must be true, shuffled, or corrupted")
        if self.hidden_dim <= 0 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if self.layers <= 0:
            raise ValueError("layers must be positive")


class SmallSpnCipherEdgeTokenTransformer(nn.Module):
    TOKEN_COUNT = 37

    def __init__(
        self,
        spec: SmallSpnEdgeTokenSpec,
        *,
        sboxes: np.ndarray,
        players: np.ndarray,
        structure_active_bits: np.ndarray,
        output_mask_bits: np.ndarray,
    ) -> None:
        super().__init__()
        self.spec = spec
        hidden_dim = spec.hidden_dim
        player_array = topology_players(players, spec.topology_mode)
        self.register_buffer("players", torch.from_numpy(player_array.copy()))
        self.register_buffer(
            "sbox_truth_bits", torch.from_numpy(_truth_table_bits(sboxes))
        )
        self.register_buffer(
            "structure_active_bits",
            torch.from_numpy(np.asarray(structure_active_bits, dtype=np.float32)),
        )
        self.register_buffer(
            "output_mask_bits",
            torch.from_numpy(np.asarray(output_mask_bits, dtype=np.float32)),
        )
        self.lane_embedding = nn.Embedding(4, hidden_dim)
        self.round_embedding = nn.Embedding(4, hidden_dim)
        self.token_type_embedding = nn.Embedding(4, hidden_dim)
        self.binary_input = nn.Linear(2, hidden_dim)
        self.sbox_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.sbox_relation_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=spec.heads,
            dim_feedforward=hidden_dim * 2,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=spec.layers,
            enable_nested_tensor=False,
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(
        self,
        variant_index: torch.Tensor,
        round_index: torch.Tensor,
        structure_index: torch.Tensor,
        mask_index: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.build_tokens(
            variant_index, round_index, structure_index, mask_index
        )
        hidden = self.transformer(self.input_norm(tokens))
        query = self.output_norm(hidden[:, -1])
        return self.head(query).squeeze(-1)

    def build_tokens(
        self,
        variant_index: torch.Tensor,
        round_index: torch.Tensor,
        structure_index: torch.Tensor,
        mask_index: torch.Tensor,
    ) -> torch.Tensor:
        device = variant_index.device
        batch = variant_index.shape[0]
        bit_index = torch.arange(16, device=device)
        active = self.structure_active_bits[structure_index]
        output_mask = self.output_mask_bits[mask_index]
        round_hidden = self.round_embedding(round_index)
        sbox_hidden = self.sbox_encoder(
            self.sbox_truth_bits[variant_index].reshape(batch, -1)
        )
        node = (
            self.lane_embedding(bit_index % 4).unsqueeze(0)
            + self.binary_input(torch.stack((active, output_mask), dim=-1))
            + round_hidden[:, None, :]
            + self.token_type_embedding.weight[0]
        )

        destination = self.players[variant_index]
        destination_node = torch.gather(
            node,
            1,
            destination.unsqueeze(-1).expand(-1, -1, self.spec.hidden_dim),
        )
        edge = self.edge_encoder(torch.cat((node, destination_node), dim=-1))
        edge = edge + self.token_type_embedding.weight[1]

        cell_node = node.reshape(batch, 4, 4, self.spec.hidden_dim).mean(dim=2)
        sbox_relation = (
            self.sbox_relation_encoder(cell_node)
            + sbox_hidden[:, None, :]
            + self.token_type_embedding.weight[2]
        )

        mask_weight = output_mask.unsqueeze(-1)
        active_weight = active.unsqueeze(-1)
        mask_pool = (node * mask_weight).sum(dim=1) / mask_weight.sum(dim=1).clamp_min(1.0)
        active_pool = (node * active_weight).sum(dim=1) / active_weight.sum(dim=1).clamp_min(1.0)
        query = (
            mask_pool
            + active_pool
            + sbox_hidden
            + round_hidden
            + self.token_type_embedding.weight[3]
        ).unsqueeze(1)
        tokens = torch.cat((node, edge, sbox_relation, query), dim=1)
        if tokens.shape[1] != self.TOKEN_COUNT:
            raise RuntimeError("CETT token count contract violated")
        return tokens


def _truth_table_bits(sboxes: np.ndarray) -> np.ndarray:
    tables = np.asarray(sboxes, dtype=np.uint8)
    inputs = np.broadcast_to(np.arange(16, dtype=np.uint8), tables.shape)
    input_bits = ((inputs[..., None] >> np.arange(4, dtype=np.uint8)) & 1).astype(
        np.float32
    )
    output_bits = ((tables[..., None] >> np.arange(4, dtype=np.uint8)) & 1).astype(
        np.float32
    )
    return np.concatenate((input_bits, output_bits), axis=-1)
