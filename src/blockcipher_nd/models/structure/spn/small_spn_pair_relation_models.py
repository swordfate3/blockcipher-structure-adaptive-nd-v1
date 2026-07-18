from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)


@dataclass(frozen=True)
class SmallSpnPairRelationSpec:
    topology_mode: str = "true"
    processor_mode: str = "triangle"
    state_bits: int = 16
    round_categories: int = 4
    round_step_offset: int = 2
    hidden_dim: int = 64
    path_rank: int = 8
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.topology_mode not in {"true", "corrupted"}:
            raise ValueError("topology_mode must be true or corrupted")
        if self.processor_mode not in {"triangle", "local"}:
            raise ValueError("processor_mode must be triangle or local")
        if self.state_bits <= 0 or self.state_bits % 4:
            raise ValueError("state_bits must be a positive multiple of four")
        if self.round_categories <= 0 or self.round_step_offset <= 0:
            raise ValueError("round_categories and round_step_offset must be positive")
        if self.hidden_dim <= 0 or self.path_rank <= 0:
            raise ValueError("hidden_dim and path_rank must be positive")


class _PairUpdateBlock(nn.Module):
    def __init__(self, hidden_dim: int, path_rank: int, dropout: float) -> None:
        super().__init__()
        self.left = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, path_rank)
        )
        self.right = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, path_rank)
        )
        self.triangle_projection = nn.Linear(path_rank, hidden_dim)
        self.update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, relation: torch.Tensor) -> torch.Tensor:
        message = self.triangle_projection(self.path_message(relation))
        update = self.update(torch.cat((relation, message), dim=-1))
        return self.output_norm(relation + update)

    def path_message(self, relation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PairTriangleBlock(_PairUpdateBlock):
    def path_message(self, relation: torch.Tensor) -> torch.Tensor:
        left = self.left(relation).permute(0, 3, 1, 2)
        right = self.right(relation).permute(0, 3, 1, 2)
        triangle = torch.matmul(left, right) / math.sqrt(relation.shape[1])
        return triangle.permute(0, 2, 3, 1)


class PairLocalBlock(_PairUpdateBlock):
    def path_message(self, relation: torch.Tensor) -> torch.Tensor:
        return self.left(relation) * self.right(relation)


class SmallSpnPairRelationReasoner(nn.Module):
    PAIR_COUNT = 16 * 16
    MAX_STEPS = 5

    def __init__(
        self,
        spec: SmallSpnPairRelationSpec,
        *,
        sboxes: np.ndarray,
        players: np.ndarray,
        structure_active_bits: np.ndarray,
        output_mask_bits: np.ndarray,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.state_bits = spec.state_bits
        self.pair_count = spec.state_bits * spec.state_bits
        self.max_steps = spec.round_step_offset + spec.round_categories - 1
        hidden_dim = spec.hidden_dim
        players = np.asarray(players, dtype=np.int64)
        structure_active_bits = np.asarray(structure_active_bits, dtype=np.float32)
        output_mask_bits = np.asarray(output_mask_bits, dtype=np.float32)
        if players.ndim != 2 or players.shape[1] != spec.state_bits:
            raise ValueError("players must have shape variants x state_bits")
        if structure_active_bits.ndim != 2 or structure_active_bits.shape[1] != spec.state_bits:
            raise ValueError("structure_active_bits must have width state_bits")
        if output_mask_bits.ndim != 2 or output_mask_bits.shape[1] != spec.state_bits:
            raise ValueError("output_mask_bits must have width state_bits")
        player_array = topology_players(players, spec.topology_mode)
        self.register_buffer("players", torch.from_numpy(player_array.copy()))
        self.register_buffer(
            "sbox_truth_bits", torch.from_numpy(_truth_table_bits(sboxes))
        )
        self.register_buffer(
            "structure_active_bits",
            torch.from_numpy(structure_active_bits),
        )
        self.register_buffer(
            "output_mask_bits",
            torch.from_numpy(output_mask_bits),
        )
        bit_index = np.arange(spec.state_bits, dtype=np.int64)
        self.register_buffer(
            "identity_relation",
            torch.from_numpy((bit_index[:, None] == bit_index[None, :]).astype(np.float32)),
        )
        self.register_buffer(
            "same_cell_relation",
            torch.from_numpy(
                ((bit_index[:, None] // 4) == (bit_index[None, :] // 4)).astype(
                    np.float32
                )
            ),
        )
        self.register_buffer(
            "same_lane_relation",
            torch.from_numpy(
                ((bit_index[:, None] % 4) == (bit_index[None, :] % 4)).astype(
                    np.float32
                )
            ),
        )
        self.source_lane_embedding = nn.Embedding(4, hidden_dim)
        self.destination_lane_embedding = nn.Embedding(4, hidden_dim)
        self.round_embedding = nn.Embedding(spec.round_categories, hidden_dim)
        self.relation_input = nn.Linear(9, hidden_dim)
        self.sbox_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        if spec.processor_mode == "triangle":
            self.triangle_block = PairTriangleBlock(
                hidden_dim, spec.path_rank, spec.dropout
            )
        else:
            self.local_block = PairLocalBlock(
                hidden_dim, spec.path_rank, spec.dropout
            )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 7),
            nn.Linear(hidden_dim * 7, hidden_dim * 2),
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
        relation, context = self.build_initial_relation(
            variant_index, round_index, structure_index, mask_index
        )
        step_count = self.step_counts(round_index)
        processor = (
            self.triangle_block
            if self.spec.processor_mode == "triangle"
            else self.local_block
        )
        for step in range(self.max_steps):
            updated = processor(relation)
            relation = torch.where(
                (step < step_count).view(-1, 1, 1, 1), updated, relation
            )
        pools = [
            relation.mean(dim=(1, 2)),
            _weighted_pair_pool(relation, context["identity"]),
            _weighted_pair_pool(relation, context["p_edge"]),
            _weighted_pair_pool(relation, context["active_to_mask"]),
            _weighted_pair_pool(relation, context["mask_to_active"]),
            context["sbox_hidden"],
            context["round_hidden"],
        ]
        return self.head(torch.cat(pools, dim=-1)).squeeze(-1)

    def build_initial_relation(
        self,
        variant_index: torch.Tensor,
        round_index: torch.Tensor,
        structure_index: torch.Tensor,
        mask_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = variant_index.device
        batch = variant_index.shape[0]
        state_bits = self.state_bits
        bit_index = torch.arange(state_bits, device=device)
        active = self.structure_active_bits[structure_index]
        output_mask = self.output_mask_bits[mask_index]
        player = self.players[variant_index]
        p_edge = (player[:, :, None] == bit_index[None, None, :]).to(active.dtype)
        identity = self.identity_relation.expand(batch, -1, -1)
        same_cell = self.same_cell_relation.expand(batch, -1, -1)
        same_lane = self.same_lane_relation.expand(batch, -1, -1)
        source_active = active[:, :, None].expand(-1, -1, state_bits)
        destination_active = active[:, None, :].expand(-1, state_bits, -1)
        source_mask = output_mask[:, :, None].expand(-1, -1, state_bits)
        destination_mask = output_mask[:, None, :].expand(-1, state_bits, -1)
        features = torch.stack(
            (
                identity,
                p_edge,
                p_edge.transpose(1, 2),
                same_cell,
                same_lane,
                source_active,
                destination_active,
                source_mask,
                destination_mask,
            ),
            dim=-1,
        )
        source_lane = self.source_lane_embedding(bit_index % 4)[:, None, :]
        destination_lane = self.destination_lane_embedding(bit_index % 4)[None, :, :]
        round_hidden = self.round_embedding(round_index)
        sbox_hidden = self.sbox_encoder(
            self.sbox_truth_bits[variant_index].reshape(batch, -1)
        )
        relation = (
            self.relation_input(features)
            + source_lane[None, :, :, :]
            + destination_lane[None, :, :, :]
            + round_hidden[:, None, None, :]
            + sbox_hidden[:, None, None, :]
        )
        active_to_mask = source_active * destination_mask
        mask_to_active = source_mask * destination_active
        return self.input_norm(relation), {
            "identity": identity,
            "p_edge": p_edge,
            "active_to_mask": active_to_mask,
            "mask_to_active": mask_to_active,
            "sbox_hidden": sbox_hidden,
            "round_hidden": round_hidden,
        }

    def step_counts(self, round_index: torch.Tensor) -> torch.Tensor:
        return round_index + self.spec.round_step_offset


def _weighted_pair_pool(relation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.unsqueeze(-1)
    return (relation * expanded).sum(dim=(1, 2)) / expanded.sum(
        dim=(1, 2)
    ).clamp_min(1.0)


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
