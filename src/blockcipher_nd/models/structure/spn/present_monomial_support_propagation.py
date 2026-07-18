from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX_ANF
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)


@dataclass(frozen=True)
class PresentMspnSpec:
    topology_mode: str = "true"
    state_bits: int = 64
    rounds: int = 4
    degree_channels: int = 9
    hidden_dim: int = 32
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.topology_mode not in {"true", "corrupted"}:
            raise ValueError("topology_mode must be true or corrupted")
        if self.state_bits <= 0 or self.state_bits % 4:
            raise ValueError("state_bits must be a positive multiple of four")
        if self.rounds <= 0 or self.degree_channels <= 1 or self.hidden_dim <= 0:
            raise ValueError("rounds and channel dimensions must be positive")


class PresentAnfSupportStep(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.term_gate = nn.Linear(hidden_dim, hidden_dim)
        self.term_size_embedding = nn.Embedding(5, hidden_dim)
        self.output_lane_embedding = nn.Embedding(4, hidden_dim)
        self.constant_token = nn.Parameter(torch.zeros(hidden_dim))
        self.term_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch, bits, hidden = state.shape
        if bits % 4 or hidden != self.hidden_dim:
            raise ValueError("state must have shape batch x (4*cells) x hidden")
        cells = state.reshape(batch, bits // 4, 4, hidden)
        output_lanes: list[torch.Tensor] = []
        for output_lane, terms in enumerate(PRESENT_SBOX_ANF):
            messages: list[torch.Tensor] = []
            for term in terms:
                input_lanes = [lane for lane in range(4) if term & (1 << lane)]
                if input_lanes:
                    selected = cells[:, :, input_lanes, :]
                    mean = selected.mean(dim=2)
                    product = torch.sigmoid(self.term_gate(selected)).prod(dim=2)
                else:
                    mean = self.constant_token.expand(batch, bits // 4, -1)
                    product = torch.sigmoid(mean)
                message = self.term_mlp(torch.cat((mean, product), dim=-1))
                message = message + self.term_size_embedding.weight[len(input_lanes)]
                messages.append(message)
            aggregated = torch.stack(messages, dim=0).mean(dim=0)
            aggregated = (
                aggregated
                + self.output_lane_embedding.weight[output_lane]
                + self.constant_token
            )
            output_lanes.append(self.output_norm(aggregated))
        return torch.stack(output_lanes, dim=2).reshape(batch, bits, hidden)


class PresentMonomialSupportPropagationNetwork(nn.Module):
    def __init__(
        self,
        spec: PresentMspnSpec,
        *,
        players: np.ndarray,
        structure_active_bits: np.ndarray,
        output_mask_bits: np.ndarray,
    ) -> None:
        super().__init__()
        self.spec = spec
        players = np.asarray(players, dtype=np.int64)
        structure_active_bits = np.asarray(structure_active_bits, dtype=np.float32)
        output_mask_bits = np.asarray(output_mask_bits, dtype=np.float32)
        if players.ndim != 2 or players.shape[1] != spec.state_bits:
            raise ValueError("players must have shape variants x state_bits")
        if (
            structure_active_bits.ndim != 2
            or structure_active_bits.shape[1] != spec.state_bits
        ):
            raise ValueError("structure_active_bits must have width state_bits")
        if output_mask_bits.ndim != 2 or output_mask_bits.shape[1] != spec.state_bits:
            raise ValueError("output_mask_bits must have width state_bits")
        player_array = topology_players(players, spec.topology_mode)
        self.register_buffer("players", torch.from_numpy(player_array.copy()))
        self.register_buffer(
            "structure_active_bits", torch.from_numpy(structure_active_bits.copy())
        )
        self.register_buffer(
            "output_mask_bits", torch.from_numpy(output_mask_bits.copy())
        )
        self.degree_seed_projection = nn.Linear(
            spec.degree_channels, spec.hidden_dim
        )
        self.input_lane_embedding = nn.Embedding(4, spec.hidden_dim)
        self.input_norm = nn.LayerNorm(spec.hidden_dim)
        self.shared_step = PresentAnfSupportStep(spec.hidden_dim, spec.dropout)
        self.degree_decoder = nn.Linear(spec.hidden_dim, spec.degree_channels)
        head_width = spec.hidden_dim * 4 + spec.degree_channels
        self.head = nn.Sequential(
            nn.LayerNorm(head_width),
            nn.Linear(head_width, spec.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim * 2, 1),
        )

    def forward(
        self,
        variant_index: torch.Tensor,
        structure_index: torch.Tensor,
        mask_index: torch.Tensor,
    ) -> torch.Tensor:
        state, active, output_mask = self.build_initial_state(
            structure_index, mask_index
        )
        initial = state
        for _ in range(self.spec.rounds):
            state = self.shared_step(state)
            state = self.transport(state, variant_index)
        query_pool = _weighted_bit_pool(state, output_mask)
        active_pool = _weighted_bit_pool(initial, active)
        global_pool = state.mean(dim=1)
        interaction = query_pool * active_pool
        degree_state = self.degree_decoder(state)
        degree_pool = _weighted_bit_pool(degree_state, output_mask)
        return self.head(
            torch.cat(
                (query_pool, active_pool, global_pool, interaction, degree_pool),
                dim=-1,
            )
        ).squeeze(-1)

    def build_initial_state(
        self, structure_index: torch.Tensor, mask_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = self.structure_active_bits[structure_index]
        output_mask = self.output_mask_bits[mask_index]
        batch = len(structure_index)
        degree_seed = torch.zeros(
            batch,
            self.spec.state_bits,
            self.spec.degree_channels,
            device=active.device,
            dtype=active.dtype,
        )
        degree_seed[:, :, 0] = 1.0
        degree_seed[:, :, 1] = active
        lanes = torch.arange(self.spec.state_bits, device=active.device) % 4
        state = self.degree_seed_projection(degree_seed)
        state = state + self.input_lane_embedding(lanes)[None, :, :]
        return self.input_norm(state), active, output_mask

    def transport(
        self, state: torch.Tensor, variant_index: torch.Tensor
    ) -> torch.Tensor:
        player = self.players[variant_index]
        inverse = torch.argsort(player, dim=1)
        return torch.gather(
            state,
            1,
            inverse[:, :, None].expand(-1, -1, state.shape[-1]),
        )


def _weighted_bit_pool(state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.unsqueeze(-1)
    return (state * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1.0)
