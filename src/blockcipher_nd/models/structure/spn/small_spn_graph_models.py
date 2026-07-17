from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SmallSpnModelSpec:
    model_name: str
    topology_mode: str = "true"
    position_mode: str = "absolute"
    hidden_dim: int = 64
    blocks: int = 3
    heads: int = 4
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.model_name not in {"graphgps", "scgt"}:
            raise ValueError("model_name must be graphgps or scgt")
        if self.topology_mode not in {"true", "shuffled"}:
            raise ValueError("topology_mode must be true or shuffled")
        if self.position_mode not in {"absolute", "cell_equivariant"}:
            raise ValueError("position_mode must be absolute or cell_equivariant")
        if self.hidden_dim <= 0 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if self.blocks <= 0:
            raise ValueError("blocks must be positive")


class SmallSpnGpsBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        incoming: torch.Tensor,
        outgoing: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, hidden_dim = hidden.shape
        cell_mean = (
            hidden.reshape(batch, 4, 4, hidden_dim)
            .mean(dim=2, keepdim=True)
            .expand(-1, -1, 4, -1)
            .reshape(batch, 16, hidden_dim)
        )
        local = self.local(torch.cat((hidden, cell_mean, incoming, outgoing), dim=-1))
        global_message, _ = self.attention(hidden, hidden, hidden, need_weights=False)
        hidden = self.message_norm(hidden + local + global_message)
        return self.output_norm(hidden + self.ffn(hidden))


class BasisSetEncoder(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(16, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, basis: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.input(basis)
        attended, _ = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=~valid,
            need_weights=False,
        )
        hidden = self.norm(hidden + attended)
        hidden = self.output_norm(hidden + self.ffn(hidden))
        weights = valid.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class SmallSpnTopologyPredictor(nn.Module):
    def __init__(
        self,
        spec: SmallSpnModelSpec,
        *,
        sboxes: np.ndarray,
        players: np.ndarray,
        structure_active_bits: np.ndarray,
        structure_basis: np.ndarray,
        structure_basis_valid: np.ndarray,
        output_mask_bits: np.ndarray,
    ) -> None:
        super().__init__()
        self.spec = spec
        hidden_dim = spec.hidden_dim
        truth_bits = _truth_table_bits(sboxes)
        player_array = np.asarray(players, dtype=np.int64)
        if spec.topology_mode == "shuffled":
            player_array = np.roll(player_array, shift=1, axis=0)
        inverse = np.argsort(player_array, axis=1)
        self.register_buffer("sbox_truth_bits", torch.from_numpy(truth_bits))
        self.register_buffer("players", torch.from_numpy(player_array.copy()))
        self.register_buffer("inverse_players", torch.from_numpy(inverse.copy()))
        self.register_buffer(
            "structure_active_bits",
            torch.from_numpy(np.asarray(structure_active_bits, dtype=np.float32)),
        )
        self.register_buffer(
            "structure_basis",
            torch.from_numpy(np.asarray(structure_basis, dtype=np.float32)),
        )
        self.register_buffer(
            "structure_basis_valid",
            torch.from_numpy(np.asarray(structure_basis_valid, dtype=np.bool_)),
        )
        self.register_buffer(
            "output_mask_bits",
            torch.from_numpy(np.asarray(output_mask_bits, dtype=np.float32)),
        )
        self.bit_embedding = (
            nn.Embedding(16, hidden_dim) if spec.position_mode == "absolute" else None
        )
        self.nibble_embedding = (
            nn.Embedding(4, hidden_dim) if spec.position_mode == "absolute" else None
        )
        self.lane_embedding = nn.Embedding(4, hidden_dim)
        self.round_embedding = nn.Embedding(4, hidden_dim)
        self.binary_input = nn.Linear(2, hidden_dim)
        self.sbox_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.basis_encoder = (
            BasisSetEncoder(hidden_dim, spec.heads, spec.dropout)
            if spec.model_name == "scgt"
            else None
        )
        self.basis_to_nodes = nn.Linear(hidden_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            SmallSpnGpsBlock(hidden_dim, spec.heads, spec.dropout)
            for _ in range(spec.blocks)
        )
        readout_parts = 6 if self.basis_encoder is not None else 5
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * readout_parts),
            nn.Linear(hidden_dim * readout_parts, hidden_dim * 2),
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
        device = variant_index.device
        batch = variant_index.shape[0]
        bit_index = torch.arange(16, device=device)
        base = self.lane_embedding(bit_index % 4)
        if self.bit_embedding is not None and self.nibble_embedding is not None:
            base = (
                base
                + self.bit_embedding(bit_index)
                + self.nibble_embedding(bit_index // 4)
            )
        active = self.structure_active_bits[structure_index]
        output_mask = self.output_mask_bits[mask_index]
        binary = self.binary_input(torch.stack((active, output_mask), dim=-1))
        round_hidden = self.round_embedding(round_index)
        sbox_hidden = self.sbox_encoder(
            self.sbox_truth_bits[variant_index].reshape(batch, -1)
        )
        hidden = (
            base.unsqueeze(0)
            + binary
            + round_hidden[:, None, :]
            + sbox_hidden[:, None, :]
        )
        basis_pool: torch.Tensor | None = None
        if self.basis_encoder is not None:
            basis_pool = self.basis_encoder(
                self.structure_basis[structure_index],
                self.structure_basis_valid[structure_index],
            )
            hidden = hidden + self.basis_to_nodes(basis_pool)[:, None, :]
        hidden = self.input_norm(hidden)
        incoming_index = self.inverse_players[variant_index]
        outgoing_index = self.players[variant_index]
        gather_shape = (-1, -1, self.spec.hidden_dim)
        for block in self.blocks:
            incoming = torch.gather(
                hidden,
                1,
                incoming_index.unsqueeze(-1).expand(*gather_shape),
            )
            outgoing = torch.gather(
                hidden,
                1,
                outgoing_index.unsqueeze(-1).expand(*gather_shape),
            )
            hidden = block(hidden, incoming, outgoing)
        mask_weight = output_mask.unsqueeze(-1)
        active_weight = active.unsqueeze(-1)
        mask_pool = (hidden * mask_weight).sum(dim=1) / mask_weight.sum(dim=1).clamp_min(1.0)
        active_pool = (hidden * active_weight).sum(dim=1) / active_weight.sum(dim=1).clamp_min(1.0)
        parts = [hidden.mean(dim=1), mask_pool, active_pool, sbox_hidden, round_hidden]
        if basis_pool is not None:
            parts.append(basis_pool)
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


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
