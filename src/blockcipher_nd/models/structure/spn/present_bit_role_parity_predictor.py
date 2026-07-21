from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def present_player() -> tuple[int, ...]:
    return tuple((16 * bit) % 63 if bit < 63 else 63 for bit in range(64))


def wrong_player() -> tuple[int, ...]:
    return tuple((8 * bit) % 63 if bit < 63 else 63 for bit in range(64))


@dataclass(frozen=True)
class PresentBitRoleParityPredictorSpec:
    bit_channels: int = 13
    routing_depth: int = 2
    head_hidden_dim: int = 64
    p_topology: str = "true"

    def __post_init__(self) -> None:
        if min(self.bit_channels, self.routing_depth, self.head_hidden_dim) <= 0:
            raise ValueError("bit-role predictor dimensions must be positive")
        if self.p_topology not in {"true", "wrong"}:
            raise ValueError("p_topology must be true or wrong")


class _BitRoleSboxBlock(nn.Module):
    def __init__(self, bit_channels: int) -> None:
        super().__init__()
        nibble_dim = 4 * bit_channels
        hidden_dim = 2 * nibble_dim
        self.norm = nn.LayerNorm(nibble_dim)
        self.update = nn.Sequential(
            nn.Linear(nibble_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, nibble_dim),
        )

    def forward(self, bit_state: torch.Tensor) -> torch.Tensor:
        batch, bits, channels = bit_state.shape
        if bits != 64:
            raise ValueError("bit_state must contain 64 bit positions")
        nibbles = bit_state.reshape(batch, 16, 4 * channels)
        updated = nibbles + self.update(self.norm(nibbles))
        return updated.reshape(batch, 64, channels)


class PresentBitRoleParityPredictor(nn.Module):
    def __init__(self, spec: PresentBitRoleParityPredictorSpec) -> None:
        super().__init__()
        self.spec = spec
        player = present_player() if spec.p_topology == "true" else wrong_player()
        player_tensor = torch.tensor(player, dtype=torch.long)
        inverse_player = torch.empty_like(player_tensor)
        inverse_player[player_tensor] = torch.arange(64)
        self.register_buffer("player", player_tensor, persistent=False)
        self.register_buffer("inverse_player", inverse_player, persistent=False)
        self.input_projection = nn.Linear(1, spec.bit_channels)
        self.bit_embedding = nn.Parameter(torch.zeros(1, 64, spec.bit_channels))
        self.stage_nibble_embedding = nn.Parameter(
            torch.zeros(spec.routing_depth, 16, spec.bit_channels)
        )
        nn.init.trunc_normal_(self.bit_embedding, std=0.02)
        nn.init.trunc_normal_(self.stage_nibble_embedding, std=0.02)
        self.routing_blocks = nn.ModuleList(
            [_BitRoleSboxBlock(spec.bit_channels) for _ in range(spec.routing_depth)]
        )
        nibble_dim = 4 * spec.bit_channels
        self.output_head = nn.Sequential(
            nn.LayerNorm(nibble_dim),
            nn.Linear(nibble_dim, spec.head_hidden_dim),
            nn.GELU(),
            nn.Linear(spec.head_hidden_dim, 1),
        )
        self.output_nibble_bias = nn.Parameter(torch.zeros(16))

    def forward(self, plaintext_bits: torch.Tensor) -> torch.Tensor:
        if plaintext_bits.ndim != 2 or plaintext_bits.shape[1] != 64:
            raise ValueError("plaintext_bits must have shape batch x 64")
        state = self.input_projection(plaintext_bits[:, :, None]) + self.bit_embedding
        for stage, block in enumerate(self.routing_blocks):
            stage_context = self.stage_nibble_embedding[stage]
            stage_context = (
                stage_context[:, None, :]
                .expand(-1, 4, -1)
                .reshape(1, 64, self.spec.bit_channels)
            )
            state = block(state + stage_context)
            state = state[:, self.inverse_player]
        nibble_state = state.reshape(state.shape[0], 16, 4 * self.spec.bit_channels)
        return self.output_head(nibble_state).squeeze(-1) + self.output_nibble_bias


__all__ = [
    "PresentBitRoleParityPredictor",
    "PresentBitRoleParityPredictorSpec",
    "present_player",
    "wrong_player",
]
