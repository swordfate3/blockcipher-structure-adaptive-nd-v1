from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RectangleRowTypedShiftOperatorSpec:
    input_dim: int = 13
    hidden_dim: int = 32
    steps: int = 2
    dropout: float = 0.10
    row_mode: str = "true"

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim, self.steps) <= 0:
            raise ValueError("operator dimensions must be positive")
        if self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by four")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.row_mode not in {"untyped", "true", "wrong"}:
            raise ValueError("unsupported row mode")


class _SharedTypedProfileBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        cell_context: torch.Tensor,
        player_context: torch.Tensor,
    ) -> torch.Tensor:
        update = self.update(
            torch.cat((state, cell_context, player_context), dim=-1)
        )
        return self.norm(state + update)


class RectangleRowTypedShiftOperator(nn.Module):
    def __init__(
        self,
        spec: RectangleRowTypedShiftOperatorSpec,
        player: torch.Tensor,
    ) -> None:
        super().__init__()
        player = torch.as_tensor(player, dtype=torch.long)
        if player.shape != (64,) or not torch.equal(
            torch.sort(player).values, torch.arange(64)
        ):
            raise ValueError("player must be a 64-bit permutation")
        inverse = torch.empty_like(player)
        inverse[player] = torch.arange(64)
        self.spec = spec
        self.register_buffer("player", player.clone())
        self.register_buffer("inverse_player", inverse)
        self.register_buffer(
            "typed_channel_index",
            _typed_channel_index(spec.hidden_dim, spec.row_mode),
        )
        self.input_norm = nn.LayerNorm(spec.input_dim)
        self.input_projection = nn.Linear(spec.input_dim, spec.hidden_dim)
        self.block = _SharedTypedProfileBlock(spec.hidden_dim, spec.dropout)
        self.output_norm = nn.LayerNorm(spec.hidden_dim)
        self.output_head = nn.Linear(spec.hidden_dim, 1)

    def forward(self, prefix_features: torch.Tensor) -> torch.Tensor:
        if prefix_features.ndim != 3 or prefix_features.shape[1:] != (
            64,
            self.spec.input_dim,
        ):
            raise ValueError("prefix_features must have shape batch x 64 x input_dim")
        state = self.input_projection(self.input_norm(prefix_features))
        for _ in range(self.spec.steps):
            cell_mean = state.reshape(
                state.shape[0], 16, 4, state.shape[-1]
            ).mean(dim=2)
            cell_context = (
                cell_mean[:, :, None, :]
                .expand(-1, -1, 4, -1)
                .reshape_as(state)
            )
            player_context = state[:, self.inverse_player]
            channel_index = self.typed_channel_index[None].expand(
                state.shape[0], -1, -1
            )
            player_context = torch.gather(player_context, -1, channel_index)
            state = self.block(state, cell_context, player_context)
        return self.output_head(self.output_norm(state)).squeeze(-1)


def _typed_channel_index(hidden_dim: int, row_mode: str) -> torch.Tensor:
    base = torch.arange(hidden_dim)
    indices = torch.empty((64, hidden_dim), dtype=torch.long)
    for node in range(64):
        cell = node // 4
        row = node % 4
        if row_mode == "untyped":
            row_type = 0
        elif row_mode == "wrong":
            row_type = (row + 1 + cell % 3) % 4
        else:
            row_type = row
        shift = row_type * (hidden_dim // 4)
        indices[node] = (base - shift) % hidden_dim
    return indices


__all__ = [
    "RectangleRowTypedShiftOperator",
    "RectangleRowTypedShiftOperatorSpec",
]
