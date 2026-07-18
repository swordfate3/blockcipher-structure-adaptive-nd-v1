from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PresentRoundRecurrentProfileOperatorSpec:
    round_input_dim: int = 13
    hidden_dim: int = 22
    prefix_rounds: int = 3
    dropout: float = 0.10
    relation_mode: str = "true"
    round_order: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if min(self.round_input_dim, self.hidden_dim, self.prefix_rounds) <= 0:
            raise ValueError("round-recurrent dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.relation_mode not in {"true", "corrupted"}:
            raise ValueError("relation_mode must be true or corrupted")
        if tuple(sorted(self.round_order)) != tuple(range(self.prefix_rounds)):
            raise ValueError("round_order must permute every prefix round")


class _SharedRoundProfileBlock(nn.Module):
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
        update = self.update(torch.cat((state, cell_context, player_context), dim=-1))
        return self.norm(state + update)


class PresentRoundRecurrentProfileOperator(nn.Module):
    def __init__(
        self,
        spec: PresentRoundRecurrentProfileOperatorSpec,
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
        self.input_norm = nn.LayerNorm(spec.round_input_dim)
        self.input_projection = nn.Linear(spec.round_input_dim, spec.hidden_dim)
        self.recurrent = nn.GRUCell(spec.hidden_dim, spec.hidden_dim)
        self.block = _SharedRoundProfileBlock(spec.hidden_dim, spec.dropout)
        self.output_norm = nn.LayerNorm(spec.hidden_dim)
        self.output_head = nn.Linear(spec.hidden_dim, 1)

    def forward(self, prefix_features: torch.Tensor) -> torch.Tensor:
        expected_dim = self.spec.prefix_rounds * self.spec.round_input_dim
        if prefix_features.ndim != 3 or prefix_features.shape[1:] != (
            64,
            expected_dim,
        ):
            raise ValueError(
                "prefix_features must have shape batch x 64 x "
                f"{expected_dim}"
            )
        rounds = prefix_features.reshape(
            prefix_features.shape[0],
            64,
            self.spec.prefix_rounds,
            self.spec.round_input_dim,
        )
        state = prefix_features.new_zeros(
            prefix_features.shape[0], 64, self.spec.hidden_dim
        )
        for round_index in self.spec.round_order:
            projected = self.input_projection(
                self.input_norm(rounds[:, :, round_index, :])
            )
            state = self.recurrent(
                projected.reshape(-1, self.spec.hidden_dim),
                state.reshape(-1, self.spec.hidden_dim),
            ).reshape_as(state)
            cell_context, player_context = self._relation_context(state)
            state = self.block(state, cell_context, player_context)
        return self.output_head(self.output_norm(state)).squeeze(-1)

    def _relation_context(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cell_mean = state.reshape(state.shape[0], 16, 4, state.shape[-1]).mean(dim=2)
        cell_context = (
            cell_mean[:, :, None, :].expand(-1, -1, 4, -1).reshape_as(state)
        )
        return cell_context, state[:, self.inverse_player]


__all__ = [
    "PresentRoundRecurrentProfileOperator",
    "PresentRoundRecurrentProfileOperatorSpec",
]
