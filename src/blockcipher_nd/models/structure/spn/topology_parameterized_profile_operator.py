from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TopologyParameterizedProfileOperatorSpec:
    input_dim: int = 13
    hidden_dim: int = 32
    steps: int = 2
    dropout: float = 0.10
    relation_mode: str = "true"

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim, self.steps) <= 0:
            raise ValueError("profile operator dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.relation_mode not in {"independent", "true", "corrupted"}:
            raise ValueError("unsupported relation mode")


class _SharedProfileBlock(nn.Module):
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
        topology_context: torch.Tensor,
    ) -> torch.Tensor:
        update = self.update(
            torch.cat((state, cell_context, topology_context), dim=-1)
        )
        return self.norm(state + update)


class TopologyParameterizedProfileOperator(nn.Module):
    """One shared profile operator whose SPN permutation is a runtime input."""

    def __init__(self, spec: TopologyParameterizedProfileOperatorSpec) -> None:
        super().__init__()
        self.spec = spec
        self.input_norm = nn.LayerNorm(spec.input_dim)
        self.input_projection = nn.Linear(spec.input_dim, spec.hidden_dim)
        self.block = _SharedProfileBlock(spec.hidden_dim, spec.dropout)
        self.output_norm = nn.LayerNorm(spec.hidden_dim)
        self.output_head = nn.Linear(spec.hidden_dim, 1)

    def forward(
        self,
        prefix_features: torch.Tensor,
        inverse_player: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(prefix_features, inverse_player)
        state = self.input_projection(self.input_norm(prefix_features))
        for _ in range(self.spec.steps):
            cell_context, topology_context = self._relation_context(
                state, inverse_player
            )
            state = self.block(state, cell_context, topology_context)
        return self.output_head(self.output_norm(state)).squeeze(-1)

    def _relation_context(
        self,
        state: torch.Tensor,
        inverse_player: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spec.relation_mode == "independent":
            return state, state
        cell_mean = state.reshape(state.shape[0], 16, 4, state.shape[-1]).mean(
            dim=2
        )
        cell_context = (
            cell_mean[:, :, None, :].expand(-1, -1, 4, -1).reshape_as(state)
        )
        indices = inverse_player
        if indices.ndim == 1:
            indices = indices[None, :].expand(state.shape[0], -1)
        topology_context = torch.gather(
            state,
            1,
            indices[:, :, None].expand(-1, -1, state.shape[-1]),
        )
        return cell_context, topology_context

    def _validate_inputs(
        self,
        prefix_features: torch.Tensor,
        inverse_player: torch.Tensor,
    ) -> None:
        if prefix_features.ndim != 3 or prefix_features.shape[1:] != (
            64,
            self.spec.input_dim,
        ):
            raise ValueError("prefix_features must have shape batch x 64 x input_dim")
        if inverse_player.dtype != torch.long:
            raise ValueError("inverse_player must use torch.long indices")
        if inverse_player.ndim == 1:
            valid_shape = inverse_player.shape == (64,)
        else:
            valid_shape = inverse_player.shape == (prefix_features.shape[0], 64)
        if not valid_shape:
            raise ValueError("inverse_player must have shape 64 or batch x 64")
        expected = torch.arange(64, device=inverse_player.device)
        rows = inverse_player[None, :] if inverse_player.ndim == 1 else inverse_player
        if not torch.all(torch.sort(rows, dim=1).values == expected[None, :]):
            raise ValueError("each inverse_player row must be a 64-bit permutation")


__all__ = [
    "TopologyParameterizedProfileOperator",
    "TopologyParameterizedProfileOperatorSpec",
]
