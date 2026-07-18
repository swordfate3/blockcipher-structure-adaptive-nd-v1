from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SparseLinearProfileOperatorSpec:
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


class _SparseProfileBlock(nn.Module):
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
        linear_context: torch.Tensor,
    ) -> torch.Tensor:
        update = self.update(torch.cat((state, cell_context, linear_context), dim=-1))
        return self.norm(state + update)


class SparseLinearProfileOperator(nn.Module):
    def __init__(
        self,
        spec: SparseLinearProfileOperatorSpec,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.shape != (64, 64):
            raise ValueError("adjacency must have shape 64 x 64")
        if not torch.all((adjacency == 0) | (adjacency == 1)):
            raise ValueError("adjacency must be binary")
        degree = adjacency.sum(dim=1)
        if not torch.all(degree > 0):
            raise ValueError("every target node must have a predecessor")
        self.spec = spec
        self.register_buffer("adjacency", adjacency.clone())
        self.register_buffer("normalized_adjacency", adjacency / degree[:, None])
        self.input_norm = nn.LayerNorm(spec.input_dim)
        self.input_projection = nn.Linear(spec.input_dim, spec.hidden_dim)
        self.block = _SparseProfileBlock(spec.hidden_dim, spec.dropout)
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
            cell_context, linear_context = self._relation_context(state)
            state = self.block(state, cell_context, linear_context)
        return self.output_head(self.output_norm(state)).squeeze(-1)

    def _relation_context(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spec.relation_mode == "independent":
            return state, state
        cell_mean = state.reshape(state.shape[0], 16, 4, state.shape[-1]).mean(dim=2)
        cell_context = cell_mean[:, :, None, :].expand(-1, -1, 4, -1).reshape_as(state)
        linear_context = torch.einsum(
            "ts,bsh->bth", self.normalized_adjacency, state
        )
        return cell_context, linear_context


__all__ = ["SparseLinearProfileOperator", "SparseLinearProfileOperatorSpec"]
