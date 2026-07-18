from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RidgeGuidedSparseResidualSpec:
    input_dim: int = 13
    hidden_dim: int = 32
    steps: int = 2
    dropout: float = 0.10
    relation_mode: str = "true"
    residual_bound: float = 0.25

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim, self.steps) <= 0:
            raise ValueError("residual dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.relation_mode not in {"independent", "true", "corrupted"}:
            raise ValueError("unsupported relation mode")
        if self.residual_bound <= 0:
            raise ValueError("residual_bound must be positive")


class _ResidualBlock(nn.Module):
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


class RidgeGuidedSparseProfileResidual(nn.Module):
    def __init__(
        self,
        spec: RidgeGuidedSparseResidualSpec,
        *,
        base_adjacency: np.ndarray,
        residual_adjacency: np.ndarray,
        ridge_mean: np.ndarray,
        ridge_scale: np.ndarray,
        ridge_weights: np.ndarray,
    ) -> None:
        super().__init__()
        self.spec = spec
        base = _normalized_adjacency(base_adjacency)
        residual = _normalized_adjacency(residual_adjacency)
        mean = np.asarray(ridge_mean, dtype=np.float32)
        scale = np.asarray(ridge_scale, dtype=np.float32)
        weights = np.asarray(ridge_weights, dtype=np.float32)
        if mean.shape != (39,) or scale.shape != (39,) or np.any(scale <= 0):
            raise ValueError("ridge mean/scale must contain 39 valid entries")
        if weights.shape != (40,):
            raise ValueError("ridge weights must contain intercept plus 39 weights")
        self.register_buffer("base_adjacency", torch.from_numpy(base))
        self.register_buffer("residual_adjacency", torch.from_numpy(residual))
        self.register_buffer("ridge_mean", torch.from_numpy(mean.copy()))
        self.register_buffer("ridge_scale", torch.from_numpy(scale.copy()))
        self.register_buffer("ridge_weights", torch.from_numpy(weights.copy()))
        self.input_norm = nn.LayerNorm(spec.input_dim)
        self.input_projection = nn.Linear(spec.input_dim, spec.hidden_dim)
        self.block = _ResidualBlock(spec.hidden_dim, spec.dropout)
        self.output_norm = nn.LayerNorm(spec.hidden_dim)
        self.output_head = nn.Linear(spec.hidden_dim, 1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, prefix_features: torch.Tensor) -> torch.Tensor:
        return self.base_score(prefix_features) + self.residual_score(prefix_features)

    def base_score(self, prefix_features: torch.Tensor) -> torch.Tensor:
        expanded = self.expanded_base_features(prefix_features)
        standardized = (expanded - self.ridge_mean) / self.ridge_scale
        return self.ridge_weights[0] + standardized @ self.ridge_weights[1:]

    def residual_score(self, prefix_features: torch.Tensor) -> torch.Tensor:
        embedding = self.residual_embedding(prefix_features)
        residual = self.output_head(self.output_norm(embedding)).squeeze(-1)
        return self.spec.residual_bound * torch.tanh(residual)

    def residual_embedding(self, prefix_features: torch.Tensor) -> torch.Tensor:
        self._validate_features(prefix_features)
        state = self.input_projection(self.input_norm(prefix_features))
        for _ in range(self.spec.steps):
            cell_context, linear_context = self._relation_context(state)
            state = self.block(state, cell_context, linear_context)
        return state

    def expanded_base_features(self, prefix_features: torch.Tensor) -> torch.Tensor:
        self._validate_features(prefix_features)
        cell_context = _cell_mean(prefix_features)
        linear_context = torch.einsum(
            "ts,bsh->bth", self.base_adjacency, prefix_features
        )
        return torch.cat((prefix_features, cell_context, linear_context), dim=-1)

    def _relation_context(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spec.relation_mode == "independent":
            return state, state
        return _cell_mean(state), torch.einsum(
            "ts,bsh->bth", self.residual_adjacency, state
        )

    def _validate_features(self, prefix_features: torch.Tensor) -> None:
        if prefix_features.ndim != 3 or prefix_features.shape[1:] != (
            64,
            self.spec.input_dim,
        ):
            raise ValueError("prefix_features must have shape batch x 64 x input_dim")


def _normalized_adjacency(adjacency: np.ndarray) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=np.float32)
    if matrix.shape != (64, 64) or not np.isin(matrix, (0, 1)).all():
        raise ValueError("adjacency must be a binary 64 x 64 matrix")
    degree = matrix.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("every target node must have a predecessor")
    return matrix / degree[:, None]


def _cell_mean(state: torch.Tensor) -> torch.Tensor:
    mean = state.reshape(state.shape[0], 16, 4, state.shape[-1]).mean(dim=2)
    return mean[:, :, None, :].expand(-1, -1, 4, -1).reshape_as(state)


__all__ = ["RidgeGuidedSparseProfileResidual", "RidgeGuidedSparseResidualSpec"]
