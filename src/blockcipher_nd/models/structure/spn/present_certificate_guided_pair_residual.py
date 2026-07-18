from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.small_spn_pair_relation_models import (
    SmallSpnPairRelationReasoner,
    SmallSpnPairRelationSpec,
)


@dataclass(frozen=True)
class PresentCgprSpec:
    residual_mode: str = "pair"
    topology_mode: str = "true"
    state_bits: int = 64
    prefix_dimensions: int = 39
    hidden_dim: int = 16
    path_rank: int = 2
    dropout: float = 0.10
    residual_bound: float = 0.25

    def __post_init__(self) -> None:
        if self.residual_mode not in {"prefix", "pair"}:
            raise ValueError("residual_mode must be prefix or pair")
        if self.topology_mode not in {"true", "corrupted"}:
            raise ValueError("topology_mode must be true or corrupted")
        if self.state_bits != 64 or self.prefix_dimensions != 39:
            raise ValueError("E50 requires 64 state bits and 39 prefix features")
        if self.hidden_dim <= 0 or self.path_rank <= 0:
            raise ValueError("hidden dimensions must be positive")
        if self.residual_bound <= 0:
            raise ValueError("residual_bound must be positive")


class PresentCertificateGuidedPairResidual(nn.Module):
    def __init__(
        self,
        spec: PresentCgprSpec,
        *,
        sboxes: np.ndarray,
        players: np.ndarray,
        structure_active_bits: np.ndarray,
        output_mask_bits: np.ndarray,
        ridge_mean: np.ndarray,
        ridge_scale: np.ndarray,
        ridge_weights: np.ndarray,
    ) -> None:
        super().__init__()
        self.spec = spec
        mean = np.asarray(ridge_mean, dtype=np.float32)
        scale = np.asarray(ridge_scale, dtype=np.float32)
        weights = np.asarray(ridge_weights, dtype=np.float32)
        if mean.shape != (spec.prefix_dimensions,):
            raise ValueError("ridge_mean must have 39 entries")
        if scale.shape != mean.shape or np.any(scale <= 0):
            raise ValueError("ridge_scale must be positive with 39 entries")
        if weights.shape != (spec.prefix_dimensions + 1,):
            raise ValueError("ridge_weights must contain intercept plus 39 weights")
        self.register_buffer("ridge_mean", torch.from_numpy(mean.copy()))
        self.register_buffer("ridge_scale", torch.from_numpy(scale.copy()))
        self.register_buffer("ridge_weights", torch.from_numpy(weights.copy()))
        if spec.residual_mode == "prefix":
            self.prefix_residual = nn.Sequential(
                nn.LayerNorm(spec.prefix_dimensions),
                nn.Linear(spec.prefix_dimensions, 128),
                nn.SiLU(),
                nn.Dropout(spec.dropout),
                nn.Linear(128, 42),
                nn.SiLU(),
                nn.Dropout(spec.dropout),
                nn.Linear(42, 1),
            )
            residual_head = self.prefix_residual[-1]
        else:
            self.pair_residual = SmallSpnPairRelationReasoner(
                SmallSpnPairRelationSpec(
                    topology_mode=spec.topology_mode,
                    processor_mode="triangle",
                    state_bits=spec.state_bits,
                    round_categories=1,
                    round_step_offset=4,
                    hidden_dim=spec.hidden_dim,
                    path_rank=spec.path_rank,
                    dropout=spec.dropout,
                ),
                sboxes=sboxes,
                players=players,
                structure_active_bits=structure_active_bits,
                output_mask_bits=output_mask_bits,
            )
            residual_head = self.pair_residual.head[-1]
        nn.init.zeros_(residual_head.weight)
        nn.init.zeros_(residual_head.bias)

    def forward(
        self,
        variant_index: torch.Tensor,
        round_index: torch.Tensor,
        structure_index: torch.Tensor,
        mask_index: torch.Tensor,
        prefix_features: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_score(prefix_features)
        if self.spec.residual_mode == "prefix":
            standardized = self.standardize_prefix(prefix_features)
            residual = self.prefix_residual(standardized).squeeze(-1)
        else:
            residual = self.pair_residual(
                variant_index, round_index, structure_index, mask_index
            )
        return base + self.spec.residual_bound * torch.tanh(residual)

    def standardize_prefix(self, prefix_features: torch.Tensor) -> torch.Tensor:
        return (prefix_features - self.ridge_mean) / self.ridge_scale

    def base_score(self, prefix_features: torch.Tensor) -> torch.Tensor:
        standardized = self.standardize_prefix(prefix_features)
        return self.ridge_weights[0] + standardized @ self.ridge_weights[1:]

    def pair_embedding(
        self,
        variant_index: torch.Tensor,
        round_index: torch.Tensor,
        structure_index: torch.Tensor,
        mask_index: torch.Tensor,
    ) -> torch.Tensor:
        if self.spec.residual_mode != "pair":
            raise RuntimeError("prefix-only residual has no pair embedding")
        return self.pair_residual.encode(
            variant_index, round_index, structure_index, mask_index
        )
