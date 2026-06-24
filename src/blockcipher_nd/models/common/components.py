from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class Identity(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


class RmsNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        rms = features.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return features / rms * self.weight


def build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    if key == "mish":
        return nn.Mish()
    raise ValueError(f"unsupported activation: {name}")


def build_norm(name: str, normalized_shape: int) -> nn.Module:
    key = name.lower()
    if key in {"none", "identity"}:
        return Identity()
    if key == "layernorm":
        return nn.LayerNorm(normalized_shape)
    if key == "rmsnorm":
        return RmsNorm(normalized_shape)
    raise ValueError(f"unsupported norm: {name}")


class GatedAttentionPooling(nn.Module):
    """Attention pooling with a tanh/sigmoid gate, common in MIL models."""

    def __init__(self, embedding_bits: int, hidden_bits: int = 128) -> None:
        super().__init__()
        self.value = nn.Linear(embedding_bits, hidden_bits)
        self.gate = nn.Linear(embedding_bits, hidden_bits)
        self.score = nn.Linear(hidden_bits, 1)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3:
            raise ValueError(f"expected [batch, items, embedding], got {tuple(embeddings.shape)}")
        hidden = torch.tanh(self.value(embeddings)) * torch.sigmoid(self.gate(embeddings))
        logits = self.score(hidden).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class AttentionPooling(nn.Module):
    def __init__(
        self,
        embedding_bits: int,
        hidden_bits: int = 128,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            build_norm(norm, embedding_bits),
            nn.Linear(embedding_bits, hidden_bits),
            build_activation(activation),
            nn.Linear(hidden_bits, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.layers(embeddings).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class EvidencePooling(nn.Module):
    """MIL-style pooling for weak evidence concentrated in a few pair embeddings."""

    def __init__(
        self,
        embedding_bits: int,
        hidden_bits: int = 128,
        mode: str = "topk_logsumexp",
        top_k: int = 4,
        lse_temperature: float = 1.0,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        if mode not in {"topk_mean", "logsumexp", "topk_logsumexp"}:
            raise ValueError(f"unsupported evidence pooling mode: {mode}")
        if top_k < 1:
            raise ValueError("EvidencePooling top_k must be >= 1")
        if lse_temperature <= 0.0:
            raise ValueError("EvidencePooling lse_temperature must be > 0")
        self.mode = mode
        self.top_k = top_k
        self.lse_temperature = lse_temperature
        self.scorer = nn.Sequential(
            build_norm(norm, embedding_bits),
            nn.Linear(embedding_bits, hidden_bits),
            build_activation(activation),
            nn.Linear(hidden_bits, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3:
            raise ValueError(f"expected [batch, items, embedding], got {tuple(embeddings.shape)}")
        logits = self.scorer(embeddings).squeeze(-1)
        if self.mode == "logsumexp":
            weights = torch.softmax(logits / self.lse_temperature, dim=1)
            pooled = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
            return pooled, weights

        k = min(self.top_k, embeddings.shape[1])
        top_values, top_indices = torch.topk(logits, k=k, dim=1)
        if self.mode == "topk_mean":
            weights = torch.zeros_like(logits)
            weights.scatter_(1, top_indices, 1.0 / float(k))
        else:
            top_weights = torch.softmax(top_values / self.lse_temperature, dim=1)
            weights = torch.zeros_like(logits)
            weights.scatter_(1, top_indices, top_weights)
        pooled = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        return pooled, weights
