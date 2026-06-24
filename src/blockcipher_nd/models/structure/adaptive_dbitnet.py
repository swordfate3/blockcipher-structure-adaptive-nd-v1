from __future__ import annotations

import math

import torch
from torch import nn

from blockcipher_nd.models.common.components import (
    AttentionPooling,
    GatedAttentionPooling,
    build_activation,
    build_norm,
)


def adaptive_dbitnet_dilations(input_bits: int) -> list[int]:
    if input_bits % 2 != 0:
        raise ValueError("AdaptiveDBitNet requires an even number of input bits")
    if input_bits < 16:
        raise ValueError("AdaptiveDBitNet requires at least 16 input bits")

    rates: list[int] = []
    dilation = input_bits // 2 - 1
    while dilation >= 3:
        rates.append(dilation)
        dilation = (dilation + 1) // 2 - 1
    return rates


def structure_conditioned_dilations(input_bits: int, structure: str) -> list[int]:
    """Return DBitNet dilation rates with structure-specific priors first.

    The generic DBitNet schedule starts from long-range interactions.  For
    innovation-one experiments we prepend small, interpretable receptive fields
    that match common cipher structures while keeping the generic wide schedule.
    """

    base = adaptive_dbitnet_dilations(input_bits)
    structure_priors = {
        "ARX": [15, 5],
        "SPN": [3, 7],
        "Feistel-like": [max(3, input_bits // 2 - 1), max(3, input_bits // 4 - 1)],
    }
    priors = structure_priors.get(structure, [])
    candidates = [rate for rate in [*priors, *base] if 1 <= rate < input_bits]
    rates: list[int] = []
    width = input_bits
    for rate in dict.fromkeys(candidates):
        if width - rate <= 0:
            continue
        rates.append(rate)
        width -= rate
    return rates


def structure_bit_mask(pair_bits: int, structure: str) -> torch.Tensor:
    """Create a deterministic, learnable prior over pair feature positions."""

    if pair_bits < 16 or pair_bits % 2 != 0:
        raise ValueError("structure bit masks require an even pair_bits >= 16")
    mask = torch.ones(pair_bits, dtype=torch.float32)
    if structure == "ARX":
        word = max(1, pair_bits // 6)
        for offset in (0, word // 2, word, 2 * word):
            mask[offset::word] = 1.35
    elif structure == "SPN":
        cell = 4
        for offset in range(0, pair_bits, cell * 2):
            mask[offset : min(offset + cell, pair_bits)] = 1.30
    elif structure == "Feistel-like":
        branch = max(1, pair_bits // 6)
        mask[branch : min(branch * 2, pair_bits)] = 1.25
        mask[branch * 3 : min(branch * 4, pair_bits)] = 1.25
        mask[branch * 5 :] = 1.35
    return mask


class AdaptiveDBitNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=2,
                dilation=dilation,
                padding=0,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class AdaptiveDBitNetEncoder(nn.Module):
    def __init__(
        self,
        input_bits: int,
        base_channels: int = 32,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        if input_bits % 2 != 0:
            raise ValueError("AdaptiveDBitNet requires an even number of input bits")
        if input_bits < 16:
            raise ValueError("AdaptiveDBitNet requires at least 16 input bits")
        self.input_bits = input_bits
        self.dilations = adaptive_dbitnet_dilations(input_bits) if dilations is None else dilations
        self.output_width = self._output_width(input_bits, self.dilations)
        self.output_channels = base_channels + (len(self.dilations) - 1) * 16
        self.embedding_bits = self.output_channels * self.output_width

        in_channels = 1
        blocks: list[nn.Module] = []
        for index, dilation in enumerate(self.dilations):
            out_channels = base_channels + index * 16
            blocks.append(AdaptiveDBitNetBlock(in_channels, out_channels, dilation))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        hidden = features.float().unsqueeze(1)
        hidden = self.features(hidden)
        return hidden.flatten(start_dim=1)

    @staticmethod
    def _output_width(input_bits: int, dilations: list[int]) -> int:
        width = input_bits
        for dilation in dilations:
            width -= dilation
            if width <= 0:
                raise ValueError(
                    "AdaptiveDBitNet dilation schedule collapsed the feature width"
                )
        return width


class AdaptiveDBitNetDistinguisher(nn.Module):
    """Input-size adaptive DBitNet-style dilated CNN.

    This follows the DBitNet idea of deriving long-range dilation rates from the
    input width, then using a fixed strong prediction head across input sizes.
    """

    def __init__(self, input_bits: int, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = AdaptiveDBitNetEncoder(input_bits, base_channels)
        self.input_bits = self.encoder.input_bits
        self.dilations = self.encoder.dilations
        self.output_width = self.encoder.output_width
        self.output_channels = self.encoder.output_channels
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.embedding_bits, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))


class PairwiseAdaptiveDBitNetDistinguisher(nn.Module):
    """Shared pair encoder plus cross-pair pooling for multi-pair inputs."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 96,
        base_channels: int = 32,
        pooling: str = "mean_max",
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError("PairwiseAdaptiveDBitNet input_bits must be a multiple of pair_bits")
        if pooling not in {"mean", "max", "mean_max"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pooling = pooling
        self.pairs_per_sample = input_bits // pair_bits
        self.encoder = AdaptiveDBitNetEncoder(pair_bits, base_channels)
        pooling_multiplier = 2 if pooling == "mean_max" else 1
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.embedding_bits * pooling_multiplier, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        pair_features = features.float().reshape(
            features.shape[0] * self.pairs_per_sample,
            self.pair_bits,
        )
        embeddings = self.encoder(pair_features).reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.encoder.embedding_bits,
        )
        mean_embedding = embeddings.mean(dim=1)
        max_embedding = embeddings.max(dim=1).values
        if self.pooling == "mean":
            pooled = mean_embedding
        elif self.pooling == "max":
            pooled = max_embedding
        else:
            pooled = torch.cat([mean_embedding, max_embedding], dim=1)
        return self.classifier(pooled)


class StructureConditionedDBitNetEncoder(nn.Module):
    """DBitNet encoder with structure-conditioned dilation and bit-mask priors."""

    def __init__(
        self,
        input_bits: int,
        base_channels: int = 32,
        structure: str = "generic",
    ) -> None:
        super().__init__()
        self.input_bits = input_bits
        self.base_channels = base_channels
        self.structure = structure
        self.dilations = structure_conditioned_dilations(input_bits, structure)
        self.backbone = AdaptiveDBitNetEncoder(
            input_bits,
            base_channels,
            dilations=self.dilations,
        )
        self.embedding_bits = self.backbone.embedding_bits
        self.output_width = self.backbone.output_width
        self.output_channels = self.backbone.output_channels
        self.mask_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.register_buffer("bit_mask", structure_bit_mask(input_bits, structure))

    def set_cipher_structure(self, structure: str) -> None:
        self.structure = structure
        self.bit_mask.copy_(structure_bit_mask(self.input_bits, structure).to(self.bit_mask))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        scale = 1.0 + self.mask_scale.tanh() * (self.bit_mask - 1.0)
        return self.backbone(features.float() * scale)


class StructureAdaptivePairSetDBitNetDistinguisher(nn.Module):
    """Structure-conditioned pair-set DBitNet distinguisher.

    The model keeps pair encoding permutation-aware through a shared encoder,
    then aggregates pair embeddings with attention, mean, and max statistics.
    """

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 96,
        base_channels: int = 32,
        structure: str = "generic",
        pooling: str = "attention_mean_max",
    ) -> None:
        super().__init__()
        if input_bits % pair_bits != 0:
            raise ValueError(
                "StructureAdaptivePairSetDBitNet input_bits must be a multiple of pair_bits"
            )
        if pooling not in {"attention", "attention_mean_max", "mean_max"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.structure = structure
        self.pooling = pooling
        self.encoder = StructureConditionedDBitNetEncoder(
            pair_bits,
            base_channels=base_channels,
            structure=structure,
        )
        self.attention = nn.Sequential(
            nn.Linear(self.encoder.embedding_bits, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        pooling_multiplier = 3 if pooling == "attention_mean_max" else 2 if pooling == "mean_max" else 1
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.embedding_bits * pooling_multiplier, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def set_cipher_structure(self, structure: str) -> None:
        self.structure = structure
        self.encoder.set_cipher_structure(structure)

    def set_structure_features(self, features: torch.Tensor) -> None:
        structure_index = int(features[:3].argmax().item()) if features.numel() >= 3 else -1
        mapping = {0: "ARX", 1: "SPN", 2: "Feistel-like"}
        if structure_index in mapping:
            self.set_cipher_structure(mapping[structure_index])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        pair_features = features.float().reshape(
            features.shape[0] * self.pairs_per_sample,
            self.pair_bits,
        )
        embeddings = self.encoder(pair_features).reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.encoder.embedding_bits,
        )
        attention_logits = self.attention(embeddings).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        self.last_attention_weights = attention_weights.detach()
        attention_embedding = torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=1)
        mean_embedding = embeddings.mean(dim=1)
        max_embedding = embeddings.max(dim=1).values

        if self.pooling == "attention":
            pooled = attention_embedding
        elif self.pooling == "mean_max":
            pooled = torch.cat([mean_embedding, max_embedding], dim=1)
        else:
            pooled = torch.cat([attention_embedding, mean_embedding, max_embedding], dim=1)
        return self.classifier(pooled)
