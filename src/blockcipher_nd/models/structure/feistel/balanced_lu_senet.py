from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.structure.feistel.balanced_round_relation import (
    balanced_feistel_relation_channels,
)


class _LuSeResidualBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: int) -> None:
        super().__init__()
        squeeze_bits = max(1, channels // se_ratio)
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.excitation = nn.Sequential(
            nn.Linear(channels, squeeze_bits, bias=False),
            nn.ReLU(),
            nn.Linear(squeeze_bits, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.layers(features)
        weights = self.excitation(residual.mean(dim=2)).unsqueeze(2)
        return features + residual * weights


class BalancedFeistelLuSeNetDistinguisher(nn.Module):
    """Lu-style SE-ResNet over eight position-preserving relation groups."""

    def __init__(
        self,
        input_bits: int,
        *,
        round_function: str,
        mapping_mode: str = "true",
        pair_bits: int = 128,
        base_channels: int = 32,
        blocks: int = 5,
        classifier_bits: int = 64,
        se_ratio: int = 16,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("Lu-layout Feistel models require 128 bits per pair")
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of 128")
        if input_bits // pair_bits != 8:
            raise ValueError("Lu-layout Feistel models require exactly eight pairs")
        if round_function not in {"simon", "simeck"}:
            raise ValueError(
                f"unsupported balanced Feistel round function: {round_function}"
            )
        if mapping_mode not in {"true", "shuffled"}:
            raise ValueError(f"unsupported mapping mode: {mapping_mode}")
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        if se_ratio < 1:
            raise ValueError("se_ratio must be >= 1")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = 8
        self.round_function = round_function
        self.mapping_mode = mapping_mode
        relation_bits_per_pair = 8 * 32
        self.stem = nn.Sequential(
            nn.Conv1d(relation_bits_per_pair, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *(_LuSeResidualBlock(base_channels, se_ratio) for _ in range(blocks))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * self.pairs_per_sample, classifier_bits),
            nn.BatchNorm1d(classifier_bits),
            nn.ReLU(),
            nn.Linear(classifier_bits, classifier_bits),
            nn.BatchNorm1d(classifier_bits),
            nn.ReLU(),
            nn.Linear(classifier_bits, 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def relation_sequence(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pairs = features.reshape(features.shape[0], self.pairs_per_sample, 2, 64)
        channels = balanced_feistel_relation_channels(
            pairs,
            round_function=self.round_function,
            mapping_mode=self.mapping_mode,
        )
        return channels.reshape(features.shape[0], self.pairs_per_sample, 256)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence = self.relation_sequence(features).transpose(1, 2)
        hidden = self.residual_blocks(self.stem(sequence))
        return self.classifier(hidden)


__all__ = ["BalancedFeistelLuSeNetDistinguisher"]
