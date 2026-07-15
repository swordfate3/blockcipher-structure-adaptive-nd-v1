from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from blockcipher_nd.ciphers.feistel.des import DES_IP


_SHUFFLED_MAPPING_SEED = 20260715


def des_canonical_bit_indices(mapping_mode: str = "true") -> torch.Tensor:
    """Map external DES ciphertext bits to the final internal (L, R) state."""

    if mapping_mode == "raw":
        return torch.arange(64, dtype=torch.long)
    if mapping_mode == "shuffled":
        generator = torch.Generator().manual_seed(_SHUFFLED_MAPPING_SEED)
        return torch.randperm(64, generator=generator)
    if mapping_mode != "true":
        raise ValueError(f"unsupported DES mapping mode: {mapping_mode}")

    # IP(ciphertext) is the DES preoutput state R_r || L_r. Swap the halves so
    # every model sees the round-function state in the canonical L_r || R_r order.
    return torch.tensor(
        [position - 1 for position in DES_IP[32:]]
        + [position - 1 for position in DES_IP[:32]],
        dtype=torch.long,
    )


class _SameLengthConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        return self.conv(F.pad(features, (left, right)))


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _SameLengthConv1d(channels, channels, kernel_size),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            _SameLengthConv1d(channels, channels, kernel_size),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


class _OfficialResidualBlock(nn.Module):
    """Residual block matching the public Zhang/Wang DES implementation."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _SameLengthConv1d(channels, channels, kernel_size),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            _SameLengthConv1d(channels, channels, kernel_size),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.layers(features)


class DesFeistelBranchInceptionPairSetDistinguisher(nn.Module):
    """DES pair-set model with explicit Feistel branch roles and interactions."""

    def __init__(
        self,
        input_bits: int,
        *,
        mapping_mode: str = "true",
        pair_bits: int = 128,
        base_channels: int = 32,
        blocks: int = 3,
        initial_kernel_sizes: tuple[int, ...] = (1, 4, 6),
        classifier_bits: int = 128,
        dropout: float = 0.0,
        include_branch_interactions: bool = True,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("DES Feistel models require 128 bits per ciphertext pair")
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of 128")
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        if not initial_kernel_sizes:
            raise ValueError("initial_kernel_sizes must not be empty")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.mapping_mode = mapping_mode
        self.include_branch_interactions = include_branch_interactions
        self.register_buffer(
            "mapping_indices",
            des_canonical_bit_indices(mapping_mode),
            persistent=False,
        )

        input_channels = 8 if include_branch_interactions else 4
        self.initial_branches = nn.ModuleList(
            [
                _SameLengthConv1d(input_channels, base_channels, kernel_size)
                for kernel_size in initial_kernel_sizes
            ]
        )
        channels = base_channels * len(initial_kernel_sizes)
        self.initial_norm = nn.BatchNorm1d(channels)
        self.residual_blocks = nn.Sequential(
            *(
                _ResidualBlock(channels, 3 + 2 * block_index, dropout)
                for block_index in range(blocks)
            )
        )
        pair_embedding_bits = channels * 2
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(pair_embedding_bits),
            nn.Linear(pair_embedding_bits, classifier_bits),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_bits * 2),
            nn.Linear(classifier_bits * 2, classifier_bits),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_bits, 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def canonical_pairs(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pairs = features.float().reshape(
            features.shape[0], self.pairs_per_sample, 2, 64
        )
        return pairs.index_select(3, self.mapping_indices.to(features.device)).reshape(
            features.shape[0], self.pairs_per_sample, 2, 2, 32
        )

    def branch_channels(self, features: torch.Tensor) -> torch.Tensor:
        pairs = self.canonical_pairs(features)
        first_left = pairs[:, :, 0, 0]
        first_right = pairs[:, :, 0, 1]
        second_left = pairs[:, :, 1, 0]
        second_right = pairs[:, :, 1, 1]
        base_channels = [first_left, first_right, second_left, second_right]
        if self.include_branch_interactions:
            base_channels.extend(
                [
                    (first_left - second_left).abs(),
                    (first_right - second_right).abs(),
                    (first_left - first_right).abs(),
                    (second_left - second_right).abs(),
                ]
            )
        return torch.stack(base_channels, dim=2)

    def encode_pairs(self, features: torch.Tensor) -> torch.Tensor:
        channels = self.branch_channels(features)
        batch = features.shape[0]
        hidden = channels.reshape(
            batch * self.pairs_per_sample, channels.shape[2], 32
        )
        hidden = torch.cat(
            [branch(hidden) for branch in self.initial_branches], dim=1
        )
        hidden = torch.relu(self.initial_norm(hidden))
        hidden = self.residual_blocks(hidden)
        embedding = torch.cat(
            [hidden.mean(dim=2), hidden.max(dim=2).values], dim=1
        )
        return self.pair_projection(embedding).reshape(
            batch, self.pairs_per_sample, -1
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pair_embeddings = self.encode_pairs(features)
        pooled = torch.cat(
            [
                pair_embeddings.mean(dim=1),
                pair_embeddings.max(dim=1).values,
            ],
            dim=1,
        )
        return self.classifier(pooled)


class DesZhangWangInceptionPairSetDistinguisher(
    DesFeistelBranchInceptionPairSetDistinguisher
):
    """Same-protocol Zhang/Wang-style Inception-ResNet family baseline."""

    def __init__(self, input_bits: int, **kwargs: object) -> None:
        super().__init__(
            input_bits,
            mapping_mode="true",
            include_branch_interactions=False,
            **kwargs,
        )


class DesZhangWangOfficialLayoutDistinguisher(nn.Module):
    """PyTorch port of the public Zhang/Wang DES Inception-ResNet layout."""

    def __init__(
        self,
        input_bits: int,
        *,
        mapping_mode: str = "true",
        pair_bits: int = 128,
        base_channels: int = 32,
        blocks: int = 5,
        initial_kernel_sizes: tuple[int, ...] = (1, 4, 6),
        include_branch_interactions: bool = False,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("official DES layout requires 128 bits per pair")
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of 128")
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        if not initial_kernel_sizes:
            raise ValueError("initial_kernel_sizes must not be empty")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.mapping_mode = mapping_mode
        self.include_branch_interactions = include_branch_interactions
        self.register_buffer(
            "mapping_indices",
            des_canonical_bit_indices(mapping_mode),
            persistent=False,
        )

        input_channels = 8 if include_branch_interactions else 4
        self.initial_branches = nn.ModuleList(
            [
                _SameLengthConv1d(input_channels, base_channels, kernel_size)
                for kernel_size in initial_kernel_sizes
            ]
        )
        channels = base_channels * len(initial_kernel_sizes)
        self.initial_norm = nn.BatchNorm1d(channels)
        self.residual_blocks = nn.Sequential(
            *(
                _OfficialResidualBlock(channels, 3 + 2 * block_index)
                for block_index in range(blocks)
            )
        )
        self.classifier = nn.Linear(channels, 1)

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def canonical_pairs(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pairs = features.float().reshape(
            features.shape[0], self.pairs_per_sample, 2, 64
        )
        return pairs.index_select(3, self.mapping_indices.to(features.device)).reshape(
            features.shape[0], self.pairs_per_sample, 2, 2, 32
        )

    def branch_channels(self, features: torch.Tensor) -> torch.Tensor:
        pairs = self.canonical_pairs(features)
        first_left = pairs[:, :, 0, 0]
        first_right = pairs[:, :, 0, 1]
        second_left = pairs[:, :, 1, 0]
        second_right = pairs[:, :, 1, 1]
        channels = [first_left, first_right, second_left, second_right]
        if self.include_branch_interactions:
            channels.extend(
                [
                    (first_left - second_left).abs(),
                    (first_right - second_right).abs(),
                    (first_left - first_right).abs(),
                    (second_left - second_right).abs(),
                ]
            )
        return torch.stack(channels, dim=2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channels = self.branch_channels(features)
        batch = features.shape[0]
        hidden = channels.reshape(
            batch * self.pairs_per_sample, channels.shape[2], 32
        )
        hidden = torch.cat(
            [branch(hidden) for branch in self.initial_branches], dim=1
        )
        hidden = torch.relu(self.initial_norm(hidden))
        hidden = self.residual_blocks(hidden)
        hidden = hidden.reshape(batch, self.pairs_per_sample, hidden.shape[1], 32)
        return self.classifier(hidden.mean(dim=(1, 3)))


class DesLstmPairSetDistinguisher(nn.Module):
    """Bidirectional LSTM baseline over canonical DES left/right bit positions."""

    def __init__(
        self,
        input_bits: int,
        *,
        pair_bits: int = 128,
        hidden_bits: int = 128,
        classifier_bits: int = 128,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("DES LSTM requires 128 bits per ciphertext pair")
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of 128")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.register_buffer(
            "mapping_indices",
            des_canonical_bit_indices("true"),
            persistent=False,
        )
        self.encoder = nn.LSTM(
            input_size=4,
            hidden_size=hidden_bits,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_bits * 4),
            nn.Linear(hidden_bits * 4, classifier_bits),
            nn.ReLU(),
            nn.Linear(classifier_bits, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        batch = features.shape[0]
        pairs = features.float().reshape(batch, self.pairs_per_sample, 2, 64)
        pairs = pairs.index_select(3, self.mapping_indices.to(features.device)).reshape(
            batch, self.pairs_per_sample, 2, 2, 32
        )
        sequence = pairs.permute(0, 1, 4, 2, 3).reshape(
            batch * self.pairs_per_sample, 32, 4
        )
        _, (hidden, _) = self.encoder(sequence)
        pair_embeddings = torch.cat([hidden[-2], hidden[-1]], dim=1).reshape(
            batch, self.pairs_per_sample, -1
        )
        pooled = torch.cat(
            [pair_embeddings.mean(dim=1), pair_embeddings.max(dim=1).values],
            dim=1,
        )
        return self.classifier(pooled)
