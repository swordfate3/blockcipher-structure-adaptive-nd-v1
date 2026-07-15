from __future__ import annotations

import torch
from torch import nn


def sm4_state_mapping_indices(mapping_mode: str = "true") -> torch.Tensor:
    """Map serialized SM4 output bits to four chronological state words."""

    if mapping_mode == "true":
        # SM4 serializes X[r+3], X[r+2], X[r+1], X[r].
        return torch.tensor(
            [index for word in (3, 2, 1, 0) for index in range(32 * word, 32 * (word + 1))],
            dtype=torch.long,
        )
    if mapping_mode == "shuffled":
        generator = torch.Generator().manual_seed(0x534D34)
        return torch.randperm(128, generator=generator)
    raise ValueError(f"unsupported SM4 mapping mode: {mapping_mode}")


class _CircularResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


class Sm4WordRecurrenceDistinguisher(nn.Module):
    """SM4 distinguisher preserving four-word recurrence and rotation roles."""

    def __init__(
        self,
        input_bits: int,
        *,
        mapping_mode: str = "true",
        pair_bits: int = 256,
        base_channels: int = 32,
        blocks: int = 3,
        classifier_bits: int = 64,
        dropout: float = 0.5,
        rotation_offsets: tuple[int, ...] = (2, 10, 18, 24),
    ) -> None:
        super().__init__()
        if pair_bits != 256:
            raise ValueError("SM4 word-recurrence input requires 256 bits per pair")
        if input_bits % pair_bits != 0:
            raise ValueError("input_bits must contain complete SM4 ciphertext pairs")
        if blocks < 1:
            raise ValueError("SM4 word-recurrence model requires at least one block")
        if not rotation_offsets or any(offset <= 0 or offset >= 32 for offset in rotation_offsets):
            raise ValueError("SM4 rotation offsets must be between 1 and 31")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.mapping_mode = mapping_mode
        self.rotation_offsets = tuple(rotation_offsets)
        self.register_buffer(
            "mapping_indices",
            sm4_state_mapping_indices(mapping_mode),
            persistent=True,
        )

        # 12 raw word-role channels, 3 feedback channels, 3 three-word XOR
        # channels, and one three-channel copy per SM4 linear rotation.
        semantic_channels = 18 + 3 * len(self.rotation_offsets)
        self.stem = nn.Sequential(
            nn.Conv1d(semantic_channels, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[_CircularResidualBlock(base_channels) for _ in range(blocks)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4, classifier_bits),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_bits, 1),
        )

    def canonical_words(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pair_tensor = features.float().reshape(
            features.shape[0], self.pairs_per_sample, 2, 128
        )
        mapped = pair_tensor.index_select(-1, self.mapping_indices)
        return mapped.reshape(features.shape[0], self.pairs_per_sample, 2, 4, 32)

    @staticmethod
    def _xor3(words: torch.Tensor) -> torch.Tensor:
        return torch.logical_xor(
            torch.logical_xor(words[:, :, 1].bool(), words[:, :, 2].bool()),
            words[:, :, 3].bool(),
        ).float()

    def semantic_channels(self, features: torch.Tensor) -> torch.Tensor:
        words = self.canonical_words(features)
        first = words[:, :, 0]
        second = words[:, :, 1]
        difference = torch.logical_xor(first.bool(), second.bool()).float()

        raw_roles = torch.stack((first, second, difference), dim=2).flatten(2, 3)
        feedback = torch.stack(
            (first[:, :, 0], second[:, :, 0], difference[:, :, 0]), dim=2
        )
        triples = torch.stack(
            (self._xor3(first), self._xor3(second), self._xor3(difference)),
            dim=2,
        )
        rotated = torch.cat(
            [
                torch.roll(triples, shifts=-offset, dims=-1)
                for offset in self.rotation_offsets
            ],
            dim=2,
        )
        return torch.cat((raw_roles, feedback, triples, rotated), dim=2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        semantic = self.semantic_channels(features)
        batch, pairs, channels, width = semantic.shape
        hidden = self.stem(semantic.reshape(batch * pairs, channels, width))
        hidden = self.blocks(hidden)
        per_pair = torch.cat((hidden.mean(dim=-1), hidden.amax(dim=-1)), dim=-1)
        per_pair = per_pair.reshape(batch, pairs, -1)
        summary = torch.cat((per_pair.mean(dim=1), per_pair.amax(dim=1)), dim=-1)
        return self.classifier(summary)


__all__ = ["Sm4WordRecurrenceDistinguisher", "sm4_state_mapping_indices"]
