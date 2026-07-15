from __future__ import annotations

import torch
from torch import nn


def _xor(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.logical_xor(left, right)


def simon_round_function_bits(word: torch.Tensor) -> torch.Tensor:
    """Apply the keyless SIMON round function to MSB-first bit tensors."""

    bits = word.bool()
    rotate_8 = torch.roll(bits, shifts=-8, dims=-1)
    rotate_1 = torch.roll(bits, shifts=-1, dims=-1)
    rotate_2 = torch.roll(bits, shifts=-2, dims=-1)
    return _xor(rotate_8 & rotate_1, rotate_2)


def simeck_round_function_bits(word: torch.Tensor) -> torch.Tensor:
    """Apply the keyless SIMECK round function to MSB-first bit tensors."""

    bits = word.bool()
    rotate_5 = torch.roll(bits, shifts=-5, dims=-1)
    rotate_1 = torch.roll(bits, shifts=-1, dims=-1)
    return _xor(rotate_5 & bits, rotate_1)


def balanced_feistel_relation_channels(
    features: torch.Tensor,
    *,
    round_function: str,
    mapping_mode: str = "true",
) -> torch.Tensor:
    """Derive Lu et al.'s eight 32-bit relation channels per ciphertext pair."""

    if features.ndim != 4 or features.shape[2:] != (2, 64):
        raise ValueError(
            "expected [batch, pairs, 2, 64] ciphertext-pair bits, "
            f"got {tuple(features.shape)}"
        )
    if mapping_mode not in {"true", "shuffled"}:
        raise ValueError(f"unsupported mapping mode: {mapping_mode}")
    if round_function == "simon":
        cipher_function = simon_round_function_bits
    elif round_function == "simeck":
        cipher_function = simeck_round_function_bits
    else:
        raise ValueError(
            f"unsupported balanced Feistel round function: {round_function}"
        )

    ciphertexts = features.bool().reshape(*features.shape[:3], 2, 32)
    if mapping_mode == "shuffled":
        ciphertexts = ciphertexts.flip(dims=(3,))

    first_left = ciphertexts[:, :, 0, 0]
    first_right = ciphertexts[:, :, 0, 1]
    second_left = ciphertexts[:, :, 1, 0]
    second_right = ciphertexts[:, :, 1, 1]

    delta_left = _xor(first_left, second_left)
    delta_right = _xor(first_right, second_right)
    first_previous = _xor(first_left, cipher_function(first_right))
    second_previous = _xor(second_left, cipher_function(second_right))
    delta_previous = _xor(first_previous, second_previous)
    first_pseudo_previous2 = _xor(first_right, cipher_function(first_previous))
    second_pseudo_previous2 = _xor(second_right, cipher_function(second_previous))
    delta_pseudo_previous2 = _xor(first_pseudo_previous2, second_pseudo_previous2)

    return torch.stack(
        (
            delta_left,
            delta_right,
            first_left,
            first_right,
            second_left,
            second_right,
            delta_previous,
            delta_pseudo_previous2,
        ),
        dim=2,
    ).float()


class _RelationResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


class BalancedFeistelRoundRelationDistinguisher(nn.Module):
    """Pair-set model using cipher-specific previous-round Feistel relations."""

    def __init__(
        self,
        input_bits: int,
        *,
        round_function: str,
        mapping_mode: str = "true",
        pair_bits: int = 128,
        base_channels: int = 32,
        blocks: int = 3,
        classifier_bits: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError(
                "balanced Feistel relation models require 128 bits per pair"
            )
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of 128")
        if round_function not in {"simon", "simeck"}:
            raise ValueError(
                f"unsupported balanced Feistel round function: {round_function}"
            )
        if mapping_mode not in {"true", "shuffled"}:
            raise ValueError(f"unsupported mapping mode: {mapping_mode}")
        if blocks < 1:
            raise ValueError("blocks must be >= 1")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.round_function = round_function
        self.mapping_mode = mapping_mode
        self.stem = nn.Sequential(
            nn.Conv1d(8, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *(_RelationResidualBlock(base_channels, dropout) for _ in range(blocks))
        )
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(base_channels * 2),
            nn.Linear(base_channels * 2, classifier_bits),
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

    def relation_channels(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pairs = features.reshape(features.shape[0], self.pairs_per_sample, 2, 64)
        return balanced_feistel_relation_channels(
            pairs,
            round_function=self.round_function,
            mapping_mode=self.mapping_mode,
        )

    def encode_pairs(self, features: torch.Tensor) -> torch.Tensor:
        channels = self.relation_channels(features)
        batch = features.shape[0]
        hidden = channels.reshape(batch * self.pairs_per_sample, 8, 32)
        hidden = self.residual_blocks(self.stem(hidden))
        pooled = torch.cat(
            (hidden.mean(dim=2), hidden.max(dim=2).values),
            dim=1,
        )
        return self.pair_projection(pooled).reshape(batch, self.pairs_per_sample, -1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pair_embeddings = self.encode_pairs(features)
        pooled = torch.cat(
            (
                pair_embeddings.mean(dim=1),
                pair_embeddings.max(dim=1).values,
            ),
            dim=1,
        )
        return self.classifier(pooled)


__all__ = [
    "BalancedFeistelRoundRelationDistinguisher",
    "balanced_feistel_relation_channels",
    "simeck_round_function_bits",
    "simon_round_function_bits",
]
