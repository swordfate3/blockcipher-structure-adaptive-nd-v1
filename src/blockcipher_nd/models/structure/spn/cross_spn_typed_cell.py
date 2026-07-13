from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import (
    AttentionPooling,
    build_activation,
    build_norm,
)
from blockcipher_nd.models.structure.spn.token_mixer_pairset import (
    SpnTokenMixerBlock,
    SpnTokenMixerPairSetDistinguisher,
)
from blockcipher_nd.registry.cipher_factory import build_cipher


_SHUFFLED_MAPPING_SEED = 20260627


def cipher_inverse_permutation_indices(
    cipher_key: str,
    mapping_mode: str,
) -> torch.Tensor:
    if mapping_mode == "raw":
        return torch.arange(64, dtype=torch.long)
    if mapping_mode == "shuffled":
        generator = torch.Generator().manual_seed(_SHUFFLED_MAPPING_SEED)
        return torch.randperm(64, generator=generator)
    if mapping_mode != "true":
        raise ValueError(f"unsupported mapping_mode: {mapping_mode}")

    cipher = build_cipher(cipher_key, rounds=1, key=0)
    inverse_permutation = getattr(cipher, "inverse_permutation_layer", None)
    if cipher.block_bits != 64 or inverse_permutation is None:
        raise ValueError(
            f"cipher does not expose a 64-bit inverse permutation: {cipher_key}"
        )

    indices = [-1] * 64
    for source_msb_index in range(64):
        source_state = 1 << (63 - source_msb_index)
        mapped_state = inverse_permutation(source_state)
        if mapped_state <= 0 or mapped_state & (mapped_state - 1):
            raise ValueError(f"inverse permutation is not one-hot for {cipher_key}")
        target_msb_index = 63 - (mapped_state.bit_length() - 1)
        indices[target_msb_index] = source_msb_index
    if sorted(indices) != list(range(64)):
        raise ValueError(f"inverse permutation is not bijective for {cipher_key}")
    return torch.tensor(indices, dtype=torch.long)


class CrossSpnTypedCellPairSetDistinguisher(nn.Module):
    def __init__(
        self,
        input_bits: int,
        cipher_key: str,
        mapping_mode: str,
        pair_bits: int = 128,
        base_channels: int = 32,
        token_dim: int | None = None,
        mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        pooling: str = "attention_mean_max",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError(
                "CrossSpnTypedCell expects raw 128-bit ciphertext pairs"
            )
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of pair_bits")
        if mixer_depth < 1:
            raise ValueError("mixer_depth must be >= 1")
        if pooling not in {"attention", "attention_mean_max", "mean_max"}:
            raise ValueError(f"unsupported pooling: {pooling}")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.cipher_key = cipher_key
        self.mapping_mode = mapping_mode
        self.token_dim = token_dim or max(16, base_channels * 2)
        self.pooling = pooling
        self.embedding_bits = max(32, base_channels * 4)
        self.register_buffer(
            "mapping_indices",
            cipher_inverse_permutation_indices(cipher_key, mapping_mode),
            persistent=False,
        )

        self.current_cell_encoder = nn.Sequential(
            nn.Linear(4, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.previous_cell_encoder = nn.Sequential(
            nn.Linear(4, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.typed_fusion = nn.Sequential(
            nn.Linear(self.token_dim * 2, self.token_dim),
            build_activation(activation),
            build_norm(norm, self.token_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 16, self.token_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.mixer_blocks = nn.ModuleList(
            [
                SpnTokenMixerBlock(
                    nibbles_per_pair=16,
                    token_dim=self.token_dim,
                    token_mlp_ratio=token_mlp_ratio,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for _ in range(mixer_depth)
            ]
        )
        self.sequence_norm = build_norm(norm, self.token_dim)
        self.pair_projection = nn.Sequential(
            nn.Linear(self.token_dim * 3, self.embedding_bits),
            build_activation(activation),
            nn.Dropout(dropout),
        )
        self.attention = AttentionPooling(
            self.embedding_bits,
            hidden_bits=max(32, base_channels * 4),
            activation=activation,
            norm=norm,
        )
        pooling_multiplier = 3 if pooling == "attention_mean_max" else 2 if pooling == "mean_max" else 1
        classifier_bits = self.embedding_bits * pooling_multiplier
        self.classifier = nn.Sequential(
            build_norm(norm, classifier_bits),
            nn.Linear(classifier_bits, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), 1),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def typed_cell_view(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pairs = features.float().reshape(
            features.shape[0], self.pairs_per_sample, 2, 64
        )
        difference = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
        previous_difference = difference.index_select(2, self.mapping_indices)
        return (
            difference.reshape(features.shape[0], self.pairs_per_sample, 16, 4),
            previous_difference.reshape(
                features.shape[0], self.pairs_per_sample, 16, 4
            ),
        )

    def encode_pairs(self, features: torch.Tensor) -> torch.Tensor:
        current, previous = self.typed_cell_view(features)
        batch = features.shape[0]
        current = current.reshape(batch * self.pairs_per_sample, 16, 4)
        previous = previous.reshape(batch * self.pairs_per_sample, 16, 4)
        current_hidden = self.current_cell_encoder(current)
        previous_hidden = self.previous_cell_encoder(previous)
        hidden = self.typed_fusion(
            torch.cat([current_hidden, previous_hidden], dim=2)
        )
        hidden = hidden + self.position_embedding
        for block in self.mixer_blocks:
            hidden = block(hidden)
        hidden = self.sequence_norm(hidden)
        mean_embedding = hidden.mean(dim=1)
        max_embedding = hidden.max(dim=1).values
        activity = current.mean(dim=2, keepdim=True)
        active_embedding = torch.sum(hidden * activity, dim=1) / (
            activity.sum(dim=1).clamp_min(1.0)
        )
        return self.pair_projection(
            torch.cat([mean_embedding, max_embedding, active_embedding], dim=1)
        ).reshape(batch, self.pairs_per_sample, self.embedding_bits)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pair_embeddings = self.encode_pairs(features)
        attention_embedding, attention_weights = self.attention(pair_embeddings)
        self.last_attention_weights = attention_weights.detach()
        if self.pooling == "attention":
            pooled = attention_embedding
        else:
            mean_embedding = pair_embeddings.mean(dim=1)
            max_embedding = pair_embeddings.max(dim=1).values
            if self.pooling == "mean_max":
                pooled = torch.cat([mean_embedding, max_embedding], dim=1)
            else:
                pooled = torch.cat(
                    [attention_embedding, mean_embedding, max_embedding], dim=1
                )
        return self.classifier(pooled)


def _set_fixed_adapter(
    kwargs: dict[str, object],
    *,
    cipher_key: str,
    mapping_mode: str,
) -> None:
    requested_cipher = kwargs.get("cipher_key", cipher_key)
    requested_mapping = kwargs.get("mapping_mode", mapping_mode)
    if requested_cipher != cipher_key:
        raise ValueError(
            f"fixed cipher {cipher_key!r} received conflicting value "
            f"{requested_cipher!r}"
        )
    if requested_mapping != mapping_mode:
        raise ValueError(
            f"fixed mapping {mapping_mode!r} received conflicting value "
            f"{requested_mapping!r}"
        )
    kwargs["cipher_key"] = cipher_key
    kwargs["mapping_mode"] = mapping_mode


class PresentCrossSpnTypedCellTrueDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="true")
        super().__init__(*args, **kwargs)


class PresentCrossSpnTypedCellShuffledDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="shuffled")
        super().__init__(*args, **kwargs)


class PresentCrossSpnTypedCellRawDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="present80", mapping_mode="raw")
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellTrueDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="gift64", mapping_mode="true")
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellShuffledDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="gift64", mapping_mode="shuffled")
        super().__init__(*args, **kwargs)


class GiftCrossSpnTypedCellRawDistinguisher(
    CrossSpnTypedCellPairSetDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_adapter(kwargs, cipher_key="gift64", mapping_mode="raw")
        super().__init__(*args, **kwargs)


class GiftAlignedTokenMixerRawInputDistinguisher(nn.Module):
    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        token_dim: int | None = None,
        mixer_depth: int = 1,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        pooling: str = "topk_logsumexp",
        dropout: float = 0.0,
        top_k: int = 2,
        lse_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError(
                "GiftAlignedTokenMixerRawInput expects raw 128-bit ciphertext pairs"
            )
        if input_bits <= 0 or input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a positive multiple of pair_bits")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.register_buffer(
            "mapping_indices",
            cipher_inverse_permutation_indices("gift64", "true"),
            persistent=False,
        )
        self.delegate = SpnTokenMixerPairSetDistinguisher(
            input_bits=self.pairs_per_sample * 256,
            pair_bits=256,
            base_channels=base_channels,
            token_dim=token_dim,
            mixer_depth=mixer_depth,
            token_mlp_ratio=token_mlp_ratio,
            activation=activation,
            norm=norm,
            pooling=pooling,
            dropout=dropout,
            top_k=top_k,
            lse_temperature=lse_temperature,
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def aligned_view(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        pairs = features.float().reshape(
            features.shape[0], self.pairs_per_sample, 2, 64
        )
        difference = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
        mapped_difference = difference.index_select(2, self.mapping_indices)
        return torch.cat(
            [pairs[:, :, 0], pairs[:, :, 1], difference, mapped_difference],
            dim=2,
        ).reshape(features.shape[0], self.pairs_per_sample * 256)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.delegate(self.aligned_view(features))


__all__ = [
    "CrossSpnTypedCellPairSetDistinguisher",
    "GiftAlignedTokenMixerRawInputDistinguisher",
    "GiftCrossSpnTypedCellRawDistinguisher",
    "GiftCrossSpnTypedCellShuffledDistinguisher",
    "GiftCrossSpnTypedCellTrueDistinguisher",
    "PresentCrossSpnTypedCellRawDistinguisher",
    "PresentCrossSpnTypedCellShuffledDistinguisher",
    "PresentCrossSpnTypedCellTrueDistinguisher",
    "cipher_inverse_permutation_indices",
]
