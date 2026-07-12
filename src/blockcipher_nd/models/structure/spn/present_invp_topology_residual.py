from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_inception_blocks import conv2d_norm
from blockcipher_nd.models.structure.spn.present_invp_state_matrix_conv2d import (
    PresentStateMatrixResidualBlock,
)
from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
    _PresentNibblePAlignedSpnEncoder,
    present_inverse_p_indices,
)


class PresentNibbleTopologyResidualSpnOnlyDistinguisher(nn.Module):
    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        spn_token_dim: int | None = None,
        spn_mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        local_channels: int = 16,
        local_depth: int = 1,
        local_kernel_size: int = 3,
        local_residual_scale_init: float = 0.1,
        activation: str = "relu",
        norm: str = "layernorm",
        local_norm: str = "batchnorm2d",
        dropout: float = 0.0,
        mapping_mode: str = "true",
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError(
                "PresentNibbleTopologyResidual expects raw 128-bit ciphertext pairs"
            )
        if input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a multiple of pair_bits")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if local_channels <= 0:
            raise ValueError("local_channels must be positive")
        if local_depth != 1:
            raise ValueError("local_depth must equal 1")
        if local_kernel_size < 1 or local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be a positive odd integer")
        if mapping_mode not in {"true", "shuffled", "delta"}:
            raise ValueError(f"unsupported mapping_mode: {mapping_mode}")

        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.mapping_mode = mapping_mode

        self.spn_encoder = _PresentNibblePAlignedSpnEncoder(
            input_bits=input_bits,
            pair_bits=pair_bits,
            base_channels=base_channels,
            spn_token_dim=spn_token_dim,
            spn_mixer_depth=spn_mixer_depth,
            token_mlp_ratio=token_mlp_ratio,
            activation=activation,
            norm=norm,
            dropout=dropout,
            view_mode="inv_p",
            p_alignment="true",
        )
        embedding_bits = self.spn_encoder.embedding_bits
        classifier_hidden = max(64, base_channels * 8)
        self.classifier = nn.Sequential(
            build_norm(norm, embedding_bits * 2),
            nn.Linear(embedding_bits * 2, classifier_hidden),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

        mapping_indices = (
            torch.arange(64, dtype=torch.long)
            if mapping_mode == "delta"
            else present_inverse_p_indices(mapping_mode)
        )
        self.register_buffer("mapping_indices", mapping_indices, persistent=False)
        self.local_stem = nn.Sequential(
            nn.Conv2d(1, local_channels, kernel_size=1),
            conv2d_norm(local_norm, local_channels),
            build_activation(activation),
        )
        self.local_blocks = nn.ModuleList(
            [
                PresentStateMatrixResidualBlock(
                    channels=local_channels,
                    kernel_size=local_kernel_size,
                    activation=activation,
                    norm=local_norm,
                    dropout=dropout,
                )
            ]
        )
        self.local_projection = nn.Linear(local_channels * 2, embedding_bits)
        self.alpha = nn.Parameter(torch.tensor(float(local_residual_scale_init)))

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def local_state_matrix_view(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        raw_pairs = features.float().reshape(
            features.shape[0], self.pairs_per_sample, 2, 64
        )
        difference = (raw_pairs[:, :, 0] - raw_pairs[:, :, 1]).abs()
        mapped = difference.index_select(dim=2, index=self.mapping_indices)
        return mapped.reshape(features.shape[0], self.pairs_per_sample, 16, 4).permute(
            0, 1, 3, 2
        )

    def encode_local_pairs(self, state_matrices: torch.Tensor) -> torch.Tensor:
        expected_shape = (self.pairs_per_sample, 4, 16)
        if (
            state_matrices.ndim != 4
            or tuple(state_matrices.shape[1:]) != expected_shape
        ):
            raise ValueError(
                "expected local state matrices with shape "
                f"[batch, {self.pairs_per_sample}, 4, 16], "
                f"got {tuple(state_matrices.shape)}"
            )
        batch = state_matrices.shape[0]
        hidden = state_matrices.reshape(batch * self.pairs_per_sample, 1, 4, 16)
        hidden = self.local_stem(hidden)
        for block in self.local_blocks:
            hidden = block(hidden)
        pooled = torch.cat(
            [hidden.mean(dim=(2, 3)), hidden.amax(dim=(2, 3))],
            dim=1,
        )
        return self.local_projection(pooled).reshape(
            batch,
            self.pairs_per_sample,
            self.spn_encoder.embedding_bits,
        )

    def encode_fused_pairs(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()
        token_pair_embedding = self.spn_encoder(features)
        local_pair_embedding = self.encode_local_pairs(
            self.local_state_matrix_view(features)
        )
        return token_pair_embedding + self.alpha * local_pair_embedding

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        fused = self.encode_fused_pairs(features)
        summary = torch.cat(
            [fused.mean(dim=1), fused.max(dim=1).values],
            dim=1,
        )
        return self.classifier(summary)


class PresentNibbleInvPTopologyResidualSpnOnlyDistinguisher(
    PresentNibbleTopologyResidualSpnOnlyDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_mapping(kwargs, "true")
        super().__init__(*args, **kwargs)


class PresentNibbleShuffledPTopologyResidualSpnOnlyDistinguisher(
    PresentNibbleTopologyResidualSpnOnlyDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_mapping(kwargs, "shuffled")
        super().__init__(*args, **kwargs)


class PresentNibbleDeltaTopologyResidualSpnOnlyDistinguisher(
    PresentNibbleTopologyResidualSpnOnlyDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        _set_fixed_mapping(kwargs, "delta")
        super().__init__(*args, **kwargs)


def _set_fixed_mapping(kwargs: dict, fixed_mapping: str) -> None:
    requested_mapping = kwargs.get("mapping_mode", fixed_mapping)
    if requested_mapping != fixed_mapping:
        raise ValueError(
            f"fixed mapping {fixed_mapping!r} received conflicting value "
            f"{requested_mapping!r}"
        )
    kwargs["mapping_mode"] = fixed_mapping


__all__ = [
    "PresentNibbleTopologyResidualSpnOnlyDistinguisher",
    "PresentNibbleInvPTopologyResidualSpnOnlyDistinguisher",
    "PresentNibbleShuffledPTopologyResidualSpnOnlyDistinguisher",
    "PresentNibbleDeltaTopologyResidualSpnOnlyDistinguisher",
]
