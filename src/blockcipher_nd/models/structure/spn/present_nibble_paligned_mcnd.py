from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import EvidencePooling, build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_zhang_wang_keras import (
    PresentZhangWangKerasMCNDDistinguisher,
)
from blockcipher_nd.models.structure.spn.token_mixer_pairset import SpnTokenMixerBlock


class PresentNibblePAlignedMCNDDistinguisher(nn.Module):
    """Zhang/Wang MCND backbone fused with a minimal PRESENT SPN cell view."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        blocks: int = 5,
        spn_token_dim: int | None = None,
        spn_mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        initial_kernel_sizes: tuple[int, ...] = (1, 2, 4),
        residual_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("PresentNibblePAlignedMCND expects raw 128-bit ciphertext pairs")
        if input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a multiple of pair_bits")
        if spn_mixer_depth < 1:
            raise ValueError("spn_mixer_depth must be >= 1")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits

        self.raw_branch = PresentZhangWangKerasMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits,
            base_channels=base_channels,
            blocks=blocks,
            activation=activation,
            dropout=dropout,
            initial_kernel_sizes=initial_kernel_sizes,
            residual_kernel_size=residual_kernel_size,
        )
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
        )
        self.spn_embedding_dim = self.spn_encoder.embedding_bits
        self.classifier = nn.Sequential(
            build_norm(norm, self.raw_branch.embedding_bits + self.spn_embedding_dim * 2),
            nn.Linear(self.raw_branch.embedding_bits + self.spn_embedding_dim * 2, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        raw_embedding = self.raw_branch.encode(features)
        spn_pair_embeddings = self.spn_encoder(features)
        spn_mean = spn_pair_embeddings.mean(dim=1)
        spn_max = spn_pair_embeddings.max(dim=1).values
        return self.classifier(torch.cat([raw_embedding, spn_mean, spn_max], dim=1))

    def _present_nibble_paligned_view(self, features: torch.Tensor) -> torch.Tensor:
        return self.spn_encoder.present_nibble_paligned_view(features)

    def _encode_spn_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.spn_encoder.encode_spn_pairs(pair_features)


class PresentNibblePAlignedSpnOnlyDistinguisher(nn.Module):
    """PRESENT Delta/InvP nibble view without the Zhang/Wang raw MCND branch."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        spn_token_dim: int | None = None,
        spn_mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        view_mode: str = "delta_inv_p",
        p_alignment: str = "true",
    ) -> None:
        super().__init__()
        self.input_bits = input_bits
        self.pair_bits = pair_bits
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
            view_mode=view_mode,
            p_alignment=p_alignment,
        )
        self.classifier = nn.Sequential(
            build_norm(norm, self.spn_encoder.embedding_bits * 2),
            nn.Linear(self.spn_encoder.embedding_bits * 2, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        spn_pair_embeddings = self.spn_encoder(features)
        spn_mean = spn_pair_embeddings.mean(dim=1)
        spn_max = spn_pair_embeddings.max(dim=1).values
        return self.classifier(torch.cat([spn_mean, spn_max], dim=1))


class PresentNibblePAlignedGatedMCNDDistinguisher(nn.Module):
    """Zhang/Wang MCND backbone modulated by a PRESENT Delta/InvP nibble gate."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        blocks: int = 5,
        spn_token_dim: int | None = None,
        spn_mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        initial_kernel_sizes: tuple[int, ...] = (1, 2, 4),
        residual_kernel_size: int = 3,
        gate_scale: float = 0.25,
        p_alignment: str = "true",
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("PresentNibblePAlignedGatedMCND expects raw 128-bit ciphertext pairs")
        if input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a multiple of pair_bits")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.gate_scale = float(gate_scale)

        self.raw_branch = PresentZhangWangKerasMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits,
            base_channels=base_channels,
            blocks=blocks,
            activation=activation,
            dropout=dropout,
            initial_kernel_sizes=initial_kernel_sizes,
            residual_kernel_size=residual_kernel_size,
        )
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
            p_alignment=p_alignment,
        )
        self.gate = nn.Sequential(
            build_norm(norm, self.spn_encoder.embedding_bits * 2),
            nn.Linear(self.spn_encoder.embedding_bits * 2, self.raw_branch.embedding_bits),
        )
        classifier_bits = self.raw_branch.embedding_bits + self.spn_encoder.embedding_bits * 2
        self.classifier = nn.Sequential(
            build_norm(norm, classifier_bits),
            nn.Linear(classifier_bits, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        raw_embedding = self.raw_branch.encode(features)
        spn_pair_embeddings = self.spn_encoder(features)
        spn_mean = spn_pair_embeddings.mean(dim=1)
        spn_max = spn_pair_embeddings.max(dim=1).values
        spn_summary = torch.cat([spn_mean, spn_max], dim=1)
        gate = torch.sigmoid(self.gate(spn_summary))
        gated_raw = raw_embedding * (1.0 + self.gate_scale * gate)
        return self.classifier(torch.cat([gated_raw, spn_summary], dim=1))


class PresentNibbleShuffledPAlignedGatedMCNDDistinguisher(PresentNibblePAlignedGatedMCNDDistinguisher):
    """Gated MCND control with a fixed shuffled pseudo P-layer alignment."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["p_alignment"] = "shuffled"
        super().__init__(*args, **kwargs)


class PresentNibbleDeltaOnlySpnOnlyDistinguisher(PresentNibblePAlignedSpnOnlyDistinguisher):
    """SPN-only attribution control that keeps only DeltaC nibble tokens."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["view_mode"] = "delta"
        super().__init__(*args, **kwargs)


class PresentNibbleInvPOnlySpnOnlyDistinguisher(PresentNibblePAlignedSpnOnlyDistinguisher):
    """SPN-only attribution control that keeps only InvP(DeltaC) nibble tokens."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["view_mode"] = "inv_p"
        super().__init__(*args, **kwargs)


class PresentNibbleShuffledPAlignedSpnOnlyDistinguisher(PresentNibblePAlignedSpnOnlyDistinguisher):
    """SPN-only attribution control with deterministic shuffled pseudo P alignment."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["p_alignment"] = "shuffled"
        super().__init__(*args, **kwargs)


class PresentNibblePAlignedTransitionDistinguisher(nn.Module):
    """PRESENT SPN-only backbone with pair-level evidence pooling."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        spn_token_dim: int | None = None,
        spn_mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        pooling: str = "topk_logsumexp",
        top_k: int = 4,
        lse_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_bits = input_bits
        self.pair_bits = pair_bits
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
        )
        self.evidence_pool = EvidencePooling(
            self.spn_encoder.embedding_bits,
            hidden_bits=max(32, base_channels * 4),
            mode=pooling,
            top_k=top_k,
            lse_temperature=lse_temperature,
            activation=activation,
            norm=norm,
        )
        self.classifier = nn.Sequential(
            build_norm(norm, self.spn_encoder.embedding_bits),
            nn.Linear(self.spn_encoder.embedding_bits, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), 1),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        pair_embeddings = self.spn_encoder(features)
        pooled, weights = self.evidence_pool(pair_embeddings)
        self.last_attention_weights = weights.detach()
        return self.classifier(pooled)


class PresentNibblePAlignedTransitionResidualDistinguisher(nn.Module):
    """PRESENT SPN backbone over DeltaC -> InvP(DeltaC) nibble transitions."""

    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        transition_token_dim: int | None = None,
        transition_mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        pooling: str = "topk_logsumexp",
        top_k: int = 4,
        lse_temperature: float = 1.0,
        p_alignment: str = "true",
    ) -> None:
        super().__init__()
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.transition_encoder = _PresentNibbleTransitionResidualEncoder(
            input_bits=input_bits,
            pair_bits=pair_bits,
            base_channels=base_channels,
            transition_token_dim=transition_token_dim,
            transition_mixer_depth=transition_mixer_depth,
            token_mlp_ratio=token_mlp_ratio,
            activation=activation,
            norm=norm,
            dropout=dropout,
            p_alignment=p_alignment,
        )
        self.evidence_pool = EvidencePooling(
            self.transition_encoder.embedding_bits,
            hidden_bits=max(32, base_channels * 4),
            mode=pooling,
            top_k=top_k,
            lse_temperature=lse_temperature,
            activation=activation,
            norm=norm,
        )
        self.classifier = nn.Sequential(
            build_norm(norm, self.transition_encoder.embedding_bits),
            nn.Linear(self.transition_encoder.embedding_bits, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), 1),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        pair_embeddings = self.transition_encoder(features)
        pooled, weights = self.evidence_pool(pair_embeddings)
        self.last_attention_weights = weights.detach()
        return self.classifier(pooled)


class PresentNibbleShuffledTransitionResidualDistinguisher(
    PresentNibblePAlignedTransitionResidualDistinguisher
):
    """Transition-residual control with deterministic shuffled pseudo P alignment."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["p_alignment"] = "shuffled"
        super().__init__(*args, **kwargs)


class _PresentNibblePAlignedSpnEncoder(nn.Module):
    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        spn_token_dim: int | None = None,
        spn_mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        p_alignment: str = "true",
        view_mode: str = "delta_inv_p",
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("PRESENT nibble P-aligned encoder expects raw 128-bit ciphertext pairs")
        if input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a multiple of pair_bits")
        if spn_mixer_depth < 1:
            raise ValueError("spn_mixer_depth must be >= 1")
        if p_alignment not in {"true", "shuffled"}:
            raise ValueError(f"unsupported p_alignment: {p_alignment}")
        if view_mode not in {"delta_inv_p", "delta", "inv_p"}:
            raise ValueError(f"unsupported view_mode: {view_mode}")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.view_mode = view_mode
        self.spn_pair_bits = 128 if view_mode == "delta_inv_p" else 64
        self.spn_nibbles_per_pair = self.spn_pair_bits // 4
        self.spn_token_dim = spn_token_dim or max(16, base_channels * 2)
        self.embedding_bits = max(32, base_channels * 4)

        self.spn_cell_encoder = nn.Sequential(
            nn.Linear(4, self.spn_token_dim),
            build_activation(activation),
            build_norm(norm, self.spn_token_dim),
        )
        self.spn_position_embedding = nn.Parameter(
            torch.zeros(1, self.spn_nibbles_per_pair, self.spn_token_dim)
        )
        nn.init.trunc_normal_(self.spn_position_embedding, std=0.02)
        self.spn_mixers = nn.ModuleList(
            [
                SpnTokenMixerBlock(
                    nibbles_per_pair=self.spn_nibbles_per_pair,
                    token_dim=self.spn_token_dim,
                    token_mlp_ratio=token_mlp_ratio,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for _ in range(spn_mixer_depth)
            ]
        )
        self.spn_norm = build_norm(norm, self.spn_token_dim)
        self.spn_pair_projection = nn.Sequential(
            nn.Linear(self.spn_token_dim * 3, self.embedding_bits),
            build_activation(activation),
            nn.Dropout(dropout),
        )

        if p_alignment == "true":
            inverse_p = [_present_inverse_p_index(index) for index in range(64)]
        else:
            generator = torch.Generator().manual_seed(20260627)
            inverse_p = torch.randperm(64, generator=generator).tolist()
        self.register_buffer("inverse_p_indices", torch.tensor(inverse_p, dtype=torch.long), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        return self.encode_spn_pairs(self.present_nibble_paligned_view(features.float()))

    def present_nibble_paligned_view(self, features: torch.Tensor) -> torch.Tensor:
        raw_pairs = features.reshape(features.shape[0], self.pairs_per_sample, 2, 64)
        difference = (raw_pairs[:, :, 0, :] - raw_pairs[:, :, 1, :]).abs()
        aligned_difference = difference.index_select(dim=2, index=self.inverse_p_indices)
        if self.view_mode == "delta":
            view = difference
        elif self.view_mode == "inv_p":
            view = aligned_difference
        else:
            view = torch.cat([difference, aligned_difference], dim=2)
        cells = view.reshape(
            features.shape[0],
            self.pairs_per_sample,
            self.spn_nibbles_per_pair,
            4,
        )
        return cells.permute(0, 1, 3, 2).reshape(features.shape[0], self.pairs_per_sample, self.spn_pair_bits)

    def encode_spn_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        nibbles = pair_features.reshape(
            pair_features.shape[0] * self.pairs_per_sample,
            4,
            self.spn_nibbles_per_pair,
        ).transpose(1, 2).reshape(
            pair_features.shape[0] * self.pairs_per_sample,
            self.spn_nibbles_per_pair,
            4,
        )
        hidden = self.spn_cell_encoder(nibbles) + self.spn_position_embedding
        for mixer in self.spn_mixers:
            hidden = mixer(hidden)
        hidden = self.spn_norm(hidden)
        mean_embedding = hidden.mean(dim=1)
        max_embedding = hidden.max(dim=1).values
        active_embedding = torch.sum(hidden * nibbles.mean(dim=2, keepdim=True), dim=1) / (
            nibbles.mean(dim=2, keepdim=True).sum(dim=1).clamp_min(1.0)
        )
        projected = self.spn_pair_projection(
            torch.cat([mean_embedding, max_embedding, active_embedding], dim=1)
        )
        return projected.reshape(pair_features.shape[0], self.pairs_per_sample, self.embedding_bits)


class _PresentNibbleTransitionResidualEncoder(nn.Module):
    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        transition_token_dim: int | None = None,
        transition_mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        activation: str = "relu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        p_alignment: str = "true",
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("PRESENT transition residual encoder expects raw 128-bit ciphertext pairs")
        if input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a multiple of pair_bits")
        if transition_mixer_depth < 1:
            raise ValueError("transition_mixer_depth must be >= 1")
        if p_alignment not in {"true", "shuffled"}:
            raise ValueError(f"unsupported p_alignment: {p_alignment}")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.nibbles_per_pair = 16
        self.transition_token_dim = transition_token_dim or max(16, base_channels * 2)
        self.embedding_bits = max(32, base_channels * 4)

        self.transition_encoder = nn.Sequential(
            nn.Linear(12, self.transition_token_dim),
            build_activation(activation),
            build_norm(norm, self.transition_token_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.nibbles_per_pair, self.transition_token_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.mixers = nn.ModuleList(
            [
                SpnTokenMixerBlock(
                    nibbles_per_pair=self.nibbles_per_pair,
                    token_dim=self.transition_token_dim,
                    token_mlp_ratio=token_mlp_ratio,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for _ in range(transition_mixer_depth)
            ]
        )
        self.norm = build_norm(norm, self.transition_token_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.transition_token_dim * 4, self.embedding_bits),
            build_activation(activation),
            nn.Dropout(dropout),
        )

        if p_alignment == "true":
            inverse_p = [_present_inverse_p_index(index) for index in range(64)]
        else:
            generator = torch.Generator().manual_seed(20260627)
            inverse_p = torch.randperm(64, generator=generator).tolist()
        self.register_buffer("inverse_p_indices", torch.tensor(inverse_p, dtype=torch.long), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw_pairs = features.float().reshape(features.shape[0], self.pairs_per_sample, 2, 64)
        difference = (raw_pairs[:, :, 0, :] - raw_pairs[:, :, 1, :]).abs()
        aligned_difference = difference.index_select(dim=2, index=self.inverse_p_indices)
        source = difference.reshape(features.shape[0] * self.pairs_per_sample, 16, 4)
        target = aligned_difference.reshape(features.shape[0] * self.pairs_per_sample, 16, 4)
        transition = torch.cat([source, target, target - source], dim=2)
        hidden = self.transition_encoder(transition) + self.position_embedding
        for mixer in self.mixers:
            hidden = mixer(hidden)
        hidden = self.norm(hidden)
        mean_embedding = hidden.mean(dim=1)
        max_embedding = hidden.max(dim=1).values
        active = (source + target).mean(dim=2, keepdim=True)
        active_embedding = torch.sum(hidden * active, dim=1) / active.sum(dim=1).clamp_min(1.0)
        transition_embedding = (hidden[:, 8:, :].mean(dim=1) - hidden[:, :8, :].mean(dim=1))
        projected = self.projection(
            torch.cat([mean_embedding, max_embedding, active_embedding, transition_embedding], dim=1)
        )
        return projected.reshape(features.shape[0], self.pairs_per_sample, self.embedding_bits)


def _present_inverse_p_index(target_bit_index: int) -> int:
    source_lsb_index = _present_inverse_p_lsb_index(63 - target_bit_index)
    return 63 - source_lsb_index


def _present_inverse_p_lsb_index(target_lsb_index: int) -> int:
    if target_lsb_index == 63:
        return 63
    return (16 * target_lsb_index) % 63


__all__ = [
    "PresentNibblePAlignedMCNDDistinguisher",
    "PresentNibblePAlignedSpnOnlyDistinguisher",
    "PresentNibbleDeltaOnlySpnOnlyDistinguisher",
    "PresentNibbleInvPOnlySpnOnlyDistinguisher",
    "PresentNibblePAlignedGatedMCNDDistinguisher",
    "PresentNibbleShuffledPAlignedGatedMCNDDistinguisher",
    "PresentNibbleShuffledPAlignedSpnOnlyDistinguisher",
    "PresentNibblePAlignedTransitionDistinguisher",
    "PresentNibblePAlignedTransitionResidualDistinguisher",
    "PresentNibbleShuffledTransitionResidualDistinguisher",
]
