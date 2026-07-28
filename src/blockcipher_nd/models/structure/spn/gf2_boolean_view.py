from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn

from blockcipher_nd.models.common.components import AttentionPooling
from blockcipher_nd.models.structure.spn.operator_tied_latent import (
    invariant_pool,
    operator_routing_fingerprint,
    segment_mean,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


VIEW_NAMES = ("raw", "single_0", "single_1", "composed_0_after_1")


@dataclass(frozen=True)
class Gf2BooleanViewSpnSpec:
    hidden_dim: int = 32
    pair_embedding_dim: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.pair_embedding_dim) <= 0:
            raise ValueError("GF(2) Boolean-view dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("GF(2) Boolean-view dropout must be in [0, 1)")


class Gf2BooleanViewSpnDistinguisher(nn.Module):
    """Width-independent SPN head over deterministic runtime GF(2) views."""

    def __init__(self, spec: Gf2BooleanViewSpnSpec) -> None:
        super().__init__()
        self.spec = spec
        hidden = spec.hidden_dim
        pair_dim = spec.pair_embedding_dim
        self.bit_encoder = nn.Sequential(
            nn.Linear(12, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(hidden * 6, pair_dim),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
        )
        self.pair_attention = AttentionPooling(
            pair_dim,
            hidden_bits=pair_dim,
            activation="relu",
            norm="layernorm",
        )
        classifier_hidden = hidden * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_dim * 3),
            nn.Linear(pair_dim * 3, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(classifier_hidden, 1),
        )
        self.last_pair_attention: torch.Tensor | None = None

    def forward(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
    ) -> torch.Tensor:
        return self.classifier(self.encode(ciphertext_pairs, structure))

    def encode(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
    ) -> torch.Tensor:
        views = gf2_boolean_views(ciphertext_pairs, structure)
        batch, pair_count, bit_count, _ = views.shape
        hidden = self.bit_encoder(views).reshape(
            batch * pair_count,
            bit_count,
            self.spec.hidden_dim,
        )
        membership = structure.cell_membership.to(hidden.device)
        cell_hidden = segment_mean(hidden, membership, structure.cells)
        pooled = torch.cat(
            (invariant_pool(hidden), invariant_pool(cell_hidden)),
            dim=-1,
        )
        pair_embeddings = self.pair_projection(pooled).reshape(
            batch,
            pair_count,
            self.spec.pair_embedding_dim,
        )
        attended, attention = self.pair_attention(pair_embeddings)
        self.last_pair_attention = attention.detach()
        return torch.cat(
            (
                attended,
                pair_embeddings.mean(dim=1),
                pair_embeddings.max(dim=1).values,
            ),
            dim=-1,
        )


class FixedGf2BooleanViewSpnProtocolAdapter(nn.Module):
    """Bind a shared Boolean-view head to one external two-transition window."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        structure: RuntimeSpnStructure,
        spec: Gf2BooleanViewSpnSpec,
        descriptor_name: str,
        descriptor_path: str,
        descriptor_sha256: str,
        descriptor_round_start: int,
        descriptor_available_rounds: int,
        runtime_structure_mode: str,
        runtime_structure_window_control: str,
    ) -> None:
        super().__init__()
        if pair_bits != 2 * structure.block_bits:
            raise ValueError("GF(2) Boolean-view pair_bits must encode two blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError(
                "GF(2) Boolean-view input_bits must contain complete pairs"
            )
        if structure.rounds != 2:
            raise ValueError("K1-I adapter requires exactly two runtime transitions")
        self.backbone = Gf2BooleanViewSpnDistinguisher(spec)
        self.runtime_structure = structure
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_loaded_rounds = structure.rounds
        self.runtime_round_window_mode = "gf2_boolean_view"
        self.runtime_structure_window_control = runtime_structure_window_control
        self.runtime_structure_descriptor_name = descriptor_name
        self.runtime_structure_descriptor_path = descriptor_path
        self.runtime_structure_descriptor_sha256 = descriptor_sha256
        self.runtime_structure_round_start = descriptor_round_start
        self.runtime_structure_available_rounds = descriptor_available_rounds
        self.runtime_structure_mode = runtime_structure_mode
        self.runtime_structure_transition_sha256s = structure.transition_sha256s()
        self.runtime_structure_window_sha256 = structure.window_sha256()
        self.runtime_structure_unique_transition_count = (
            structure.unique_transition_count
        )
        self.runtime_structure_homogeneous = structure.is_homogeneous
        self.operator_routing_sha256 = operator_routing_fingerprint(structure)
        self.boolean_view_sha256 = boolean_view_fingerprint(structure)
        self.deterministic_gf2_views = True
        self.boolean_view_names = VIEW_NAMES
        self.boolean_channels_per_bit = 12
        self.uses_raw_bypass = False
        self.uses_learned_message_passing = False
        self.uses_path_tokens = False
        self.uses_absolute_cell_or_bit_identity = False
        self.uses_cipher_identity = False
        self.uses_sbox_semantics = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        runtime = features.reshape(
            features.shape[0],
            -1,
            2,
            self.runtime_structure.block_bits,
        ).flip(-1)
        return self.backbone(runtime, self.runtime_structure)


def gf2_boolean_views(
    ciphertext_pairs: torch.Tensor,
    structure: RuntimeSpnStructure,
) -> torch.Tensor:
    if ciphertext_pairs.ndim != 4 or ciphertext_pairs.shape[2] != 2:
        raise ValueError("ciphertext pairs must have shape [batch, pairs, 2, bits]")
    if ciphertext_pairs.shape[-1] != structure.block_bits:
        raise ValueError("ciphertext pair width does not match runtime structure")
    if structure.rounds != 2:
        raise ValueError("GF(2) Boolean views require exactly two transitions")
    left = ciphertext_pairs[:, :, 0]
    right = ciphertext_pairs[:, :, 1]
    raw = torch.stack((left, right, torch.remainder(left + right, 2.0)), dim=-1)
    first, second = structure.inverse_linear_matrices
    single_0 = apply_gf2_operator(raw, first)
    single_1 = apply_gf2_operator(raw, second)
    composed = apply_gf2_operator(single_1, first)
    return torch.cat((raw, single_0, single_1, composed), dim=-1)


def apply_gf2_operator(values: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("GF(2) values must include bit and channel dimensions")
    bits = values.shape[-2]
    matrix = torch.as_tensor(operator, dtype=torch.uint8, device="cpu")
    if matrix.shape != (bits, bits):
        raise ValueError("GF(2) operator dimensions do not match values")
    if not torch.all((matrix == 0) | (matrix == 1)):
        raise ValueError("GF(2) operator must be binary")
    if not torch.all((values == 0) | (values == 1)):
        raise ValueError("GF(2) Boolean views require binary input values")
    transformed = torch.matmul(matrix.to(values.device, values.dtype), values)
    return torch.remainder(transformed, 2.0)


def boolean_view_fingerprint(structure: RuntimeSpnStructure) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(VIEW_NAMES).encode("ascii"))
    digest.update(operator_routing_fingerprint(structure).encode("ascii"))
    return digest.hexdigest()


__all__ = [
    "FixedGf2BooleanViewSpnProtocolAdapter",
    "Gf2BooleanViewSpnDistinguisher",
    "Gf2BooleanViewSpnSpec",
    "VIEW_NAMES",
    "apply_gf2_operator",
    "boolean_view_fingerprint",
    "gf2_boolean_views",
]
