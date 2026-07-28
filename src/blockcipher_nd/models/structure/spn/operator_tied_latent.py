from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn

from blockcipher_nd.models.common.components import AttentionPooling
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


@dataclass(frozen=True)
class OperatorTiedLatentSpnSpec:
    hidden_dim: int = 32
    pair_embedding_dim: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.pair_embedding_dim) <= 0:
            raise ValueError("operator-tied latent dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("operator-tied latent dropout must be in [0, 1)")


class SharedOperatorResidualBlock(nn.Module):
    """Update bit latents after fixed, non-trainable operator transport."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        inverse_operator: torch.Tensor,
        cell_membership: torch.Tensor,
        *,
        cell_count: int,
    ) -> torch.Tensor:
        normalized = self.state_norm(state)
        message = operator_support_mean(normalized, inverse_operator)
        cell_values = segment_mean(normalized, cell_membership, cell_count)
        cell_context = cell_values[:, cell_membership.to(state.device)]
        delta = self.update(torch.cat((normalized, message, cell_context), dim=-1))
        return self.output_norm(state + delta)


class OperatorTiedLatentSpnDistinguisher(nn.Module):
    """Permutation-equivariant bit processor routed by exact runtime operators."""

    def __init__(self, spec: OperatorTiedLatentSpnSpec) -> None:
        super().__init__()
        self.spec = spec
        hidden = spec.hidden_dim
        pair_dim = spec.pair_embedding_dim
        self.bit_encoder = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.shared_residual = SharedOperatorResidualBlock(hidden, spec.dropout)
        self.sequence_norm = nn.LayerNorm(hidden)
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
        if ciphertext_pairs.ndim != 4 or ciphertext_pairs.shape[2] != 2:
            raise ValueError("ciphertext pairs must have shape [batch, pairs, 2, bits]")
        if ciphertext_pairs.shape[-1] != structure.block_bits:
            raise ValueError("ciphertext pair width does not match runtime structure")
        left = ciphertext_pairs[:, :, 0]
        right = ciphertext_pairs[:, :, 1]
        bit_inputs = torch.stack(
            (left, right, torch.remainder(left + right, 2.0)), dim=-1
        )
        batch, pair_count, bit_count, _ = bit_inputs.shape
        hidden = self.bit_encoder(bit_inputs).reshape(
            batch * pair_count,
            bit_count,
            self.spec.hidden_dim,
        )
        membership = structure.cell_membership.to(hidden.device)
        for round_index in reversed(range(structure.rounds)):
            hidden = self.shared_residual(
                hidden,
                structure.inverse_linear_matrices[round_index],
                membership,
                cell_count=structure.cells,
            )
        hidden = self.sequence_norm(hidden)
        cell_hidden = segment_mean(hidden, membership, structure.cells)
        pooled = torch.cat(
            (
                invariant_pool(hidden),
                invariant_pool(cell_hidden),
            ),
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


class FixedOperatorTiedLatentSpnProtocolAdapter(nn.Module):
    """Bind K1-H to one externally supplied two-transition runtime window."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        structure: RuntimeSpnStructure,
        spec: OperatorTiedLatentSpnSpec,
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
            raise ValueError("operator-tied latent pair_bits must encode two blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError(
                "operator-tied latent input_bits must contain complete pairs"
            )
        if structure.rounds != 2:
            raise ValueError("K1-H adapter requires exactly two runtime transitions")
        self.backbone = OperatorTiedLatentSpnDistinguisher(spec)
        self.runtime_structure = structure
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_loaded_rounds = structure.rounds
        self.runtime_round_window_mode = "operator_tied_latent"
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
        self.operator_routing_only = True
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


def operator_support_mean(
    values: torch.Tensor,
    inverse_operator: torch.Tensor,
) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("operator transport requires [batch, source_bits, channels]")
    operator = torch.as_tensor(inverse_operator, dtype=torch.uint8, device="cpu")
    if operator.shape != (values.shape[1], values.shape[1]):
        raise ValueError("operator dimensions do not match bit latents")
    if not torch.all((operator == 0) | (operator == 1)):
        raise ValueError("operator transport requires a binary matrix")
    edges = torch.nonzero(operator, as_tuple=False)
    if edges.numel() == 0:
        raise ValueError("operator transport requires nonzero target degree")
    targets = edges[:, 0].to(values.device)
    sources = edges[:, 1].to(values.device)
    if int(torch.bincount(targets, minlength=values.shape[1]).min()) <= 0:
        raise ValueError("operator transport requires nonzero degree for every target")
    return segment_mean(values[:, sources], targets, values.shape[1])


def segment_mean(
    values: torch.Tensor,
    indices: torch.Tensor,
    segment_count: int,
) -> torch.Tensor:
    if values.ndim != 3 or indices.ndim != 1 or values.shape[1] != indices.numel():
        raise ValueError("segment mean requires [batch, items, channels] and [items]")
    index = indices.to(values.device)
    output = values.new_zeros(values.shape[0], segment_count, values.shape[2])
    expanded = index.reshape(1, -1, 1).expand(values.shape[0], -1, values.shape[2])
    output.scatter_add_(1, expanded, values)
    counts = values.new_zeros(segment_count)
    counts.scatter_add_(0, index, values.new_ones(index.numel()))
    return output / counts.clamp_min(1.0).reshape(1, -1, 1)


def invariant_pool(values: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            values.mean(dim=1),
            values.max(dim=1).values,
            torch.sqrt(values.square().mean(dim=1).clamp_min(1e-8)),
        ),
        dim=-1,
    )


def operator_routing_fingerprint(structure: RuntimeSpnStructure) -> str:
    digest = hashlib.sha256()
    for matrix in structure.inverse_linear_matrices:
        digest.update(matrix.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "FixedOperatorTiedLatentSpnProtocolAdapter",
    "OperatorTiedLatentSpnDistinguisher",
    "OperatorTiedLatentSpnSpec",
    "SharedOperatorResidualBlock",
    "invariant_pool",
    "operator_routing_fingerprint",
    "operator_support_mean",
    "segment_mean",
]
