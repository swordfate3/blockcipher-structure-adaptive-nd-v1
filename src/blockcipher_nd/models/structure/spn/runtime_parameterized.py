from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from blockcipher_nd.models.common.components import AttentionPooling
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
)


@dataclass(frozen=True)
class RuntimeParameterizedSpnSpec:
    hidden_dim: int = 64
    pair_embedding_dim: int = 128
    processor_steps: int = 2
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.pair_embedding_dim, self.processor_steps) <= 0:
            raise ValueError("runtime SPN model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class _RuntimeSpnBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        cell_context: torch.Tensor,
        graph_context: torch.Tensor,
        exact_context: torch.Tensor,
        sbox_context: torch.Tensor,
    ) -> torch.Tensor:
        update = self.update(
            torch.cat(
                (
                    hidden,
                    cell_context,
                    graph_context,
                    exact_context,
                    sbox_context,
                ),
                dim=-1,
            )
        )
        return self.output_norm(hidden + update)


class _RuntimeSpnEncoderBase(nn.Module):
    def __init__(self, spec: RuntimeParameterizedSpnSpec) -> None:
        super().__init__()
        self.spec = spec
        hidden_dim = spec.hidden_dim
        self.input_encoder = nn.Linear(3, hidden_dim)
        self.bit_role_embedding = nn.Embedding(4, hidden_dim)
        self.exact_value_encoder = nn.Linear(1, hidden_dim)
        self.sbox_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.processor = _RuntimeSpnBlock(hidden_dim, spec.dropout)

    def _encode_bits(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        relation_mode: str,
    ) -> torch.Tensor:
        pairs = self._normalize_pairs(ciphertext_pairs, structure.block_bits)
        if relation_mode not in {"true", "independent"}:
            raise ValueError("relation_mode must be true or independent")
        if not torch.all((pairs == 0) | (pairs == 1)):
            raise ValueError("ciphertext pair tensors must be binary")

        pairs = pairs.to(dtype=self.input_encoder.weight.dtype)
        left = pairs[:, :, 0]
        right = pairs[:, :, 1]
        difference = torch.remainder(left + right, 2.0)
        hidden = self.input_encoder(torch.stack((left, right, difference), dim=-1))

        roles = structure.bit_role.to(device=hidden.device)
        if relation_mode == "true":
            hidden = hidden + self.bit_role_embedding(roles)[None, None, :, :]
        hidden = self.input_norm(hidden)

        signal = difference
        round_indices = tuple(
            range(
                structure.rounds - 1,
                max(-1, structure.rounds - 1 - self.spec.processor_steps),
                -1,
            )
        )
        for round_index in round_indices:
            if relation_mode == "true":
                exact = structure.exact_inverse(signal, round_index)
                cell_context = self._cell_context(hidden, structure)
                graph_context = self._graph_context(
                    hidden, structure.inverse_linear_matrices[round_index]
                )
                sbox_context = self._sbox_context(hidden, structure, round_index)
            else:
                exact = signal
                cell_context = torch.zeros_like(hidden)
                graph_context = torch.zeros_like(hidden)
                sbox_context = torch.zeros_like(hidden)
            exact_context = self.exact_value_encoder(exact.unsqueeze(-1))
            hidden = self.processor(
                hidden,
                cell_context,
                graph_context,
                exact_context,
                sbox_context,
            )
            signal = exact
        return hidden

    @staticmethod
    def _normalize_pairs(features: torch.Tensor, block_bits: int) -> torch.Tensor:
        if features.ndim == 4:
            if features.shape[2:] != (2, block_bits):
                raise ValueError(
                    "ciphertext pairs must have shape batch x pairs x 2 x bits"
                )
            return features
        if features.ndim == 2:
            pair_bits = 2 * block_bits
            if features.shape[1] <= 0 or features.shape[1] % pair_bits:
                raise ValueError(
                    "flat ciphertext features must contain complete ciphertext pairs"
                )
            return features.reshape(features.shape[0], -1, 2, block_bits)
        raise ValueError("ciphertext pairs must be two- or four-dimensional")

    @staticmethod
    def _cell_context(
        hidden: torch.Tensor, structure: RuntimeSpnStructure
    ) -> torch.Tensor:
        membership = torch.nn.functional.one_hot(
            structure.cell_membership.to(device=hidden.device),
            num_classes=structure.cells,
        ).to(dtype=hidden.dtype)
        cell_sum = torch.einsum("nc,bpnh->bpch", membership, hidden)
        counts = membership.sum(dim=0).clamp_min(1.0)
        cell_mean = cell_sum / counts[None, None, :, None]
        return torch.einsum("nc,bpch->bpnh", membership, cell_mean)

    @staticmethod
    def _graph_context(hidden: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        adjacency = matrix.to(device=hidden.device, dtype=hidden.dtype)
        degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        normalized = adjacency / degree
        return torch.einsum("ts,bpsh->bpth", normalized, hidden)

    def _sbox_context(
        self,
        hidden: torch.Tensor,
        structure: RuntimeSpnStructure,
        round_index: int,
    ) -> torch.Tensor:
        membership = torch.nn.functional.one_hot(
            structure.cell_membership.to(device=hidden.device),
            num_classes=structure.cells,
        ).to(dtype=hidden.dtype)
        truth = structure.sbox_truth_bits[round_index].to(
            device=hidden.device, dtype=hidden.dtype
        )
        encoded = self.sbox_encoder(truth)
        per_bit = torch.einsum("nc,ch->nh", membership, encoded)
        return per_bit[None, None, :, :].expand_as(hidden)


class RuntimeParameterizedSpnDistinguisher(_RuntimeSpnEncoderBase):
    """Cipher-name-free SPN distinguisher driven by a runtime structure object."""

    def __init__(self, spec: RuntimeParameterizedSpnSpec) -> None:
        super().__init__(spec)
        hidden_dim = spec.hidden_dim
        pair_dim = spec.pair_embedding_dim
        self.node_attention = AttentionPooling(
            hidden_dim,
            hidden_bits=hidden_dim,
            activation="gelu",
            norm="layernorm",
        )
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, pair_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
        )
        self.pair_attention = AttentionPooling(
            pair_dim,
            hidden_bits=pair_dim,
            activation="gelu",
            norm="layernorm",
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_dim * 3),
            nn.Linear(pair_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden_dim * 2, 1),
        )
        self.last_node_attention: torch.Tensor | None = None
        self.last_pair_attention: torch.Tensor | None = None

    def forward(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: str = "true",
    ) -> torch.Tensor:
        hidden = self._encode_bits(ciphertext_pairs, structure, relation_mode)

        batch, pair_count, bit_count, hidden_dim = hidden.shape
        node_sequence = hidden.reshape(batch * pair_count, bit_count, hidden_dim)
        attended_nodes, node_attention = self.node_attention(node_sequence)
        self.last_node_attention = node_attention.detach().reshape(
            batch, pair_count, bit_count
        )
        node_mean = node_sequence.mean(dim=1)
        node_max = node_sequence.max(dim=1).values
        pair_embeddings = self.pair_projection(
            torch.cat((attended_nodes, node_mean, node_max), dim=-1)
        ).reshape(batch, pair_count, self.spec.pair_embedding_dim)

        attended_pairs, pair_attention = self.pair_attention(pair_embeddings)
        self.last_pair_attention = pair_attention.detach()
        pair_mean = pair_embeddings.mean(dim=1)
        pair_max = pair_embeddings.max(dim=1).values
        return self.classifier(torch.cat((attended_pairs, pair_mean, pair_max), dim=-1))


class RuntimeCellTokenSpnDistinguisher(_RuntimeSpnEncoderBase):
    """Preserve same-cell evidence across pairs before global pooling."""

    def __init__(self, spec: RuntimeParameterizedSpnSpec) -> None:
        super().__init__(spec)
        hidden_dim = spec.hidden_dim
        pair_dim = spec.pair_embedding_dim
        self.cell_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cell_graph_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cell_graph_norm = nn.LayerNorm(hidden_dim)
        self.pair_within_cell_attention = AttentionPooling(
            hidden_dim,
            hidden_bits=hidden_dim,
            activation="gelu",
            norm="layernorm",
        )
        self.cell_set_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, pair_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
        )
        self.cell_attention = AttentionPooling(
            pair_dim,
            hidden_bits=pair_dim,
            activation="gelu",
            norm="layernorm",
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_dim * 3),
            nn.Linear(pair_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden_dim * 2, 1),
        )
        self.last_pair_within_cell_attention: torch.Tensor | None = None
        self.last_cell_attention: torch.Tensor | None = None

    def forward(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: str = "true",
    ) -> torch.Tensor:
        hidden = self._encode_bits(ciphertext_pairs, structure, relation_mode)
        cell_tokens = self._ordered_cell_tokens(hidden, structure)
        if relation_mode == "true":
            graph_context = self._cell_graph_context(cell_tokens, structure)
        else:
            graph_context = torch.zeros_like(cell_tokens)
        cell_tokens = self.cell_graph_norm(
            cell_tokens
            + self.cell_graph_update(torch.cat((cell_tokens, graph_context), dim=-1))
        )

        batch, pair_count, cell_count, hidden_dim = cell_tokens.shape
        pair_sequences = cell_tokens.permute(0, 2, 1, 3).reshape(
            batch * cell_count,
            pair_count,
            hidden_dim,
        )
        attended_pairs, pair_attention = self.pair_within_cell_attention(pair_sequences)
        self.last_pair_within_cell_attention = pair_attention.detach().reshape(
            batch,
            cell_count,
            pair_count,
        )
        pair_mean = pair_sequences.mean(dim=1)
        pair_max = pair_sequences.max(dim=1).values
        cell_embeddings = self.cell_set_projection(
            torch.cat((attended_pairs, pair_mean, pair_max), dim=-1)
        ).reshape(batch, cell_count, self.spec.pair_embedding_dim)

        attended_cells, cell_attention = self.cell_attention(cell_embeddings)
        self.last_cell_attention = cell_attention.detach()
        cell_mean = cell_embeddings.mean(dim=1)
        cell_max = cell_embeddings.max(dim=1).values
        return self.classifier(
            torch.cat((attended_cells, cell_mean, cell_max), dim=-1)
        )

    def _ordered_cell_tokens(
        self,
        hidden: torch.Tensor,
        structure: RuntimeSpnStructure,
    ) -> torch.Tensor:
        indices = torch.empty(
            structure.cells,
            4,
            dtype=torch.long,
            device=hidden.device,
        )
        bit_indices = torch.arange(structure.block_bits, device=hidden.device)
        indices[
            structure.cell_membership.to(hidden.device),
            structure.bit_role.to(hidden.device),
        ] = bit_indices
        ordered = hidden[:, :, indices, :]
        return self.cell_projection(ordered.flatten(start_dim=-2))

    @staticmethod
    def _cell_graph_context(
        cell_tokens: torch.Tensor,
        structure: RuntimeSpnStructure,
    ) -> torch.Tensor:
        membership = torch.nn.functional.one_hot(
            structure.cell_membership.to(device=cell_tokens.device),
            num_classes=structure.cells,
        ).to(dtype=cell_tokens.dtype)
        adjacency = structure.inverse_linear_matrices[-1].to(
            device=cell_tokens.device,
            dtype=cell_tokens.dtype,
        )
        cell_adjacency = membership.transpose(0, 1) @ adjacency @ membership
        degree = cell_adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        normalized = cell_adjacency / degree
        return torch.einsum("cs,bpsh->bpch", normalized, cell_tokens)


class FixedRuntimeSpnProtocolAdapter(nn.Module):
    """Bind an external runtime structure for legacy single-input trainers."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        structure: RuntimeSpnStructure,
        relation_mode: str,
        spec: RuntimeParameterizedSpnSpec,
        aggregation_mode: str = "bit_pair",
    ) -> None:
        super().__init__()
        if pair_bits != 2 * structure.block_bits:
            raise ValueError("pair_bits must encode two runtime SPN blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError("input_bits must contain complete ciphertext pairs")
        if relation_mode not in {"true", "independent"}:
            raise ValueError("relation_mode must be true or independent")
        if aggregation_mode == "bit_pair":
            self.backbone = RuntimeParameterizedSpnDistinguisher(spec)
        elif aggregation_mode == "cell_pair":
            self.backbone = RuntimeCellTokenSpnDistinguisher(spec)
        else:
            raise ValueError("aggregation_mode must be bit_pair or cell_pair")
        self.runtime_structure = structure
        self.relation_mode = relation_mode
        self.mapping_mode = relation_mode
        self.aggregation_mode = aggregation_mode

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone(
            features,
            self.runtime_structure,
            relation_mode=self.relation_mode,
        )


__all__ = [
    "FixedRuntimeSpnProtocolAdapter",
    "RuntimeCellTokenSpnDistinguisher",
    "RuntimeParameterizedSpnDistinguisher",
    "RuntimeParameterizedSpnSpec",
]
