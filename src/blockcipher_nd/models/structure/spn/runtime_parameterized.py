from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from blockcipher_nd.models.common.components import AttentionPooling
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
)
from blockcipher_nd.models.structure.spn.token_mixer_pairset import (
    EquivariantSpnTokenMixerBlock,
)


@dataclass(frozen=True)
class RuntimeParameterizedSpnSpec:
    hidden_dim: int = 64
    pair_embedding_dim: int = 128
    processor_steps: int = 2
    dropout: float = 0.10
    sbox_context_scale: float = 1.0
    sbox_context_mode: Literal["early_add", "late_pair", "late_cell", "edge_gate"] = (
        "early_add"
    )
    cell_input_mode: Literal[
        "difference_only",
        "state_triplet",
        "inverse_sbox_triplet",
        "dual_view_triplet",
        "state_triplet_delta_v_query",
        "state_triplet_delta_u_query",
    ] = "difference_only"
    round_window_mode: Literal["last_transition", "recurrent_window"] = (
        "last_transition"
    )
    primitive_adapter_mode: Literal[
        "none",
        "dense",
        "correct",
        "uniform",
        "shuffled",
    ] = "none"
    primitive_adapter_rank: int = 8
    primitive_adapter_scale: float = 0.1
    primitive_adapter_effect: Literal["additive", "multiplicative_gate"] = "additive"
    primitive_film_mode: Literal[
        "none",
        "dense",
        "correct",
        "uniform",
        "shuffled",
    ] = "none"
    primitive_film_rank: int = 10
    primitive_film_scale: float = 0.1
    typed_relation_mode: Literal[
        "none",
        "dense",
        "correct",
        "agnostic",
        "shuffled",
    ] = "none"
    typed_relation_scale: float = 0.1

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.pair_embedding_dim, self.processor_steps) <= 0:
            raise ValueError("runtime SPN model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.sbox_context_scale < 0.0:
            raise ValueError("sbox_context_scale must be non-negative")
        if self.sbox_context_mode not in {
            "early_add",
            "late_pair",
            "late_cell",
            "edge_gate",
        }:
            raise ValueError(
                "sbox_context_mode must be early_add, late_pair, late_cell, or edge_gate"
            )
        if self.cell_input_mode not in {
            "difference_only",
            "state_triplet",
            "inverse_sbox_triplet",
            "dual_view_triplet",
            "state_triplet_delta_v_query",
            "state_triplet_delta_u_query",
        }:
            raise ValueError(
                "cell_input_mode must be difference_only, state_triplet, "
                "inverse_sbox_triplet, dual_view_triplet, "
                "state_triplet_delta_v_query, or state_triplet_delta_u_query"
            )
        if self.round_window_mode not in {"last_transition", "recurrent_window"}:
            raise ValueError(
                "round_window_mode must be last_transition or recurrent_window"
            )
        if self.primitive_adapter_mode not in {
            "none",
            "dense",
            "correct",
            "uniform",
            "shuffled",
        }:
            raise ValueError(
                "primitive_adapter_mode must be none, dense, correct, uniform, "
                "or shuffled"
            )
        if self.primitive_adapter_rank <= 0:
            raise ValueError("primitive_adapter_rank must be positive")
        if self.primitive_adapter_scale < 0.0:
            raise ValueError("primitive_adapter_scale must be non-negative")
        if self.primitive_adapter_effect not in {"additive", "multiplicative_gate"}:
            raise ValueError(
                "primitive_adapter_effect must be additive or multiplicative_gate"
            )
        if self.primitive_film_mode not in {
            "none",
            "dense",
            "correct",
            "uniform",
            "shuffled",
        }:
            raise ValueError(
                "primitive_film_mode must be none, dense, correct, uniform, or shuffled"
            )
        if self.primitive_film_rank <= 0:
            raise ValueError("primitive_film_rank must be positive")
        if self.primitive_film_scale < 0.0:
            raise ValueError("primitive_film_scale must be non-negative")
        if self.typed_relation_mode not in {
            "none",
            "dense",
            "correct",
            "agnostic",
            "shuffled",
        }:
            raise ValueError(
                "typed_relation_mode must be none, dense, correct, agnostic, "
                "or shuffled"
            )
        if self.typed_relation_scale < 0.0:
            raise ValueError("typed_relation_scale must be non-negative")
        active_conditioners = sum(
            mode != "none"
            for mode in (
                self.primitive_adapter_mode,
                self.primitive_film_mode,
                self.typed_relation_mode,
            )
        )
        if active_conditioners > 1:
            raise ValueError(
                "primitive Adapter, True FiLM, and typed relation conditioning "
                "are mutually exclusive"
            )


class _LowRankResidualAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, hidden_dim, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(hidden)))


class _PrimitiveTrueFiLM(nn.Module):
    def __init__(self, descriptor_dim: int, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(descriptor_dim, rank, bias=False)
        self.activation = nn.GELU()
        self.affine = nn.Linear(rank, hidden_dim * 2, bias=True)

    def forward(self, descriptor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.affine(self.activation(self.down(descriptor))).chunk(2, -1)
        return gamma, beta


class _TypedRelationGnnFiLM(nn.Module):
    """Feature-wise affine messages for the 16 target/source bit-role pairs."""

    RELATION_TYPES = 16

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(self.RELATION_TYPES, hidden_dim))
        self.beta = nn.Parameter(torch.empty(self.RELATION_TYPES, hidden_dim))
        nn.init.normal_(self.gamma, mean=0.0, std=0.02)
        nn.init.normal_(self.beta, mean=0.0, std=0.02)

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        if adjacency.shape != (
            self.RELATION_TYPES,
            hidden.shape[1],
            hidden.shape[1],
        ):
            raise ValueError("typed relation adjacency does not match cell tokens")
        messages = hidden[:, None, :, :] * torch.tanh(self.gamma)[None, :, None, :]
        messages = messages + torch.tanh(self.beta)[None, :, None, :]
        incoming = torch.einsum("rts,brsd->btd", adjacency, messages)
        degree = adjacency.sum(dim=(0, 2)).clamp_min(1.0)
        return incoming / degree[None, :, None]


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
        return self.classifier(torch.cat((attended_cells, cell_mean, cell_max), dim=-1))

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


class RuntimeE4EquivariantSpnDistinguisher(nn.Module):
    """Runtime-width E4-style frontend with cell-permutation-equivariant mixing."""

    def __init__(self, spec: RuntimeParameterizedSpnSpec) -> None:
        super().__init__()
        self.spec = spec
        token_dim = max(16, spec.hidden_dim * 2)
        pair_dim = spec.pair_embedding_dim
        self.token_dim = token_dim
        self.cell_encoder = nn.Sequential(
            nn.Linear(4, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )
        fusion_views = (
            3
            if spec.cell_input_mode
            in {"state_triplet_delta_v_query", "state_triplet_delta_u_query"}
            else 2
        )
        self.typed_fusion = nn.Sequential(
            nn.Linear(token_dim * fusion_views, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )
        self.sbox_encoder = nn.Sequential(
            nn.Linear(64, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )
        self.mixer_blocks = nn.ModuleList(
            [
                EquivariantSpnTokenMixerBlock(
                    nibbles_per_pair=16,
                    token_dim=token_dim,
                    token_mlp_ratio=2,
                    activation="relu",
                    norm="layernorm",
                    dropout=spec.dropout,
                )
                for _ in range(spec.processor_steps)
            ]
        )
        self.primitive_adapters: nn.ModuleDict | None = None
        self.dense_primitive_adapter: _LowRankResidualAdapter | None = None
        if spec.primitive_adapter_mode == "dense":
            self.dense_primitive_adapter = _LowRankResidualAdapter(
                token_dim,
                2 * spec.primitive_adapter_rank,
            )
        elif spec.primitive_adapter_mode != "none":
            self.primitive_adapters = nn.ModuleDict(
                {
                    "fan_in_1": _LowRankResidualAdapter(
                        token_dim,
                        spec.primitive_adapter_rank,
                    ),
                    "multi_source": _LowRankResidualAdapter(
                        token_dim,
                        spec.primitive_adapter_rank,
                    ),
                }
            )
        self.primitive_film_conditioner: _PrimitiveTrueFiLM | None = None
        if spec.primitive_film_mode != "none":
            self.primitive_film_conditioner = _PrimitiveTrueFiLM(
                descriptor_dim=128,
                hidden_dim=token_dim,
                rank=spec.primitive_film_rank,
            )
        self.typed_relation_message: _TypedRelationGnnFiLM | None = None
        if spec.typed_relation_mode != "none":
            self.typed_relation_message = _TypedRelationGnnFiLM(token_dim)
        self.sequence_norm = nn.LayerNorm(token_dim)
        self.pair_projection = nn.Sequential(
            nn.Linear(token_dim * 3, pair_dim),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
        )
        self.pair_attention = AttentionPooling(
            pair_dim,
            hidden_bits=pair_dim,
            activation="relu",
            norm="layernorm",
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_dim * 3),
            nn.Linear(pair_dim * 3, spec.hidden_dim * 8),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim * 8, 1),
        )
        self.last_pair_attention: torch.Tensor | None = None
        self.last_primitive_adapter_traffic: dict[str, float] = {}

    def forward(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: str = "true",
        query_input_mode: Literal["delta_v", "delta_u"] | None = None,
        query_structure: RuntimeSpnStructure | None = None,
    ) -> torch.Tensor:
        return self.classifier(
            self.encode(
                ciphertext_pairs,
                structure,
                relation_mode=relation_mode,
                query_input_mode=query_input_mode,
                query_structure=query_structure,
            )
        )

    def encode(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: str = "true",
        query_input_mode: Literal["delta_v", "delta_u"] | None = None,
        query_structure: RuntimeSpnStructure | None = None,
    ) -> torch.Tensor:
        """Return the invariant pooled representation consumed by the classifier."""
        pairs = _RuntimeSpnEncoderBase._normalize_pairs(
            ciphertext_pairs, structure.block_bits
        )
        if relation_mode not in {"true", "independent"}:
            raise ValueError("relation_mode must be true or independent")
        if not torch.all((pairs == 0) | (pairs == 1)):
            raise ValueError("ciphertext pair tensors must be binary")
        pairs = pairs.to(dtype=self.cell_encoder[0].weight.dtype)
        self.last_primitive_adapter_traffic = {}
        difference = torch.remainder(pairs[:, :, 0] + pairs[:, :, 1], 2.0)
        configured_query_mode = {
            "state_triplet_delta_v_query": "delta_v",
            "state_triplet_delta_u_query": "delta_u",
        }.get(self.spec.cell_input_mode)
        if query_input_mode is not None and configured_query_mode is None:
            raise ValueError("query overrides require a three-input delta-query model")
        active_query_mode = query_input_mode or configured_query_mode
        if query_structure is not None:
            if active_query_mode != "delta_u":
                raise ValueError("query_structure is valid only for a delta_u query")
            self._validate_query_structure(structure, query_structure)
        if self.spec.round_window_mode == "recurrent_window":
            if active_query_mode is not None or query_structure is not None:
                raise ValueError(
                    "recurrent_window does not support final-transition delta queries"
                )
            return self._forward_recurrent_window(
                pairs,
                structure,
                relation_mode=relation_mode,
            )
        previous = (
            structure.exact_inverse(difference, -1)
            if relation_mode == "true"
            else difference
        )
        current_cells = self._ordered_cell_values(difference, structure)
        previous_cells = self._ordered_cell_values(previous, structure)
        batch, pair_count, cell_count, _ = current_cells.shape
        if self.spec.cell_input_mode in {
            "state_triplet",
            "inverse_sbox_triplet",
            "dual_view_triplet",
            "state_triplet_delta_v_query",
            "state_triplet_delta_u_query",
        }:
            left = pairs[:, :, 0]
            right = pairs[:, :, 1]
            previous_left = (
                structure.exact_inverse(left, -1) if relation_mode == "true" else left
            )
            previous_right = (
                structure.exact_inverse(right, -1) if relation_mode == "true" else right
            )
            current_hidden = self._state_triplet_cell_hidden(
                left,
                right,
                difference,
                structure,
            )
            previous_hidden = self._state_triplet_cell_hidden(
                previous_left,
                previous_right,
                previous,
                structure,
            )
            if (
                self.spec.cell_input_mode
                in {"inverse_sbox_triplet", "dual_view_triplet"}
                and relation_mode == "true"
            ):
                inverse_left = structure.apply_inverse_sboxes(previous_left, -1)
                inverse_right = structure.apply_inverse_sboxes(previous_right, -1)
                inverse_difference = torch.remainder(
                    inverse_left + inverse_right,
                    2.0,
                )
                inverse_hidden = self._state_triplet_cell_hidden(
                    inverse_left,
                    inverse_right,
                    inverse_difference,
                    structure,
                )
                previous_hidden = (
                    inverse_hidden
                    if self.spec.cell_input_mode == "inverse_sbox_triplet"
                    else 0.5 * (previous_hidden + inverse_hidden)
                )
        else:
            current_hidden = self.cell_encoder(
                current_cells.reshape(batch * pair_count, cell_count, 4)
            )
            previous_hidden = self.cell_encoder(
                previous_cells.reshape(batch * pair_count, cell_count, 4)
            )
        fusion_inputs = [current_hidden, previous_hidden]
        if active_query_mode is not None:
            query_difference = previous
            if active_query_mode == "delta_u" and relation_mode == "true":
                query_runtime = query_structure or structure
                inverse_left = query_runtime.apply_inverse_sboxes(previous_left, -1)
                inverse_right = query_runtime.apply_inverse_sboxes(previous_right, -1)
                query_difference = torch.remainder(
                    inverse_left + inverse_right,
                    2.0,
                )
            query_cells = self._ordered_cell_values(query_difference, structure)
            query_hidden = self.cell_encoder(
                query_cells.reshape(batch * pair_count, cell_count, 4)
            )
            fusion_inputs.append(query_hidden)
        hidden = self.typed_fusion(torch.cat(fusion_inputs, dim=-1)).reshape(
            batch, pair_count, cell_count, self.token_dim
        )
        sbox_context = self.sbox_encoder(
            structure.sbox_truth_bits[-1].to(device=hidden.device, dtype=hidden.dtype)
        )
        if self.spec.sbox_context_mode == "early_add":
            hidden = (
                hidden + self.spec.sbox_context_scale * sbox_context[None, None, :, :]
            )
        token_dim = hidden.shape[-1]
        sequence = hidden.reshape(batch * pair_count, cell_count, token_dim)
        if self.spec.sbox_context_mode == "edge_gate":
            sequence = self._apply_sbox_edge_gate(
                sequence,
                sbox_context,
                structure,
                relation_mode=relation_mode,
            )
        sequence = self._apply_typed_relation_message(
            sequence,
            structure,
            relation_mode=relation_mode,
            round_index=-1,
        )
        sequence = self._apply_primitive_film(
            sequence,
            structure,
            round_index=-1,
        )
        for block in self.mixer_blocks:
            sequence = block(sequence)
        sequence = self._apply_primitive_adapter(
            sequence,
            structure,
            round_index=-1,
        )
        if self.spec.sbox_context_mode == "late_cell":
            sequence = (
                sequence + self.spec.sbox_context_scale * sbox_context[None, :, :]
            )
        sequence = self.sequence_norm(sequence)
        return self._pool_sequence(
            sequence,
            current_cells=current_cells,
            sbox_context=sbox_context,
            batch=batch,
            pair_count=pair_count,
        )

    def _forward_recurrent_window(
        self,
        pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: str,
    ) -> torch.Tensor:
        """Consume every loaded transition with one shared E4 parameter stack."""
        left = pairs[:, :, 0]
        right = pairs[:, :, 1]
        difference = torch.remainder(left + right, 2.0)
        output_cells = self._ordered_cell_values(difference, structure)
        batch, pair_count, cell_count, _ = output_cells.shape
        sequence: torch.Tensor | None = None
        round_sbox_contexts: list[torch.Tensor] = []

        for round_index in reversed(range(structure.rounds)):
            previous_left = (
                structure.exact_inverse(left, round_index)
                if relation_mode == "true"
                else left
            )
            previous_right = (
                structure.exact_inverse(right, round_index)
                if relation_mode == "true"
                else right
            )
            previous_difference = torch.remainder(
                previous_left + previous_right,
                2.0,
            )
            current_cells = self._ordered_cell_values(difference, structure)
            previous_cells = self._ordered_cell_values(
                previous_difference,
                structure,
            )

            if self.spec.cell_input_mode in {
                "state_triplet",
                "inverse_sbox_triplet",
                "dual_view_triplet",
            }:
                current_hidden = self._state_triplet_cell_hidden(
                    left,
                    right,
                    difference,
                    structure,
                )
                previous_hidden = self._state_triplet_cell_hidden(
                    previous_left,
                    previous_right,
                    previous_difference,
                    structure,
                )
                if (
                    self.spec.cell_input_mode
                    in {"inverse_sbox_triplet", "dual_view_triplet"}
                    and relation_mode == "true"
                ):
                    inverse_left = structure.apply_inverse_sboxes(
                        previous_left,
                        round_index,
                    )
                    inverse_right = structure.apply_inverse_sboxes(
                        previous_right,
                        round_index,
                    )
                    inverse_difference = torch.remainder(
                        inverse_left + inverse_right,
                        2.0,
                    )
                    inverse_hidden = self._state_triplet_cell_hidden(
                        inverse_left,
                        inverse_right,
                        inverse_difference,
                        structure,
                    )
                    previous_hidden = (
                        inverse_hidden
                        if self.spec.cell_input_mode == "inverse_sbox_triplet"
                        else 0.5 * (previous_hidden + inverse_hidden)
                    )
            elif self.spec.cell_input_mode == "difference_only":
                current_hidden = self.cell_encoder(
                    current_cells.reshape(batch * pair_count, cell_count, 4)
                )
                previous_hidden = self.cell_encoder(
                    previous_cells.reshape(batch * pair_count, cell_count, 4)
                )
            else:
                raise ValueError(
                    "recurrent_window supports difference_only, state_triplet, "
                    "inverse_sbox_triplet, or dual_view_triplet"
                )

            transition = self.typed_fusion(
                torch.cat((current_hidden, previous_hidden), dim=-1)
            )
            sbox_context = self.sbox_encoder(
                structure.sbox_truth_bits[round_index].to(
                    device=transition.device,
                    dtype=transition.dtype,
                )
            )
            round_sbox_contexts.append(sbox_context)
            if self.spec.sbox_context_mode == "early_add":
                transition = (
                    transition + self.spec.sbox_context_scale * sbox_context[None, :, :]
                )
            if sequence is None:
                sequence = transition
            else:
                sequence = self.sequence_norm(sequence + transition)
            if self.spec.sbox_context_mode == "edge_gate":
                sequence = self._apply_sbox_edge_gate(
                    sequence,
                    sbox_context,
                    structure,
                    relation_mode=relation_mode,
                    round_index=round_index,
                )
            sequence = self._apply_typed_relation_message(
                sequence,
                structure,
                relation_mode=relation_mode,
                round_index=round_index,
            )
            sequence = self._apply_primitive_film(
                sequence,
                structure,
                round_index=round_index,
            )
            for block in self.mixer_blocks:
                sequence = block(sequence)
            sequence = self._apply_primitive_adapter(
                sequence,
                structure,
                round_index=round_index,
            )
            if self.spec.sbox_context_mode == "late_cell":
                sequence = (
                    sequence + self.spec.sbox_context_scale * sbox_context[None, :, :]
                )
            sequence = self.sequence_norm(sequence)
            left = previous_left
            right = previous_right
            difference = previous_difference

        if sequence is None:
            raise ValueError("runtime SPN structure must contain at least one round")
        aggregate_sbox_context = torch.stack(round_sbox_contexts, dim=0).mean(dim=0)
        return self._pool_sequence(
            sequence,
            current_cells=output_cells,
            sbox_context=aggregate_sbox_context,
            batch=batch,
            pair_count=pair_count,
        )

    def _apply_primitive_adapter(
        self,
        sequence: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        round_index: int,
    ) -> torch.Tensor:
        mode = self.spec.primitive_adapter_mode
        if mode == "none":
            return sequence
        if mode == "dense":
            if self.dense_primitive_adapter is None:
                raise RuntimeError("dense primitive adapter was not initialized")
            self.last_primitive_adapter_traffic["dense"] = (
                self.last_primitive_adapter_traffic.get("dense", 0.0)
                + float(sequence.shape[0] * sequence.shape[1])
            )
            return self._combine_primitive_effect(
                sequence,
                self.dense_primitive_adapter(sequence),
            )
        if self.primitive_adapters is None:
            raise RuntimeError("routed primitive adapters were not initialized")

        weights = self.primitive_routing_weights(
            structure,
            round_index=round_index,
            mode=mode,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        fan_in_1 = self.primitive_adapters["fan_in_1"](sequence)
        multi_source = self.primitive_adapters["multi_source"](sequence)
        residual = (
            weights[None, :, 0:1] * fan_in_1 + weights[None, :, 1:2] * multi_source
        )
        token_multiplier = float(sequence.shape[0])
        for name, index in (("fan_in_1", 0), ("multi_source", 1)):
            self.last_primitive_adapter_traffic[name] = (
                self.last_primitive_adapter_traffic.get(name, 0.0)
                + float(weights[:, index].sum().detach().cpu()) * token_multiplier
            )
        return self._combine_primitive_effect(sequence, residual)

    def _apply_primitive_film(
        self,
        sequence: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        round_index: int,
    ) -> torch.Tensor:
        mode = self.spec.primitive_film_mode
        if mode == "none":
            return sequence
        if self.primitive_film_conditioner is None:
            raise RuntimeError("primitive True FiLM was not initialized")
        descriptor = self.primitive_film_descriptor(
            structure,
            round_index=round_index,
            mode=mode,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        gamma, beta = self.primitive_film_conditioner(descriptor)
        scale = self.spec.primitive_film_scale
        self.last_primitive_adapter_traffic["film"] = (
            self.last_primitive_adapter_traffic.get("film", 0.0)
            + float(sequence.shape[0] * sequence.shape[1])
        )
        return sequence * (1.0 + scale * torch.tanh(gamma)[None, :, :]) + (
            scale * torch.tanh(beta)[None, :, :]
        )

    def _apply_typed_relation_message(
        self,
        sequence: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: str,
        round_index: int,
    ) -> torch.Tensor:
        mode = self.spec.typed_relation_mode
        if mode == "none" or relation_mode == "independent":
            return sequence
        if self.typed_relation_message is None:
            raise RuntimeError("typed relation message module was not initialized")
        adjacency = self.typed_relation_adjacency(
            structure,
            round_index=round_index,
            mode=mode,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        residual = self.typed_relation_message(sequence, adjacency)
        self.last_primitive_adapter_traffic["typed_relation"] = (
            self.last_primitive_adapter_traffic.get("typed_relation", 0.0)
            + float(adjacency.sum().detach().cpu()) * float(sequence.shape[0])
        )
        return sequence + self.spec.typed_relation_scale * residual

    @staticmethod
    def typed_relation_adjacency(
        structure: RuntimeSpnStructure,
        *,
        round_index: int,
        mode: Literal["dense", "correct", "agnostic", "shuffled"],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if mode not in {"dense", "correct", "agnostic", "shuffled"}:
            raise ValueError(
                "typed relation mode must be dense, correct, agnostic, or shuffled"
            )
        cells = structure.cells
        if mode == "dense":
            return torch.full(
                (16, cells, cells),
                1.0 / 16.0,
                device=device,
                dtype=dtype,
            )

        membership = structure.cell_membership.to(device=device)
        bit_role = structure.bit_role.to(device=device)
        bit_indices = torch.arange(structure.block_bits, device=device)
        cell_role_indices = torch.empty(
            cells,
            4,
            dtype=torch.long,
            device=device,
        )
        cell_role_indices[membership, bit_role] = bit_indices
        inverse = structure.inverse_linear_matrices[round_index].to(
            device=device,
            dtype=dtype,
        )
        relation = inverse[
            cell_role_indices[:, :, None, None],
            cell_role_indices[None, None, :, :],
        ].permute(1, 3, 0, 2).reshape(16, cells, cells)
        if mode == "agnostic":
            return relation.sum(dim=0, keepdim=True).expand_as(relation) / 16.0
        if mode == "shuffled":
            return torch.roll(relation, shifts=1, dims=0)
        return relation

    @staticmethod
    def primitive_film_descriptor(
        structure: RuntimeSpnStructure,
        *,
        round_index: int,
        mode: Literal["dense", "correct", "uniform", "shuffled"],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if mode not in {"dense", "correct", "uniform", "shuffled"}:
            raise ValueError(
                "primitive True FiLM mode must be dense, correct, uniform, or shuffled"
            )
        if mode == "dense":
            fixed = torch.linspace(-1.0, 1.0, 128, device=device, dtype=dtype)
            return fixed[None, :].expand(structure.cells, -1)

        membership = structure.cell_membership.to(device=device)
        bit_role = structure.bit_role.to(device=device)
        bit_indices = torch.arange(structure.block_bits, device=device)
        cell_role_indices = torch.empty(
            structure.cells,
            4,
            dtype=torch.long,
            device=device,
        )
        cell_role_indices[membership, bit_role] = bit_indices

        inverse = structure.inverse_linear_matrices[round_index].to(
            device=device,
            dtype=dtype,
        )
        target_rows = inverse[cell_role_indices]
        source_roles = torch.nn.functional.one_hot(bit_role, num_classes=4).to(dtype)
        edge_counts = torch.einsum("ctb,bs->cts", target_rows, source_roles)
        normalized_counts = edge_counts / float(structure.cells)
        count_parity = torch.remainder(edge_counts, 2.0)
        same_cell = inverse[
            cell_role_indices[:, :, None],
            cell_role_indices[:, None, :],
        ]
        source_fanout = inverse.sum(dim=0) / float(structure.block_bits)
        fanout_sum = torch.einsum(
            "ctb,b,bs->cts",
            target_rows,
            source_fanout,
            source_roles,
        )
        mean_source_fanout = fanout_sum / edge_counts.clamp_min(1.0)
        diffusion = torch.stack(
            (
                normalized_counts,
                count_parity,
                same_cell,
                mean_source_fanout,
            ),
            dim=-1,
        ).reshape(structure.cells, 64)
        sbox = structure.sbox_truth_bits[round_index].to(device=device, dtype=dtype)
        descriptor = torch.cat((sbox, diffusion), dim=-1)
        if mode == "uniform":
            return descriptor.mean(dim=0, keepdim=True).expand_as(descriptor)
        if mode == "shuffled":
            return torch.cat(
                (
                    torch.roll(descriptor[:, :64], shifts=1, dims=-1),
                    torch.roll(descriptor[:, 64:], shifts=1, dims=-1),
                ),
                dim=-1,
            )
        return descriptor

    def _combine_primitive_effect(
        self,
        sequence: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.spec.primitive_adapter_scale
        if self.spec.primitive_adapter_effect == "additive":
            return sequence + scale * residual
        return sequence * (1.0 + scale * torch.tanh(residual))

    @staticmethod
    def primitive_routing_weights(
        structure: RuntimeSpnStructure,
        *,
        round_index: int,
        mode: Literal["correct", "uniform", "shuffled"],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if mode == "uniform":
            return torch.full(
                (structure.cells, 2),
                0.5,
                device=device,
                dtype=dtype,
            )
        if mode not in {"correct", "shuffled"}:
            raise ValueError(
                "routed primitive mode must be correct, uniform, or shuffled"
            )

        inverse = structure.inverse_linear_matrices[round_index].to(device=device)
        row_fan_in = inverse.sum(dim=1)
        membership = structure.cell_membership.to(device=device)
        fan_in_1_cells = torch.ones(structure.cells, dtype=torch.bool, device=device)
        for cell in range(structure.cells):
            fan_in_1_cells[cell] = torch.all(row_fan_in[membership == cell] == 1)
        weights = torch.stack((fan_in_1_cells, ~fan_in_1_cells), dim=-1).to(dtype)
        return weights.flip(-1) if mode == "shuffled" else weights

    def primitive_adapter_summary(self) -> dict[str, object]:
        return {
            "mode": self.spec.primitive_adapter_mode,
            "rank": self.spec.primitive_adapter_rank,
            "scale": self.spec.primitive_adapter_scale,
            "effect": self.spec.primitive_adapter_effect,
            "active_adapter_evaluations": (
                0
                if self.spec.primitive_adapter_mode == "none"
                else 1
                if self.spec.primitive_adapter_mode == "dense"
                else 2
            ),
            "traffic": dict(self.last_primitive_adapter_traffic),
        }

    def primitive_film_summary(self) -> dict[str, object]:
        return {
            "mode": self.spec.primitive_film_mode,
            "rank": self.spec.primitive_film_rank,
            "scale": self.spec.primitive_film_scale,
            "descriptor_dim": 128,
            "active_conditioner_evaluations": (
                0 if self.spec.primitive_film_mode == "none" else 1
            ),
            "traffic": dict(self.last_primitive_adapter_traffic),
        }

    def typed_relation_summary(self) -> dict[str, object]:
        return {
            "mode": self.spec.typed_relation_mode,
            "scale": self.spec.typed_relation_scale,
            "relation_types": 16,
            "active_message_evaluations": (
                0 if self.spec.typed_relation_mode == "none" else 1
            ),
            "traffic": dict(self.last_primitive_adapter_traffic),
        }

    def _pool_sequence(
        self,
        sequence: torch.Tensor,
        *,
        current_cells: torch.Tensor,
        sbox_context: torch.Tensor,
        batch: int,
        pair_count: int,
    ) -> torch.Tensor:
        cell_count = current_cells.shape[2]
        current_activity = current_cells.mean(dim=-1, keepdim=True).reshape(
            batch * pair_count, cell_count, 1
        )
        mean_embedding = sequence.mean(dim=1)
        max_embedding = sequence.max(dim=1).values
        active_embedding = torch.sum(sequence * current_activity, dim=1) / (
            current_activity.sum(dim=1).clamp_min(1.0)
        )
        pair_embeddings = self.pair_projection(
            torch.cat((mean_embedding, max_embedding, active_embedding), dim=-1)
        ).reshape(batch, pair_count, self.spec.pair_embedding_dim)
        if self.spec.sbox_context_mode == "late_pair":
            late_context = sbox_context.to(torch.float64).mean(dim=0).to(sequence.dtype)
            if late_context.shape[0] != self.spec.pair_embedding_dim:
                late_context = torch.nn.functional.adaptive_avg_pool1d(
                    late_context.reshape(1, 1, -1),
                    self.spec.pair_embedding_dim,
                ).reshape(-1)
            pair_embeddings = (
                pair_embeddings
                + self.spec.sbox_context_scale * late_context[None, None, :]
            )
        attended_pairs, pair_attention = self.pair_attention(pair_embeddings)
        self.last_pair_attention = pair_attention.detach()
        return torch.cat(
            (
                attended_pairs,
                pair_embeddings.mean(dim=1),
                pair_embeddings.max(dim=1).values,
            ),
            dim=-1,
        )

    @staticmethod
    def _validate_query_structure(
        main: RuntimeSpnStructure,
        query: RuntimeSpnStructure,
    ) -> None:
        if not (
            torch.equal(main.cell_membership, query.cell_membership)
            and torch.equal(main.bit_role, query.bit_role)
            and torch.equal(main.linear_matrices, query.linear_matrices)
            and torch.equal(
                main.inverse_linear_matrices,
                query.inverse_linear_matrices,
            )
        ):
            raise ValueError("query_structure may change only per-cell S-box ownership")

    @staticmethod
    def _ordered_cell_values(
        values: torch.Tensor,
        structure: RuntimeSpnStructure,
    ) -> torch.Tensor:
        indices = torch.empty(
            structure.cells,
            4,
            dtype=torch.long,
            device=values.device,
        )
        bit_indices = torch.arange(structure.block_bits, device=values.device)
        indices[
            structure.cell_membership.to(values.device),
            structure.bit_role.to(values.device),
        ] = bit_indices
        return values[:, :, indices]

    def _state_triplet_cell_hidden(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        difference: torch.Tensor,
        structure: RuntimeSpnStructure,
    ) -> torch.Tensor:
        cell_values = tuple(
            self._ordered_cell_values(values, structure)
            for values in (left, right, difference)
        )
        batch, pair_count, cell_count, _ = cell_values[0].shape
        encoded = tuple(
            self.cell_encoder(values.reshape(batch * pair_count, cell_count, 4))
            for values in cell_values
        )
        endpoint_mean = 0.5 * (encoded[0] + encoded[1])
        return 0.5 * (endpoint_mean + encoded[2])

    def _apply_sbox_edge_gate(
        self,
        sequence: torch.Tensor,
        sbox_context: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: str,
        round_index: int = -1,
    ) -> torch.Tensor:
        membership = torch.nn.functional.one_hot(
            structure.cell_membership.to(sequence.device),
            num_classes=structure.cells,
        ).to(sequence.dtype)
        if relation_mode == "true":
            bit_adjacency = structure.inverse_linear_matrices[round_index].to(
                device=sequence.device,
                dtype=sequence.dtype,
            )
            cell_adjacency = membership.transpose(0, 1) @ bit_adjacency @ membership
        elif relation_mode == "independent":
            cell_adjacency = torch.eye(
                structure.cells,
                device=sequence.device,
                dtype=sequence.dtype,
            )
        else:
            raise ValueError("relation_mode must be true or independent")
        normalized = cell_adjacency / cell_adjacency.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        graph_message = torch.einsum("ts,bsd->btd", normalized, sequence)
        neighbor_sbox = torch.einsum("ts,sd->td", normalized, sbox_context)
        gate = torch.sigmoid(0.5 * (sbox_context + neighbor_sbox))
        return (
            sequence + self.spec.sbox_context_scale * gate[None, :, :] * graph_message
        )


class RuntimeE5GatedResidualSpnDistinguisher(nn.Module):
    """Preserve an independent E4 prediction and add bounded topology residuals."""

    def __init__(self, spec: RuntimeParameterizedSpnSpec) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = RuntimeE4EquivariantSpnDistinguisher(spec)
        representation_dim = spec.pair_embedding_dim * 3
        self.topology_residual_head = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, spec.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim * 2, 1),
        )
        self.topology_gate = nn.Parameter(torch.zeros(()))
        self.last_base_logit: torch.Tensor | None = None
        self.last_topology_residual_logit: torch.Tensor | None = None

    def forward(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: str = "true",
        topology_gate_override: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        base, topology_residual = self.encode_components(
            ciphertext_pairs,
            structure,
            relation_mode=relation_mode,
        )
        base_logit = self.encoder.classifier(base)
        residual_logit = (
            torch.zeros_like(base_logit)
            if relation_mode == "independent"
            else self.topology_residual_head(topology_residual)
        )
        gate_source = (
            self.topology_gate
            if topology_gate_override is None
            else torch.as_tensor(
                topology_gate_override,
                device=base_logit.device,
                dtype=base_logit.dtype,
            )
        )
        gate = torch.tanh(gate_source)
        self.last_base_logit = base_logit.detach()
        self.last_topology_residual_logit = residual_logit.detach()
        return base_logit + gate * torch.tanh(residual_logit)

    def base_logits(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
    ) -> torch.Tensor:
        base = self.encoder.encode(
            ciphertext_pairs,
            structure,
            relation_mode="independent",
        )
        return self.encoder.classifier(base)

    def encode_components(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if relation_mode not in {"true", "independent"}:
            raise ValueError("relation_mode must be true or independent")
        base = self.encoder.encode(
            ciphertext_pairs,
            structure,
            relation_mode="independent",
        )
        if relation_mode == "independent":
            topology = base
        else:
            topology = self.encoder.encode(
                ciphertext_pairs,
                structure,
                relation_mode="true",
            )
        return base, topology - base


class FixedRuntimeSpnProtocolAdapter(nn.Module):
    """Bind an external runtime structure to the project's MSB-first features."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        structure: RuntimeSpnStructure,
        relation_mode: str,
        spec: RuntimeParameterizedSpnSpec,
        aggregation_mode: str = "bit_pair",
        descriptor_name: str | None = None,
        descriptor_path: str | None = None,
        descriptor_sha256: str | None = None,
        descriptor_round_start: int | None = None,
        descriptor_available_rounds: int | None = None,
        runtime_structure_mode: str | None = None,
        runtime_structure_window_control: str = "full",
    ) -> None:
        super().__init__()
        if pair_bits != 2 * structure.block_bits:
            raise ValueError("pair_bits must encode two runtime SPN blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError("input_bits must contain complete ciphertext pairs")
        if relation_mode not in {"true", "independent"}:
            raise ValueError("relation_mode must be true or independent")
        if runtime_structure_window_control not in {"full", "repeat_last"}:
            raise ValueError(
                "runtime_structure_window_control must be full or repeat_last"
            )
        if aggregation_mode == "bit_pair":
            self.backbone = RuntimeParameterizedSpnDistinguisher(spec)
        elif aggregation_mode == "cell_pair":
            self.backbone = RuntimeCellTokenSpnDistinguisher(spec)
        elif aggregation_mode == "e4_equivariant":
            self.backbone = RuntimeE4EquivariantSpnDistinguisher(spec)
        elif aggregation_mode == "e5_gated_residual":
            self.backbone = RuntimeE5GatedResidualSpnDistinguisher(spec)
        else:
            raise ValueError(
                "aggregation_mode must be bit_pair, cell_pair, e4_equivariant, "
                "or e5_gated_residual"
            )
        self.runtime_structure = structure
        self.relation_mode = relation_mode
        self.mapping_mode = relation_mode
        self.aggregation_mode = aggregation_mode
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_loaded_rounds = structure.rounds
        self.runtime_round_window_mode = spec.round_window_mode
        self.runtime_structure_window_control = runtime_structure_window_control
        transition_sha256s = structure.transition_sha256s()
        unique_transition_count = len(set(transition_sha256s))
        self.runtime_structure_transition_sha256s = transition_sha256s
        self.runtime_structure_window_sha256 = structure.window_sha256()
        self.runtime_structure_unique_transition_count = unique_transition_count
        self.runtime_structure_homogeneous = unique_transition_count == 1
        if aggregation_mode == "e5_gated_residual":
            self.topology_residual_mode = (
                "independent_base_plus_bounded_topology_logit_residual"
            )
            self.topology_gate_initial = 0.0
        if descriptor_name is not None:
            self.runtime_structure_descriptor_name = descriptor_name
        if descriptor_path is not None:
            self.runtime_structure_descriptor_path = descriptor_path
        if descriptor_sha256 is not None:
            self.runtime_structure_descriptor_sha256 = descriptor_sha256
        if descriptor_round_start is not None:
            self.runtime_structure_round_start = descriptor_round_start
        if descriptor_available_rounds is not None:
            self.runtime_structure_available_rounds = descriptor_available_rounds
        if runtime_structure_mode is not None:
            self.runtime_structure_mode = runtime_structure_mode

    def forward(
        self,
        features: torch.Tensor,
        *,
        query_input_mode: Literal["delta_v", "delta_u"] | None = None,
        query_structure: RuntimeSpnStructure | None = None,
    ) -> torch.Tensor:
        runtime_features = self._to_runtime_coordinates(features)
        if query_input_mode is None and query_structure is None:
            return self.backbone(
                runtime_features,
                self.runtime_structure,
                relation_mode=self.relation_mode,
            )
        if not isinstance(self.backbone, RuntimeE4EquivariantSpnDistinguisher):
            raise ValueError("query overrides require the E4-equivariant backbone")
        return self.backbone(
            runtime_features,
            self.runtime_structure,
            relation_mode=self.relation_mode,
            query_input_mode=query_input_mode,
            query_structure=query_structure,
        )

    def encode(
        self,
        features: torch.Tensor,
        *,
        query_input_mode: Literal["delta_v", "delta_u"] | None = None,
        query_structure: RuntimeSpnStructure | None = None,
    ) -> torch.Tensor:
        if not isinstance(self.backbone, RuntimeE4EquivariantSpnDistinguisher):
            raise TypeError("RuntimeE4 encoding requires the E4-equivariant backbone")
        return self.backbone.encode(
            self._to_runtime_coordinates(features),
            self.runtime_structure,
            relation_mode=self.relation_mode,
            query_input_mode=query_input_mode,
            query_structure=query_structure,
        )

    def _to_runtime_coordinates(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        return features.reshape(
            batch,
            -1,
            2,
            self.runtime_structure.block_bits,
        ).flip(-1)


__all__ = [
    "FixedRuntimeSpnProtocolAdapter",
    "RuntimeCellTokenSpnDistinguisher",
    "RuntimeE4EquivariantSpnDistinguisher",
    "RuntimeE5GatedResidualSpnDistinguisher",
    "RuntimeParameterizedSpnDistinguisher",
    "RuntimeParameterizedSpnSpec",
]
