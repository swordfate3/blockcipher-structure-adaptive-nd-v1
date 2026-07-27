from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Literal

import torch
from torch import nn

from blockcipher_nd.models.common.components import AttentionPooling
from blockcipher_nd.models.structure.spn.canonical_relative_path import (
    PATH_FEATURE_SCHEMA,
    RelativeCrossTransitionSpnDistinguisher,
    RelativePathTopology,
    build_relative_path_topology,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


IncidenceMode = Literal["true", "shuffled"]


@dataclass(frozen=True)
class CellPathHypergraphSpnSpec:
    hidden_dim: int = 64
    pair_embedding_dim: int = 128
    processor_steps: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.pair_embedding_dim, self.processor_steps) <= 0:
            raise ValueError("cell/path hypergraph dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("cell/path hypergraph dropout must be in [0, 1)")


class EquivariantCellPathMessageBlock(nn.Module):
    """Exchange messages through shared boundary cells without numeric cell features."""

    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        message_hidden = token_dim * 2
        self.path_norm = nn.LayerNorm(token_dim)
        self.incidence_mixer = nn.Sequential(
            nn.Linear(token_dim * 4, message_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(message_hidden, token_dim),
        )
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(
        self,
        paths: torch.Tensor,
        topology: RelativePathTopology,
        *,
        cell_count: int,
    ) -> torch.Tensor:
        normalized = self.path_norm(paths)
        source, middle, target = self.aggregate_nodes(
            normalized,
            topology,
            cell_count=cell_count,
        )
        source_index = topology.source_cells.to(paths.device)
        middle_index = topology.middle_cells.to(paths.device)
        target_index = topology.target_cells.to(paths.device)
        incidence_message = self.incidence_mixer(
            torch.cat(
                (
                    normalized,
                    source[:, source_index],
                    middle[:, middle_index],
                    target[:, target_index],
                ),
                dim=-1,
            )
        )
        paths = paths + incidence_message
        return paths + self.channel_mixer(self.channel_norm(paths))

    @staticmethod
    def aggregate_nodes(
        paths: torch.Tensor,
        topology: RelativePathTopology,
        *,
        cell_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            _segment_mean(paths, indices.to(paths.device), cell_count)
            for indices in (
                topology.source_cells,
                topology.middle_cells,
                topology.target_cells,
            )
        )  # type: ignore[return-value]


class CellPathHypergraphSpnDistinguisher(nn.Module):
    """Permutation-invariant classifier with equivariant cell/path message passing."""

    def __init__(self, spec: CellPathHypergraphSpnSpec) -> None:
        super().__init__()
        self.spec = spec
        self.token_dim = spec.hidden_dim
        self.path_input_dim = len(PATH_FEATURE_SCHEMA)
        pair_dim = spec.pair_embedding_dim
        self.path_encoder = nn.Sequential(
            nn.Linear(self.path_input_dim, self.token_dim),
            nn.ReLU(),
            nn.LayerNorm(self.token_dim),
        )
        self.message_blocks = nn.ModuleList(
            [
                EquivariantCellPathMessageBlock(
                    token_dim=self.token_dim,
                    dropout=spec.dropout,
                )
                for _ in range(spec.processor_steps)
            ]
        )
        self.sequence_norm = nn.LayerNorm(self.token_dim)
        self.pair_projection = nn.Sequential(
            nn.Linear(self.token_dim * 3, pair_dim),
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

    def forward(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: Literal["true", "independent"] = "true",
        incidence_mode: IncidenceMode = "true",
    ) -> torch.Tensor:
        return self.classifier(
            self.encode(
                ciphertext_pairs,
                structure,
                relation_mode=relation_mode,
                incidence_mode=incidence_mode,
            )
        )

    def encode(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: Literal["true", "independent"] = "true",
        incidence_mode: IncidenceMode = "true",
    ) -> torch.Tensor:
        paths, topology = self.path_views_and_routing(
            ciphertext_pairs,
            structure,
            relation_mode=relation_mode,
            incidence_mode=incidence_mode,
        )
        batch, pair_count, path_count, _ = paths.shape
        hidden = self.path_encoder(paths).reshape(
            batch * pair_count,
            path_count,
            self.token_dim,
        )
        for block in self.message_blocks:
            hidden = block(hidden, topology, cell_count=structure.cells)
        hidden = self.sequence_norm(hidden)
        path_mean = hidden.mean(dim=1)
        path_max = hidden.max(dim=1).values
        path_rms = torch.sqrt(hidden.square().mean(dim=1).clamp_min(1e-8))
        pair_embeddings = self.pair_projection(
            torch.cat((path_mean, path_max, path_rms), dim=-1)
        ).reshape(batch, pair_count, self.spec.pair_embedding_dim)
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

    @staticmethod
    def path_views_and_routing(
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: Literal["true", "independent"],
        incidence_mode: IncidenceMode,
    ) -> tuple[torch.Tensor, RelativePathTopology]:
        if incidence_mode not in {"true", "shuffled"}:
            raise ValueError("incidence mode must be true or shuffled")
        paths = RelativeCrossTransitionSpnDistinguisher.relative_path_views(
            ciphertext_pairs,
            structure,
            relation_mode=relation_mode,
        )
        topology = build_relative_path_topology(
            structure,
            relation_mode=relation_mode,
        )
        if incidence_mode == "shuffled":
            topology = shuffle_path_incidence(topology, seed=20260728)
        return paths, topology


def shuffle_path_incidence(
    topology: RelativePathTopology,
    *,
    seed: int,
) -> RelativePathTopology:
    generator = torch.Generator().manual_seed(seed)
    middle_order = torch.randperm(topology.path_count, generator=generator)
    target_order = torch.randperm(topology.path_count, generator=generator)
    middle = topology.middle_cells[middle_order]
    target = topology.target_cells[target_order]
    digest = hashlib.sha256()
    for values in (topology.source_cells, middle, target):
        digest.update(values.detach().cpu().numpy().tobytes())
    return replace(
        topology,
        middle_cells=middle,
        target_cells=target,
        fingerprint_sha256=digest.hexdigest(),
    )


def routing_fingerprint(topology: RelativePathTopology) -> str:
    digest = hashlib.sha256()
    for values in (
        topology.source_cells,
        topology.middle_cells,
        topology.target_cells,
    ):
        digest.update(values.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _segment_mean(
    values: torch.Tensor,
    indices: torch.Tensor,
    segment_count: int,
) -> torch.Tensor:
    if values.ndim != 3 or indices.ndim != 1 or values.shape[1] != indices.numel():
        raise ValueError("segment mean requires [batch, paths, channels] and [paths]")
    output = values.new_zeros(values.shape[0], segment_count, values.shape[2])
    expanded = indices.reshape(1, -1, 1).expand(
        values.shape[0],
        -1,
        values.shape[2],
    )
    output.scatter_add_(1, expanded, values)
    counts = values.new_zeros(segment_count)
    counts.scatter_add_(0, indices, values.new_ones(indices.numel()))
    return output / counts.clamp_min(1.0).reshape(1, -1, 1)


class FixedCellPathHypergraphSpnProtocolAdapter(nn.Module):
    """Bind K1-F to one externally supplied two-transition runtime window."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        structure: RuntimeSpnStructure,
        relation_mode: Literal["true", "independent"],
        incidence_mode: IncidenceMode,
        spec: CellPathHypergraphSpnSpec,
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
            raise ValueError("cell/path hypergraph pair_bits must encode two blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError(
                "cell/path hypergraph input_bits must contain complete pairs"
            )
        if structure.rounds != 2:
            raise ValueError("K1-F adapter requires exactly two runtime transitions")
        self.backbone = CellPathHypergraphSpnDistinguisher(spec)
        self.runtime_structure = structure
        self.relation_mode = relation_mode
        self.incidence_mode = incidence_mode
        self.mapping_mode = relation_mode
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_loaded_rounds = structure.rounds
        self.runtime_round_window_mode = "cell_path_hypergraph"
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
        topology = build_relative_path_topology(structure, relation_mode=relation_mode)
        routing = (
            shuffle_path_incidence(topology, seed=20260728)
            if incidence_mode == "shuffled"
            else topology
        )
        self.relative_path_count = topology.path_count
        self.relative_path_topology_sha256 = topology.fingerprint_sha256
        self.cell_path_routing_sha256 = routing_fingerprint(routing)
        self.relative_path_feature_schema = PATH_FEATURE_SCHEMA
        self.relative_path_uses_absolute_cell_identity = False
        self.cell_indices_are_numeric_features = False
        self.cell_indices_are_routing_only = True
        self.relative_path_compositions = 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        runtime = features.reshape(
            features.shape[0],
            -1,
            2,
            self.runtime_structure.block_bits,
        ).flip(-1)
        return self.backbone(
            runtime,
            self.runtime_structure,
            relation_mode=self.relation_mode,
            incidence_mode=self.incidence_mode,
        )


__all__ = [
    "CellPathHypergraphSpnDistinguisher",
    "CellPathHypergraphSpnSpec",
    "EquivariantCellPathMessageBlock",
    "FixedCellPathHypergraphSpnProtocolAdapter",
    "routing_fingerprint",
    "shuffle_path_incidence",
]
