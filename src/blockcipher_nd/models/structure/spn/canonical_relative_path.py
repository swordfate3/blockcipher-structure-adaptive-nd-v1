from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal

import torch
from torch import nn

from blockcipher_nd.models.common.components import AttentionPooling
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    apply_gf2,
)
from blockcipher_nd.models.structure.spn.token_mixer_pairset import (
    EquivariantSpnTokenMixerBlock,
)
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    _RuntimeSpnEncoderBase,
)


PATH_FEATURE_SCHEMA = (
    *(
        f"source_{channel}_role{role}"
        for channel in ("left", "right", "xor")
        for role in range(4)
    ),
    *(
        f"middle_{channel}_role{role}"
        for channel in ("left", "right", "xor")
        for role in range(4)
    ),
    *(
        f"target_{channel}_role{role}"
        for channel in ("left", "right", "xor")
        for role in range(4)
    ),
    *(
        f"source_target_product_{channel}_role{role}"
        for channel in ("left", "right", "xor")
        for role in range(4)
    ),
    *(
        f"source_target_xor_{channel}_role{role}"
        for channel in ("left", "right", "xor")
        for role in range(4)
    ),
    *(
        f"reachable_source_role{source}_target_role{target}"
        for source in range(4)
        for target in range(4)
    ),
)


@dataclass(frozen=True)
class RelativePathTopology:
    source_cells: torch.Tensor
    middle_cells: torch.Tensor
    target_cells: torch.Tensor
    reachability: torch.Tensor
    fingerprint_sha256: str

    @property
    def path_count(self) -> int:
        return int(self.source_cells.numel())


@dataclass(frozen=True)
class RelativePathSpnSpec:
    hidden_dim: int = 64
    pair_embedding_dim: int = 128
    processor_steps: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.pair_embedding_dim, self.processor_steps) <= 0:
            raise ValueError("relative-path SPN dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("relative-path SPN dropout must be in [0, 1)")


class RelativeCrossTransitionSpnDistinguisher(nn.Module):
    """Permutation-invariant K1-D model over one directed two-transition path set."""

    def __init__(self, spec: RelativePathSpnSpec) -> None:
        super().__init__()
        self.spec = spec
        self.token_dim = max(16, spec.hidden_dim * 2)
        self.path_input_dim = len(PATH_FEATURE_SCHEMA)
        pair_dim = spec.pair_embedding_dim
        self.path_encoder = nn.Sequential(
            nn.Linear(self.path_input_dim, self.token_dim),
            nn.ReLU(),
            nn.LayerNorm(self.token_dim),
        )
        self.mixer_blocks = nn.ModuleList(
            [
                EquivariantSpnTokenMixerBlock(
                    nibbles_per_pair=16,
                    token_dim=self.token_dim,
                    token_mlp_ratio=2,
                    activation="relu",
                    norm="layernorm",
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
    ) -> torch.Tensor:
        return self.classifier(
            self.encode(ciphertext_pairs, structure, relation_mode=relation_mode)
        )

    def encode(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: Literal["true", "independent"] = "true",
    ) -> torch.Tensor:
        paths = self.relative_path_views(
            ciphertext_pairs,
            structure,
            relation_mode=relation_mode,
        )
        batch, pair_count, path_count, _ = paths.shape
        hidden = self.path_encoder(paths).reshape(
            batch * pair_count,
            path_count,
            self.token_dim,
        )
        for block in self.mixer_blocks:
            hidden = block(hidden)
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
    def relative_path_views(
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        relation_mode: Literal["true", "independent"],
    ) -> torch.Tensor:
        if relation_mode not in {"true", "independent"}:
            raise ValueError("relative-path relation mode must be true or independent")
        if structure.rounds != 2:
            raise ValueError("K1-D requires exactly two runtime transitions")
        pairs = _RuntimeSpnEncoderBase._normalize_pairs(
            ciphertext_pairs, structure.block_bits
        )
        if not torch.all((pairs == 0) | (pairs == 1)):
            raise ValueError("relative-path ciphertext pair tensors must be binary")
        pairs = pairs.to(dtype=torch.float32)
        topology = build_relative_path_topology(structure, relation_mode=relation_mode)
        boundaries = _native_boundary_states(pairs, structure, relation_mode)
        endpoints = tuple(
            _cell_endpoint_values(state, structure) for state in boundaries
        )
        device = pairs.device
        source_cells = topology.source_cells.to(device)
        middle_cells = topology.middle_cells.to(device)
        target_cells = topology.target_cells.to(device)
        source = endpoints[0][..., source_cells, :, :].flatten(-2)
        middle = endpoints[1][..., middle_cells, :, :].flatten(-2)
        target = endpoints[2][..., target_cells, :, :].flatten(-2)
        source_target_product = source * target
        source_target_xor = torch.remainder(source + target, 2.0)
        reachability = topology.reachability.to(device=device, dtype=pairs.dtype)
        reachability = reachability.reshape(1, 1, topology.path_count, 16).expand(
            *source.shape[:-1], 16
        )
        return torch.cat(
            (
                source,
                middle,
                target,
                source_target_product,
                source_target_xor,
                reachability,
            ),
            dim=-1,
        )


def build_relative_path_topology(
    structure: RuntimeSpnStructure,
    *,
    relation_mode: Literal["true", "independent"],
) -> RelativePathTopology:
    if relation_mode not in {"true", "independent"}:
        raise ValueError("relative-path relation mode must be true or independent")
    if structure.rounds != 2:
        raise ValueError("relative-path topology requires exactly two transitions")
    lookup = _cell_role_lookup(structure)
    if relation_mode == "true":
        first, second = structure.linear_matrices
    else:
        identity = torch.eye(structure.block_bits, dtype=torch.uint8)
        first = second = identity
    source_cells: list[int] = []
    middle_cells: list[int] = []
    target_cells: list[int] = []
    reachability_rows: list[torch.Tensor] = []
    for source_cell in range(structure.cells):
        source_bits = lookup[source_cell]
        for middle_cell in range(structure.cells):
            middle_bits = lookup[middle_cell]
            first_relation = first[middle_bits][:, source_bits].to(torch.bool)
            active_source_roles = first_relation.any(dim=0)
            if not bool(active_source_roles.any()):
                continue
            for target_cell in range(structure.cells):
                target_bits = lookup[target_cell]
                second_relation = second[target_bits][:, middle_bits].to(torch.bool)
                active_target_roles = second_relation.any(dim=1)
                reachability = (
                    active_source_roles[:, None] & active_target_roles[None, :]
                )
                if not bool(reachability.any()):
                    continue
                source_cells.append(source_cell)
                middle_cells.append(middle_cell)
                target_cells.append(target_cell)
                reachability_rows.append(reachability.to(torch.uint8))
    if not reachability_rows:
        raise ValueError("relative-path topology produced no connected paths")
    reachability = torch.stack(reachability_rows)
    payload = sorted(
        tuple(int(value) for value in row.flatten().tolist()) for row in reachability
    )
    fingerprint = hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return RelativePathTopology(
        source_cells=torch.tensor(source_cells, dtype=torch.long),
        middle_cells=torch.tensor(middle_cells, dtype=torch.long),
        target_cells=torch.tensor(target_cells, dtype=torch.long),
        reachability=reachability,
        fingerprint_sha256=fingerprint,
    )


def _native_boundary_states(
    pairs: torch.Tensor,
    structure: RuntimeSpnStructure,
    relation_mode: Literal["true", "independent"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current = pairs
    reversed_states = [current]
    for round_index in reversed(range(structure.rounds)):
        previous = (
            apply_gf2(structure.inverse_linear_matrices[round_index], current)
            if relation_mode == "true"
            else current
        )
        reversed_states.append(previous)
        current = previous
    return tuple(reversed(reversed_states))  # type: ignore[return-value]


def _cell_endpoint_values(
    pair_state: torch.Tensor,
    structure: RuntimeSpnStructure,
) -> torch.Tensor:
    left = pair_state[:, :, 0]
    right = pair_state[:, :, 1]
    difference = torch.remainder(left + right, 2.0)
    endpoints = torch.stack((left, right, difference), dim=-1)
    lookup = _cell_role_lookup(structure).to(pair_state.device)
    return endpoints[..., lookup, :]


def _cell_role_lookup(structure: RuntimeSpnStructure) -> torch.Tensor:
    indices = torch.empty(structure.cells, 4, dtype=torch.long)
    bit_indices = torch.arange(structure.block_bits)
    indices[structure.cell_membership, structure.bit_role] = bit_indices
    return indices


class FixedRelativePathSpnProtocolAdapter(nn.Module):
    """Bind the K1-D path model to one externally supplied two-transition window."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        structure: RuntimeSpnStructure,
        relation_mode: Literal["true", "independent"],
        spec: RelativePathSpnSpec,
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
            raise ValueError("relative-path pair_bits must encode two runtime blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError("relative-path input_bits must contain complete pairs")
        if structure.rounds != 2:
            raise ValueError("K1-D adapter requires exactly two runtime transitions")
        self.backbone = RelativeCrossTransitionSpnDistinguisher(spec)
        self.runtime_structure = structure
        self.relation_mode = relation_mode
        self.mapping_mode = relation_mode
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_loaded_rounds = structure.rounds
        self.runtime_round_window_mode = "relative_cross_transition_path"
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
        self.relative_path_count = topology.path_count
        self.relative_path_topology_sha256 = topology.fingerprint_sha256
        self.relative_path_feature_schema = PATH_FEATURE_SCHEMA
        self.relative_path_uses_absolute_cell_identity = False
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
        )


__all__ = [
    "FixedRelativePathSpnProtocolAdapter",
    "PATH_FEATURE_SCHEMA",
    "RelativeCrossTransitionSpnDistinguisher",
    "RelativePathSpnSpec",
    "RelativePathTopology",
    "build_relative_path_topology",
]
