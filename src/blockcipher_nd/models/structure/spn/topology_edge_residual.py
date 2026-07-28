from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn

from blockcipher_nd.models.structure.spn.gf2_boolean_view import (
    Gf2BooleanViewSpnDistinguisher,
    Gf2BooleanViewSpnSpec,
    boolean_view_fingerprint,
    gf2_boolean_views,
)
from blockcipher_nd.models.structure.spn.operator_tied_latent import (
    invariant_pool,
    operator_routing_fingerprint,
    segment_mean,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


@dataclass(frozen=True)
class TopologyEdgeResidualSpnSpec:
    hidden_dim: int = 32
    pair_embedding_dim: int = 128
    dropout: float = 0.0
    initial_effective_gate: float = 0.0

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.pair_embedding_dim) <= 0:
            raise ValueError("topology edge-residual dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("topology edge-residual dropout must be in [0, 1)")
        if not -1.0 < self.initial_effective_gate < 1.0:
            raise ValueError("topology edge-residual effective gate must be in (-1, 1)")


class TopologyEdgeResidualSpnDistinguisher(nn.Module):
    """K1-I plus a bounded equivariant residual over explicit GF(2) edges."""

    def __init__(self, spec: TopologyEdgeResidualSpnSpec) -> None:
        super().__init__()
        self.spec = spec
        hidden = spec.hidden_dim
        pair_dim = spec.pair_embedding_dim
        self.base = Gf2BooleanViewSpnDistinguisher(
            Gf2BooleanViewSpnSpec(
                hidden_dim=hidden,
                pair_embedding_dim=pair_dim,
                dropout=spec.dropout,
            )
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        edge_input_dim = hidden * 4 + 10
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.cell_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.cell_update_norm = nn.LayerNorm(hidden)
        self.residual_pair_projection = nn.Sequential(
            nn.Linear(hidden * 3, pair_dim),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
        )
        self.residual_gate = nn.Parameter(
            torch.tensor(math.atanh(spec.initial_effective_gate), dtype=torch.float32)
        )

    def forward(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
    ) -> torch.Tensor:
        base_embedding = self.base.encode(ciphertext_pairs, structure)
        residual = self.edge_residual_embedding(ciphertext_pairs, structure)
        combined = base_embedding + torch.tanh(self.residual_gate) * torch.tanh(
            residual
        )
        return self.base.classifier(combined)

    def edge_residual_embedding(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        slot_mask: tuple[bool, bool] = (True, True),
    ) -> torch.Tensor:
        if len(slot_mask) != 2 or not any(slot_mask):
            raise ValueError("edge residual must retain at least one transition slot")
        views = gf2_boolean_views(ciphertext_pairs, structure)
        batch, pair_count, bit_count, _ = views.shape
        bit_hidden = self.base.bit_encoder(views).reshape(
            batch * pair_count,
            bit_count,
            self.spec.hidden_dim,
        )
        lookup = ordered_cell_role_lookup(structure).to(bit_hidden.device)
        cell_input = bit_hidden[:, lookup].flatten(-2)
        initial_cells = self.cell_encoder(cell_input)
        cell_state = initial_cells
        membership = structure.cell_membership.to(bit_hidden.device)
        roles = structure.bit_role.to(bit_hidden.device)
        for slot, enabled in enumerate(slot_mask):
            if not enabled:
                continue
            matrix = structure.inverse_linear_matrices[slot]
            edges = torch.nonzero(matrix, as_tuple=False)
            if edges.numel() == 0:
                raise ValueError("topology edge residual requires nonempty operators")
            targets = edges[:, 0].to(bit_hidden.device)
            sources = edges[:, 1].to(bit_hidden.device)
            target_cells = membership[targets]
            source_cells = membership[sources]
            edge_count = int(edges.shape[0])
            source_role = torch.nn.functional.one_hot(roles[sources], num_classes=4).to(
                bit_hidden.dtype
            )
            target_role = torch.nn.functional.one_hot(roles[targets], num_classes=4).to(
                bit_hidden.dtype
            )
            slot_code = torch.nn.functional.one_hot(
                torch.tensor(slot, device=bit_hidden.device), num_classes=2
            ).to(bit_hidden.dtype)
            fixed = torch.cat((source_role, target_role), dim=-1)
            fixed = torch.cat(
                (fixed, slot_code.reshape(1, 2).expand(edge_count, -1)), dim=-1
            )
            fixed = fixed.reshape(1, edge_count, 10).expand(bit_hidden.shape[0], -1, -1)
            edge_input = torch.cat(
                (
                    cell_state[:, source_cells],
                    cell_state[:, target_cells],
                    bit_hidden[:, sources],
                    bit_hidden[:, targets],
                    fixed,
                ),
                dim=-1,
            )
            messages = self.edge_encoder(edge_input)
            target_messages = segment_mean(
                messages,
                target_cells,
                structure.cells,
            )
            update = self.cell_update(torch.cat((cell_state, target_messages), dim=-1))
            cell_state = self.cell_update_norm(cell_state + update)
        topology_delta = cell_state - initial_cells
        pair_residual = self.residual_pair_projection(
            invariant_pool(topology_delta)
        ).reshape(batch, pair_count, self.spec.pair_embedding_dim)
        return torch.cat(
            (
                pair_residual.mean(dim=1),
                pair_residual.max(dim=1).values,
                torch.sqrt(pair_residual.square().mean(dim=1).clamp_min(1e-8)),
            ),
            dim=-1,
        )


class FixedTopologyEdgeResidualSpnProtocolAdapter(nn.Module):
    """Bind K1-K to one external two-transition runtime descriptor."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        structure: RuntimeSpnStructure,
        spec: TopologyEdgeResidualSpnSpec,
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
            raise ValueError("K1-K pair_bits must encode two runtime blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError("K1-K input_bits must contain complete pairs")
        if structure.rounds != 2:
            raise ValueError("K1-K adapter requires exactly two transitions")
        self.backbone = TopologyEdgeResidualSpnDistinguisher(spec)
        self.runtime_structure = structure
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_loaded_rounds = structure.rounds
        self.runtime_round_window_mode = "gf2_topology_edge_residual"
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
        self.topology_edge_sha256 = topology_edge_fingerprint(structure)
        self.topology_edge_counts = tuple(
            int(matrix.sum()) for matrix in structure.inverse_linear_matrices
        )
        self.deterministic_gf2_views = True
        self.boolean_view_names = (
            "raw",
            "single_0",
            "single_1",
            "composed_0_after_1",
        )
        self.boolean_channels_per_bit = 12
        self.uses_raw_bypass = False
        self.uses_learned_message_passing = True
        self.uses_path_tokens = False
        self.uses_absolute_cell_or_bit_identity = False
        self.uses_cipher_identity = False
        self.uses_sbox_semantics = False
        self.uses_ordered_cell_roles = True
        self.uses_explicit_source_target_edges = True
        self.uses_transition_slot_identity = True
        self.residual_gate_bounded = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        runtime = features.reshape(
            features.shape[0],
            -1,
            2,
            self.runtime_structure.block_bits,
        ).flip(-1)
        return self.backbone(runtime, self.runtime_structure)


def ordered_cell_role_lookup(structure: RuntimeSpnStructure) -> torch.Tensor:
    lookup = torch.empty(structure.cells, 4, dtype=torch.long)
    indices = torch.arange(structure.block_bits)
    lookup[structure.cell_membership, structure.bit_role] = indices
    return lookup


def topology_edge_fingerprint(structure: RuntimeSpnStructure) -> str:
    digest = hashlib.sha256()
    digest.update(structure.cell_membership.numpy().tobytes())
    digest.update(structure.bit_role.numpy().tobytes())
    for slot, matrix in enumerate(structure.inverse_linear_matrices):
        digest.update(slot.to_bytes(1, "little"))
        digest.update(matrix.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "FixedTopologyEdgeResidualSpnProtocolAdapter",
    "TopologyEdgeResidualSpnDistinguisher",
    "TopologyEdgeResidualSpnSpec",
    "ordered_cell_role_lookup",
    "topology_edge_fingerprint",
]
