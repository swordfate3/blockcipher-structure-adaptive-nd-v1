from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


EDGE_TOKEN_DIM = 17
SBOX_TOKEN_DIM = 70


@dataclass(frozen=True)
class StructureProgramEncoderSpec:
    hidden_dim: int = 48
    embedding_dim: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.embedding_dim) <= 0:
            raise ValueError("structure-program dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("structure-program dropout must be in [0, 1)")


class RuntimeSpnProgramEncoder(nn.Module):
    """Encode a runtime SPN as an ordered program of S-box and GF(2) stages."""

    def __init__(self, spec: StructureProgramEncoderSpec) -> None:
        super().__init__()
        self.spec = spec
        hidden = spec.hidden_dim
        self.sbox_encoder = nn.Sequential(
            nn.Linear(SBOX_TOKEN_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(EDGE_TOKEN_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.stage_fusion = nn.Sequential(
            nn.Linear(hidden * 6, hidden * 2),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.program_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden * 3, spec.embedding_dim),
            nn.LayerNorm(spec.embedding_dim),
        )
        self.uses_cipher_identity = False
        self.uses_cipher_name = False
        self.uses_actual_source_target_connectivity = True
        self.uses_sbox_truth_tables = True
        self.preserves_transition_order = True

    def forward(self, structures: Sequence[RuntimeSpnStructure]) -> torch.Tensor:
        if not structures:
            raise ValueError("structure-program encoder requires at least one structure")
        embeddings = [self.encode_structure(structure) for structure in structures]
        return torch.stack(embeddings)

    def encode_structure(
        self,
        structure: RuntimeSpnStructure,
        *,
        cell_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = self.sbox_encoder[0].weight.device
        dtype = self.sbox_encoder[0].weight.dtype
        position_ids = _validated_position_ids(
            structure,
            cell_position_ids,
            device=device,
        )
        membership = position_ids[structure.cell_membership.to(device)]
        roles = structure.bit_role.to(device)
        stage_rows: list[torch.Tensor] = []
        for stage in range(structure.rounds):
            stage_position = _position_triplet(
                torch.full((structure.cells,), stage, device=device),
                structure.rounds,
                dtype=dtype,
            )
            cell_position = _position_triplet(
                position_ids,
                structure.cells,
                dtype=dtype,
            )
            truth = structure.sbox_truth_bits[stage].to(device=device, dtype=dtype)
            sbox_tokens = torch.cat((stage_position, cell_position, truth), dim=-1)
            if sbox_tokens.shape[-1] != SBOX_TOKEN_DIM:
                raise RuntimeError("S-box structure token width drifted")
            sbox_hidden = self.sbox_encoder(sbox_tokens)

            matrix = structure.inverse_linear_matrices[stage]
            edges = torch.nonzero(matrix, as_tuple=False)
            if edges.numel() == 0:
                raise ValueError("structure program requires nonempty GF(2) edges")
            targets = edges[:, 0].to(device)
            sources = edges[:, 1].to(device)
            edge_count = int(edges.shape[0])
            edge_stage = _position_triplet(
                torch.full((edge_count,), stage, device=device),
                structure.rounds,
                dtype=dtype,
            )
            source_cell = _position_triplet(
                membership[sources],
                structure.cells,
                dtype=dtype,
            )
            target_cell = _position_triplet(
                membership[targets],
                structure.cells,
                dtype=dtype,
            )
            source_role = F.one_hot(roles[sources], num_classes=4).to(dtype)
            target_role = F.one_hot(roles[targets], num_classes=4).to(dtype)
            edge_tokens = torch.cat(
                (edge_stage, source_cell, target_cell, source_role, target_role),
                dim=-1,
            )
            if edge_tokens.shape[-1] != EDGE_TOKEN_DIM:
                raise RuntimeError("GF(2) structure token width drifted")
            edge_hidden = self.edge_encoder(edge_tokens)
            stage_rows.append(
                self.stage_fusion(
                    torch.cat((_pool(sbox_hidden), _pool(edge_hidden)), dim=-1)
                )
            )

        program = torch.stack(stage_rows).unsqueeze(0)
        recurrent, final = self.program_gru(program)
        summary = torch.cat(
            (
                final.squeeze(0).squeeze(0),
                recurrent.mean(dim=1).squeeze(0),
                recurrent.max(dim=1).values.squeeze(0),
            ),
            dim=-1,
        )
        return F.normalize(self.output_projection(summary), dim=-1)


def _validated_position_ids(
    structure: RuntimeSpnStructure,
    values: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    if values is None:
        return torch.arange(structure.cells, device=device)
    position_ids = torch.as_tensor(values, dtype=torch.long, device=device)
    if position_ids.shape != (structure.cells,):
        raise ValueError("cell position IDs must contain one value per cell")
    if not torch.equal(
        torch.sort(position_ids).values,
        torch.arange(structure.cells, device=device),
    ):
        raise ValueError("cell position IDs must be a permutation")
    return position_ids


def _position_triplet(
    positions: torch.Tensor,
    count: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    denominator = max(count - 1, 1)
    normalized = positions.to(dtype) / float(denominator)
    angle = normalized * (2.0 * math.pi)
    return torch.stack((normalized, torch.sin(angle), torch.cos(angle)), dim=-1)


def _pool(values: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            values.mean(dim=0),
            values.max(dim=0).values,
            torch.sqrt(values.square().mean(dim=0).clamp_min(1e-8)),
        ),
        dim=-1,
    )


__all__ = [
    "EDGE_TOKEN_DIM",
    "SBOX_TOKEN_DIM",
    "RuntimeSpnProgramEncoder",
    "StructureProgramEncoderSpec",
]
