from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from blockcipher_nd.models.structure.spn.gf2_boolean_view import gf2_boolean_views
from blockcipher_nd.models.structure.spn.operator_tied_latent import segment_mean
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


OPERATOR_TOKEN_DIM = 18


@dataclass(frozen=True)
class PositionPreservingOperatorSpec:
    hidden_dim: int = 32
    pair_embedding_dim: int = 128
    dropout: float = 0.0
    modulation_scale: float = 0.05

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.pair_embedding_dim) <= 0:
            raise ValueError("position-preserving operator dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("position-preserving operator dropout must be in [0, 1)")
        if not 0.0 < self.modulation_scale < 1.0:
            raise ValueError("position-preserving modulation scale must be in (0, 1)")


@dataclass(frozen=True)
class OperatorTokenBatch:
    values: torch.Tensor
    slots: torch.Tensor
    sources: torch.Tensor
    targets: torch.Tensor


class SharedPositionPreservingOperatorEncoder(nn.Module):
    """Encode actual GF(2) edges before target aggregation.

    Every learned function is shared across transition slots, cells, edges and
    block widths. Native endpoint positions enter each edge message before any
    invariant reduction, so distinct operators cannot collide merely because
    they share row/column-weight statistics.
    """

    def __init__(self, spec: PositionPreservingOperatorSpec) -> None:
        super().__init__()
        self.spec = spec
        hidden = spec.hidden_dim
        pair_dim = spec.pair_embedding_dim
        self.bit_encoder = nn.Sequential(
            nn.Linear(12, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(OPERATOR_TOKEN_DIM, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.edge_message = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.bit_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.bit_update_norm = nn.LayerNorm(hidden)
        self.pair_projection = nn.Sequential(
            nn.Linear(hidden * 3, pair_dim),
            nn.ReLU(),
            nn.LayerNorm(pair_dim),
        )
        self.structure_projection = nn.Sequential(
            nn.Linear(hidden * 3, pair_dim),
            nn.ReLU(),
            nn.LayerNorm(pair_dim),
        )

    def operator_tokens(
        self,
        structure: RuntimeSpnStructure,
        *,
        cell_position_ids: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> OperatorTokenBatch:
        target_device = torch.device("cpu") if device is None else torch.device(device)
        target_dtype = self.token_encoder[0].weight.dtype if dtype is None else dtype
        position_ids = _validated_cell_position_ids(
            structure,
            cell_position_ids,
            device=target_device,
        )
        membership = structure.cell_membership.to(target_device)
        roles = structure.bit_role.to(target_device)
        slot_rows: list[torch.Tensor] = []
        source_rows: list[torch.Tensor] = []
        target_rows: list[torch.Tensor] = []
        token_rows: list[torch.Tensor] = []
        for slot, matrix in enumerate(structure.inverse_linear_matrices):
            edges = torch.nonzero(matrix, as_tuple=False)
            if edges.numel() == 0:
                raise ValueError("position-preserving operator requires nonempty edges")
            targets = edges[:, 0].to(target_device)
            sources = edges[:, 1].to(target_device)
            edge_count = int(edges.shape[0])
            slot_feature = _position_triplet(
                torch.full((edge_count,), slot, device=target_device),
                structure.rounds,
                dtype=target_dtype,
            )
            source_cells = position_ids[membership[sources]]
            target_cells = position_ids[membership[targets]]
            source_feature = _position_triplet(
                source_cells,
                structure.cells,
                dtype=target_dtype,
            )
            target_feature = _position_triplet(
                target_cells,
                structure.cells,
                dtype=target_dtype,
            )
            source_role = F.one_hot(roles[sources], num_classes=4).to(target_dtype)
            target_role = F.one_hot(roles[targets], num_classes=4).to(target_dtype)
            relation = torch.ones(edge_count, 1, device=target_device, dtype=target_dtype)
            token_rows.append(
                torch.cat(
                    (
                        slot_feature,
                        source_feature,
                        target_feature,
                        source_role,
                        target_role,
                        relation,
                    ),
                    dim=-1,
                )
            )
            slot_rows.append(
                torch.full((edge_count,), slot, dtype=torch.long, device=target_device)
            )
            source_rows.append(sources)
            target_rows.append(targets)
        values = torch.cat(token_rows, dim=0)
        if values.shape[1] != OPERATOR_TOKEN_DIM:
            raise RuntimeError("position-preserving operator token width drifted")
        return OperatorTokenBatch(
            values=values,
            slots=torch.cat(slot_rows),
            sources=torch.cat(source_rows),
            targets=torch.cat(target_rows),
        )

    def structure_embedding(
        self,
        structure: RuntimeSpnStructure,
        *,
        cell_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.operator_tokens(
            structure,
            cell_position_ids=cell_position_ids,
            device=self.token_encoder[0].weight.device,
            dtype=self.token_encoder[0].weight.dtype,
        )
        hidden = self.token_encoder(tokens.values)
        return self.structure_projection(_invariant_pool_2d(hidden))

    def sample_modulation(
        self,
        ciphertext_pairs: torch.Tensor,
        runtime_structure: RuntimeSpnStructure,
        operator_structure: RuntimeSpnStructure,
        *,
        cell_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if runtime_structure.block_bits != operator_structure.block_bits:
            raise ValueError("runtime and operator widths must match")
        if runtime_structure.rounds != operator_structure.rounds:
            raise ValueError("runtime and operator transition counts must match")
        views = gf2_boolean_views(ciphertext_pairs, runtime_structure)
        batch, pair_count, bit_count, _ = views.shape
        bit_state = self.bit_encoder(views).reshape(
            batch * pair_count,
            bit_count,
            self.spec.hidden_dim,
        )
        tokens = self.operator_tokens(
            operator_structure,
            cell_position_ids=cell_position_ids,
            device=bit_state.device,
            dtype=bit_state.dtype,
        )
        token_hidden = self.token_encoder(tokens.values)
        for slot in range(operator_structure.rounds):
            selected = tokens.slots == slot
            sources = tokens.sources[selected]
            targets = tokens.targets[selected]
            fixed = token_hidden[selected].reshape(1, -1, self.spec.hidden_dim)
            fixed = fixed.expand(bit_state.shape[0], -1, -1)
            messages = self.edge_message(
                torch.cat(
                    (
                        bit_state[:, sources],
                        bit_state[:, targets],
                        fixed,
                    ),
                    dim=-1,
                )
            )
            target_messages = segment_mean(messages, targets, bit_count)
            update = self.bit_update(torch.cat((bit_state, target_messages), dim=-1))
            bit_state = self.bit_update_norm(bit_state + update)
        pair_embedding = self.pair_projection(_invariant_pool_3d(bit_state)).reshape(
            batch,
            pair_count,
            self.spec.pair_embedding_dim,
        )
        return torch.cat(
            (
                pair_embedding.mean(dim=1),
                pair_embedding.max(dim=1).values,
                torch.sqrt(pair_embedding.square().mean(dim=1).clamp_min(1e-8)),
            ),
            dim=-1,
        )


class PositionPreservingOperatorK1AzProbe(nn.Module):
    """K1-BB readiness wrapper with an exact disabled K1-AZ path."""

    def __init__(
        self,
        anchor: nn.Module,
        spec: PositionPreservingOperatorSpec,
    ) -> None:
        super().__init__()
        self.anchor = anchor
        self.operator_encoder = SharedPositionPreservingOperatorEncoder(spec)
        self.spec = spec
        self.uses_cipher_identity = False
        self.uses_per_cipher_parameters = False
        self.uses_invariant_linear_summary = False
        self.uses_actual_source_target_connectivity = True
        self.operator_interaction_before_pooling = True

    def disabled_logits(
        self,
        features: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        gate_summary: torch.Tensor,
    ) -> torch.Tensor:
        return self.anchor.logits_with_runtime(
            features,
            structure,
            apply_sboxes=True,
            transition_branch_enabled=True,
            gate_summary=gate_summary,
            dual_path_enabled=True,
            component_separation_enabled=True,
        )

    def logits_with_operator(
        self,
        features: torch.Tensor,
        runtime_structure: RuntimeSpnStructure,
        operator_structure: RuntimeSpnStructure,
        *,
        gate_summary: torch.Tensor,
        cell_position_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return self.disabled_logits(
                features,
                runtime_structure,
                gate_summary=gate_summary,
            )
        runtime = features.reshape(
            features.shape[0],
            -1,
            2,
            runtime_structure.block_bits,
        ).flip(-1)
        base_embedding = self.anchor.backbone.base.encode(runtime, runtime_structure)
        edge_residual = self.anchor.backbone.edge_residual_embedding(
            runtime,
            runtime_structure,
            apply_sboxes=True,
        )
        modulation = self.operator_encoder.sample_modulation(
            runtime,
            runtime_structure,
            operator_structure,
            cell_position_ids=cell_position_ids,
        )
        global_edge_gate = torch.tanh(self.anchor.backbone.residual_gate)
        edge_scale = global_edge_gate + self.spec.modulation_scale * torch.tanh(
            modulation
        )
        combined = base_embedding + edge_scale * torch.tanh(edge_residual)
        transition = self.anchor.backbone.transition_embedding(
            runtime,
            runtime_structure,
            apply_sboxes=True,
        )
        _, transition_gate = self.anchor.effective_path_gates(
            runtime_structure,
            summary=gate_summary,
            dual_path_enabled=True,
            component_separation_enabled=True,
        )
        combined = combined + transition_gate * torch.tanh(transition.repeat(1, 3))
        return self.anchor.backbone.base.classifier(combined)


def transported_position_ids(
    cell_permutation: torch.Tensor | list[int] | tuple[int, ...],
) -> torch.Tensor:
    permutation = torch.as_tensor(cell_permutation, dtype=torch.long, device="cpu")
    if permutation.ndim != 1 or not torch.equal(
        torch.sort(permutation).values,
        torch.arange(permutation.numel()),
    ):
        raise ValueError("cell permutation must contain every cell exactly once")
    transported = torch.empty_like(permutation)
    transported[permutation] = torch.arange(permutation.numel())
    return transported


def relabel_runtime_pairs(
    pairs: torch.Tensor,
    bit_permutation: torch.Tensor,
) -> torch.Tensor:
    permutation = torch.as_tensor(
        bit_permutation,
        dtype=torch.long,
        device=pairs.device,
    )
    if pairs.shape[-1] != permutation.numel():
        raise ValueError("pair width does not match bit permutation")
    relabeled = torch.empty_like(pairs)
    relabeled[..., permutation] = pairs
    return relabeled


def trainable_parameter_geometry(module: nn.Module) -> Mapping[str, tuple[int, ...]]:
    return {
        name: tuple(parameter.shape)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _validated_cell_position_ids(
    structure: RuntimeSpnStructure,
    values: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    position_ids = (
        torch.arange(structure.cells, dtype=torch.long, device=device)
        if values is None
        else torch.as_tensor(values, dtype=torch.long, device=device)
    )
    if position_ids.shape != (structure.cells,) or not torch.equal(
        torch.sort(position_ids).values,
        torch.arange(structure.cells, device=device),
    ):
        raise ValueError("cell position ids must be a permutation of native cells")
    return position_ids


def _position_triplet(
    indices: torch.Tensor,
    count: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("position count must be positive")
    values = indices.to(dtype)
    normalized = torch.zeros_like(values) if count == 1 else 2.0 * values / (count - 1) - 1.0
    angle = math.pi * normalized
    return torch.stack((normalized, torch.sin(angle), torch.cos(angle)), dim=-1)


def _invariant_pool_2d(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("operator pooling requires [items, channels]")
    return torch.cat(
        (
            values.mean(dim=0),
            values.max(dim=0).values,
            torch.sqrt(values.square().mean(dim=0).clamp_min(1e-8)),
        ),
        dim=-1,
    )


def _invariant_pool_3d(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("sample pooling requires [batch, items, channels]")
    return torch.cat(
        (
            values.mean(dim=1),
            values.max(dim=1).values,
            torch.sqrt(values.square().mean(dim=1).clamp_min(1e-8)),
        ),
        dim=-1,
    )


__all__ = [
    "OPERATOR_TOKEN_DIM",
    "OperatorTokenBatch",
    "PositionPreservingOperatorK1AzProbe",
    "PositionPreservingOperatorSpec",
    "SharedPositionPreservingOperatorEncoder",
    "relabel_runtime_pairs",
    "trainable_parameter_geometry",
    "transported_position_ids",
]
