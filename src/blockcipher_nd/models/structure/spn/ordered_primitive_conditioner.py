from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from blockcipher_nd.models.common.components import AttentionPooling
from blockcipher_nd.models.structure.spn.gf2_boolean_view import apply_gf2_operator
from blockcipher_nd.models.structure.spn.ordered_primitive_program import (
    GF2_EXPERT,
    PERMUTATION_EXPERT,
    CompiledSpnProgram,
    materialize_ordered_primitive_payload,
    replay_ordered_primitive_program,
)


@dataclass(frozen=True)
class OrderedPrimitiveConditionerSpec:
    hidden_dim: int = 32
    pair_embedding_dim: int = 128
    dropout: float = 0.0
    initial_effective_gate: float = 0.05

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.pair_embedding_dim) <= 0:
            raise ValueError("ordered primitive dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("ordered primitive dropout must be in [0, 1)")
        if not -1.0 < self.initial_effective_gate < 1.0:
            raise ValueError("ordered primitive gate must be in (-1, 1)")


class WidthIndependentRawPairBackbone(nn.Module):
    """Raw pair encoder whose parameter shapes do not depend on state width."""

    def __init__(self, spec: OrderedPrimitiveConditionerSpec) -> None:
        super().__init__()
        hidden = spec.hidden_dim
        pair_dim = spec.pair_embedding_dim
        self.spec = spec
        self.bit_encoder = nn.Sequential(
            nn.Linear(3, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(hidden * 3, pair_dim),
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
            nn.Linear(pair_dim * 3, hidden * 4),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden * 4, 1),
        )

    def encode(self, ciphertext_pairs: torch.Tensor) -> torch.Tensor:
        left = ciphertext_pairs[:, :, 0]
        right = ciphertext_pairs[:, :, 1]
        views = torch.stack((left, right, torch.remainder(left + right, 2.0)), -1)
        bit_hidden = self.bit_encoder(views)
        pooled_bits = torch.cat(
            (
                bit_hidden.mean(dim=2),
                bit_hidden.max(dim=2).values,
                torch.sqrt(bit_hidden.square().mean(dim=2).clamp_min(1e-8)),
            ),
            dim=-1,
        )
        pair_hidden = self.pair_projection(pooled_bits)
        attended, _attention = self.pair_attention(pair_hidden)
        return torch.cat(
            (attended, pair_hidden.mean(dim=1), pair_hidden.max(dim=1).values),
            dim=-1,
        )


class OrderedPrimitiveConditioner(nn.Module):
    """Execute compiled primitives and encode them with shared learned experts."""

    def __init__(
        self,
        program: CompiledSpnProgram,
        spec: OrderedPrimitiveConditionerSpec,
    ) -> None:
        super().__init__()
        hidden = spec.hidden_dim
        pair_dim = spec.pair_embedding_dim
        self.spec = spec
        self.program = program
        self.runtime_structure = replay_ordered_primitive_program(program)
        truth, inverse = materialize_ordered_primitive_payload(program)
        self.register_buffer("sbox_truth_bits", truth.to(torch.float32))
        self.register_buffer("inverse_linear_matrices", inverse.to(torch.float32))
        self.register_buffer(
            "semantic_cell_bits",
            torch.tensor(program.semantic_cell_to_native_bits, dtype=torch.long),
        )
        edge_tokens, edge_masks, expert_types = _padded_edge_descriptors(program)
        self.register_buffer("edge_tokens", edge_tokens)
        self.register_buffer("edge_masks", edge_masks)
        self.register_buffer("linear_expert_types", expert_types)

        self.histogram_encoder = nn.Sequential(
            nn.Linear(16, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.sbox_descriptor_encoder = nn.Sequential(
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.edge_descriptor_encoder = nn.Sequential(
            nn.Linear(10, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.sbox_expert = _primitive_expert(hidden)
        self.permutation_expert = _primitive_expert(hidden)
        self.gf2_expert = _primitive_expert(hidden)
        self.cell_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.cell_attention = AttentionPooling(
            hidden,
            hidden_bits=hidden,
            activation="relu",
            norm="layernorm",
        )
        self.stage_projection = nn.Sequential(
            nn.Linear(hidden * 3, pair_dim),
            nn.ReLU(),
            nn.LayerNorm(pair_dim),
        )
        self.stage_recurrence = nn.GRUCell(pair_dim, pair_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.ReLU(),
            nn.LayerNorm(pair_dim),
        )

    def forward(self, ciphertext_pairs: torch.Tensor) -> torch.Tensor:
        if ciphertext_pairs.ndim != 4 or ciphertext_pairs.shape[2] != 2:
            raise ValueError("ciphertext pairs must have shape [batch, pairs, 2, bits]")
        if ciphertext_pairs.shape[-1] != self.program.block_bits:
            raise ValueError("ciphertext pair width does not match compiled program")
        if not torch.all((ciphertext_pairs == 0) | (ciphertext_pairs == 1)):
            raise ValueError("ordered primitive conditioner requires binary input")

        current = ciphertext_pairs
        recurrent = torch.zeros(
            current.shape[0],
            self.spec.pair_embedding_dim,
            dtype=current.dtype,
            device=current.device,
        )
        for stage_index in reversed(range(self.program.rounds)):
            triplet = torch.stack(
                (
                    current[:, :, 0],
                    current[:, :, 1],
                    torch.remainder(current[:, :, 0] + current[:, :, 1], 2.0),
                ),
                dim=-1,
            )
            linear_triplet = apply_gf2_operator(
                triplet,
                self.inverse_linear_matrices[stage_index],
            )
            linear_state = linear_triplet[..., :2].permute(0, 1, 3, 2)
            linear_histogram = self._difference_histogram(linear_state)
            linear_hidden = self.histogram_encoder(linear_histogram)
            edge_hidden = self._edge_embedding(stage_index, current.device)
            linear_inputs = torch.cat(
                (
                    linear_hidden,
                    edge_hidden.unsqueeze(0).expand(current.shape[0], -1, -1),
                ),
                dim=-1,
            )
            permutation = self.permutation_expert(linear_inputs)
            gf2 = self.gf2_expert(linear_inputs)
            expert_types = self.linear_expert_types[stage_index].to(current.device)
            linear_encoded = torch.where(
                expert_types.reshape(1, -1, 1) == 0,
                permutation,
                gf2,
            )

            left = self.runtime_structure.apply_inverse_sboxes(
                linear_state[:, :, 0], stage_index
            )
            right = self.runtime_structure.apply_inverse_sboxes(
                linear_state[:, :, 1], stage_index
            )
            current = torch.stack((left, right), dim=2)
            sbox_histogram = self._difference_histogram(current)
            sbox_hidden = self.histogram_encoder(sbox_histogram)
            descriptor = self.sbox_descriptor_encoder(
                self.sbox_truth_bits[stage_index].to(current.device)
            )
            sbox_encoded = self.sbox_expert(
                torch.cat(
                    (
                        sbox_hidden,
                        descriptor.unsqueeze(0).expand(current.shape[0], -1, -1),
                    ),
                    dim=-1,
                )
            )
            cells = self.cell_fusion(torch.cat((linear_encoded, sbox_encoded), dim=-1))
            attended, _attention = self.cell_attention(cells)
            stage = self.stage_projection(
                torch.cat(
                    (
                        attended,
                        cells.mean(dim=1),
                        cells.max(dim=1).values,
                    ),
                    dim=-1,
                )
            )
            recurrent = self.stage_recurrence(stage, recurrent)
        return self.output_projection(recurrent)

    def _difference_histogram(self, values: torch.Tensor) -> torch.Tensor:
        difference = torch.remainder(values[:, :, 0] + values[:, :, 1], 2.0)
        bits = difference[..., self.semantic_cell_bits.to(values.device)]
        weights = torch.tensor((8, 4, 2, 1), device=values.device, dtype=torch.long)
        cell_values = torch.sum(bits.to(torch.long) * weights, dim=-1)
        return F.one_hot(cell_values, num_classes=16).to(values.dtype).mean(dim=1)

    def _edge_embedding(self, stage_index: int, device: torch.device) -> torch.Tensor:
        encoded = self.edge_descriptor_encoder(self.edge_tokens[stage_index].to(device))
        mask = self.edge_masks[stage_index].to(device).unsqueeze(-1)
        return (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class FixedOrderedPrimitiveConditionedSpnProtocolAdapter(nn.Module):
    """Bind one compiled runtime program to a shared raw-pair distinguisher."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        program: CompiledSpnProgram,
        spec: OrderedPrimitiveConditionerSpec,
        descriptor_name: str,
        descriptor_path: str,
        descriptor_sha256: str,
        descriptor_round_start: int,
        descriptor_available_rounds: int,
        conditioner_enabled: bool,
    ) -> None:
        super().__init__()
        if pair_bits != 2 * program.block_bits:
            raise ValueError("K1-BY1 pair_bits must encode two runtime blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError("K1-BY1 input_bits must contain complete pairs")
        self.backbone = WidthIndependentRawPairBackbone(spec)
        self.conditioner = OrderedPrimitiveConditioner(program, spec)
        self.conditioner_enabled = bool(conditioner_enabled)
        self.primitive_gate = nn.Parameter(
            torch.tensor(math.atanh(spec.initial_effective_gate), dtype=torch.float32)
        )
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_descriptor_name = descriptor_name
        self.runtime_structure_descriptor_path = descriptor_path
        self.runtime_structure_descriptor_sha256 = descriptor_sha256
        self.runtime_structure_round_start = descriptor_round_start
        self.runtime_structure_available_rounds = descriptor_available_rounds
        self.runtime_structure_loaded_rounds = program.rounds
        self.runtime_round_window_mode = "ordered_learnable_primitive_conditioner"
        self.runtime_structure_window_control = program.control
        self.runtime_structure_mode = program.control
        self.compiled_program_semantic_sha256 = program.semantic_sha256
        self.compiled_program_expert_usage = program.expert_usage
        self.primitive_expert_set = ("sbox4_table", PERMUTATION_EXPERT, GF2_EXPERT)
        self.uses_cipher_identity = False
        self.uses_absolute_cell_or_bit_identity = False
        self.shared_experts_across_cells_and_stages = True
        self.state_width_independent_parameter_shapes = True
        self.ordered_stage_recurrence = True
        self.primitive_conditioner_enabled = self.conditioner_enabled
        self.primitive_gate_bounded = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        runtime = features.reshape(
            features.shape[0],
            -1,
            2,
            self.conditioner.program.block_bits,
        ).flip(-1)
        base = self.backbone.encode(runtime)
        primitive = self.conditioner(runtime).repeat(1, 3)
        scale = torch.tanh(self.primitive_gate)
        if not self.conditioner_enabled:
            scale = scale * 0.0
        return self.backbone.classifier(base + scale * torch.tanh(primitive))


def _primitive_expert(hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden * 2, hidden * 2),
        nn.ReLU(),
        nn.Linear(hidden * 2, hidden),
        nn.LayerNorm(hidden),
    )


def _padded_edge_descriptors(
    program: CompiledSpnProgram,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_edges = max(
        len(cell.edges) for stage in program.stages for cell in stage.linear_cells
    )
    tokens = torch.zeros(program.rounds, program.cells, max_edges, 10)
    masks = torch.zeros(program.rounds, program.cells, max_edges)
    expert_types = torch.empty(program.rounds, program.cells, dtype=torch.long)
    for stage_index, stage in enumerate(program.stages):
        for cell in stage.linear_cells:
            expert_types[stage_index, cell.target_cell] = (
                0 if cell.expert == PERMUTATION_EXPERT else 1
            )
            for edge_index, (target_role, _source_cell, source_role) in enumerate(
                cell.edges
            ):
                tokens[stage_index, cell.target_cell, edge_index, target_role] = 1.0
                tokens[
                    stage_index,
                    cell.target_cell,
                    edge_index,
                    4 + source_role,
                ] = 1.0
                tokens[
                    stage_index,
                    cell.target_cell,
                    edge_index,
                    8 + int(cell.expert == GF2_EXPERT),
                ] = 1.0
                masks[stage_index, cell.target_cell, edge_index] = 1.0
    return tokens, masks, expert_types


__all__ = [
    "FixedOrderedPrimitiveConditionedSpnProtocolAdapter",
    "OrderedPrimitiveConditioner",
    "OrderedPrimitiveConditionerSpec",
    "WidthIndependentRawPairBackbone",
]
