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


LINEAR_HISTOGRAM_LOCAL = "local"
LINEAR_HISTOGRAM_SOURCE_BUNDLE_MEAN = "source_bundle_mean"
LINEAR_HISTOGRAM_EDGE_CONTEXT_COVARIANCE = "edge_context_covariance"
LINEAR_HISTOGRAM_MODES = {
    LINEAR_HISTOGRAM_LOCAL,
    LINEAR_HISTOGRAM_SOURCE_BUNDLE_MEAN,
    LINEAR_HISTOGRAM_EDGE_CONTEXT_COVARIANCE,
}
POST_EXPERT_RESIDUAL_NONE = "none"
POST_EXPERT_RESIDUAL_EDGE_GATED_LAPLACIAN = "edge_gated_laplacian"
POST_EXPERT_RESIDUAL_MODES = {
    POST_EXPERT_RESIDUAL_NONE,
    POST_EXPERT_RESIDUAL_EDGE_GATED_LAPLACIAN,
}


@dataclass(frozen=True)
class OrderedPrimitiveConditionerSpec:
    hidden_dim: int = 32
    pair_embedding_dim: int = 128
    dropout: float = 0.0
    initial_effective_gate: float = 0.05
    linear_histogram_mode: str = LINEAR_HISTOGRAM_LOCAL
    post_expert_residual_mode: str = POST_EXPERT_RESIDUAL_NONE

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.pair_embedding_dim) <= 0:
            raise ValueError("ordered primitive dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("ordered primitive dropout must be in [0, 1)")
        if not -1.0 < self.initial_effective_gate < 1.0:
            raise ValueError("ordered primitive gate must be in (-1, 1)")
        if self.linear_histogram_mode not in LINEAR_HISTOGRAM_MODES:
            raise ValueError(
                "ordered primitive linear histogram mode must be local, "
                "source_bundle_mean, or edge_context_covariance"
            )
        if self.post_expert_residual_mode not in POST_EXPERT_RESIDUAL_MODES:
            raise ValueError(
                "ordered primitive post-expert residual mode must be none or "
                "edge_gated_laplacian"
            )


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


class PostExpertStructuralResidual(nn.Module):
    """Apply one zero-parameter runtime-edge residual after expert selection."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in POST_EXPERT_RESIDUAL_MODES:
            raise ValueError("unknown post-expert structural residual mode")
        self.mode = mode

    def forward(
        self,
        expert_output: torch.Tensor,
        edge_role_embedding: torch.Tensor,
        edge_source_cells: torch.Tensor | None,
        edge_masks: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == POST_EXPERT_RESIDUAL_NONE:
            return expert_output
        if edge_source_cells is None:
            raise ValueError("post-expert edge residual requires source-cell bindings")
        return post_expert_edge_gated_laplacian(
            expert_output,
            edge_role_embedding,
            edge_source_cells,
            edge_masks,
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
        if spec.linear_histogram_mode == LINEAR_HISTOGRAM_SOURCE_BUNDLE_MEAN:
            self.register_buffer(
                "linear_source_bundle_equivalence",
                source_bundle_equivalence_matrices(program),
            )
        elif spec.linear_histogram_mode == LINEAR_HISTOGRAM_EDGE_CONTEXT_COVARIANCE:
            edge_source_cells, edge_source_roles = _padded_edge_context(program)
            self.register_buffer("linear_edge_source_cells", edge_source_cells)
            self.register_buffer("linear_edge_source_roles", edge_source_roles)
        if (
            spec.post_expert_residual_mode
            == POST_EXPERT_RESIDUAL_EDGE_GATED_LAPLACIAN
        ):
            edge_source_cells, _edge_source_roles = _padded_edge_context(program)
            self.register_buffer(
                "post_expert_edge_source_cells",
                edge_source_cells,
            )

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
        self.post_expert_structural_residual = PostExpertStructuralResidual(
            spec.post_expert_residual_mode
        )
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
            linear_histogram = self._difference_histogram(
                linear_state,
                stage_index=stage_index,
                source_values=current,
            )
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
            post_expert_source_cells = getattr(
                self,
                "post_expert_edge_source_cells",
                None,
            )
            if post_expert_source_cells is not None:
                post_expert_source_cells = post_expert_source_cells[stage_index].to(
                    current.device
                )
            linear_encoded = self.post_expert_structural_residual(
                linear_encoded,
                edge_hidden,
                post_expert_source_cells,
                self.edge_masks[stage_index].to(current.device),
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

    def _difference_histogram(
        self,
        values: torch.Tensor,
        *,
        stage_index: int | None = None,
        source_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        difference = torch.remainder(values[:, :, 0] + values[:, :, 1], 2.0)
        bits = difference[..., self.semantic_cell_bits.to(values.device)]
        weights = torch.tensor((8, 4, 2, 1), device=values.device, dtype=torch.long)
        cell_values = torch.sum(bits.to(torch.long) * weights, dim=-1)
        histogram = F.one_hot(cell_values, num_classes=16).to(values.dtype).mean(dim=1)
        if (
            stage_index is None
            or self.spec.linear_histogram_mode == LINEAR_HISTOGRAM_LOCAL
        ):
            return histogram
        if (
            self.spec.linear_histogram_mode
            == LINEAR_HISTOGRAM_EDGE_CONTEXT_COVARIANCE
        ):
            if source_values is None:
                raise ValueError(
                    "edge-context covariance requires the pre-linear source state"
                )
            if (
                source_values.ndim != 4
                or source_values.shape[:2] != values.shape[:2]
                or source_values.shape[2] != 2
                or source_values.shape[3] != self.program.block_bits
            ):
                raise ValueError("edge-context source-state geometry drifted")
            source_difference = torch.remainder(
                source_values[:, :, 0] + source_values[:, :, 1],
                2.0,
            )
            source_cell_bits = source_difference[
                ..., self.semantic_cell_bits.to(values.device)
            ]
            return edge_context_covariance_histogram(
                cell_values,
                source_cell_bits,
                self.linear_edge_source_cells[stage_index].to(values.device),
                self.linear_edge_source_roles[stage_index].to(values.device),
                self.edge_masks[stage_index].to(values.device),
            )
        bundle_mean = torch.einsum(
            "ij,bjk->bik",
            self.linear_source_bundle_equivalence[stage_index].to(values.device),
            histogram,
        )
        return 0.5 * histogram + 0.5 * bundle_mean

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
        self.linear_histogram_mode = spec.linear_histogram_mode
        self.post_expert_residual_mode = spec.post_expert_residual_mode

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


def _padded_edge_context(
    program: CompiledSpnProgram,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_edges = max(
        len(cell.edges) for stage in program.stages for cell in stage.linear_cells
    )
    source_cells = torch.zeros(
        program.rounds,
        program.cells,
        max_edges,
        dtype=torch.long,
    )
    source_roles = torch.zeros_like(source_cells)
    for stage_index, stage in enumerate(program.stages):
        for cell in stage.linear_cells:
            for edge_index, (_target_role, source_cell, source_role) in enumerate(
                cell.edges
            ):
                source_cells[stage_index, cell.target_cell, edge_index] = source_cell
                source_roles[stage_index, cell.target_cell, edge_index] = source_role
    return source_cells, source_roles


def edge_context_covariance_histogram(
    target_cell_values: torch.Tensor,
    source_cell_bits: torch.Tensor,
    edge_source_cells: torch.Tensor,
    edge_source_roles: torch.Tensor,
    edge_masks: torch.Tensor,
) -> torch.Tensor:
    """Add a zero-mass per-cell residual from non-transported source-bit parity."""

    if target_cell_values.ndim != 3:
        raise ValueError("target cell values must have shape [batch, pairs, cells]")
    if source_cell_bits.shape != (*target_cell_values.shape, 4):
        raise ValueError("source cell bits must have shape [batch, pairs, cells, 4]")
    cells = target_cell_values.shape[-1]
    if (
        edge_source_cells.ndim != 2
        or edge_source_roles.shape != edge_source_cells.shape
        or edge_masks.shape != edge_source_cells.shape
        or edge_source_cells.shape[0] != cells
    ):
        raise ValueError("edge-context buffers must have shape [cells, edges]")
    if edge_source_cells.numel() == 0 or not torch.all(
        (edge_source_cells >= 0) & (edge_source_cells < cells)
    ):
        raise ValueError("edge-context source cell is out of range")
    if not torch.all((edge_source_roles >= 0) & (edge_source_roles < 4)):
        raise ValueError("edge-context source role is out of range")
    if not torch.all((edge_masks == 0) | (edge_masks == 1)) or not torch.all(
        edge_masks.sum(dim=-1) > 0
    ):
        raise ValueError("edge-context masks must select at least one edge per cell")

    selected_cells = source_cell_bits[:, :, edge_source_cells, :]
    selected_roles = edge_source_roles.reshape(1, 1, cells, -1, 1).expand(
        target_cell_values.shape[0],
        target_cell_values.shape[1],
        -1,
        -1,
        1,
    )
    transported = torch.gather(selected_cells, dim=-1, index=selected_roles).squeeze(-1)
    other_parity = torch.remainder(selected_cells.sum(dim=-1) - transported, 2.0)
    edge_sign = 1.0 - 2.0 * other_parity
    mask = edge_masks.to(dtype=source_cell_bits.dtype).reshape(1, 1, cells, -1)
    context = (edge_sign * mask).sum(dim=-1) / mask.sum(dim=-1)

    indicator = F.one_hot(target_cell_values.to(torch.long), num_classes=16).to(
        source_cell_bits.dtype
    )
    histogram = indicator.mean(dim=1)
    joint = (indicator * context.unsqueeze(-1)).mean(dim=1)
    residual = joint - histogram * context.mean(dim=1, keepdim=False).unsqueeze(-1)
    return histogram + residual


def post_expert_edge_gated_laplacian(
    expert_output: torch.Tensor,
    edge_role_embedding: torch.Tensor,
    edge_source_cells: torch.Tensor,
    edge_masks: torch.Tensor,
) -> torch.Tensor:
    """Add a bounded edge-source contrast gated by the frozen role embedding."""

    if expert_output.ndim != 3:
        raise ValueError("expert output must have shape [batch, cells, hidden]")
    batch, cells, hidden = expert_output.shape
    if edge_role_embedding.shape != (cells, hidden):
        raise ValueError("edge-role embedding must have shape [cells, hidden]")
    if (
        edge_source_cells.ndim != 2
        or edge_masks.shape != edge_source_cells.shape
        or edge_source_cells.shape[0] != cells
    ):
        raise ValueError("post-expert edge buffers must have shape [cells, edges]")
    if edge_source_cells.numel() == 0 or not torch.all(
        (edge_source_cells >= 0) & (edge_source_cells < cells)
    ):
        raise ValueError("post-expert source cell is out of range")
    if not torch.all((edge_masks == 0) | (edge_masks == 1)) or not torch.all(
        edge_masks.sum(dim=-1) > 0
    ):
        raise ValueError("post-expert masks must select at least one edge per cell")

    gathered = expert_output[:, edge_source_cells, :]
    mask = edge_masks.to(dtype=expert_output.dtype).reshape(1, cells, -1, 1)
    source_mean = (gathered * mask).sum(dim=2) / mask.sum(dim=2)
    residual = torch.tanh(source_mean - expert_output) * torch.tanh(
        edge_role_embedding
    ).reshape(1, cells, hidden)
    if residual.shape != (batch, cells, hidden):
        raise ValueError("post-expert residual geometry drifted")
    return expert_output + residual


def source_bundle_equivalence_matrices(
    program: CompiledSpnProgram,
) -> torch.Tensor:
    """Average cells whose incoming edges use the same source-cell set."""

    matrices = torch.zeros(
        program.rounds,
        program.cells,
        program.cells,
        dtype=torch.float32,
    )
    for stage_index, stage in enumerate(program.stages):
        signatures = tuple(
            frozenset(
                source_cell for _target_role, source_cell, _source_role in cell.edges
            )
            for cell in stage.linear_cells
        )
        for target_cell, signature in enumerate(signatures):
            peers = [
                peer_cell
                for peer_cell, peer_signature in enumerate(signatures)
                if peer_signature == signature
            ]
            matrices[stage_index, target_cell, peers] = 1.0 / len(peers)
    return matrices


__all__ = [
    "FixedOrderedPrimitiveConditionedSpnProtocolAdapter",
    "LINEAR_HISTOGRAM_EDGE_CONTEXT_COVARIANCE",
    "LINEAR_HISTOGRAM_LOCAL",
    "LINEAR_HISTOGRAM_SOURCE_BUNDLE_MEAN",
    "OrderedPrimitiveConditioner",
    "OrderedPrimitiveConditionerSpec",
    "POST_EXPERT_RESIDUAL_EDGE_GATED_LAPLACIAN",
    "POST_EXPERT_RESIDUAL_NONE",
    "PostExpertStructuralResidual",
    "WidthIndependentRawPairBackbone",
    "edge_context_covariance_histogram",
    "post_expert_edge_gated_laplacian",
    "source_bundle_equivalence_matrices",
]
