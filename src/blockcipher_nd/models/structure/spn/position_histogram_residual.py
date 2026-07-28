from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from blockcipher_nd.models.structure.spn.exact_operator_composition import (
    COMPOSITION_STAGE_NAMES,
    ExactOperatorCompositionSpnDistinguisher,
    composition_fingerprint,
    exact_operator_composition_views,
)
from blockcipher_nd.models.structure.spn.operator_tied_latent import (
    operator_routing_fingerprint,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.models.structure.spn.topology_edge_residual import (
    TopologyEdgeResidualSpnSpec,
    ordered_cell_role_lookup,
    topology_edge_fingerprint,
)


@dataclass(frozen=True)
class PositionHistogramResidualSpnSpec:
    hidden_dim: int = 32
    pair_embedding_dim: int = 128
    histogram_value_dim: int = 8
    dropout: float = 0.0
    initial_edge_gate: float = 0.05
    initial_histogram_gate: float = 0.05

    def __post_init__(self) -> None:
        if (
            min(
                self.hidden_dim,
                self.pair_embedding_dim,
                self.histogram_value_dim,
            )
            <= 0
        ):
            raise ValueError("position-histogram dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("position-histogram dropout must be in [0, 1)")
        if not -1.0 < self.initial_edge_gate < 1.0:
            raise ValueError("position-histogram edge gate must be in (-1, 1)")
        if not -1.0 < self.initial_histogram_gate < 1.0:
            raise ValueError("position-histogram gate must be in (-1, 1)")


class VirtualSlotSummedLinear(nn.Module):
    """Linear projection with fixed optimizer slots and a summed forward weight."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        virtual_slots: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(in_features, out_features, virtual_slots) <= 0:
            raise ValueError("virtual-slot linear dimensions must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.virtual_slots = int(virtual_slots)
        self.virtual_slot_weights = nn.Parameter(
            torch.empty(
                self.virtual_slots,
                self.out_features,
                self.in_features,
            )
        )
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.virtual_slots * self.in_features)
        nn.init.uniform_(self.virtual_slot_weights, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def effective_weight(self) -> torch.Tensor:
        return self.virtual_slot_weights.sum(dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.effective_weight(), self.bias)


class PositionHistogramResidualSpnDistinguisher(
    ExactOperatorCompositionSpnDistinguisher
):
    """K1-N plus a bounded exact stage/cell histogram residual."""

    def __init__(self, spec: PositionHistogramResidualSpnSpec) -> None:
        super().__init__(
            TopologyEdgeResidualSpnSpec(
                hidden_dim=spec.hidden_dim,
                pair_embedding_dim=spec.pair_embedding_dim,
                dropout=spec.dropout,
                initial_effective_gate=spec.initial_edge_gate,
            )
        )
        self.histogram_spec = spec
        self.histogram_value_encoder = nn.Sequential(
            nn.Linear(16, spec.histogram_value_dim),
            nn.ReLU(),
        )
        histogram_slots = len(COMPOSITION_STAGE_NAMES) * 16
        self.histogram_projection = nn.Sequential(
            nn.Linear(
                histogram_slots * spec.histogram_value_dim,
                spec.pair_embedding_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(spec.pair_embedding_dim),
        )
        self.histogram_gate = nn.Parameter(
            torch.tensor(
                math.atanh(spec.initial_histogram_gate),
                dtype=torch.float32,
            )
        )

    def histogram_embedding(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        apply_sboxes: bool = True,
        invariant_cells: bool = False,
    ) -> torch.Tensor:
        histogram = deterministic_position_histogram(
            ciphertext_pairs,
            structure,
            apply_sboxes=apply_sboxes,
            invariant_cells=invariant_cells,
        )
        encoded = self.histogram_value_encoder(histogram)
        return self.histogram_projection(encoded.flatten(1))


class CompactInvariantHistogramResidualSpnDistinguisher(
    ExactOperatorCompositionSpnDistinguisher
):
    """K1-N plus a cell-count-invariant exact histogram residual."""

    def __init__(
        self,
        spec: PositionHistogramResidualSpnSpec,
        *,
        virtual_projection_slots: int | None = None,
    ) -> None:
        super().__init__(
            TopologyEdgeResidualSpnSpec(
                hidden_dim=spec.hidden_dim,
                pair_embedding_dim=spec.pair_embedding_dim,
                dropout=spec.dropout,
                initial_effective_gate=spec.initial_edge_gate,
            )
        )
        self.histogram_spec = spec
        self.histogram_value_encoder = nn.Sequential(
            nn.Linear(16, spec.histogram_value_dim),
            nn.ReLU(),
        )
        projection_input_dim = len(COMPOSITION_STAGE_NAMES) * spec.histogram_value_dim
        projection = (
            nn.Linear(projection_input_dim, spec.pair_embedding_dim)
            if virtual_projection_slots is None
            else VirtualSlotSummedLinear(
                projection_input_dim,
                spec.pair_embedding_dim,
                virtual_projection_slots,
            )
        )
        self.histogram_projection = nn.Sequential(
            projection,
            nn.ReLU(),
            nn.LayerNorm(spec.pair_embedding_dim),
        )
        self.histogram_gate = nn.Parameter(
            torch.tensor(
                math.atanh(spec.initial_histogram_gate),
                dtype=torch.float32,
            )
        )

    def histogram_embedding(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        apply_sboxes: bool = True,
    ) -> torch.Tensor:
        histogram = deterministic_position_histogram(
            ciphertext_pairs,
            structure,
            apply_sboxes=apply_sboxes,
        ).mean(dim=2)
        encoded = self.histogram_value_encoder(histogram)
        return self.histogram_projection(encoded.flatten(1))


@dataclass(frozen=True)
class SboxTransitionResidualSpnSpec:
    hidden_dim: int = 32
    pair_embedding_dim: int = 128
    transition_value_dim: int = 20
    dropout: float = 0.0
    initial_edge_gate: float = 0.05
    initial_transition_gate: float = 0.05
    virtual_projection_slots: int = 16

    def __post_init__(self) -> None:
        if (
            min(
                self.hidden_dim,
                self.pair_embedding_dim,
                self.transition_value_dim,
                self.virtual_projection_slots,
            )
            <= 0
        ):
            raise ValueError("S-box transition dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("S-box transition dropout must be in [0, 1)")
        if not -1.0 < self.initial_edge_gate < 1.0:
            raise ValueError("S-box transition edge gate must be in (-1, 1)")
        if not -1.0 < self.initial_transition_gate < 1.0:
            raise ValueError("S-box transition gate must be in (-1, 1)")


class CompactSboxTransitionResidualSpnDistinguisher(
    ExactOperatorCompositionSpnDistinguisher
):
    """K1-AA trunk with a shared per-cell nonlinear-transition readout."""

    def __init__(self, spec: SboxTransitionResidualSpnSpec) -> None:
        super().__init__(
            TopologyEdgeResidualSpnSpec(
                hidden_dim=spec.hidden_dim,
                pair_embedding_dim=spec.pair_embedding_dim,
                dropout=spec.dropout,
                initial_effective_gate=spec.initial_edge_gate,
            )
        )
        self.transition_spec = spec
        self.sbox_transition_encoder = nn.Sequential(
            nn.Linear(16 * 16, spec.transition_value_dim),
            nn.ReLU(),
        )
        self.transition_projection = nn.Sequential(
            VirtualSlotSummedLinear(
                2 * spec.transition_value_dim,
                spec.pair_embedding_dim,
                spec.virtual_projection_slots,
            ),
            nn.ReLU(),
            nn.LayerNorm(spec.pair_embedding_dim),
        )
        self.transition_gate = nn.Parameter(
            torch.tensor(
                math.atanh(spec.initial_transition_gate),
                dtype=torch.float32,
            )
        )

    def transition_embedding(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        apply_sboxes: bool = True,
    ) -> torch.Tensor:
        histogram = deterministic_sbox_transition_histogram(
            ciphertext_pairs,
            structure,
            apply_sboxes=apply_sboxes,
        )
        encoded_per_cell = self.sbox_transition_encoder(histogram)
        invariant = encoded_per_cell.mean(dim=2)
        return self.transition_projection(invariant.flatten(1))


class FixedPositionHistogramResidualSpnProtocolAdapter(nn.Module):
    """Bind K1-T to one external two-transition runtime descriptor."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        structure: RuntimeSpnStructure,
        spec: PositionHistogramResidualSpnSpec,
        descriptor_name: str,
        descriptor_path: str,
        descriptor_sha256: str,
        descriptor_round_start: int,
        descriptor_available_rounds: int,
        runtime_structure_mode: str,
        apply_sboxes: bool,
        invariant_cells: bool,
    ) -> None:
        super().__init__()
        if pair_bits != 2 * structure.block_bits:
            raise ValueError("K1-T pair_bits must encode two runtime blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError("K1-T input_bits must contain complete pairs")
        if structure.rounds != 2 or structure.cells != 16:
            raise ValueError("K1-T requires two transitions and sixteen cells")
        self.backbone = PositionHistogramResidualSpnDistinguisher(spec)
        self.runtime_structure = structure
        self.apply_sboxes = bool(apply_sboxes)
        self.invariant_histogram_cells = bool(invariant_cells)
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_loaded_rounds = structure.rounds
        self.runtime_round_window_mode = "deterministic_position_histogram_residual"
        self.runtime_structure_window_control = runtime_structure_mode
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
        self.topology_edge_sha256 = topology_edge_fingerprint(structure)
        self.composition_sha256 = composition_fingerprint(
            structure,
            apply_sboxes=self.apply_sboxes,
        )
        self.histogram_semantics_sha256 = histogram_semantics_fingerprint(
            structure,
            apply_sboxes=self.apply_sboxes,
            invariant_cells=self.invariant_histogram_cells,
        )
        self.composition_stage_names = COMPOSITION_STAGE_NAMES
        self.histogram_shape = (len(COMPOSITION_STAGE_NAMES), structure.cells, 16)
        self.histogram_value_dim = spec.histogram_value_dim
        self.deterministic_exact_composition = True
        self.deterministic_position_histogram = True
        self.uses_absolute_cell_or_bit_identity = False
        self.uses_runtime_native_cell_slots = not self.invariant_histogram_cells
        self.uses_sbox_semantics = self.apply_sboxes
        self.uses_cipher_identity = False
        self.histogram_gate_bounded = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        runtime = features.reshape(
            features.shape[0],
            -1,
            2,
            self.runtime_structure.block_bits,
        ).flip(-1)
        base_embedding = self.backbone.base.encode(runtime, self.runtime_structure)
        edge_residual = self.backbone.edge_residual_embedding(
            runtime,
            self.runtime_structure,
            apply_sboxes=self.apply_sboxes,
        )
        combined = base_embedding + torch.tanh(
            self.backbone.residual_gate
        ) * torch.tanh(edge_residual)
        histogram = self.backbone.histogram_embedding(
            runtime,
            self.runtime_structure,
            apply_sboxes=self.apply_sboxes,
            invariant_cells=self.invariant_histogram_cells,
        )
        histogram_residual = histogram.repeat(1, 3)
        combined = combined + torch.tanh(self.backbone.histogram_gate) * torch.tanh(
            histogram_residual
        )
        return self.backbone.base.classifier(combined)


class FixedCompactInvariantHistogramResidualSpnProtocolAdapter(nn.Module):
    """Bind K1-W to a variable-cell external two-transition descriptor."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        structure: RuntimeSpnStructure,
        spec: PositionHistogramResidualSpnSpec,
        descriptor_name: str,
        descriptor_path: str,
        descriptor_sha256: str,
        descriptor_round_start: int,
        descriptor_available_rounds: int,
        runtime_structure_mode: str,
        apply_sboxes: bool,
        virtual_projection_slots: int | None = None,
    ) -> None:
        super().__init__()
        if pair_bits != 2 * structure.block_bits:
            raise ValueError("K1-W pair_bits must encode two runtime blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError("K1-W input_bits must contain complete pairs")
        if structure.rounds != 2:
            raise ValueError("K1-W requires exactly two transitions")
        self.backbone = CompactInvariantHistogramResidualSpnDistinguisher(
            spec,
            virtual_projection_slots=virtual_projection_slots,
        )
        self.runtime_structure = structure
        self.apply_sboxes = bool(apply_sboxes)
        self.invariant_histogram_cells = True
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_loaded_rounds = structure.rounds
        self.runtime_round_window_mode = (
            "deterministic_compact_invariant_histogram_residual"
        )
        self.runtime_structure_window_control = runtime_structure_mode
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
        self.topology_edge_sha256 = topology_edge_fingerprint(structure)
        self.composition_sha256 = composition_fingerprint(
            structure,
            apply_sboxes=self.apply_sboxes,
        )
        self.histogram_semantics_sha256 = histogram_semantics_fingerprint(
            structure,
            apply_sboxes=self.apply_sboxes,
            invariant_cells=True,
        )
        self.composition_stage_names = COMPOSITION_STAGE_NAMES
        self.histogram_shape = (len(COMPOSITION_STAGE_NAMES), structure.cells, 16)
        self.histogram_value_dim = spec.histogram_value_dim
        self.deterministic_exact_composition = True
        self.deterministic_position_histogram = True
        self.compact_invariant_histogram = True
        self.uses_absolute_cell_or_bit_identity = False
        self.uses_runtime_native_cell_slots = False
        self.uses_sbox_semantics = self.apply_sboxes
        self.uses_cipher_identity = False
        self.histogram_gate_bounded = True
        if virtual_projection_slots is not None:
            self.virtual_projection_slots = int(virtual_projection_slots)
            self.virtual_projection_parameter = (
                "backbone.histogram_projection.0.virtual_slot_weights"
            )
            self.virtual_projection_effective_weight_shape = (
                spec.pair_embedding_dim,
                len(COMPOSITION_STAGE_NAMES) * spec.histogram_value_dim,
            )
            self.runtime_round_window_mode = (
                "deterministic_virtual_slot_compact_invariant_histogram_residual"
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        runtime = features.reshape(
            features.shape[0],
            -1,
            2,
            self.runtime_structure.block_bits,
        ).flip(-1)
        base_embedding = self.backbone.base.encode(runtime, self.runtime_structure)
        edge_residual = self.backbone.edge_residual_embedding(
            runtime,
            self.runtime_structure,
            apply_sboxes=self.apply_sboxes,
        )
        combined = base_embedding + torch.tanh(
            self.backbone.residual_gate
        ) * torch.tanh(edge_residual)
        histogram = self.backbone.histogram_embedding(
            runtime,
            self.runtime_structure,
            apply_sboxes=self.apply_sboxes,
        )
        combined = combined + torch.tanh(self.backbone.histogram_gate) * torch.tanh(
            histogram.repeat(1, 3)
        )
        return self.backbone.base.classifier(combined)


class FixedCompactSboxTransitionResidualSpnProtocolAdapter(nn.Module):
    """Bind the K1-AK cell-invariant S-box-transition readout."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        structure: RuntimeSpnStructure,
        spec: SboxTransitionResidualSpnSpec,
        descriptor_name: str,
        descriptor_path: str,
        descriptor_sha256: str,
        descriptor_round_start: int,
        descriptor_available_rounds: int,
        runtime_structure_mode: str,
        apply_sboxes: bool,
    ) -> None:
        super().__init__()
        if pair_bits != 2 * structure.block_bits:
            raise ValueError("K1-AK pair_bits must encode two runtime blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError("K1-AK input_bits must contain complete pairs")
        if structure.rounds != 2:
            raise ValueError("K1-AK requires exactly two transitions")
        self.backbone = CompactSboxTransitionResidualSpnDistinguisher(spec)
        self.runtime_structure = structure
        self.apply_sboxes = bool(apply_sboxes)
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_loaded_rounds = structure.rounds
        self.runtime_round_window_mode = (
            "deterministic_compact_sbox_transition_residual"
        )
        self.runtime_structure_window_control = runtime_structure_mode
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
        self.topology_edge_sha256 = topology_edge_fingerprint(structure)
        self.composition_sha256 = composition_fingerprint(
            structure,
            apply_sboxes=self.apply_sboxes,
        )
        self.sbox_transition_semantics_sha256 = sbox_transition_semantics_fingerprint(
            structure,
            apply_sboxes=self.apply_sboxes,
        )
        self.composition_stage_names = COMPOSITION_STAGE_NAMES
        self.sbox_transition_stage_pairs = ((1, 2), (3, 4))
        self.sbox_transition_histogram_shape = (2, structure.cells, 16, 16)
        self.sbox_transition_value_dim = spec.transition_value_dim
        self.virtual_projection_slots = spec.virtual_projection_slots
        self.virtual_projection_parameter = (
            "backbone.transition_projection.0.virtual_slot_weights"
        )
        self.virtual_projection_effective_weight_shape = (
            spec.pair_embedding_dim,
            2 * spec.transition_value_dim,
        )
        self.deterministic_exact_composition = True
        self.deterministic_sbox_transition_histogram = True
        self.compact_invariant_sbox_transition = True
        self.uses_absolute_cell_or_bit_identity = False
        self.uses_runtime_native_cell_slots = False
        self.uses_sbox_semantics = self.apply_sboxes
        self.uses_sbox_transition_semantics = self.apply_sboxes
        self.uses_cipher_identity = False
        self.transition_gate_bounded = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        runtime = features.reshape(
            features.shape[0],
            -1,
            2,
            self.runtime_structure.block_bits,
        ).flip(-1)
        base_embedding = self.backbone.base.encode(runtime, self.runtime_structure)
        edge_residual = self.backbone.edge_residual_embedding(
            runtime,
            self.runtime_structure,
            apply_sboxes=self.apply_sboxes,
        )
        combined = base_embedding + torch.tanh(
            self.backbone.residual_gate
        ) * torch.tanh(edge_residual)
        transition = self.backbone.transition_embedding(
            runtime,
            self.runtime_structure,
            apply_sboxes=self.apply_sboxes,
        )
        combined = combined + torch.tanh(self.backbone.transition_gate) * torch.tanh(
            transition.repeat(1, 3)
        )
        return self.backbone.base.classifier(combined)


def deterministic_position_histogram(
    ciphertext_pairs: torch.Tensor,
    structure: RuntimeSpnStructure,
    *,
    apply_sboxes: bool = True,
    invariant_cells: bool = False,
) -> torch.Tensor:
    views = exact_operator_composition_views(
        ciphertext_pairs,
        structure,
        apply_sboxes=apply_sboxes,
    )
    batch, pairs, bits, channels = views.shape
    stages_count = len(COMPOSITION_STAGE_NAMES)
    if channels != stages_count * 3:
        raise ValueError("exact composition histogram geometry is invalid")
    stages = views.reshape(batch, pairs, bits, stages_count, 3)[..., 2].permute(
        0, 1, 3, 2
    )
    lookup = ordered_cell_role_lookup(structure).to(stages.device)
    cell_bits = stages[..., lookup].to(torch.long)
    weights = torch.tensor((8, 4, 2, 1), dtype=torch.long, device=stages.device)
    cell_values = torch.sum(cell_bits * weights, dim=-1)
    histogram = F.one_hot(cell_values, num_classes=16).to(stages.dtype).mean(dim=1)
    expected = (batch, stages_count, structure.cells, 16)
    if histogram.shape != expected:
        raise ValueError("K1-T position histogram geometry is invalid")
    if invariant_cells:
        histogram = histogram.mean(dim=2, keepdim=True).expand_as(histogram)
    return histogram


def deterministic_sbox_transition_histogram(
    ciphertext_pairs: torch.Tensor,
    structure: RuntimeSpnStructure,
    *,
    apply_sboxes: bool = True,
) -> torch.Tensor:
    views = exact_operator_composition_views(
        ciphertext_pairs,
        structure,
        apply_sboxes=apply_sboxes,
    )
    batch, pairs, bits, channels = views.shape
    stages_count = len(COMPOSITION_STAGE_NAMES)
    if channels != stages_count * 3:
        raise ValueError("S-box transition composition geometry is invalid")
    stages = views.reshape(batch, pairs, bits, stages_count, 3)[..., 2].permute(
        0, 1, 3, 2
    )
    lookup = ordered_cell_role_lookup(structure).to(stages.device)
    cell_bits = stages[..., lookup].to(torch.long)
    weights = torch.tensor((8, 4, 2, 1), dtype=torch.long, device=stages.device)
    cell_values = torch.sum(cell_bits * weights, dim=-1)
    before = cell_values[:, :, (1, 3)]
    after = cell_values[:, :, (2, 4)]
    transitions = before * 16 + after
    histogram = F.one_hot(transitions, num_classes=16 * 16).to(stages.dtype)
    histogram = histogram.mean(dim=1)
    expected = (batch, 2, structure.cells, 16 * 16)
    if histogram.shape != expected:
        raise ValueError("S-box transition histogram geometry is invalid")
    return histogram


def histogram_semantics_fingerprint(
    structure: RuntimeSpnStructure,
    *,
    apply_sboxes: bool,
    invariant_cells: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        composition_fingerprint(structure, apply_sboxes=apply_sboxes).encode("ascii")
    )
    digest.update(bytes((int(invariant_cells),)))
    digest.update(b"stage-cell-nibble-histogram-v1")
    return digest.hexdigest()


def sbox_transition_semantics_fingerprint(
    structure: RuntimeSpnStructure,
    *,
    apply_sboxes: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        composition_fingerprint(structure, apply_sboxes=apply_sboxes).encode("ascii")
    )
    digest.update(b"cell-shared-sbox-delta-transition-histogram-v1")
    return digest.hexdigest()


__all__ = [
    "CompactInvariantHistogramResidualSpnDistinguisher",
    "CompactSboxTransitionResidualSpnDistinguisher",
    "FixedCompactInvariantHistogramResidualSpnProtocolAdapter",
    "FixedCompactSboxTransitionResidualSpnProtocolAdapter",
    "FixedPositionHistogramResidualSpnProtocolAdapter",
    "PositionHistogramResidualSpnDistinguisher",
    "PositionHistogramResidualSpnSpec",
    "SboxTransitionResidualSpnSpec",
    "VirtualSlotSummedLinear",
    "deterministic_position_histogram",
    "deterministic_sbox_transition_histogram",
    "histogram_semantics_fingerprint",
    "sbox_transition_semantics_fingerprint",
]
