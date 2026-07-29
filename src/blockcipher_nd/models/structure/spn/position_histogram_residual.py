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
from blockcipher_nd.models.structure.spn.structure_conditioned_gate import (
    SharedStructureTransitionGate,
    runtime_structure_summary,
)
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


@dataclass(frozen=True)
class CanonicalWalshTransitionResidualSpnSpec:
    hidden_dim: int = 32
    pair_embedding_dim: int = 128
    walsh_features_per_stage: int = 64
    dropout: float = 0.0
    initial_edge_gate: float = 0.05
    initial_transition_gate: float = 0.05

    def __post_init__(self) -> None:
        if (
            min(
                self.hidden_dim,
                self.pair_embedding_dim,
                self.walsh_features_per_stage,
            )
            <= 0
        ):
            raise ValueError("canonical Walsh dimensions must be positive")
        if self.walsh_features_per_stage > 255:
            raise ValueError("canonical Walsh features must exclude the DC term")
        if self.pair_embedding_dim != 2 * self.walsh_features_per_stage:
            raise ValueError(
                "canonical Walsh embedding must concatenate exactly two stages"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("canonical Walsh dropout must be in [0, 1)")
        if not -1.0 < self.initial_edge_gate < 1.0:
            raise ValueError("canonical Walsh edge gate must be in (-1, 1)")
        if not -1.0 < self.initial_transition_gate < 1.0:
            raise ValueError("canonical Walsh transition gate must be in (-1, 1)")


class CanonicalWalshTransitionResidualSpnDistinguisher(
    ExactOperatorCompositionSpnDistinguisher
):
    """Parameter-free low-degree Walsh readout for two S-box transitions."""

    def __init__(self, spec: CanonicalWalshTransitionResidualSpnSpec) -> None:
        super().__init__(
            TopologyEdgeResidualSpnSpec(
                hidden_dim=spec.hidden_dim,
                pair_embedding_dim=spec.pair_embedding_dim,
                dropout=spec.dropout,
                initial_effective_gate=spec.initial_edge_gate,
            )
        )
        self.walsh_spec = spec
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
        features = deterministic_sbox_transition_walsh_features(
            ciphertext_pairs,
            structure,
            apply_sboxes=apply_sboxes,
            feature_count=self.walsh_spec.walsh_features_per_stage,
        )
        invariant = features.mean(dim=2).flatten(1)
        return F.layer_norm(invariant, (self.walsh_spec.pair_embedding_dim,))


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
        canonical_walsh_features: int | None = None,
        transition_branch_enabled: bool = True,
    ) -> None:
        super().__init__()
        if pair_bits != 2 * structure.block_bits:
            raise ValueError("K1-AK pair_bits must encode two runtime blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError("K1-AK input_bits must contain complete pairs")
        if structure.rounds != 2:
            raise ValueError("K1-AK requires exactly two transitions")
        self.canonical_walsh_transition = canonical_walsh_features is not None
        if canonical_walsh_features is None:
            self.backbone = CompactSboxTransitionResidualSpnDistinguisher(spec)
        else:
            self.backbone = CanonicalWalshTransitionResidualSpnDistinguisher(
                CanonicalWalshTransitionResidualSpnSpec(
                    hidden_dim=spec.hidden_dim,
                    pair_embedding_dim=spec.pair_embedding_dim,
                    walsh_features_per_stage=canonical_walsh_features,
                    dropout=spec.dropout,
                    initial_edge_gate=spec.initial_edge_gate,
                    initial_transition_gate=spec.initial_transition_gate,
                )
            )
        self.runtime_structure = structure
        self.apply_sboxes = bool(apply_sboxes)
        self.transition_branch_enabled = bool(transition_branch_enabled)
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
        if canonical_walsh_features is None:
            self.sbox_transition_value_dim = spec.transition_value_dim
            self.virtual_projection_slots = spec.virtual_projection_slots
            self.virtual_projection_parameter = (
                "backbone.transition_projection.0.virtual_slot_weights"
            )
            self.virtual_projection_effective_weight_shape = (
                spec.pair_embedding_dim,
                2 * spec.transition_value_dim,
            )
        else:
            self.sbox_transition_value_dim = canonical_walsh_features
            self.canonical_walsh_features_per_stage = canonical_walsh_features
            self.canonical_walsh_mask_pairs = canonical_walsh_mask_pairs(
                canonical_walsh_features
            )
            self.canonical_walsh_fingerprint = canonical_walsh_fingerprint(
                canonical_walsh_features
            )
            self.runtime_round_window_mode = (
                "deterministic_canonical_walsh_sbox_transition_residual"
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
        self.semantic_contrast_orientation: str | None = None
        self.semantic_contrast_scale = 0.0
        self.semantic_contrast_margin = 0.0
        self.semantic_counterfactual_structure: RuntimeSpnStructure | None = None
        self.semantic_counterfactual_apply_sboxes = True
        self.last_auxiliary_loss: torch.Tensor | None = None
        self.last_auxiliary_metrics: dict[str, torch.Tensor] = {}
        self._last_semantic_counterfactual_logits: torch.Tensor | None = None

    def configure_semantic_contrast(
        self,
        *,
        orientation: str,
        counterfactual_structure: RuntimeSpnStructure,
        counterfactual_apply_sboxes: bool,
        scale: float,
        margin: float,
    ) -> None:
        if orientation not in {"correct_vs_wrong", "wrong_vs_correct"}:
            raise ValueError("unsupported K1-AM semantic contrast orientation")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("semantic contrast scale must be finite and positive")
        if not math.isfinite(margin) or not 0.0 <= margin <= 1.0:
            raise ValueError("semantic contrast margin must be in [0, 1]")
        if (
            counterfactual_structure.block_bits != self.runtime_structure.block_bits
            or counterfactual_structure.rounds != self.runtime_structure.rounds
            or counterfactual_structure.cells != self.runtime_structure.cells
        ):
            raise ValueError("semantic counterfactual runtime geometry changed")
        counterfactual_fingerprint = sbox_transition_semantics_fingerprint(
            counterfactual_structure,
            apply_sboxes=counterfactual_apply_sboxes,
        )
        if counterfactual_fingerprint == self.sbox_transition_semantics_sha256:
            raise ValueError("semantic counterfactual must change S-box semantics")
        self.semantic_contrast_orientation = orientation
        self.semantic_contrast_scale = float(scale)
        self.semantic_contrast_margin = float(margin)
        self.semantic_counterfactual_structure = counterfactual_structure
        self.semantic_counterfactual_apply_sboxes = bool(
            counterfactual_apply_sboxes
        )

    def logits_with_runtime(
        self,
        features: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        apply_sboxes: bool,
        transition_branch_enabled: bool | None = None,
    ) -> torch.Tensor:
        use_transition_branch = (
            self.transition_branch_enabled
            if transition_branch_enabled is None
            else bool(transition_branch_enabled)
        )
        runtime = features.reshape(
            features.shape[0],
            -1,
            2,
            structure.block_bits,
        ).flip(-1)
        base_embedding = self.backbone.base.encode(runtime, structure)
        edge_residual = self.backbone.edge_residual_embedding(
            runtime,
            structure,
            apply_sboxes=apply_sboxes,
        )
        combined = base_embedding + torch.tanh(
            self.backbone.residual_gate
        ) * torch.tanh(edge_residual)
        if use_transition_branch:
            transition = self.backbone.transition_embedding(
                runtime,
                structure,
                apply_sboxes=apply_sboxes,
            )
            combined = combined + torch.tanh(
                self.backbone.transition_gate
            ) * torch.tanh(transition.repeat(1, 3))
        return self.backbone.base.classifier(combined)

    def compute_auxiliary_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_name: str,
    ) -> torch.Tensor | None:
        if self.semantic_contrast_orientation is None:
            return self.last_auxiliary_loss
        if self._last_semantic_counterfactual_logits is None:
            raise RuntimeError("semantic counterfactual logits are unavailable")
        primary_loss = self._per_sample_classification_loss(
            logits,
            labels,
            loss_name,
        )
        counterfactual_loss = self._per_sample_classification_loss(
            self._last_semantic_counterfactual_logits.squeeze(1),
            labels,
            loss_name,
        )
        margin_values = F.relu(
            self.semantic_contrast_margin + primary_loss - counterfactual_loss
        )
        auxiliary_loss = self.semantic_contrast_scale * margin_values.mean()
        self.last_auxiliary_loss = auxiliary_loss
        self.last_auxiliary_metrics = {
            "semantic_primary_loss": primary_loss.detach().mean(),
            "semantic_counterfactual_loss": counterfactual_loss.detach().mean(),
            "semantic_loss_gap": (
                counterfactual_loss.detach().mean() - primary_loss.detach().mean()
            ),
            "semantic_margin_loss": margin_values.detach().mean(),
            "semantic_violation_rate": (margin_values.detach() > 0.0).float().mean(),
        }
        return auxiliary_loss

    @staticmethod
    def _per_sample_classification_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_name: str,
    ) -> torch.Tensor:
        if loss_name == "mse":
            return F.mse_loss(torch.sigmoid(logits), labels, reduction="none")
        if loss_name == "bce":
            return F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
        raise ValueError(f"unsupported loss: {loss_name}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.logits_with_runtime(
            features,
            self.runtime_structure,
            apply_sboxes=self.apply_sboxes,
        )
        self.last_auxiliary_loss = None
        self.last_auxiliary_metrics = {}
        self._last_semantic_counterfactual_logits = None
        if self.training and self.semantic_contrast_orientation is not None:
            if self.semantic_counterfactual_structure is None:
                raise RuntimeError("semantic counterfactual runtime is unavailable")
            self._last_semantic_counterfactual_logits = self.logits_with_runtime(
                features,
                self.semantic_counterfactual_structure,
                apply_sboxes=self.semantic_counterfactual_apply_sboxes,
            )
        return logits


class FixedStructureConditionedSboxTransitionResidualSpnProtocolAdapter(
    FixedCompactSboxTransitionResidualSpnProtocolAdapter
):
    """K1-AS extension that leaves the frozen K1-AK adapter API unchanged."""

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
        structure_gate_hidden_dim: int = 12,
    ) -> None:
        super().__init__(
            input_bits=input_bits,
            pair_bits=pair_bits,
            structure=structure,
            spec=spec,
            descriptor_name=descriptor_name,
            descriptor_path=descriptor_path,
            descriptor_sha256=descriptor_sha256,
            descriptor_round_start=descriptor_round_start,
            descriptor_available_rounds=descriptor_available_rounds,
            runtime_structure_mode=runtime_structure_mode,
            apply_sboxes=apply_sboxes,
        )
        self.backbone.structure_gate = SharedStructureTransitionGate(
            hidden_dim=structure_gate_hidden_dim
        )
        self._structure_summary_cache: dict[
            int, tuple[RuntimeSpnStructure, torch.Tensor]
        ] = {}
        self.structure_gate_inputs = (
            "sbox_ddt_lat_distribution",
            "gf2_row_column_weight_rank_diversity",
        )
        self.structure_gate_shared = True
        self.structure_gate_uses_cipher_identity = False

    def logits_with_runtime(
        self,
        features: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        apply_sboxes: bool,
        transition_branch_enabled: bool | None = None,
        gate_structure: RuntimeSpnStructure | None = None,
        gate_summary: torch.Tensor | None = None,
        structure_gate_enabled: bool | None = None,
    ) -> torch.Tensor:
        use_transition_branch = (
            self.transition_branch_enabled
            if transition_branch_enabled is None
            else bool(transition_branch_enabled)
        )
        runtime = features.reshape(
            features.shape[0],
            -1,
            2,
            structure.block_bits,
        ).flip(-1)
        base_embedding = self.backbone.base.encode(runtime, structure)
        edge_residual = self.backbone.edge_residual_embedding(
            runtime,
            structure,
            apply_sboxes=apply_sboxes,
        )
        combined = base_embedding + torch.tanh(
            self.backbone.residual_gate
        ) * torch.tanh(edge_residual)
        if use_transition_branch:
            transition = self.backbone.transition_embedding(
                runtime,
                structure,
                apply_sboxes=apply_sboxes,
            )
            effective_gate = self.effective_transition_gate(
                structure if gate_structure is None else gate_structure,
                summary=gate_summary,
                enabled=structure_gate_enabled,
            )
            combined = combined + effective_gate * torch.tanh(transition.repeat(1, 3))
        return self.backbone.base.classifier(combined)

    def effective_transition_gate(
        self,
        structure: RuntimeSpnStructure,
        *,
        summary: torch.Tensor | None = None,
        enabled: bool | None = None,
    ) -> torch.Tensor:
        if enabled is False:
            return torch.tanh(self.backbone.transition_gate)
        descriptor = summary
        if descriptor is None:
            cache_key = id(structure)
            cached = self._structure_summary_cache.get(cache_key)
            if cached is None or cached[0] is not structure:
                cached = (structure, runtime_structure_summary(structure))
                self._structure_summary_cache[cache_key] = cached
            descriptor = cached[1]
        return self.backbone.structure_gate(
            self.backbone.transition_gate,
            descriptor,
            enabled=True,
        )


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


def canonical_walsh_mask_pairs(feature_count: int = 64) -> tuple[tuple[int, int], ...]:
    if not 1 <= feature_count <= 255:
        raise ValueError("canonical Walsh feature count must be in [1, 255]")
    pairs = [(left, right) for left in range(16) for right in range(16)]
    pairs.remove((0, 0))
    pairs.sort(
        key=lambda pair: (
            pair[0].bit_count() + pair[1].bit_count(),
            pair[0].bit_count(),
            pair[1].bit_count(),
            pair[0],
            pair[1],
        )
    )
    return tuple(pairs[:feature_count])


def deterministic_sbox_transition_walsh_features(
    ciphertext_pairs: torch.Tensor,
    structure: RuntimeSpnStructure,
    *,
    apply_sboxes: bool = True,
    feature_count: int = 64,
) -> torch.Tensor:
    histogram = deterministic_sbox_transition_histogram(
        ciphertext_pairs,
        structure,
        apply_sboxes=apply_sboxes,
    ).reshape(
        ciphertext_pairs.shape[0],
        2,
        structure.cells,
        16,
        16,
    )
    bit_counts = torch.tensor(
        [value.bit_count() for value in range(16)],
        dtype=torch.long,
        device=histogram.device,
    )
    masks = torch.arange(16, dtype=torch.long, device=histogram.device)
    values = torch.arange(16, dtype=torch.long, device=histogram.device)
    parity = bit_counts[torch.bitwise_and(masks[:, None], values[None, :])] % 2
    walsh = (1 - 2 * parity).to(histogram.dtype)
    spectrum = torch.einsum(
        "ux,nscxy,vy->nscuv",
        walsh,
        histogram,
        walsh,
    ).flatten(-2)
    mask_indices = torch.tensor(
        [
            left * 16 + right
            for left, right in canonical_walsh_mask_pairs(feature_count)
        ],
        dtype=torch.long,
        device=histogram.device,
    )
    features = spectrum.index_select(-1, mask_indices)
    expected = (ciphertext_pairs.shape[0], 2, structure.cells, feature_count)
    if features.shape != expected:
        raise ValueError("canonical Walsh transition geometry is invalid")
    return features


def canonical_walsh_fingerprint(feature_count: int = 64) -> str:
    digest = hashlib.sha256()
    digest.update(b"canonical-low-degree-sbox-transition-walsh-v1")
    digest.update(feature_count.to_bytes(2, "big"))
    for left, right in canonical_walsh_mask_pairs(feature_count):
        digest.update(bytes((left, right)))
    return digest.hexdigest()


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
    "CanonicalWalshTransitionResidualSpnDistinguisher",
    "CanonicalWalshTransitionResidualSpnSpec",
    "CompactInvariantHistogramResidualSpnDistinguisher",
    "CompactSboxTransitionResidualSpnDistinguisher",
    "FixedCompactInvariantHistogramResidualSpnProtocolAdapter",
    "FixedCompactSboxTransitionResidualSpnProtocolAdapter",
    "FixedStructureConditionedSboxTransitionResidualSpnProtocolAdapter",
    "FixedPositionHistogramResidualSpnProtocolAdapter",
    "PositionHistogramResidualSpnDistinguisher",
    "PositionHistogramResidualSpnSpec",
    "SboxTransitionResidualSpnSpec",
    "VirtualSlotSummedLinear",
    "canonical_walsh_fingerprint",
    "canonical_walsh_mask_pairs",
    "deterministic_position_histogram",
    "deterministic_sbox_transition_histogram",
    "deterministic_sbox_transition_walsh_features",
    "histogram_semantics_fingerprint",
    "sbox_transition_semantics_fingerprint",
]
