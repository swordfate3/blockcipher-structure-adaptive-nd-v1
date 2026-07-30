from __future__ import annotations

import hashlib

import torch
from torch import nn

from blockcipher_nd.models.structure.spn.gf2_boolean_view import apply_gf2_operator
from blockcipher_nd.models.structure.spn.operator_tied_latent import (
    operator_routing_fingerprint,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.models.structure.spn.topology_edge_residual import (
    TopologyEdgeResidualSpnDistinguisher,
    TopologyEdgeResidualSpnSpec,
    topology_edge_fingerprint,
)


def composition_stage_names(rounds: int) -> tuple[str, ...]:
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("composition rounds must be a positive integer")
    names = ["ciphertext"]
    for slot in reversed(range(rounds)):
        names.extend((f"inverse_linear_{slot}", f"inverse_sbox_{slot}"))
    return tuple(names)


COMPOSITION_STAGE_NAMES = composition_stage_names(2)


class ExactOperatorCompositionSpnDistinguisher(
    TopologyEdgeResidualSpnDistinguisher
):
    """K1-M with exact inverse S-box/linear stages in the residual encoder."""

    def __init__(self, spec: TopologyEdgeResidualSpnSpec) -> None:
        super().__init__(spec)
        hidden = spec.hidden_dim
        self.composition_bit_encoder = nn.Sequential(
            nn.Linear(15, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

    def edge_residual_embedding(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        *,
        slot_mask: tuple[bool, bool] = (True, True),
        apply_sboxes: bool = True,
    ) -> torch.Tensor:
        if len(slot_mask) != 2 or not any(slot_mask):
            raise ValueError("composition residual must retain a transition slot")
        views = exact_operator_composition_views(
            ciphertext_pairs,
            structure,
            apply_sboxes=apply_sboxes,
        )
        batch, pair_count, bit_count, _ = views.shape
        bit_hidden = self.composition_bit_encoder(views).reshape(
            batch * pair_count,
            bit_count,
            self.spec.hidden_dim,
        )
        return self._edge_residual_from_bit_hidden(
            bit_hidden,
            batch=batch,
            pair_count=pair_count,
            structure=structure,
            slot_mask=slot_mask,
        )


class FixedExactOperatorCompositionSpnProtocolAdapter(nn.Module):
    """Bind K1-N to one external two-transition runtime descriptor."""

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
        apply_sboxes: bool,
    ) -> None:
        super().__init__()
        if pair_bits != 2 * structure.block_bits:
            raise ValueError("K1-N pair_bits must encode two runtime blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError("K1-N input_bits must contain complete pairs")
        if structure.rounds != 2:
            raise ValueError("K1-N adapter requires exactly two transitions")
        self.backbone = ExactOperatorCompositionSpnDistinguisher(spec)
        self.runtime_structure = structure
        self.apply_sboxes = bool(apply_sboxes)
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_loaded_rounds = structure.rounds
        self.runtime_round_window_mode = "exact_operator_composition"
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
        self.topology_edge_counts = tuple(
            int(matrix.sum()) for matrix in structure.inverse_linear_matrices
        )
        self.composition_stage_names = COMPOSITION_STAGE_NAMES
        self.composition_channels_per_bit = 15
        self.deterministic_exact_composition = True
        self.uses_raw_bypass = False
        self.uses_learned_message_passing = True
        self.uses_path_tokens = False
        self.uses_absolute_cell_or_bit_identity = False
        self.uses_cipher_identity = False
        self.uses_sbox_semantics = self.apply_sboxes
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
        base_embedding = self.backbone.base.encode(runtime, self.runtime_structure)
        residual = self.backbone.edge_residual_embedding(
            runtime,
            self.runtime_structure,
            apply_sboxes=self.apply_sboxes,
        )
        combined = base_embedding + torch.tanh(
            self.backbone.residual_gate
        ) * torch.tanh(residual)
        return self.backbone.base.classifier(combined)


def exact_operator_composition_views(
    ciphertext_pairs: torch.Tensor,
    structure: RuntimeSpnStructure,
    *,
    apply_sboxes: bool = True,
) -> torch.Tensor:
    if ciphertext_pairs.ndim != 4 or ciphertext_pairs.shape[2] != 2:
        raise ValueError("ciphertext pairs must have shape [batch, pairs, 2, bits]")
    if ciphertext_pairs.shape[-1] != structure.block_bits:
        raise ValueError("ciphertext pair width does not match runtime structure")
    if not torch.all((ciphertext_pairs == 0) | (ciphertext_pairs == 1)):
        raise ValueError("exact operator composition requires binary values")

    left = ciphertext_pairs[:, :, 0]
    right = ciphertext_pairs[:, :, 1]
    current = _triplet(left, right)
    stages = [current]
    for slot in reversed(range(structure.rounds)):
        current = apply_gf2_operator(
            current,
            structure.inverse_linear_matrices[slot],
        )
        stages.append(current)
        if apply_sboxes:
            left = structure.apply_inverse_sboxes(current[..., 0], slot)
            right = structure.apply_inverse_sboxes(current[..., 1], slot)
            current = _triplet(left, right)
        stages.append(current)
    return torch.cat(stages, dim=-1)


def _triplet(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.stack((left, right, torch.remainder(left + right, 2.0)), dim=-1)


def composition_fingerprint(
    structure: RuntimeSpnStructure,
    *,
    apply_sboxes: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        "|".join(composition_stage_names(structure.rounds)).encode("ascii")
    )
    digest.update(bytes((int(apply_sboxes),)))
    digest.update(structure.cell_membership.numpy().tobytes())
    digest.update(structure.bit_role.numpy().tobytes())
    if apply_sboxes:
        digest.update(structure.sbox_truth_bits.numpy().tobytes())
    digest.update(structure.inverse_linear_matrices.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "COMPOSITION_STAGE_NAMES",
    "ExactOperatorCompositionSpnDistinguisher",
    "FixedExactOperatorCompositionSpnProtocolAdapter",
    "composition_stage_names",
    "composition_fingerprint",
    "exact_operator_composition_views",
]
