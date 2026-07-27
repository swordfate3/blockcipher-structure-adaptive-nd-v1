from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from blockcipher_nd.models.common.components import AttentionPooling
from blockcipher_nd.models.structure.spn.canonical_components import (
    CanonicalLinearSchedule,
    compile_canonical_linear_schedule,
)
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    _RuntimeSpnEncoderBase,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.models.structure.spn.token_mixer_pairset import (
    EquivariantSpnTokenMixerBlock,
)


@dataclass(frozen=True)
class CanonicalTransitionSpnSpec:
    hidden_dim: int = 64
    pair_embedding_dim: int = 128
    processor_steps: int = 2
    temporal_hidden_dim: int = 76
    dropout: float = 0.0
    endpoint_identity_mode: Literal["edge_invariant", "native_cell_role"] = (
        "edge_invariant"
    )

    def __post_init__(self) -> None:
        if min(
            self.hidden_dim,
            self.pair_embedding_dim,
            self.processor_steps,
            self.temporal_hidden_dim,
        ) <= 0:
            raise ValueError("CT-SPN dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("CT-SPN dropout must be in [0, 1)")
        if self.endpoint_identity_mode not in {
            "edge_invariant",
            "native_cell_role",
        }:
            raise ValueError("unsupported CT-SPN endpoint identity mode")


class CanonicalTransitionSpnDistinguisher(nn.Module):
    """Shared K1 network over exact canonical transition state views."""

    def __init__(self, spec: CanonicalTransitionSpnSpec) -> None:
        super().__init__()
        self.spec = spec
        token_dim = max(16, spec.hidden_dim * 2)
        pair_dim = spec.pair_embedding_dim
        self.token_dim = token_dim
        edge_input_dim = 22 if spec.endpoint_identity_mode == "native_cell_role" else 12
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )
        self.temporal_depthwise = nn.Conv1d(
            token_dim,
            token_dim,
            kernel_size=3,
            padding=1,
            groups=token_dim,
        )
        self.temporal_channel = nn.Sequential(
            nn.Linear(token_dim, spec.temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.temporal_hidden_dim, token_dim),
        )
        self.temporal_gate = nn.Linear(token_dim, token_dim)
        self.temporal_norm = nn.LayerNorm(token_dim)
        self.mixer_blocks = nn.ModuleList(
            [
                EquivariantSpnTokenMixerBlock(
                    nibbles_per_pair=16,
                    token_dim=token_dim,
                    token_mlp_ratio=2,
                    activation="relu",
                    norm="layernorm",
                    dropout=spec.dropout,
                )
                for _ in range(spec.processor_steps)
            ]
        )
        self.sequence_norm = nn.LayerNorm(token_dim)
        self.pair_projection = nn.Sequential(
            nn.Linear(token_dim * 3, pair_dim),
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
        schedule: CanonicalLinearSchedule,
        *,
        relation_mode: Literal["true", "independent"] = "true",
    ) -> torch.Tensor:
        return self.classifier(
            self.encode(
                ciphertext_pairs,
                structure,
                schedule,
                relation_mode=relation_mode,
            )
        )

    def encode(
        self,
        ciphertext_pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        schedule: CanonicalLinearSchedule,
        *,
        relation_mode: Literal["true", "independent"] = "true",
    ) -> torch.Tensor:
        pairs = _RuntimeSpnEncoderBase._normalize_pairs(
            ciphertext_pairs, structure.block_bits
        )
        if relation_mode not in {"true", "independent"}:
            raise ValueError("CT-SPN relation mode must be true or independent")
        if schedule.block_bits != structure.block_bits:
            raise ValueError("CT-SPN schedule width does not match runtime structure")
        if schedule.rounds != structure.rounds:
            raise ValueError("CT-SPN schedule length does not match runtime structure")
        if not torch.all((pairs == 0) | (pairs == 1)):
            raise ValueError("CT-SPN ciphertext pair tensors must be binary")
        pairs = pairs.to(dtype=self.edge_encoder[0].weight.dtype)
        views = self.canonical_transition_edge_views(
            pairs,
            structure,
            schedule,
            relation_mode=relation_mode,
            endpoint_identity_mode=self.spec.endpoint_identity_mode,
        )
        batch, pair_count, transitions, edges, _ = views.shape
        hidden = self.edge_encoder(views).reshape(
            batch * pair_count * transitions,
            edges,
            self.token_dim,
        )
        for block in self.mixer_blocks:
            hidden = block(hidden)
        hidden = self.sequence_norm(hidden)
        edge_mean = hidden.mean(dim=1)
        edge_max = hidden.max(dim=1).values
        edge_rms = torch.sqrt(hidden.square().mean(dim=1).clamp_min(1e-8))
        transition_summary = (edge_mean + edge_max + edge_rms) / 3.0
        temporal = transition_summary.reshape(
            batch * pair_count,
            transitions,
            self.token_dim,
        ).transpose(1, 2)
        temporal = self.temporal_depthwise(temporal).transpose(1, 2)
        update = self.temporal_channel(temporal)
        temporal = self.temporal_norm(
            temporal + torch.sigmoid(self.temporal_gate(temporal)) * update
        )
        mean_embedding = temporal.mean(dim=1)
        max_embedding = temporal.max(dim=1).values
        rms_embedding = torch.sqrt(temporal.square().mean(dim=1).clamp_min(1e-8))
        pair_embeddings = self.pair_projection(
            torch.cat((mean_embedding, max_embedding, rms_embedding), dim=-1)
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
    def canonical_transition_edge_views(
        pairs: torch.Tensor,
        structure: RuntimeSpnStructure,
        schedule: CanonicalLinearSchedule,
        *,
        relation_mode: Literal["true", "independent"],
        endpoint_identity_mode: Literal[
            "edge_invariant", "native_cell_role"
        ] = "edge_invariant",
    ) -> torch.Tensor:
        left = pairs[:, :, 0]
        right = pairs[:, :, 1]
        reverse_views: list[torch.Tensor] = []
        for round_index in reversed(range(structure.rounds)):
            if relation_mode == "true":
                (
                    current_left,
                    previous_left,
                    native_previous_left,
                ) = schedule.transition(left, round_index)
                (
                    current_right,
                    previous_right,
                    native_previous_right,
                ) = schedule.transition(right, round_index)
            else:
                current_left = previous_left = left
                current_right = previous_right = right
                native_previous_left = left
                native_previous_right = right
            current_difference = torch.remainder(current_left + current_right, 2.0)
            previous_difference = torch.remainder(previous_left + previous_right, 2.0)
            if relation_mode == "true":
                targets = schedule.canonical_edge_index[0].to(left.device)
                sources = schedule.canonical_edge_index[1].to(left.device)
            else:
                targets = sources = torch.arange(
                    structure.block_bits, device=left.device
                )
            current_endpoint = torch.stack(
                (
                    current_left[..., targets],
                    current_right[..., targets],
                    current_difference[..., targets],
                ),
                dim=-1,
            )
            previous_endpoint = torch.stack(
                (
                    previous_left[..., sources],
                    previous_right[..., sources],
                    previous_difference[..., sources],
                ),
                dim=-1,
            )
            endpoint_product = current_endpoint * previous_endpoint
            endpoint_xor = torch.remainder(
                current_endpoint + previous_endpoint, 2.0
            )
            reverse_views.append(
                torch.cat(
                    (
                        current_endpoint,
                        previous_endpoint,
                        endpoint_product,
                        endpoint_xor,
                        *(
                            (
                                CanonicalTransitionSpnDistinguisher._native_endpoint_identity(
                                    structure,
                                    schedule,
                                    round_index,
                                    targets,
                                    sources,
                                    relation_mode=relation_mode,
                                    reference=current_endpoint,
                                ),
                            )
                            if endpoint_identity_mode == "native_cell_role"
                            else ()
                        ),
                    ),
                    dim=-1,
                )
            )
            if relation_mode == "true":
                left = native_previous_left
                right = native_previous_right
        return torch.stack(tuple(reversed(reverse_views)), dim=2)

    @staticmethod
    def _native_endpoint_identity(
        structure: RuntimeSpnStructure,
        schedule: CanonicalLinearSchedule,
        round_index: int,
        targets: torch.Tensor,
        sources: torch.Tensor,
        *,
        relation_mode: Literal["true", "independent"],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if relation_mode == "true":
            canonical_input_native, canonical_output_native = schedule.factors[
                round_index
            ]
            native_targets = torch.tensor(
                canonical_output_native,
                dtype=torch.long,
                device=reference.device,
            )[targets]
            native_sources = torch.tensor(
                canonical_input_native,
                dtype=torch.long,
                device=reference.device,
            )[sources]
        else:
            native_targets = targets
            native_sources = sources
        membership = structure.cell_membership.to(reference.device)
        roles = structure.bit_role.to(reference.device)
        denominator = max(1, structure.cells - 1)
        target_cell = (
            membership[native_targets].to(reference.dtype) * (2.0 / denominator)
            - 1.0
        )
        source_cell = (
            membership[native_sources].to(reference.dtype) * (2.0 / denominator)
            - 1.0
        )
        target_role = torch.nn.functional.one_hot(
            roles[native_targets], num_classes=4
        ).to(reference.dtype)
        source_role = torch.nn.functional.one_hot(
            roles[native_sources], num_classes=4
        ).to(reference.dtype)
        identity = torch.cat(
            (
                target_cell[:, None],
                source_cell[:, None],
                target_role,
                source_role,
            ),
            dim=-1,
        )
        return identity.reshape(1, 1, identity.shape[0], 10).expand(
            *reference.shape[:-1],
            10,
        )


class FixedCanonicalTransitionSpnProtocolAdapter(nn.Module):
    """Bind CT-SPN to an external runtime descriptor and project input bit order."""

    def __init__(
        self,
        *,
        input_bits: int,
        pair_bits: int,
        structure: RuntimeSpnStructure,
        relation_mode: Literal["true", "independent"],
        spec: CanonicalTransitionSpnSpec,
        descriptor_name: str,
        descriptor_path: str,
        descriptor_sha256: str,
        descriptor_round_start: int,
        descriptor_available_rounds: int,
        runtime_structure_mode: str,
        runtime_structure_window_control: str,
        canonical_schedule_control: str = "ordered",
    ) -> None:
        super().__init__()
        if pair_bits != 2 * structure.block_bits:
            raise ValueError("CT-SPN pair_bits must encode two runtime blocks")
        if input_bits <= 0 or input_bits % pair_bits:
            raise ValueError("CT-SPN input_bits must contain complete pairs")
        self.backbone = CanonicalTransitionSpnDistinguisher(spec)
        self.runtime_structure = structure
        self.canonical_schedule = compile_canonical_linear_schedule(
            structure, control=canonical_schedule_control
        )
        self.relation_mode = relation_mode
        self.mapping_mode = relation_mode
        self.input_bit_order = "project_msb_to_runtime_lsb"
        self.runtime_structure_loaded_rounds = structure.rounds
        self.runtime_round_window_mode = "canonical_transition_window"
        self.runtime_structure_window_control = runtime_structure_window_control
        self.runtime_structure_descriptor_name = descriptor_name
        self.runtime_structure_descriptor_path = descriptor_path
        self.runtime_structure_descriptor_sha256 = descriptor_sha256
        self.runtime_structure_round_start = descriptor_round_start
        self.runtime_structure_available_rounds = descriptor_available_rounds
        self.runtime_structure_mode = runtime_structure_mode
        self.runtime_structure_transition_sha256s = structure.transition_sha256s()
        self.runtime_structure_window_sha256 = structure.window_sha256()
        self.runtime_structure_unique_transition_count = structure.unique_transition_count
        self.runtime_structure_homogeneous = structure.is_homogeneous
        self.canonical_primitive = self.canonical_schedule.primitive
        self.canonical_factor_manifest_sha256 = (
            self.canonical_schedule.manifest_sha256
        )
        self.canonical_schedule_control = self.canonical_schedule.control
        self.canonical_endpoint_identity_mode = spec.endpoint_identity_mode

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
            self.canonical_schedule,
            relation_mode=self.relation_mode,
        )


__all__ = [
    "CanonicalTransitionSpnDistinguisher",
    "CanonicalTransitionSpnSpec",
    "FixedCanonicalTransitionSpnProtocolAdapter",
]
