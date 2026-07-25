from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    FixedRuntimeSpnProtocolAdapter,
    RuntimeE4EquivariantSpnDistinguisher,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


_RUNTIME_METADATA_FIELDS = (
    "input_bit_order",
    "runtime_structure_descriptor_name",
    "runtime_structure_descriptor_path",
    "runtime_structure_descriptor_sha256",
    "runtime_structure_round_start",
    "runtime_structure_available_rounds",
    "runtime_structure_loaded_rounds",
    "runtime_round_window_mode",
    "runtime_structure_window_control",
    "runtime_structure_transition_sha256s",
    "runtime_structure_window_sha256",
    "runtime_structure_unique_transition_count",
    "runtime_structure_homogeneous",
    "runtime_structure_mode",
)


@dataclass(frozen=True)
class RuntimeE4RepresentationBatch:
    """RuntimeE4 logits paired with the exact pre-classifier representation."""

    representation: torch.Tensor
    logits: torch.Tensor


class FrozenRuntimeE4HeadAdapter(nn.Module):
    """Train a target head on a frozen RuntimeE4 representation extractor."""

    def __init__(
        self,
        feature_extractor: FixedRuntimeSpnProtocolAdapter,
        target_head: nn.Module,
    ) -> None:
        super().__init__()
        extractor = _require_runtime_e4_adapter(feature_extractor)
        extractor_parameter_ids = {
            id(parameter) for parameter in extractor.parameters()
        }
        target_parameters = tuple(target_head.parameters())
        if not target_parameters:
            raise ValueError("RuntimeE4 target head must own trainable parameters")
        if any(
            id(parameter) in extractor_parameter_ids for parameter in target_parameters
        ):
            raise ValueError(
                "RuntimeE4 target head must not share extractor parameters"
            )

        self.feature_extractor = extractor
        self.target_head = target_head
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_head.parameters():
            parameter.requires_grad_(True)
        self.feature_extractor.eval()
        for field in _RUNTIME_METADATA_FIELDS:
            if hasattr(self.feature_extractor, field):
                setattr(self, field, getattr(self.feature_extractor, field))
        self.adapter_mode = "frozen_runtime_e4_target_head"
        self.feature_extractor_frozen = True
        self.source_classifier_preserved = True
        self.target_head_parameter_count = sum(
            parameter.numel() for parameter in self.target_head.parameters()
        )

    @property
    def representation_width(self) -> int:
        return 3 * self.feature_extractor.backbone.spec.pair_embedding_dim

    @property
    def runtime_structure(self) -> RuntimeSpnStructure:
        return self.feature_extractor.runtime_structure

    @property
    def relation_mode(self) -> str:
        return self.feature_extractor.relation_mode

    @property
    def aggregation_mode(self) -> str:
        return self.feature_extractor.aggregation_mode

    def train(self, mode: bool = True) -> FrozenRuntimeE4HeadAdapter:
        super().train(mode)
        self.feature_extractor.eval()
        self.target_head.train(mode)
        return self

    def forward(
        self,
        features: torch.Tensor,
        *,
        query_input_mode: Literal["delta_v", "delta_u"] | None = None,
        query_structure: RuntimeSpnStructure | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            batch = extract_runtime_e4_representation(
                self.feature_extractor,
                features,
                query_input_mode=query_input_mode,
                query_structure=query_structure,
            )
        return self.target_head(batch.representation)


def extract_runtime_e4_representation(
    model: nn.Module,
    features: torch.Tensor,
    *,
    query_input_mode: Literal["delta_v", "delta_u"] | None = None,
    query_structure: RuntimeSpnStructure | None = None,
) -> RuntimeE4RepresentationBatch:
    """Run a fixed RuntimeE4 adapter and expose its invariant pooled embedding."""
    adapter = _require_runtime_e4_adapter(model)
    representation = adapter.encode(
        features,
        query_input_mode=query_input_mode,
        query_structure=query_structure,
    )
    logits = adapter.backbone.classifier(representation)
    expected_width = 3 * adapter.backbone.spec.pair_embedding_dim
    if representation.ndim != 2 or representation.shape != (
        features.shape[0],
        expected_width,
    ):
        raise RuntimeError(
            "RuntimeE4 representation shape must be "
            f"batch x {expected_width}, got {tuple(representation.shape)}"
        )
    return RuntimeE4RepresentationBatch(
        representation=representation,
        logits=logits,
    )


def _require_runtime_e4_adapter(
    model: nn.Module,
) -> FixedRuntimeSpnProtocolAdapter:
    if not isinstance(model, FixedRuntimeSpnProtocolAdapter):
        raise TypeError(
            "RuntimeE4 representation extraction requires "
            "FixedRuntimeSpnProtocolAdapter"
        )
    if not isinstance(model.backbone, RuntimeE4EquivariantSpnDistinguisher):
        raise TypeError(
            "RuntimeE4 representation extraction requires the E4-equivariant backbone"
        )
    return model


__all__ = [
    "FrozenRuntimeE4HeadAdapter",
    "RuntimeE4RepresentationBatch",
    "extract_runtime_e4_representation",
]
