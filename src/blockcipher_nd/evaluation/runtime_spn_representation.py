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


@dataclass(frozen=True)
class RuntimeE4RepresentationBatch:
    """RuntimeE4 logits paired with the exact pre-classifier representation."""

    representation: torch.Tensor
    logits: torch.Tensor


def extract_runtime_e4_representation(
    model: nn.Module,
    features: torch.Tensor,
    *,
    query_input_mode: Literal["delta_v", "delta_u"] | None = None,
    query_structure: RuntimeSpnStructure | None = None,
) -> RuntimeE4RepresentationBatch:
    """Run a fixed RuntimeE4 adapter and expose its invariant pooled embedding."""
    adapter = _require_runtime_e4_adapter(model)
    captured: list[torch.Tensor] = []

    def capture_classifier_input(
        _module: nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("RuntimeE4 classifier must receive one tensor")
        captured.append(inputs[0])

    handle = adapter.backbone.classifier.register_forward_pre_hook(
        capture_classifier_input
    )
    try:
        logits = adapter(
            features,
            query_input_mode=query_input_mode,
            query_structure=query_structure,
        )
    finally:
        handle.remove()

    if len(captured) != 1:
        raise RuntimeError("RuntimeE4 classifier must run exactly once per extraction")
    representation = captured[0]
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
    "RuntimeE4RepresentationBatch",
    "extract_runtime_e4_representation",
]
