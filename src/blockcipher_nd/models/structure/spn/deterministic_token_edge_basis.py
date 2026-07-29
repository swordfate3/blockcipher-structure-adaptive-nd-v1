from __future__ import annotations

import math

import torch
from torch import nn

from blockcipher_nd.models.structure.spn.gf2_boolean_view import gf2_boolean_views
from blockcipher_nd.models.structure.spn.mandatory_token_gate_operator import (
    MandatoryTokenGateK1AzProbe,
    MandatoryTokenGateOperatorEncoder,
    _pool_items,
)
from blockcipher_nd.models.structure.spn.operator_tied_latent import segment_mean
from blockcipher_nd.models.structure.spn.position_preserving_operator import (
    OPERATOR_TOKEN_DIM,
    OperatorTokenBatch,
    PositionPreservingOperatorSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


class DeterministicTokenEdgeBasisOperatorEncoder(MandatoryTokenGateOperatorEncoder):
    """Use a fixed full-rank runtime-token basis for every edge message."""

    def __init__(self, spec: PositionPreservingOperatorSpec) -> None:
        if spec.hidden_dim != 32:
            raise ValueError("deterministic token edge basis requires hidden_dim=32")
        super().__init__(spec)
        del self.token_encoder
        self.register_buffer(
            "basis_projection",
            _normalized_hadamard(32)[:OPERATOR_TOKEN_DIM],
            persistent=True,
        )

    def operator_tokens(
        self,
        structure: RuntimeSpnStructure,
        *,
        cell_position_ids: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> OperatorTokenBatch:
        target_dtype = self.basis_projection.dtype if dtype is None else dtype
        return super().operator_tokens(
            structure,
            cell_position_ids=cell_position_ids,
            device=device,
            dtype=target_dtype,
        )

    def fixed_edge_basis(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != OPERATOR_TOKEN_DIM:
            raise ValueError("edge tokens must have shape [edges, 18]")
        projection = self.basis_projection.to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        projected = tokens @ projection
        rms = torch.sqrt(projected.square().mean(dim=-1, keepdim=True).clamp_min(1e-8))
        return torch.tanh(projected / rms)

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
        fixed_basis = self.fixed_edge_basis(tokens.values)
        for slot in range(operator_structure.rounds):
            selected = tokens.slots == slot
            sources = tokens.sources[selected]
            targets = tokens.targets[selected]
            sample_message = self.sample_message(
                torch.cat((bit_state[:, sources], bit_state[:, targets]), dim=-1)
            )
            edge_basis = fixed_basis[selected].reshape(
                1,
                -1,
                self.spec.hidden_dim,
            )
            messages = sample_message * edge_basis
            target_messages = segment_mean(messages, targets, bit_count)
            update = self.bit_update(torch.cat((bit_state, target_messages), dim=-1))
            bit_state = self.bit_update_norm(bit_state + update)
        pair_embedding = self.pair_projection(_pool_items(bit_state)).reshape(
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


class DeterministicTokenEdgeBasisK1AzProbe(MandatoryTokenGateK1AzProbe):
    def __init__(
        self,
        anchor: nn.Module,
        spec: PositionPreservingOperatorSpec,
    ) -> None:
        super().__init__(anchor, spec)
        self.operator_encoder = DeterministicTokenEdgeBasisOperatorEncoder(spec)
        self.sample_only_bypass = False
        self.readiness_only_projection_present = False
        self.token_encoder_present = False
        self.basis_projection_trainable = False


def _normalized_hadamard(order: int) -> torch.Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError("Sylvester-Hadamard order must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(order)


__all__ = [
    "DeterministicTokenEdgeBasisK1AzProbe",
    "DeterministicTokenEdgeBasisOperatorEncoder",
]
