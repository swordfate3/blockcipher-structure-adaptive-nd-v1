from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.structure.spn.gf2_boolean_view import gf2_boolean_views
from blockcipher_nd.models.structure.spn.operator_tied_latent import segment_mean
from blockcipher_nd.models.structure.spn.position_preserving_operator import (
    PositionPreservingOperatorK1AzProbe,
    PositionPreservingOperatorSpec,
    SharedPositionPreservingOperatorEncoder,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


class MandatoryTokenGateOperatorEncoder(SharedPositionPreservingOperatorEncoder):
    """Make the actual edge token a mandatory multiplicative message gate."""

    def __init__(self, spec: PositionPreservingOperatorSpec) -> None:
        super().__init__(spec)
        del self.edge_message
        del self.structure_projection
        hidden = spec.hidden_dim
        self.sample_message = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(hidden * 2, hidden),
        )

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
            sample_message = self.sample_message(
                torch.cat((bit_state[:, sources], bit_state[:, targets]), dim=-1)
            )
            token_gate = torch.tanh(token_hidden[selected]).reshape(
                1, -1, self.spec.hidden_dim
            )
            messages = sample_message * token_gate
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


class MandatoryTokenGateK1AzProbe(PositionPreservingOperatorK1AzProbe):
    def __init__(
        self,
        anchor: nn.Module,
        spec: PositionPreservingOperatorSpec,
    ) -> None:
        super().__init__(anchor, spec)
        self.operator_encoder = MandatoryTokenGateOperatorEncoder(spec)
        self.sample_only_bypass = False
        self.readiness_only_projection_present = False


def _pool_items(values: torch.Tensor) -> torch.Tensor:
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
    "MandatoryTokenGateK1AzProbe",
    "MandatoryTokenGateOperatorEncoder",
]
