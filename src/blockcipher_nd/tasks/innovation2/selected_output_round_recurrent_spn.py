from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    _present_topology_mapping,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
)


class _SharedPresentRoundBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        source_for_destination: torch.Tensor,
    ) -> None:
        super().__init__()
        cell_dim = token_dim * 4
        self.register_buffer(
            "source_for_destination",
            source_for_destination.detach().clone().to(dtype=torch.long),
            persistent=False,
        )
        self.local_norm = nn.LayerNorm(token_dim)
        self.local_mlp = nn.Sequential(
            nn.Linear(cell_dim, cell_dim),
            nn.GELU(),
            nn.Linear(cell_dim, cell_dim),
        )
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, positions, channels = hidden.shape
        if positions != 64:
            raise ValueError(f"expected 64 positions, got {positions}")
        normalized = self.local_norm(hidden).reshape(batch, 16, channels * 4)
        mixed = hidden.reshape(batch, 16, channels * 4) + self.local_mlp(normalized)
        routed = mixed.reshape(batch, 64, channels).index_select(
            1,
            self.source_for_destination,
        )
        return routed + self.channel_mlp(self.channel_norm(routed))


class SelectedOutputRoundRecurrentSpn(nn.Module):
    """Four-step shared SPN predictor for eight fixed PRESENT output positions."""

    def __init__(
        self,
        token_dim: int = 316,
        rounds: int = 4,
        source_for_destination: torch.Tensor | None = None,
        selected_msb_indices: tuple[int, ...] = SELECTED_MSB_INDICES,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if rounds != 4:
            raise ValueError("OPF3 requires exactly four recurrent PRESENT rounds")
        if selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("OPF3 output positions must remain preregistered")
        mapping = (
            _present_topology_mapping("exact")
            if source_for_destination is None
            else source_for_destination.detach().clone().to(dtype=torch.long)
        )
        if mapping.shape != (64,) or sorted(mapping.tolist()) != list(range(64)):
            raise ValueError("source_for_destination must be a 64-position permutation")

        self.token_dim = token_dim
        self.rounds = rounds
        self.selected_msb_indices = selected_msb_indices
        self.embedding = nn.Linear(1, token_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, 64, token_dim))
        self.round_position_contexts = nn.Parameter(
            torch.empty(rounds, 64, token_dim)
        )
        self.final_whitening_context = nn.Parameter(torch.empty(1, 64, token_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.round_position_contexts, std=0.02)
        nn.init.trunc_normal_(self.final_whitening_context, std=0.02)
        self.round_block = _SharedPresentRoundBlock(token_dim, mapping)
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(token_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            for _ in selected_msb_indices
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 64:
            raise ValueError(f"expected [batch, 64] input, got {tuple(features.shape)}")
        hidden = self.embedding(features.float().unsqueeze(-1))
        hidden = hidden + self.position_embedding
        for round_index in range(self.rounds):
            hidden = hidden + self.round_position_contexts[round_index]
            hidden = self.round_block(hidden)
        hidden = hidden + self.final_whitening_context
        selected = hidden[:, list(self.selected_msb_indices), :]
        return torch.cat(
            [head(selected[:, index, :]) for index, head in enumerate(self.heads)],
            dim=1,
        )


def build_round_recurrent_spn(
    topology_mode: str,
    *,
    token_dim: int = 316,
) -> SelectedOutputRoundRecurrentSpn:
    if topology_mode not in {"exact", "identity"}:
        raise ValueError("OPF3 topology_mode must be exact or identity")
    return SelectedOutputRoundRecurrentSpn(
        token_dim=token_dim,
        source_for_destination=_present_topology_mapping(topology_mode),
    )


def round_recurrent_parameter_count(token_dim: int = 316) -> int:
    return sum(
        parameter.numel()
        for parameter in build_round_recurrent_spn(
            "exact",
            token_dim=token_dim,
        ).parameters()
    )


__all__ = [
    "SelectedOutputRoundRecurrentSpn",
    "build_round_recurrent_spn",
    "round_recurrent_parameter_count",
]
