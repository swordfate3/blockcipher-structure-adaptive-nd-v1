from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from blockcipher_nd.models.structure.spn.present_p_layer_mixer import (
    PresentPLayerMixerBlock,
)


@dataclass(frozen=True)
class PresentOutputParityPredictorSpec:
    token_dim: int = 28
    mixer_depth: int = 2
    p_topology: str = "true"

    def __post_init__(self) -> None:
        if min(self.token_dim, self.mixer_depth) <= 0:
            raise ValueError("token_dim and mixer_depth must be positive")
        if self.p_topology not in {"true", "shuffled"}:
            raise ValueError("p_topology must be true or shuffled")


class PresentOutputParityPredictor(nn.Module):
    def __init__(self, spec: PresentOutputParityPredictorSpec) -> None:
        super().__init__()
        self.spec = spec
        self.nibble_encoder = nn.Sequential(
            nn.Linear(4, spec.token_dim),
            nn.GELU(),
            nn.LayerNorm(spec.token_dim),
        )
        self.nibble_embedding = nn.Parameter(torch.zeros(1, 16, spec.token_dim))
        nn.init.trunc_normal_(self.nibble_embedding, std=0.02)
        self.mixer_blocks = nn.ModuleList(
            [
                PresentPLayerMixerBlock(
                    words_per_pair=1,
                    token_dim=spec.token_dim,
                    token_mlp_ratio=2,
                    activation="gelu",
                    norm="layernorm",
                    dropout=0.0,
                    p_topology=spec.p_topology,
                )
                for _ in range(spec.mixer_depth)
            ]
        )
        self.output_norm = nn.LayerNorm(spec.token_dim)
        self.output_head = nn.Linear(spec.token_dim, 1)

    def forward(self, plaintext_bits: torch.Tensor) -> torch.Tensor:
        tokens = plaintext_bits_to_msb_nibble_tokens(plaintext_bits)
        hidden = self.nibble_encoder(tokens) + self.nibble_embedding
        for block in self.mixer_blocks:
            hidden = block(hidden)
        msb_logits = self.output_head(self.output_norm(hidden)).squeeze(-1)
        return msb_token_logits_to_lsb_outputs(msb_logits)


def plaintext_bits_to_msb_nibble_tokens(
    plaintext_bits: torch.Tensor,
) -> torch.Tensor:
    if plaintext_bits.ndim != 2 or plaintext_bits.shape[1] != 64:
        raise ValueError("plaintext_bits must have shape batch x 64")
    lsb_nibbles = plaintext_bits.reshape(plaintext_bits.shape[0], 16, 4)
    return torch.flip(lsb_nibbles, dims=(1,))


def msb_token_logits_to_lsb_outputs(msb_logits: torch.Tensor) -> torch.Tensor:
    if msb_logits.ndim != 2 or msb_logits.shape[1] != 16:
        raise ValueError("msb_logits must have shape batch x 16")
    return torch.flip(msb_logits, dims=(1,))


__all__ = [
    "PresentOutputParityPredictor",
    "PresentOutputParityPredictorSpec",
    "msb_token_logits_to_lsb_outputs",
    "plaintext_bits_to_msb_nibble_tokens",
]
