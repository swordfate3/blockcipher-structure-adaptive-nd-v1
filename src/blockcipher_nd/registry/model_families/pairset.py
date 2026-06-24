from __future__ import annotations

from torch import nn

from blockcipher_nd.models.structure import (
    PairwiseAdaptiveDBitNetDistinguisher,
    StructureAdaptivePairSetDBitNetDistinguisher,
)


def build_pairset_model(
    name: str,
    input_bits: int,
    hidden_bits: int,
    pair_bits: int | None,
    structure: str,
) -> nn.Module | None:
    pairwise_pooling_keys = {
        "adaptive_dbitnet_pairwise": "mean_max",
        "adaptive_dbitnet_pairwise_mean": "mean",
        "adaptive_dbitnet_pairwise_max": "max",
        "adaptive_dbitnet_pairwise_mean_max": "mean_max",
    }
    if name in pairwise_pooling_keys:
        return PairwiseAdaptiveDBitNetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 96,
            base_channels=hidden_bits,
            pooling=pairwise_pooling_keys[name],
        )

    pairset_pooling_keys = {
        "structure_adaptive_pairset_dbitnet": "attention_mean_max",
        "structure_adaptive_pairset_dbitnet_attention": "attention",
        "structure_adaptive_pairset_dbitnet_mean_max": "mean_max",
    }
    if name in pairset_pooling_keys:
        return StructureAdaptivePairSetDBitNetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 96,
            base_channels=hidden_bits,
            structure=structure,
            pooling=pairset_pooling_keys[name],
        )
    return None
