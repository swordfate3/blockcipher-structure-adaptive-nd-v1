from __future__ import annotations

from torch import nn

from blockcipher_nd.models.structure import (
    ArxCarryPositionStatsPairSetDistinguisher,
    ArxCarryRunMixerPairSetDistinguisher,
    ArxPairSetStatsHybridDistinguisher,
    ArxRoundFunctionHybridPairSetDistinguisher,
    ArxRoundStatsHybridPairSetDistinguisher,
    ArxRoundStatsPairSetDistinguisher,
    ArxStructureAdaptivePairSetDBitNetDistinguisher,
    ArxTrailMixerPairSetDistinguisher,
    ArxWordMixerPairSetDistinguisher,
)
from blockcipher_nd.registry.model_options import int_option


def build_arx_model(
    name: str,
    input_bits: int,
    hidden_bits: int,
    pair_bits: int | None,
    structure: str,
    options: dict[str, object],
) -> nn.Module | None:
    arx_pairset_pooling_keys = {
        "arx_structure_adaptive_pairset_dbitnet": "attention_mean_max",
        "arx_pairset_dbitnet": "attention_mean_max",
        "arx_pairset_dbitnet_attention": "attention",
        "arx_pairset_dbitnet_mean_max": "mean_max",
    }
    if name in arx_pairset_pooling_keys:
        return ArxStructureAdaptivePairSetDBitNetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 96,
            base_channels=hidden_bits,
            structure=structure if structure != "generic" else "ARX",
            pooling=arx_pairset_pooling_keys[name],
        )
    if name == "arx_word_mixer_pairset":
        return ArxWordMixerPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 224,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "arx_pairset_stats_hybrid":
        return ArxPairSetStatsHybridDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 224,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            stats_hidden_bits=int_option(options, "stats_hidden_bits"),
        )
    if name == "arx_trail_mixer_pairset":
        return ArxTrailMixerPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 352,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3),
            role_mixer_depth=int_option(options, "role_mixer_depth", 2),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "arx_round_function_hybrid_pairset":
        return ArxRoundFunctionHybridPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 352,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3),
            group_mixer_depth=int_option(options, "group_mixer_depth", 2),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "arx_round_stats_hybrid_pairset":
        return ArxRoundStatsHybridPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 736,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3),
            group_mixer_depth=int_option(options, "group_mixer_depth", 2),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            stats_hidden_bits=int_option(options, "stats_hidden_bits"),
        )
    if name == "arx_round_stats_pairset":
        return ArxRoundStatsPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 736,
            base_channels=hidden_bits,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            stats_hidden_bits=int_option(options, "stats_hidden_bits"),
        )
    if name == "arx_carry_position_stats_pairset":
        return ArxCarryPositionStatsPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 736,
            base_channels=hidden_bits,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            stats_hidden_bits=int_option(options, "stats_hidden_bits"),
        )
    if name == "arx_carry_run_mixer_pairset":
        return ArxCarryRunMixerPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 736,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3) or 3,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            kernel_size=int_option(options, "kernel_size", 3) or 3,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    return None
