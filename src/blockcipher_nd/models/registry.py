from __future__ import annotations

from collections.abc import Callable

from torch import nn

from blockcipher_nd.models.baseline import (
    AutoNDDBitNet2023Distinguisher,
    CnnDistinguisher,
    DBitNetDistinguisher,
    GohrSpeckDistinguisher,
    LstmRoundSeqDistinguisher,
    MlpDistinguisher,
    MultiScaleDenseResNetDistinguisher,
    ResNetBitSliceDistinguisher,
    SeResNeXtDistinguisher,
    Sm4Yu2023PositionResNetDistinguisher,
    TransformerEncoderDistinguisher,
)
from blockcipher_nd.models.structure import (
    AdaptiveDBitNetDistinguisher,
    ArxCarryPositionStatsPairSetDistinguisher,
    ArxCarryRunMixerPairSetDistinguisher,
    ArxPairSetStatsHybridDistinguisher,
    ArxRoundFunctionHybridPairSetDistinguisher,
    ArxRoundStatsHybridPairSetDistinguisher,
    ArxRoundStatsPairSetDistinguisher,
    ArxStructureAdaptivePairSetDBitNetDistinguisher,
    ArxTrailMixerPairSetDistinguisher,
    ArxWordMixerPairSetDistinguisher,
    PairwiseAdaptiveDBitNetDistinguisher,
    PresentInceptionMCNDDistinguisher,
    PresentInceptionMCNDGlobalMatrixDistinguisher,
    PresentInceptionMCNDPairStackMatrixDistinguisher,
    PresentMatrixTrailHybridPairSetDistinguisher,
    PresentNibbleInvPPLayerGraphSpnOnlyDistinguisher,
    PresentNibbleInvPShuffledPLayerGraphSpnOnlyDistinguisher,
    PresentPairSetGlobalStatsDistinguisher,
    PresentPairSetGlobalStatsHybridDistinguisher,
    PresentPairSetHistogramHybridDistinguisher,
    PresentPairSetStatsHybridDistinguisher,
    PresentActiveCellGraphPairSetDistinguisher,
    PresentPLayerMixerPairSetDistinguisher,
    PresentStateTokenResidualDistinguisher,
    PresentTrailPositionStatsPairSetDistinguisher,
    PresentTrailMixerPairSetDistinguisher,
    SpnCellPairSetDBitNetDistinguisher,
    SpnNibbleConvPairSetDistinguisher,
    SpnTokenMixerPairSetDistinguisher,
    StructureAdaptivePairSetDBitNetDistinguisher,
    StructureAwareMoEDistinguisher,
)

ModelBuilder = Callable[..., nn.Module]

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "autond_dbitnet2023": AutoNDDBitNet2023Distinguisher,
    "mlp": MlpDistinguisher,
    "cnn": CnnDistinguisher,
    "dbitnet_dilated_cnn": DBitNetDistinguisher,
    "gohr_resnet_speck": GohrSpeckDistinguisher,
    "lstm_roundseq": LstmRoundSeqDistinguisher,
    "transformer_encoder": TransformerEncoderDistinguisher,
    "resnet_bitslice": ResNetBitSliceDistinguisher,
    "senet_resnext": SeResNeXtDistinguisher,
    "multiscale_dense_resnet": MultiScaleDenseResNetDistinguisher,
    "sm4_yu2023_position_resnet": Sm4Yu2023PositionResNetDistinguisher,
    "adaptive_dbitnet": AdaptiveDBitNetDistinguisher,
    "adaptive_dbitnet_pairwise": PairwiseAdaptiveDBitNetDistinguisher,
    "structure_adaptive_pairset_dbitnet": StructureAdaptivePairSetDBitNetDistinguisher,
    "arx_structure_adaptive_pairset_dbitnet": ArxStructureAdaptivePairSetDBitNetDistinguisher,
    "arx_pairset_dbitnet": ArxStructureAdaptivePairSetDBitNetDistinguisher,
    "arx_pairset_stats_hybrid": ArxPairSetStatsHybridDistinguisher,
    "arx_carry_position_stats_pairset": ArxCarryPositionStatsPairSetDistinguisher,
    "arx_carry_run_mixer_pairset": ArxCarryRunMixerPairSetDistinguisher,
    "arx_round_function_hybrid_pairset": ArxRoundFunctionHybridPairSetDistinguisher,
    "arx_round_stats_hybrid_pairset": ArxRoundStatsHybridPairSetDistinguisher,
    "arx_round_stats_pairset": ArxRoundStatsPairSetDistinguisher,
    "arx_word_mixer_pairset": ArxWordMixerPairSetDistinguisher,
    "arx_trail_mixer_pairset": ArxTrailMixerPairSetDistinguisher,
    "present_inception_mcnd": PresentInceptionMCNDDistinguisher,
    "present_inception_mcnd_global_matrix": PresentInceptionMCNDGlobalMatrixDistinguisher,
    "present_inception_mcnd_pair_stack_matrix": PresentInceptionMCNDPairStackMatrixDistinguisher,
    "present_matrix_trail_hybrid_pairset": PresentMatrixTrailHybridPairSetDistinguisher,
    "present_matrix_trail_hybrid_pairset_invp": PresentMatrixTrailHybridPairSetDistinguisher,
    "present_matrix_trail_hybrid_pairset_invp_sinv": PresentMatrixTrailHybridPairSetDistinguisher,
    "present_nibble_invp_p_layer_graph_spn_only": PresentNibbleInvPPLayerGraphSpnOnlyDistinguisher,
    "present_nibble_invp_shuffled_p_layer_graph_spn_only": PresentNibbleInvPShuffledPLayerGraphSpnOnlyDistinguisher,
    "present_pairset_global_stats": PresentPairSetGlobalStatsDistinguisher,
    "present_active_cell_graph_pairset": PresentActiveCellGraphPairSetDistinguisher,
    "present_pairset_global_stats_hybrid": PresentPairSetGlobalStatsHybridDistinguisher,
    "present_pairset_histogram_hybrid": PresentPairSetHistogramHybridDistinguisher,
    "present_pairset_stats_hybrid": PresentPairSetStatsHybridDistinguisher,
    "present_trail_position_stats_pairset": PresentTrailPositionStatsPairSetDistinguisher,
    "present_state_token_residual": PresentStateTokenResidualDistinguisher,
    "spn_cell_pairset_dbitnet": SpnCellPairSetDBitNetDistinguisher,
    "spn_nibble_conv_pairset": SpnNibbleConvPairSetDistinguisher,
    "spn_token_mixer_pairset": SpnTokenMixerPairSetDistinguisher,
    "present_p_layer_mixer_pairset": PresentPLayerMixerPairSetDistinguisher,
    "present_trail_mixer_pairset": PresentTrailMixerPairSetDistinguisher,
    "structure_aware_moe": StructureAwareMoEDistinguisher,
}


def get_model_class(model_key: str) -> type[nn.Module]:
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        raise KeyError(f"unknown model key: {model_key}") from exc


__all__ = ["MODEL_REGISTRY", "ModelBuilder", "get_model_class"]
