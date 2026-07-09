from __future__ import annotations

from torch import nn

from blockcipher_nd.models.structure import (
    PresentInceptionMCNDDistinguisher,
    PresentInceptionMCNDGlobalMatrixDistinguisher,
    PresentInceptionMCNDMatrixDistinguisher,
    PresentInceptionMCNDPairStackMatrixDistinguisher,
    PresentMatrixTrailHybridPairSetDistinguisher,
    PresentNibbleDDTGraphDistinguisher,
    PresentNibbleNoDDTGraphDistinguisher,
    PresentPairSetGlobalStatsDistinguisher,
    PresentPairSetGlobalStatsHybridDistinguisher,
    PresentPairSetHistogramHybridDistinguisher,
    PresentPairSetStatsHybridDistinguisher,
    PresentPLayerMixerPairSetDistinguisher,
    PresentNibbleDeltaOnlySpnOnlyDistinguisher,
    PresentNibbleInvPActiveAuxSpnOnlyDistinguisher,
    PresentNibbleInvPNoDDTGateDistinguisher,
    PresentNibblePAlignedGatedMCNDDistinguisher,
    PresentNibbleInvPPLayerGraphSpnOnlyDistinguisher,
    PresentNibbleInvPPairConsistencySpnOnlyDistinguisher,
    PresentNibbleInvPPairMixerConsistencySpnOnlyDistinguisher,
    PresentNibbleInvPOnlySpnOnlyDistinguisher,
    PresentNibbleInvPShuffledPLayerGraphSpnOnlyDistinguisher,
    PresentNibbleInvPShuffledSboxPriorGateDistinguisher,
    PresentNibbleInvPSboxPriorGateDistinguisher,
    PresentNibblePAlignedMCNDDistinguisher,
    PresentNibblePAlignedSpnOnlyDistinguisher,
    PresentNibblePAlignedTransitionDistinguisher,
    PresentNibblePAlignedTransitionResidualDistinguisher,
    PresentNibbleShuffledDDTGraphDistinguisher,
    PresentNibbleShuffledPAlignedGatedMCNDDistinguisher,
    PresentNibbleShuffledPAlignedSpnOnlyDistinguisher,
    PresentNibbleShuffledTransitionResidualDistinguisher,
    PresentTrailPositionStatsPairSetDistinguisher,
    PresentTrailMixerPairSetDistinguisher,
    PresentZhangWangKerasMCNDDistinguisher,
    SpnCellPairSetDBitNetDistinguisher,
    SpnNibbleConvPairSetDistinguisher,
    SpnTokenMixerPairSetDistinguisher,
)
from blockcipher_nd.registry.model_options import (
    int_option,
    int_tuple_option,
    matrix_kernel_size_option,
)


def build_spn_model(
    name: str,
    input_bits: int,
    hidden_bits: int,
    pair_bits: int | None,
    options: dict[str, object],
) -> nn.Module | None:
    if name == "present_zhang_wang_keras_mcnd":
        return PresentZhangWangKerasMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            activation=str(options.get("activation", "relu")),
            dropout=float(options.get("dropout", 0.0)),
            initial_kernel_sizes=int_tuple_option(options, "initial_kernel_sizes", (1, 2, 4)),
            residual_kernel_size=int_option(options, "residual_kernel_size", 3) or 3,
        )
    if name == "present_nibble_paligned_mcnd":
        return PresentNibblePAlignedMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            initial_kernel_sizes=int_tuple_option(options, "initial_kernel_sizes", (1, 2, 4)),
            residual_kernel_size=int_option(options, "residual_kernel_size", 3) or 3,
        )
    if name == "present_nibble_paligned_spn_only":
        return PresentNibblePAlignedSpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
        )
    if name == "present_nibble_delta_only_spn_only":
        return PresentNibbleDeltaOnlySpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
        )
    if name == "present_nibble_invp_only_spn_only":
        return PresentNibbleInvPOnlySpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
        )
    if name == "present_nibble_invp_active_aux_spn_only":
        return PresentNibbleInvPActiveAuxSpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
        )
    if name == "present_nibble_invp_sbox_prior_gate":
        return PresentNibbleInvPSboxPriorGateDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            prior_token_dim=int_option(options, "prior_token_dim"),
            prior_mixer_depth=int_option(options, "prior_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            gate_scale=float(options.get("gate_scale", 0.25)),
        )
    if name == "present_nibble_invp_no_ddt_gate":
        return PresentNibbleInvPNoDDTGateDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            prior_token_dim=int_option(options, "prior_token_dim"),
            prior_mixer_depth=int_option(options, "prior_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            gate_scale=float(options.get("gate_scale", 0.25)),
        )
    if name == "present_nibble_invp_shuffled_sbox_prior_gate":
        return PresentNibbleInvPShuffledSboxPriorGateDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            prior_token_dim=int_option(options, "prior_token_dim"),
            prior_mixer_depth=int_option(options, "prior_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            gate_scale=float(options.get("gate_scale", 0.25)),
        )
    if name == "present_nibble_invp_pair_consistency_spn_only":
        return PresentNibbleInvPPairConsistencySpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_invp_pair_mixer_consistency_spn_only":
        return PresentNibbleInvPPairMixerConsistencySpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            pair_mixer_depth=int_option(options, "pair_mixer_depth", 1) or 1,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_shuffled_paligned_spn_only":
        return PresentNibbleShuffledPAlignedSpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
        )
    if name == "present_nibble_paligned_gated_mcnd":
        return PresentNibblePAlignedGatedMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            initial_kernel_sizes=int_tuple_option(options, "initial_kernel_sizes", (1, 2, 4)),
            residual_kernel_size=int_option(options, "residual_kernel_size", 3) or 3,
            gate_scale=float(options.get("gate_scale", 0.25)),
        )
    if name == "present_nibble_shuffled_paligned_gated_mcnd":
        return PresentNibbleShuffledPAlignedGatedMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            initial_kernel_sizes=int_tuple_option(options, "initial_kernel_sizes", (1, 2, 4)),
            residual_kernel_size=int_option(options, "residual_kernel_size", 3) or 3,
            gate_scale=float(options.get("gate_scale", 0.25)),
        )
    if name == "present_nibble_paligned_transition":
        return PresentNibblePAlignedTransitionDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_paligned_transition_residual":
        return PresentNibblePAlignedTransitionResidualDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            transition_token_dim=int_option(options, "transition_token_dim"),
            transition_mixer_depth=int_option(options, "transition_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_shuffled_transition_residual":
        return PresentNibbleShuffledTransitionResidualDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            transition_token_dim=int_option(options, "transition_token_dim"),
            transition_mixer_depth=int_option(options, "transition_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_ddt_graph":
        return PresentNibbleDDTGraphDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            ddt_token_dim=int_option(options, "ddt_token_dim"),
            ddt_mixer_depth=int_option(options, "ddt_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_no_ddt_graph":
        return PresentNibbleNoDDTGraphDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            ddt_token_dim=int_option(options, "ddt_token_dim"),
            ddt_mixer_depth=int_option(options, "ddt_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_shuffled_ddt_graph":
        return PresentNibbleShuffledDDTGraphDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            ddt_token_dim=int_option(options, "ddt_token_dim"),
            ddt_mixer_depth=int_option(options, "ddt_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_invp_p_layer_graph_spn_only":
        return PresentNibbleInvPPLayerGraphSpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            graph_token_dim=int_option(options, "graph_token_dim"),
            graph_mixer_depth=int_option(options, "graph_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_invp_shuffled_p_layer_graph_spn_only":
        return PresentNibbleInvPShuffledPLayerGraphSpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            graph_token_dim=int_option(options, "graph_token_dim"),
            graph_mixer_depth=int_option(options, "graph_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_inception_mcnd":
        return PresentInceptionMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            branches=int_option(options, "branches"),
            blocks=int_option(options, "blocks", 3) or 3,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "batchnorm1d")),
            pooling=str(options.get("pooling", "attention_mean_max")),
            dropout=float(options.get("dropout", 0.0)),
            kernel_sizes=int_tuple_option(options, "kernel_sizes", (1, 3, 5)),
        )
    if name == "present_inception_mcnd_matrix":
        return PresentInceptionMCNDMatrixDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            branches=int_option(options, "branches"),
            blocks=int_option(options, "blocks", 3) or 3,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "batchnorm2d")),
            pooling=str(options.get("pooling", "attention_mean_max")),
            dropout=float(options.get("dropout", 0.0)),
            kernel_sizes=tuple(
                matrix_kernel_size_option(item)
                for item in options.get("kernel_sizes", [[1, 1], [1, 2], [2, 4]])
            ),
            cell_bits=int_option(options, "cell_bits", 4) or 4,
        )
    if name == "present_inception_mcnd_global_matrix":
        return PresentInceptionMCNDGlobalMatrixDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            branches=int_option(options, "branches"),
            blocks=int_option(options, "blocks", 3) or 3,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "batchnorm2d")),
            dropout=float(options.get("dropout", 0.0)),
            kernel_sizes=tuple(
                matrix_kernel_size_option(item)
                for item in options.get("kernel_sizes", [[1, 1], [1, 2], [2, 4]])
            ),
            cell_bits=int_option(options, "cell_bits", 4) or 4,
        )
    if name == "present_inception_mcnd_pair_stack_matrix":
        return PresentInceptionMCNDPairStackMatrixDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            branches=int_option(options, "branches"),
            blocks=int_option(options, "blocks", 3) or 3,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "batchnorm2d")),
            dropout=float(options.get("dropout", 0.0)),
            kernel_sizes=tuple(
                matrix_kernel_size_option(item)
                for item in options.get("kernel_sizes", [[1, 1], [1, 2], [2, 4], [4, 4]])
            ),
            cell_bits=int_option(options, "cell_bits", 4) or 4,
        )
    if name == "spn_pairset_dbitnet_v2":
        return SpnCellPairSetDBitNetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 192,
            base_channels=hidden_bits,
        )
    if name == "spn_nibble_conv_pairset":
        return SpnNibbleConvPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 192,
            base_channels=hidden_bits,
            nibble_embed_dim=int_option(options, "nibble_embed_dim"),
            conv_depth=int_option(options, "conv_depth", 3),
            kernel_size=int_option(options, "kernel_size", 3),
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "attention_mean_max")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "spn_token_mixer_pairset":
        return SpnTokenMixerPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 192,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "attention_mean_max")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_trail_mixer_pairset":
        return PresentTrailMixerPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 768,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3) or 3,
            role_mixer_depth=int_option(options, "role_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name in {
        "present_matrix_trail_hybrid_pairset",
        "present_matrix_trail_hybrid_pairset_invp",
        "present_matrix_trail_hybrid_pairset_invp_sinv",
    }:
        return PresentMatrixTrailHybridPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 768,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3) or 3,
            role_mixer_depth=int_option(options, "role_mixer_depth", 2) or 2,
            matrix_depth=int_option(options, "matrix_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_pairset_stats_hybrid":
        return PresentPairSetStatsHybridDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 2496,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 2) or 2,
            role_mixer_depth=int_option(options, "role_mixer_depth", 1) or 1,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            stats_hidden_bits=int_option(options, "stats_hidden_bits"),
        )
    if name == "present_pairset_histogram_hybrid":
        return PresentPairSetHistogramHybridDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 2496,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 2) or 2,
            role_mixer_depth=int_option(options, "role_mixer_depth", 1) or 1,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            histogram_hidden_bits=int_option(options, "histogram_hidden_bits"),
        )
    if name == "present_pairset_global_stats_hybrid":
        return PresentPairSetGlobalStatsHybridDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 2496,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 2) or 2,
            role_mixer_depth=int_option(options, "role_mixer_depth", 1) or 1,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            global_hidden_bits=int_option(options, "global_hidden_bits"),
        )
    if name == "present_pairset_global_stats":
        return PresentPairSetGlobalStatsDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 2496,
            base_channels=hidden_bits,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            global_hidden_bits=int_option(options, "global_hidden_bits"),
        )
    if name == "present_trail_position_stats_pairset":
        return PresentTrailPositionStatsPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 2496,
            base_channels=hidden_bits,
            trail_depth=int_option(options, "trail_depth", 4) or 4,
            trail_words_per_depth=int_option(options, "trail_words_per_depth", 9) or 9,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            stats_hidden_bits=int_option(options, "stats_hidden_bits"),
            metadata_bits=int_option(options, "metadata_bits", 0) or 0,
            active_conditioning=str(options.get("active_conditioning", "none")),
            trail_position_control=str(options.get("trail_position_control", "none")),
            trail_normalization=str(options.get("trail_normalization", "none")),
        )
    if name == "present_p_layer_mixer_pairset":
        return PresentPLayerMixerPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
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
            metadata_bits=int_option(options, "metadata_bits", 0) or 0,
            active_conditioning=str(options.get("active_conditioning", "none")),
        )
    return None
