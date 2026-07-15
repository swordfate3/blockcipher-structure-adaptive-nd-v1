from __future__ import annotations

from torch import nn

from blockcipher_nd.models.baseline import Sm4Yu2023PositionResNetDistinguisher
from blockcipher_nd.models.structure import (
    BalancedFeistelLuSeNetDistinguisher,
    BalancedFeistelRoundRelationDistinguisher,
    DesFeistelBranchInceptionPairSetDistinguisher,
    DesLstmPairSetDistinguisher,
    DesZhangWangOfficialLayoutDistinguisher,
    DesZhangWangInceptionPairSetDistinguisher,
    Sm4WordRecurrenceDistinguisher,
)
from blockcipher_nd.registry.model_options import int_option, int_tuple_option


def build_feistel_model(
    name: str,
    input_bits: int,
    hidden_bits: int,
    pair_bits: int | None,
    options: dict[str, object],
) -> nn.Module | None:
    common = {
        "pair_bits": pair_bits or 128,
        "base_channels": hidden_bits,
        "blocks": int_option(options, "blocks", 3) or 3,
        "initial_kernel_sizes": int_tuple_option(
            options, "initial_kernel_sizes", (1, 4, 6)
        ),
        "classifier_bits": int_option(options, "classifier_bits", 128) or 128,
        "dropout": float(options.get("dropout", 0.0)),
    }
    balanced_round_relation = {
        "simon_lu_round_relation_true": ("simon", "true"),
        "simon_lu_round_relation_shuffled": ("simon", "shuffled"),
        "simeck_lu_round_relation_true": ("simeck", "true"),
        "simeck_lu_round_relation_shuffled": ("simeck", "shuffled"),
    }
    balanced_lu_senet = {
        "simon_lu_senet_layout_true": ("simon", "true"),
        "simon_lu_senet_layout_shuffled": ("simon", "shuffled"),
        "simeck_lu_senet_layout_true": ("simeck", "true"),
        "simeck_lu_senet_layout_shuffled": ("simeck", "shuffled"),
    }
    if name in balanced_lu_senet:
        round_function, mapping_mode = balanced_lu_senet[name]
        return BalancedFeistelLuSeNetDistinguisher(
            input_bits=input_bits,
            round_function=round_function,
            mapping_mode=mapping_mode,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            classifier_bits=int_option(options, "classifier_bits", 64) or 64,
            se_ratio=int_option(options, "se_ratio", 16) or 16,
        )
    if name in balanced_round_relation:
        round_function, mapping_mode = balanced_round_relation[name]
        return BalancedFeistelRoundRelationDistinguisher(
            input_bits=input_bits,
            round_function=round_function,
            mapping_mode=mapping_mode,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 3) or 3,
            classifier_bits=int_option(options, "classifier_bits", 64) or 64,
            dropout=float(options.get("dropout", 0.0)),
        )
    if name == "des_feistel_branch_inception_true":
        return DesFeistelBranchInceptionPairSetDistinguisher(
            input_bits=input_bits,
            mapping_mode="true",
            **common,
        )
    if name == "des_feistel_branch_inception_shuffled":
        return DesFeistelBranchInceptionPairSetDistinguisher(
            input_bits=input_bits,
            mapping_mode="shuffled",
            **common,
        )
    if name == "des_zhang_wang_inception_pairset":
        return DesZhangWangInceptionPairSetDistinguisher(
            input_bits=input_bits,
            **common,
        )
    if name == "des_lstm_pairset":
        return DesLstmPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            hidden_bits=int_option(options, "lstm_hidden_bits", 128) or 128,
            classifier_bits=int_option(options, "classifier_bits", 128) or 128,
        )
    if name in {
        "des_zhang_wang_official_layout",
        "des_zhang_wang_official_layout_shuffled",
        "des_feistel_official_backbone_true",
        "des_feistel_official_backbone_shuffled",
    }:
        return DesZhangWangOfficialLayoutDistinguisher(
            input_bits=input_bits,
            mapping_mode=(
                "shuffled"
                if name
                in {
                    "des_feistel_official_backbone_shuffled",
                    "des_zhang_wang_official_layout_shuffled",
                }
                else "true"
            ),
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            initial_kernel_sizes=int_tuple_option(
                options, "initial_kernel_sizes", (1, 4, 6)
            ),
            include_branch_interactions=name
            not in {
                "des_zhang_wang_official_layout",
                "des_zhang_wang_official_layout_shuffled",
            },
        )
    if name in {"sm4_word_recurrence_true", "sm4_word_recurrence_shuffled"}:
        return Sm4WordRecurrenceDistinguisher(
            input_bits=input_bits,
            mapping_mode=(
                "shuffled" if name == "sm4_word_recurrence_shuffled" else "true"
            ),
            pair_bits=pair_bits or 256,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 3) or 3,
            classifier_bits=int_option(options, "classifier_bits", 64) or 64,
            dropout=float(options.get("dropout", 0.5)),
            rotation_offsets=int_tuple_option(
                options, "rotation_offsets", (2, 10, 18, 24)
            ),
        )
    if name == "sm4_yu2023_position_resnet":
        return Sm4Yu2023PositionResNetDistinguisher(
            input_bits=input_bits,
            channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            classifier_bits=int_option(options, "classifier_bits", 64) or 64,
            dropout=float(options.get("dropout", 0.5)),
        )
    return None


__all__ = ["build_feistel_model"]
