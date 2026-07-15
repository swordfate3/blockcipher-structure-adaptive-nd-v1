from __future__ import annotations

from torch import nn

from blockcipher_nd.models.structure import (
    DesFeistelBranchInceptionPairSetDistinguisher,
    DesLstmPairSetDistinguisher,
    DesZhangWangInceptionPairSetDistinguisher,
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
    return None


__all__ = ["build_feistel_model"]
