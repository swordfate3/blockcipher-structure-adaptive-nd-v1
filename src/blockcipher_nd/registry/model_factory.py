from __future__ import annotations

from torch import nn

from blockcipher_nd.registry.model_families.arx import build_arx_model
from blockcipher_nd.registry.model_families.baseline import build_baseline_model
from blockcipher_nd.registry.model_families.feistel import build_feistel_model
from blockcipher_nd.registry.model_families.moe import build_moe_model
from blockcipher_nd.registry.model_families.pairset import build_pairset_model
from blockcipher_nd.registry.model_families.spn import build_spn_model


def build_model(
    name: str,
    input_bits: int,
    hidden_bits: int,
    pair_bits: int | None = None,
    structure: str = "generic",
    model_options: dict[str, object] | None = None,
) -> nn.Module:
    options = model_options or {}
    builders = (
        lambda: build_baseline_model(name, input_bits, hidden_bits),
        lambda: build_pairset_model(name, input_bits, hidden_bits, pair_bits, structure),
        lambda: build_arx_model(name, input_bits, hidden_bits, pair_bits, structure, options),
        lambda: build_spn_model(name, input_bits, hidden_bits, pair_bits, options),
        lambda: build_feistel_model(name, input_bits, hidden_bits, pair_bits, options),
        lambda: build_moe_model(name, input_bits, hidden_bits, pair_bits, options),
    )
    for build in builders:
        model = build()
        if model is not None:
            return model
    raise ValueError(f"unsupported model: {name}")
