from __future__ import annotations

from torch import nn

from blockcipher_nd.features.profile import STRUCTURE_FEATURE_NAMES
from blockcipher_nd.models.structure import StructureAwareMoEDistinguisher
from blockcipher_nd.registry.model_options import (
    MOE_V5_PRESENT_HPO_MULTISEED_TRIAL11_HIDDEN_BITS,
    MOE_V5_PRESENT_HPO_MULTISEED_TRIAL11_OPTIONS,
    MOE_V5_PRESENT_HPO_TRIAL20_HIDDEN_BITS,
    MOE_V5_PRESENT_HPO_TRIAL20_OPTIONS,
    moe_v5_options,
)


def build_moe_model(
    name: str,
    input_bits: int,
    hidden_bits: int,
    pair_bits: int | None,
    options: dict[str, object],
) -> nn.Module | None:
    simple_modes = {
        "moe_uniform": ("uniform", "legacy", {}),
        "moe_hard": ("hard", "legacy", {}),
        "moe_soft": ("soft", "legacy", {}),
        "moe_v2_uniform": ("uniform", "v2_adaptive", {}),
        "moe_v2_hard": ("hard", "v2_adaptive", {}),
        "moe_v2_soft": ("soft", "v2_adaptive", {}),
        "moe_v3_uniform": ("uniform", "v3_pairwise", {"pair_bits": pair_bits}),
        "moe_v3_hard": ("hard", "v3_pairwise", {"pair_bits": pair_bits}),
        "moe_v3_soft": ("soft", "v3_pairwise", {"pair_bits": pair_bits}),
        "moe_v4_uniform": ("uniform", "v4_structure_adapter", {"pair_bits": pair_bits}),
        "moe_v4_hard": ("hard", "v4_structure_adapter", {"pair_bits": pair_bits}),
        "moe_v4_soft": ("soft", "v4_structure_adapter", {"pair_bits": pair_bits}),
    }
    if name in simple_modes:
        gate_mode, expert_set, extra = simple_modes[name]
        kwargs = {} if expert_set == "legacy" else {"expert_set": expert_set}
        kwargs.update(extra)
        return StructureAwareMoEDistinguisher(
            input_bits=input_bits,
            hidden_bits=hidden_bits,
            structure_feature_bits=len(STRUCTURE_FEATURE_NAMES),
            gate_mode=gate_mode,
            **kwargs,
        )

    moe_v5_modes = {
        "moe_v5_uniform": "uniform",
        "moe_v5_hard": "hard",
        "moe_v5_soft": "soft",
    }
    if name in moe_v5_modes:
        return StructureAwareMoEDistinguisher(
            input_bits=input_bits,
            hidden_bits=hidden_bits,
            structure_feature_bits=len(STRUCTURE_FEATURE_NAMES),
            gate_mode=moe_v5_modes[name],
            expert_set="v5_structure_experts",
            pair_bits=pair_bits,
            **moe_v5_options(options),
        )

    if name == "moe_v5_soft_hpo_present_best":
        return StructureAwareMoEDistinguisher(
            input_bits=input_bits,
            hidden_bits=MOE_V5_PRESENT_HPO_TRIAL20_HIDDEN_BITS,
            structure_feature_bits=len(STRUCTURE_FEATURE_NAMES),
            gate_mode="soft",
            expert_set="v5_structure_experts",
            pair_bits=pair_bits or 192,
            **MOE_V5_PRESENT_HPO_TRIAL20_OPTIONS,
        )
    if name == "moe_v5_soft_hpo_multiseed_present_best":
        return StructureAwareMoEDistinguisher(
            input_bits=input_bits,
            hidden_bits=MOE_V5_PRESENT_HPO_MULTISEED_TRIAL11_HIDDEN_BITS,
            structure_feature_bits=len(STRUCTURE_FEATURE_NAMES),
            gate_mode="soft",
            expert_set="v5_structure_experts",
            pair_bits=pair_bits or 192,
            **MOE_V5_PRESENT_HPO_MULTISEED_TRIAL11_OPTIONS,
        )
    return None
