from __future__ import annotations

from typing import Any

import torch

from blockcipher_nd.features.profile import structure_feature_vector
from blockcipher_nd.features.registry import pair_bits_for_encoding
from blockcipher_nd.registry.cipher_profiles import CipherProfile


def configure_structure_aware_model(model: Any, cipher_key: str, rounds: int) -> None:
    if not hasattr(model, "set_structure_features"):
        return
    vector = structure_feature_vector(cipher_profile(cipher_key), rounds)
    model.set_structure_features(torch.tensor(vector, dtype=torch.float32))


def model_metadata(model: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "trainable_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
    }
    for field in (
        "dilations",
        "output_width",
        "output_channels",
        "flattened_width",
        "input_bit_order",
        "l2_coefficient",
    ):
        if hasattr(model, field):
            value = getattr(model, field)
            metadata[field] = list(value) if isinstance(value, tuple) else value
    if not hasattr(model, "gate_summary"):
        return metadata
    summary = model.gate_summary()
    gate_weights = {
        key.removeprefix("gate_weight_"): value
        for key, value in summary.items()
        if key.startswith("gate_weight_")
    }
    return {
        **metadata,
        "gate_mode": summary["gate_mode"],
        "expert_set": summary.get("expert_set", "legacy"),
        "adapter_mode": summary.get("adapter_mode", "none"),
        "adapter_name": summary.get("adapter_name", "identity"),
        "gate_weights_mean": gate_weights,
    }


def infer_pair_bits(block_bits: int, feature_encoding: str) -> int | None:
    try:
        return pair_bits_for_encoding(block_bits, feature_encoding)
    except ValueError:
        return None


def select_model_key(model_key: str, structure: str, pairs_per_sample: int) -> str:
    if model_key not in {"selector_rule", "selector_rule_v2"}:
        return model_key
    if model_key == "selector_rule_v2" and pairs_per_sample > 1:
        return "adaptive_dbitnet_pairwise"
    if structure == "ARX" and pairs_per_sample > 1:
        return "adaptive_dbitnet_pairwise"
    if structure == "ARX":
        return "resnet_bitslice"
    if structure == "SPN":
        return "senet_resnext"
    if structure == "Feistel-like":
        return "multiscale_dense_resnet"
    return "mlp"


def cipher_profile(cipher_key: str) -> CipherProfile:
    mapping = {
        "speck32": CipherProfile.speck32_64,
        "present80": CipherProfile.present80,
        "gift64": CipherProfile.gift64,
        "des": CipherProfile.des,
        "sm4": CipherProfile.sm4,
        "simon64": CipherProfile.simon64_128,
        "simeck64": CipherProfile.simeck64_128,
    }
    try:
        return mapping[cipher_key]()
    except KeyError as exc:
        raise ValueError(f"unsupported cipher key: {cipher_key}") from exc
