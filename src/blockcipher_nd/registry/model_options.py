from __future__ import annotations

MOE_V5_PRESENT_HPO_TRIAL20_OPTIONS = {
    "gate_hidden_bits": 96,
    "gate_activation": "silu",
    "gate_dropout": 0.0,
    "gate_temperature": 1.0,
    "pairwise_pooling": "mean_max",
    "spn_token_dim": 128,
    "spn_mixer_depth": 4,
    "spn_token_mlp_ratio": 2,
    "expert_activation": "gelu",
    "expert_norm": "rmsnorm",
    "spn_pooling": "gated_attention",
    "expert_dropout": 0.05,
}
MOE_V5_PRESENT_HPO_TRIAL20_HIDDEN_BITS = 96
MOE_V5_PRESENT_HPO_MULTISEED_TRIAL11_OPTIONS = {
    "gate_hidden_bits": 32,
    "gate_activation": "relu",
    "gate_dropout": 0.05,
    "gate_temperature": 0.75,
    "pairwise_pooling": "mean",
    "spn_token_dim": 64,
    "spn_mixer_depth": 3,
    "spn_token_mlp_ratio": 3,
    "expert_activation": "silu",
    "expert_norm": "rmsnorm",
    "spn_pooling": "gated_attention",
    "expert_dropout": 0.0,
}
MOE_V5_PRESENT_HPO_MULTISEED_TRIAL11_HIDDEN_BITS = 96


def moe_v5_options(options: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    int_keys = {
        "gate_hidden_bits",
        "spn_token_dim",
        "spn_mixer_depth",
        "spn_token_mlp_ratio",
    }
    float_keys = {"gate_dropout", "gate_temperature", "expert_dropout"}
    string_keys = {
        "gate_activation",
        "pairwise_pooling",
        "expert_activation",
        "expert_norm",
        "spn_pooling",
    }
    aliases = {
        "token_dim": "spn_token_dim",
        "mixer_depth": "spn_mixer_depth",
        "pooling": "spn_pooling",
        "dropout": "expert_dropout",
    }
    expanded_options = dict(options)
    for key, target in aliases.items():
        if key in expanded_options and target not in expanded_options:
            expanded_options[target] = expanded_options[key]
    for key in int_keys:
        if key in expanded_options:
            result[key] = int(expanded_options[key])
    for key in float_keys:
        if key in expanded_options:
            result[key] = float(expanded_options[key])
    for key in string_keys:
        if key in expanded_options:
            result[key] = str(expanded_options[key])
    return result


def int_tuple_option(
    options: dict[str, object],
    key: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    value = options.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"model option {key} must be a list or tuple")
    result = tuple(int(item) for item in value)
    if not result:
        raise ValueError(f"model option {key} must not be empty")
    return result


def matrix_kernel_size_option(value: object) -> tuple[int, int]:
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError("matrix kernel sizes must be ints or [height, width] pairs")


def int_option(options: dict[str, object], key: str, default: int | None = None) -> int | None:
    value = options.get(key, default)
    if value is None:
        return None
    return int(value)
