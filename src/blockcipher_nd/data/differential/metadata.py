from __future__ import annotations

from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.features.pair_features import pair_bits_for_encoding


def dataset_metadata(config: DifferentialDatasetConfig) -> dict[str, int | str | bool | list[int]]:
    block_bits = config.cipher.block_bits
    return {
        "cipher": config.cipher.name,
        "structure": config.cipher.structure,
        "rounds": config.cipher.rounds,
        "block_bits": block_bits,
        "input_difference": config.input_difference,
        "samples_per_class": config.samples_per_class,
        "seed": config.seed,
        "shuffle": config.shuffle,
        "feature_encoding": config.feature_encoding,
        "pairs_per_sample": config.pairs_per_sample,
        "negative_mode": config.negative_mode,
        "key_rotation_interval": config.key_rotation_interval,
        "key_schedule": _key_schedule(config),
        "sample_structure": config.sample_structure,
        "integral_active_nibble": config.integral_active_nibble,
        "integral_active_nibbles": list(config.integral_active_nibbles),
        "row_metadata_bits": row_metadata_bits(config, block_bits),
        "pair_bits": effective_pair_bits(config, block_bits),
        "base_pair_bits": pair_bits_for_encoding(block_bits, config.feature_encoding),
        "selected_bit_indices": list(config.selected_bit_indices),
    }


def effective_pair_bits(config: DifferentialDatasetConfig, block_bits: int) -> int:
    if config.selected_bit_indices:
        return len(config.selected_bit_indices)
    return pair_bits_for_encoding(block_bits, config.feature_encoding)


def row_metadata_bits(config: DifferentialDatasetConfig, block_bits: int) -> int:
    if (
        config.sample_structure
        == "plaintext_integral_nibble_difference_matched_negative_random_active_metadata"
    ):
        return block_bits // 4
    return 0


def _key_schedule(config: DifferentialDatasetConfig) -> str:
    if config.sample_structure == "zhang_wang_case2_official_mcnd":
        return "per_pair_random"
    if config.key_rotation_interval > 0:
        return "rotating"
    return "fixed"
