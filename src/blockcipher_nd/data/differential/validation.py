from __future__ import annotations

from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.features.pair_features import pair_bits_for_encoding


def validate_differential_config(config: DifferentialDatasetConfig) -> None:
    if config.pairs_per_sample < 1:
        raise ValueError("pairs_per_sample must be at least 1")
    if config.key_rotation_interval < 0:
        raise ValueError("key_rotation_interval must be non-negative")
    if config.negative_mode not in {"random_ciphertext", "encrypted_random_plaintexts"}:
        raise ValueError(f"unsupported negative_mode: {config.negative_mode}")
    if config.sample_structure not in {
        "independent_pairs",
        "plaintext_integral_nibble",
        "plaintext_integral_nibble_difference_matched_negative",
        "plaintext_integral_nibble_difference_matched_negative_pair_shuffled",
        "plaintext_integral_nibble_difference_matched_negative_partial8",
        "plaintext_integral_nibble_difference_matched_negative_random_active",
        "plaintext_integral_nibble_difference_matched_negative_random_active_metadata",
        "plaintext_integral_nibble_difference_matched_negative_random_active_relative",
        "plaintext_integral_nibble_matched_negative",
        "plaintext_integral_nibble_same_difference_random_negative",
        "plaintext_integral_nibble_strict_random_negative",
        "plaintext_integral_multi_nibble_difference_matched_negative",
        "plaintext_integral_nibble_scrambled_positive",
        "zhang_wang_case2_mcnd",
        "zhang_wang_case2_independent_mcnd",
        "zhang_wang_case2_official_mcnd",
    }:
        raise ValueError(f"unsupported sample_structure: {config.sample_structure}")
    if config.sample_structure in {
        "plaintext_integral_nibble",
        "plaintext_integral_nibble_difference_matched_negative",
        "plaintext_integral_nibble_difference_matched_negative_pair_shuffled",
        "plaintext_integral_nibble_difference_matched_negative_partial8",
        "plaintext_integral_nibble_difference_matched_negative_random_active",
        "plaintext_integral_nibble_difference_matched_negative_random_active_metadata",
        "plaintext_integral_nibble_difference_matched_negative_random_active_relative",
        "plaintext_integral_nibble_matched_negative",
        "plaintext_integral_nibble_same_difference_random_negative",
        "plaintext_integral_nibble_strict_random_negative",
        "plaintext_integral_nibble_scrambled_positive",
    }:
        if config.pairs_per_sample < 2 or config.pairs_per_sample & (config.pairs_per_sample - 1):
            raise ValueError("plaintext_integral_nibble requires power-of-two pairs_per_sample >= 2")
        if config.pairs_per_sample > 16:
            raise ValueError("plaintext_integral_nibble currently supports at most one active nibble")
        max_nibble = config.cipher.block_bits // 4
        if config.integral_active_nibble < 0 or config.integral_active_nibble >= max_nibble:
            raise ValueError("integral_active_nibble is outside the cipher block")
        for active_nibble in config.integral_active_nibbles:
            if active_nibble < 0 or active_nibble >= max_nibble:
                raise ValueError("integral_active_nibbles contains a nibble outside the cipher block")
        if len(set(config.integral_active_nibbles)) != len(config.integral_active_nibbles):
            raise ValueError("integral_active_nibbles must not contain duplicates")
    if (
        config.sample_structure
        == "plaintext_integral_nibble_difference_matched_negative_partial8"
        and config.pairs_per_sample != 16
    ):
        raise ValueError(
            "plaintext_integral_nibble_difference_matched_negative_partial8 "
            "requires pairs_per_sample=16"
        )
    if config.sample_structure == "plaintext_integral_multi_nibble_difference_matched_negative":
        active_nibbles = _nonzero_nibble_support(config.input_difference, config.cipher.block_bits)
        if not active_nibbles:
            raise ValueError("multi-nibble integral samples require a nonzero input_difference")
        expected_pairs = 16 ** len(active_nibbles)
        if config.pairs_per_sample != expected_pairs:
            raise ValueError(
                "plaintext_integral_multi_nibble_difference_matched_negative "
                f"requires pairs_per_sample={expected_pairs} for {len(active_nibbles)} active nibbles"
            )
    if config.sample_structure in {
        "zhang_wang_case2_mcnd",
        "zhang_wang_case2_independent_mcnd",
        "zhang_wang_case2_official_mcnd",
    } and config.pairs_per_sample < 1:
        raise ValueError(f"{config.sample_structure} requires pairs_per_sample >= 1")
    base_pair_bits = pair_bits_for_encoding(config.cipher.block_bits, config.feature_encoding)
    for index in config.selected_bit_indices:
        if index < 0 or index >= base_pair_bits:
            raise ValueError("selected_bit_indices must reference encoded pair bits")


def _nonzero_nibble_support(value: int, block_bits: int) -> tuple[int, ...]:
    return tuple(
        nibble_index
        for nibble_index in range(block_bits // 4)
        if ((value >> (4 * nibble_index)) & 0xF) != 0
    )
