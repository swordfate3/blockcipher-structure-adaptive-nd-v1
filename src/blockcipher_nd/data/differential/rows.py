from __future__ import annotations

import numpy as np

from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.data.differential.encoding import encode_pair
from blockcipher_nd.data.differential.keys import cipher_for_row, random_cipher_for_pair
from blockcipher_nd.data.differential.random import random_int


def generate_positive_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
    row_index: int = 0,
) -> list[int]:
    if config.sample_structure in {
        "plaintext_integral_nibble",
        "plaintext_integral_nibble_matched_negative",
    }:
        return _generate_integral_positive_row(config, rng, block_bits, mask, row_index)
    if config.sample_structure == "zhang_wang_case2_mcnd":
        return _generate_zhang_wang_case2_positive_row(config, rng, block_bits, mask, row_index)
    if config.sample_structure == "zhang_wang_case2_official_mcnd":
        return _generate_official_mcnd_positive_row(config, rng, block_bits, mask)
    if config.sample_structure == "zhang_wang_case2_independent_mcnd":
        return _generate_independent_positive_row(config, rng, block_bits, mask, row_index)
    return _generate_independent_positive_row(config, rng, block_bits, mask, row_index)


def generate_negative_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    row_index: int = 0,
) -> list[int]:
    if config.sample_structure == "plaintext_integral_nibble":
        return _generate_integral_negative_row(config, rng, block_bits, row_index)
    if config.sample_structure == "plaintext_integral_nibble_matched_negative":
        return _generate_integral_matched_negative_row(config, rng, block_bits, row_index)
    if config.sample_structure == "zhang_wang_case2_mcnd":
        return _generate_zhang_wang_case2_negative_row(config, rng, block_bits, row_index)
    if config.sample_structure == "zhang_wang_case2_official_mcnd":
        return _generate_official_mcnd_negative_row(config, rng, block_bits)
    if config.sample_structure == "zhang_wang_case2_independent_mcnd":
        return _generate_independent_negative_row(config, rng, block_bits, row_index)
    return _generate_independent_negative_row(config, rng, block_bits, row_index)


def _generate_independent_positive_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    for _pair_index in range(config.pairs_per_sample):
        plaintext = random_int(rng, block_bits)
        paired = (plaintext ^ config.input_difference) & mask
        ciphertext_a = cipher.encrypt(plaintext)
        ciphertext_b = cipher.encrypt(paired)
        encoded_pairs.extend(encode_pair(ciphertext_a, ciphertext_b, block_bits, config, cipher))
    return encoded_pairs


def _generate_independent_negative_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    for _pair_index in range(config.pairs_per_sample):
        if config.negative_mode == "random_ciphertext":
            ciphertext_a = random_int(rng, block_bits)
            ciphertext_b = random_int(rng, block_bits)
        else:
            plaintext_a = random_int(rng, block_bits)
            plaintext_b = random_int(rng, block_bits)
            ciphertext_a = cipher.encrypt(plaintext_a)
            ciphertext_b = cipher.encrypt(plaintext_b)
        encoded_pairs.extend(encode_pair(ciphertext_a, ciphertext_b, block_bits, config, cipher))
    return encoded_pairs


def _generate_official_mcnd_positive_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    for _pair_index in range(config.pairs_per_sample):
        cipher = random_cipher_for_pair(config, rng)
        plaintext = random_int(rng, block_bits)
        paired = (plaintext ^ config.input_difference) & mask
        ciphertext_a = cipher.encrypt(plaintext)
        ciphertext_b = cipher.encrypt(paired)
        encoded_pairs.extend(encode_pair(ciphertext_a, ciphertext_b, block_bits, config, cipher))
    return encoded_pairs


def _generate_official_mcnd_negative_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    for _pair_index in range(config.pairs_per_sample):
        if config.negative_mode == "random_ciphertext":
            ciphertext_a = random_int(rng, block_bits)
            ciphertext_b = random_int(rng, block_bits)
            cipher = config.cipher
        else:
            cipher = random_cipher_for_pair(config, rng)
            plaintext_a = random_int(rng, block_bits)
            plaintext_b = random_int(rng, block_bits)
            ciphertext_a = cipher.encrypt(plaintext_a)
            ciphertext_b = cipher.encrypt(plaintext_b)
        encoded_pairs.extend(encode_pair(ciphertext_a, ciphertext_b, block_bits, config, cipher))
    return encoded_pairs


def _generate_integral_positive_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    base = _integral_base_plaintext(config, rng, block_bits)
    for variant in _integral_variants(config):
        plaintext = base | variant
        paired = (plaintext ^ config.input_difference) & mask
        encoded_pairs.extend(
            encode_pair(cipher.encrypt(plaintext), cipher.encrypt(paired), block_bits, config, cipher)
        )
    return encoded_pairs


def _generate_integral_negative_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    base = _integral_base_plaintext(config, rng, block_bits)
    for variant in _integral_variants(config):
        plaintext = base | variant
        if config.negative_mode == "random_ciphertext":
            ciphertext_a = random_int(rng, block_bits)
            ciphertext_b = random_int(rng, block_bits)
        else:
            plaintext_b = random_int(rng, block_bits)
            ciphertext_a = cipher.encrypt(plaintext)
            ciphertext_b = cipher.encrypt(plaintext_b)
        encoded_pairs.extend(encode_pair(ciphertext_a, ciphertext_b, block_bits, config, cipher))
    return encoded_pairs


def _generate_integral_matched_negative_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    base = _integral_base_plaintext(config, rng, block_bits)
    variants = _integral_variants(config)
    paired_variants = variants[1:] + variants[:1]
    for variant, paired_variant in zip(variants, paired_variants, strict=True):
        plaintext_a = base | variant
        plaintext_b = base | paired_variant
        encoded_pairs.extend(
            encode_pair(cipher.encrypt(plaintext_a), cipher.encrypt(plaintext_b), block_bits, config, cipher)
        )
    return encoded_pairs


def _generate_zhang_wang_case2_positive_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    base = random_int(rng, block_bits)
    for mask_delta in _mcnd_plaintext_masks(config, rng, block_bits):
        plaintext = (base ^ mask_delta) & mask
        paired = (plaintext ^ config.input_difference) & mask
        encoded_pairs.extend(
            encode_pair(cipher.encrypt(plaintext), cipher.encrypt(paired), block_bits, config, cipher)
        )
    return encoded_pairs


def _generate_zhang_wang_case2_negative_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    base = random_int(rng, block_bits)
    for mask_delta in _mcnd_plaintext_masks(config, rng, block_bits):
        if config.negative_mode == "random_ciphertext":
            ciphertext_a = random_int(rng, block_bits)
            ciphertext_b = random_int(rng, block_bits)
        else:
            plaintext_a = base ^ mask_delta
            plaintext_b = random_int(rng, block_bits)
            ciphertext_a = cipher.encrypt(plaintext_a)
            ciphertext_b = cipher.encrypt(plaintext_b)
        encoded_pairs.extend(encode_pair(ciphertext_a, ciphertext_b, block_bits, config, cipher))
    return encoded_pairs


def _mcnd_plaintext_masks(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
) -> list[int]:
    if config.pairs_per_sample == 1:
        return [0]
    masks = {0}
    while len(masks) < config.pairs_per_sample:
        masks.add(random_int(rng, block_bits))
    return list(masks)


def _integral_base_plaintext(
    config: DifferentialDatasetConfig, rng: np.random.Generator, block_bits: int
) -> int:
    active_mask = _integral_active_mask(config)
    return random_int(rng, block_bits) & ~active_mask


def _integral_variants(config: DifferentialDatasetConfig) -> list[int]:
    shift = config.integral_active_nibble * 4
    return [value << shift for value in range(config.pairs_per_sample)]


def _integral_active_mask(config: DifferentialDatasetConfig) -> int:
    shift = config.integral_active_nibble * 4
    return ((1 << (config.pairs_per_sample.bit_length() - 1)) - 1) << shift
