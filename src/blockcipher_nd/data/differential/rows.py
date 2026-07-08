from __future__ import annotations

from dataclasses import replace

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
        "plaintext_integral_nibble_difference_matched_negative",
        "plaintext_integral_nibble_matched_negative",
        "plaintext_integral_nibble_same_difference_random_negative",
        "plaintext_integral_nibble_strict_random_negative",
    }:
        return _generate_integral_positive_row(config, rng, block_bits, mask, row_index)
    if config.sample_structure == "plaintext_integral_nibble_difference_matched_negative_pair_shuffled":
        row = _generate_integral_positive_row(config, rng, block_bits, mask, row_index)
        return _shuffle_encoded_pair_order(row, config, rng)
    if config.sample_structure == "plaintext_integral_nibble_difference_matched_negative_random_active":
        active_config = _with_sampled_active_nibble(config, rng, block_bits)
        return _generate_integral_positive_row(active_config, rng, block_bits, mask, row_index)
    if config.sample_structure == "plaintext_integral_nibble_difference_matched_negative_random_active_metadata":
        active_config = _with_sampled_active_nibble(config, rng, block_bits)
        row = _generate_integral_positive_row(active_config, rng, block_bits, mask, row_index)
        return row + _active_nibble_one_hot(active_config.integral_active_nibble, block_bits)
    if config.sample_structure == "plaintext_integral_nibble_difference_matched_negative_random_active_relative":
        active_config = _with_sampled_active_nibble(config, rng, block_bits)
        row = _generate_integral_positive_row(active_config, rng, block_bits, mask, row_index)
        return _relative_active_nibble_row(row, active_config, block_bits)
    if config.sample_structure == "plaintext_integral_nibble_difference_matched_negative_partial8":
        return _generate_integral_partial_positive_row(config, rng, block_bits, mask, row_index)
    if config.sample_structure == "plaintext_integral_multi_nibble_difference_matched_negative":
        return _generate_integral_multi_nibble_positive_row(config, rng, block_bits, mask, row_index)
    if config.sample_structure == "plaintext_integral_nibble_scrambled_positive":
        return _generate_integral_scrambled_positive_row(config, rng, block_bits, mask, row_index)
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
    if config.sample_structure == "plaintext_integral_nibble_difference_matched_negative":
        mask = (1 << block_bits) - 1
        return _generate_integral_difference_matched_negative_row(config, rng, block_bits, mask, row_index)
    if config.sample_structure == "plaintext_integral_nibble_difference_matched_negative_pair_shuffled":
        mask = (1 << block_bits) - 1
        row = _generate_integral_difference_matched_negative_row(config, rng, block_bits, mask, row_index)
        return _shuffle_encoded_pair_order(row, config, rng)
    if config.sample_structure == "plaintext_integral_nibble_difference_matched_negative_random_active":
        mask = (1 << block_bits) - 1
        active_config = _with_sampled_active_nibble(config, rng, block_bits)
        return _generate_integral_difference_matched_negative_row(
            active_config, rng, block_bits, mask, row_index
        )
    if config.sample_structure == "plaintext_integral_nibble_difference_matched_negative_random_active_metadata":
        mask = (1 << block_bits) - 1
        active_config = _with_sampled_active_nibble(config, rng, block_bits)
        row = _generate_integral_difference_matched_negative_row(
            active_config, rng, block_bits, mask, row_index
        )
        return row + _active_nibble_one_hot(active_config.integral_active_nibble, block_bits)
    if config.sample_structure == "plaintext_integral_nibble_difference_matched_negative_random_active_relative":
        mask = (1 << block_bits) - 1
        active_config = _with_sampled_active_nibble(config, rng, block_bits)
        row = _generate_integral_difference_matched_negative_row(
            active_config, rng, block_bits, mask, row_index
        )
        return _relative_active_nibble_row(row, active_config, block_bits)
    if config.sample_structure == "plaintext_integral_nibble_difference_matched_negative_partial8":
        mask = (1 << block_bits) - 1
        return _generate_integral_partial_negative_row(config, rng, block_bits, mask, row_index)
    if config.sample_structure == "plaintext_integral_nibble_strict_random_negative":
        return _generate_independent_negative_row(config, rng, block_bits, row_index)
    if config.sample_structure == "plaintext_integral_nibble_same_difference_random_negative":
        mask = (1 << block_bits) - 1
        return _generate_integral_same_difference_random_negative_row(
            config, rng, block_bits, mask, row_index
        )
    if config.sample_structure == "plaintext_integral_multi_nibble_difference_matched_negative":
        mask = (1 << block_bits) - 1
        return _generate_integral_multi_nibble_difference_matched_negative_row(
            config, rng, block_bits, mask, row_index
        )
    if config.sample_structure == "plaintext_integral_nibble_scrambled_positive":
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


def _generate_integral_difference_matched_negative_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    base = _integral_base_plaintext(config, rng, block_bits)
    variants = _integral_variants(config)
    paired_variants = variants[1:] + variants[:1]
    for variant, paired_variant in zip(variants, paired_variants, strict=True):
        plaintext_a = base | variant
        plaintext_b = ((base | paired_variant) ^ config.input_difference) & mask
        encoded_pairs.extend(
            encode_pair(cipher.encrypt(plaintext_a), cipher.encrypt(plaintext_b), block_bits, config, cipher)
        )
    return encoded_pairs


def _generate_integral_same_difference_random_negative_row(
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
        encoded_pairs.extend(
            encode_pair(cipher.encrypt(plaintext), cipher.encrypt(paired), block_bits, config, cipher)
        )
    return encoded_pairs


def _generate_integral_partial_positive_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    base = _integral_base_plaintext(config, rng, block_bits)
    for variant in _integral_variants(config)[:8]:
        plaintext = base | variant
        paired = (plaintext ^ config.input_difference) & mask
        encoded_pairs.extend(
            encode_pair(cipher.encrypt(plaintext), cipher.encrypt(paired), block_bits, config, cipher)
        )
    for _pair_index in range(config.pairs_per_sample - 8):
        plaintext = random_int(rng, block_bits)
        paired = (plaintext ^ config.input_difference) & mask
        encoded_pairs.extend(
            encode_pair(cipher.encrypt(plaintext), cipher.encrypt(paired), block_bits, config, cipher)
        )
    return encoded_pairs


def _generate_integral_partial_negative_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    base = _integral_base_plaintext(config, rng, block_bits)
    variants = _integral_variants(config)[:8]
    paired_variants = variants[1:] + variants[:1]
    for variant, paired_variant in zip(variants, paired_variants, strict=True):
        plaintext_a = base | variant
        plaintext_b = ((base | paired_variant) ^ config.input_difference) & mask
        encoded_pairs.extend(
            encode_pair(cipher.encrypt(plaintext_a), cipher.encrypt(plaintext_b), block_bits, config, cipher)
        )
    for _pair_index in range(config.pairs_per_sample - 8):
        plaintext = random_int(rng, block_bits)
        paired = (plaintext ^ config.input_difference) & mask
        encoded_pairs.extend(
            encode_pair(cipher.encrypt(plaintext), cipher.encrypt(paired), block_bits, config, cipher)
        )
    return encoded_pairs


def _generate_integral_multi_nibble_positive_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    base = _integral_multi_nibble_base_plaintext(config, rng, block_bits)
    for variant in _integral_multi_nibble_variants(config, block_bits):
        plaintext = base | variant
        paired = (plaintext ^ config.input_difference) & mask
        encoded_pairs.extend(
            encode_pair(cipher.encrypt(plaintext), cipher.encrypt(paired), block_bits, config, cipher)
        )
    return encoded_pairs


def _generate_integral_multi_nibble_difference_matched_negative_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    base = _integral_multi_nibble_base_plaintext(config, rng, block_bits)
    variants = _integral_multi_nibble_variants(config, block_bits)
    paired_variants = variants[1:] + variants[:1]
    for variant, paired_variant in zip(variants, paired_variants, strict=True):
        plaintext_a = base | variant
        plaintext_b = ((base | paired_variant) ^ config.input_difference) & mask
        encoded_pairs.extend(
            encode_pair(cipher.encrypt(plaintext_a), cipher.encrypt(plaintext_b), block_bits, config, cipher)
        )
    return encoded_pairs


def _generate_integral_scrambled_positive_row(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
    row_index: int,
) -> list[int]:
    encoded_pairs: list[int] = []
    cipher = cipher_for_row(config, rng, row_index)
    base = _integral_base_plaintext(config, rng, block_bits)
    variants = _integral_variants(config)
    paired_variants = variants[1:] + variants[:1]
    for variant, paired_variant in zip(variants, paired_variants, strict=True):
        plaintext_a = base | variant
        plaintext_b = ((base | paired_variant) ^ config.input_difference) & mask
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


def _with_sampled_active_nibble(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
) -> DifferentialDatasetConfig:
    choices = config.integral_active_nibbles or tuple(range(block_bits // 4))
    active_nibble = int(choices[int(rng.integers(0, len(choices)))])
    return replace(config, integral_active_nibble=active_nibble)


def _active_nibble_one_hot(active_nibble: int, block_bits: int) -> list[int]:
    nibbles = block_bits // 4
    return [1 if index == active_nibble else 0 for index in range(nibbles)]


def _relative_active_nibble_row(
    encoded_pairs: list[int],
    config: DifferentialDatasetConfig,
    block_bits: int,
) -> list[int]:
    if block_bits % 4 != 0:
        raise ValueError("relative active-nibble rows require 4-bit cells")
    pair_width = len(encoded_pairs) // config.pairs_per_sample
    if pair_width * config.pairs_per_sample != len(encoded_pairs):
        raise ValueError("encoded row width is not divisible by pairs_per_sample")
    if pair_width % block_bits != 0:
        raise ValueError("relative active-nibble rows require whole block-width feature words")

    aligned: list[int] = []
    for pair_index in range(config.pairs_per_sample):
        pair = encoded_pairs[pair_index * pair_width : (pair_index + 1) * pair_width]
        aligned.extend(_rotate_feature_cells_to_active_zero(pair, config.integral_active_nibble, block_bits))
    return aligned


def _rotate_feature_cells_to_active_zero(
    pair: list[int],
    active_nibble: int,
    block_bits: int,
) -> list[int]:
    cells_per_word = block_bits // 4
    cell_bits = 4
    word_width = cells_per_word * cell_bits
    words = len(pair) // word_width
    rotated: list[int] = []
    for word_index in range(words):
        word = pair[word_index * word_width : (word_index + 1) * word_width]
        cells = [
            word[cell_index * cell_bits : (cell_index + 1) * cell_bits]
            for cell_index in range(cells_per_word)
        ]
        shifted = [None] * cells_per_word
        for cell_index, cell in enumerate(cells):
            shifted[(cell_index + active_nibble) % cells_per_word] = cell
        for cell in shifted:
            if cell is None:
                raise ValueError("failed to rotate active-nibble feature cells")
            rotated.extend(cell)
    return rotated


def _shuffle_encoded_pair_order(
    encoded_pairs: list[int],
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
) -> list[int]:
    pair_width = len(encoded_pairs) // config.pairs_per_sample
    if pair_width * config.pairs_per_sample != len(encoded_pairs):
        raise ValueError("encoded row width is not divisible by pairs_per_sample")
    chunks = [
        encoded_pairs[index * pair_width : (index + 1) * pair_width]
        for index in range(config.pairs_per_sample)
    ]
    shuffled: list[int] = []
    for index in rng.permutation(config.pairs_per_sample):
        shuffled.extend(chunks[int(index)])
    return shuffled


def _integral_multi_nibble_base_plaintext(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
) -> int:
    return random_int(rng, block_bits) & ~_integral_multi_nibble_active_mask(config, block_bits)


def _integral_multi_nibble_variants(
    config: DifferentialDatasetConfig,
    block_bits: int,
) -> list[int]:
    active_nibbles = _nonzero_nibble_support(config.input_difference, block_bits)
    variants = [0]
    for nibble_index in active_nibbles:
        shift = 4 * nibble_index
        variants = [variant | (value << shift) for variant in variants for value in range(16)]
    return variants


def _integral_multi_nibble_active_mask(config: DifferentialDatasetConfig, block_bits: int) -> int:
    mask = 0
    for nibble_index in _nonzero_nibble_support(config.input_difference, block_bits):
        mask |= 0xF << (4 * nibble_index)
    return mask


def _nonzero_nibble_support(value: int, block_bits: int) -> tuple[int, ...]:
    return tuple(
        nibble_index
        for nibble_index in range(block_bits // 4)
        if ((value >> (4 * nibble_index)) & 0xF) != 0
    )
