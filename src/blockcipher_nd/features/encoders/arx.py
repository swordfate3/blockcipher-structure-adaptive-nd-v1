from __future__ import annotations

from blockcipher_nd.ciphers import ReducedRoundCipher
from blockcipher_nd.features.arx_aligned import (
    arx_aligned_difference,
    speck32_partial_inverse_feature_words,
    speck32_partial_inverse_rx_carrychain_plus_feature_words,
    speck32_partial_inverse_rx_carrychain_feature_words,
    speck32_partial_inverse_rx_feature_words,
)
from blockcipher_nd.features.encoders.bitwise import int_to_bits, pair_xor_bits


def arx_aligned_pair_xor_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    left_bits, right_bits, difference_bits = pair_xor_bits(left, right, width)
    aligned_difference = arx_aligned_difference(left ^ right, width, cipher)
    return left_bits + right_bits + difference_bits + int_to_bits(aligned_difference, width)


def arx_partial_inverse_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
    *,
    variant: str,
) -> list[int]:
    extractors = {
        "plain": speck32_partial_inverse_feature_words,
        "rx": speck32_partial_inverse_rx_feature_words,
        "rx_carrychain": speck32_partial_inverse_rx_carrychain_feature_words,
        "rx_carrychain_plus": speck32_partial_inverse_rx_carrychain_plus_feature_words,
    }
    left_bits, right_bits, difference_bits = pair_xor_bits(left, right, width)
    extra_bits = []
    for word in extractors[variant](left, right, width, cipher):
        extra_bits.extend(int_to_bits(word, width))
    return left_bits + right_bits + difference_bits + extra_bits
