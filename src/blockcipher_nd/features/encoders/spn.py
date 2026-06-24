from __future__ import annotations

from blockcipher_nd.ciphers import ReducedRoundCipher
from blockcipher_nd.features.encoders.bitwise import int_to_bits, pair_xor_bits
from blockcipher_nd.features.spn_aligned import inverse_permutation_difference


def spn_aligned_xor_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    return int_to_bits(difference, width) + int_to_bits(
        inverse_permutation_difference(difference, width, cipher),
        width,
    )


def spn_aligned_pair_xor_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    left_bits, right_bits, difference_bits = pair_xor_bits(left, right, width)
    aligned_difference = inverse_permutation_difference(left ^ right, width, cipher)
    return left_bits + right_bits + difference_bits + int_to_bits(aligned_difference, width)
