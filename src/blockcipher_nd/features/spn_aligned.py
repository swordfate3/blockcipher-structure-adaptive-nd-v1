from __future__ import annotations

from blockcipher_nd.ciphers import ReducedRoundCipher


def inverse_permutation_difference(
    difference: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> int:
    inverse_permutation = getattr(cipher, "inverse_permutation_layer", None)
    if inverse_permutation is None or not callable(inverse_permutation):
        raise ValueError(
            "SPN aligned feature encodings require a cipher with "
            "inverse_permutation_layer"
        )
    return int(inverse_permutation(difference)) & ((1 << width) - 1)


def aligned_difference_bits(
    difference: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    from blockcipher_nd.features.pair_features import int_to_bits

    return int_to_bits(inverse_permutation_difference(difference, width, cipher), width)
