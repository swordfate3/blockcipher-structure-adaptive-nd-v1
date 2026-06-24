from __future__ import annotations

from blockcipher_nd.ciphers import ReducedRoundCipher
from blockcipher_nd.ciphers.base import rol, ror


def speck32_rotation_aligned_difference(difference: int, width: int) -> int:
    if width != 32:
        raise ValueError("SPECK32/64 ARX aligned feature encoding requires 32-bit blocks")
    word_bits = 16
    mask = (1 << word_bits) - 1
    delta_left = (difference >> word_bits) & mask
    delta_right = difference & mask
    aligned_left = ror(delta_left, 7, word_bits)
    aligned_right = rol(delta_right, 2, word_bits)
    return (aligned_left << word_bits) | aligned_right


def _require_speck32(width: int, cipher: ReducedRoundCipher) -> None:
    if getattr(cipher, "name", "") != "SPECK32/64":
        raise ValueError(
            "ARX aligned feature encoding currently supports SPECK32/64; "
            f"got {getattr(cipher, 'name', type(cipher).__name__)}"
        )
    if width != 32:
        raise ValueError("SPECK32/64 ARX feature encoding requires 32-bit blocks")


def speck32_partial_inverse_words(left: int, right: int, width: int) -> tuple[int, int, int]:
    if width != 32:
        raise ValueError("SPECK32/64 partial-inverse encoding requires 32-bit blocks")
    mask = 0xFFFF
    x = (left >> 16) & mask
    y = left & mask
    x_prime = (right >> 16) & mask
    y_prime = right & mask
    pre_y = ror(y ^ x, 2, 16)
    pre_y_prime = ror(y_prime ^ x_prime, 2, 16)
    return pre_y, pre_y_prime, pre_y ^ pre_y_prime


def speck32_partial_inverse_feature_words(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> tuple[int, ...]:
    _require_speck32(width, cipher)
    rotation_aligned = speck32_rotation_aligned_difference(left ^ right, width)
    pre_y, pre_y_prime, delta_pre_y = speck32_partial_inverse_words(left, right, width)
    return (rotation_aligned, pre_y, pre_y_prime, delta_pre_y)


def speck32_partial_inverse_rx_feature_words(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> tuple[int, ...]:
    _require_speck32(width, cipher)
    base_words = speck32_partial_inverse_feature_words(left, right, width, cipher)
    mask = 0xFFFF
    x = (left >> 16) & mask
    y = left & mask
    x_prime = (right >> 16) & mask
    y_prime = right & mask
    rx_alpha = ((rol(x, 7, 16) ^ x_prime) << 16) | (rol(y, 7, 16) ^ y_prime)
    rx_beta = ((rol(x, 2, 16) ^ x_prime) << 16) | (rol(y, 2, 16) ^ y_prime)
    carry_left = ((x + y) & mask) ^ x ^ y
    carry_right = ((x_prime + y_prime) & mask) ^ x_prime ^ y_prime
    carry_delta = carry_left ^ carry_right
    carry_left_delta = (carry_left << 16) | carry_delta
    carry_right_delta = (carry_right << 16) | carry_delta
    return (*base_words, rx_alpha, rx_beta, carry_left_delta, carry_right_delta)


def _speck32_carry_chain_mask(a: int, b: int, word_bits: int = 16) -> int:
    mask = (1 << word_bits) - 1
    carry = 0
    carry_chain = 0
    for bit_index in range(word_bits):
        a_bit = (a >> bit_index) & 1
        b_bit = (b >> bit_index) & 1
        carry = (a_bit & b_bit) | (a_bit & carry) | (b_bit & carry)
        carry_chain |= carry << bit_index
    return carry_chain & mask


def _speck32_carry_edge_mask(carry_chain: int, word_bits: int = 16) -> int:
    mask = (1 << word_bits) - 1
    return (carry_chain ^ ((carry_chain << 1) & mask)) & mask


def speck32_partial_inverse_rx_carrychain_feature_words(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> tuple[int, ...]:
    _require_speck32(width, cipher)
    base_words = speck32_partial_inverse_rx_feature_words(left, right, width, cipher)
    mask = 0xFFFF
    x = (left >> 16) & mask
    y = left & mask
    x_prime = (right >> 16) & mask
    y_prime = right & mask
    pre_y, pre_y_prime, _delta_pre_y = speck32_partial_inverse_words(left, right, width)
    ror_x = ror(x, 7, 16)
    ror_x_prime = ror(x_prime, 7, 16)

    generate_xy = x & y
    generate_xy_prime = x_prime & y_prime
    propagate_xy = x ^ y
    propagate_xy_prime = x_prime ^ y_prime
    carry_xy = _speck32_carry_chain_mask(x, y)
    carry_xy_prime = _speck32_carry_chain_mask(x_prime, y_prime)
    carry_edge_xy = _speck32_carry_edge_mask(carry_xy)
    carry_edge_xy_prime = _speck32_carry_edge_mask(carry_xy_prime)

    generate_rot_pre = ror_x & pre_y
    generate_rot_pre_prime = ror_x_prime & pre_y_prime
    propagate_rot_pre = ror_x ^ pre_y
    propagate_rot_pre_prime = ror_x_prime ^ pre_y_prime
    carry_rot_pre = _speck32_carry_chain_mask(ror_x, pre_y)
    carry_rot_pre_prime = _speck32_carry_chain_mask(ror_x_prime, pre_y_prime)
    carry_edge_rot_pre = _speck32_carry_edge_mask(carry_rot_pre)
    carry_edge_rot_pre_prime = _speck32_carry_edge_mask(carry_rot_pre_prime)

    return (
        *base_words,
        (generate_xy << 16) | (generate_xy ^ generate_xy_prime),
        (propagate_xy << 16) | (propagate_xy ^ propagate_xy_prime),
        (carry_edge_xy << 16) | (carry_edge_xy ^ carry_edge_xy_prime),
        (generate_rot_pre << 16) | (generate_rot_pre ^ generate_rot_pre_prime),
        (propagate_rot_pre << 16) | (propagate_rot_pre ^ propagate_rot_pre_prime),
        (carry_edge_rot_pre << 16) | (carry_edge_rot_pre ^ carry_edge_rot_pre_prime),
    )


def speck32_partial_inverse_rx_carrychain_plus_feature_words(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> tuple[int, ...]:
    _require_speck32(width, cipher)
    base_words = speck32_partial_inverse_rx_carrychain_feature_words(left, right, width, cipher)
    mask = 0xFFFF
    x = (left >> 16) & mask
    y = left & mask
    x_prime = (right >> 16) & mask
    y_prime = right & mask
    pre_y, pre_y_prime, _delta_pre_y = speck32_partial_inverse_words(left, right, width)
    ror_x = ror(x, 7, 16)
    ror_x_prime = ror(x_prime, 7, 16)

    carry_xy = _speck32_carry_chain_mask(x, y)
    carry_xy_prime = _speck32_carry_chain_mask(x_prime, y_prime)
    carry_xy_delta = carry_xy ^ carry_xy_prime
    carry_rot_pre = _speck32_carry_chain_mask(ror_x, pre_y)
    carry_rot_pre_prime = _speck32_carry_chain_mask(ror_x_prime, pre_y_prime)
    carry_rot_pre_delta = carry_rot_pre ^ carry_rot_pre_prime
    add_xy = (x + y) & mask
    add_xy_prime = (x_prime + y_prime) & mask
    add_rot_pre = (ror_x + pre_y) & mask
    add_rot_pre_prime = (ror_x_prime + pre_y_prime) & mask

    return (
        *base_words,
        (carry_xy << 16) | carry_xy_delta,
        (carry_xy_prime << 16) | carry_xy_delta,
        (carry_rot_pre << 16) | carry_rot_pre_delta,
        (carry_rot_pre_prime << 16) | carry_rot_pre_delta,
        (add_xy << 16) | (add_xy ^ add_xy_prime),
        (add_rot_pre << 16) | (add_rot_pre ^ add_rot_pre_prime),
    )


def arx_aligned_difference(
    difference: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> int:
    _require_speck32(width, cipher)
    return speck32_rotation_aligned_difference(difference, width)


def aligned_difference_bits(
    difference: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    from blockcipher_nd.features.pair_features import int_to_bits

    return int_to_bits(arx_aligned_difference(difference, width, cipher), width)
