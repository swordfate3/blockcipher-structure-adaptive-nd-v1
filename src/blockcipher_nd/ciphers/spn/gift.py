"""GIFT block cipher implementations.

This module implements the 64-bit GIFT permutation with 128-bit keys.  GIFT is
an SPN lightweight block cipher and is useful for the structure-aware neural
distinguisher experiments because its bit permutation differs sharply from
nibble-oriented AES/PRESENT style diffusion.
"""

from __future__ import annotations

from dataclasses import dataclass


_SBOX = (0x1, 0xA, 0x4, 0xC, 0x6, 0xF, 0x3, 0x9, 0x2, 0xD, 0xB, 0x7, 0x5, 0x0, 0x8, 0xE)
_GIFT64_PERM = [
    0,
    17,
    34,
    51,
    48,
    1,
    18,
    35,
    32,
    49,
    2,
    19,
    16,
    33,
    50,
    3,
    4,
    21,
    38,
    55,
    52,
    5,
    22,
    39,
    36,
    53,
    6,
    23,
    20,
    37,
    54,
    7,
    8,
    25,
    42,
    59,
    56,
    9,
    26,
    43,
    40,
    57,
    10,
    27,
    24,
    41,
    58,
    11,
    12,
    29,
    46,
    63,
    60,
    13,
    30,
    47,
    44,
    61,
    14,
    31,
    28,
    45,
    62,
    15,
]
_ROUND_CONSTANTS = (
    0x01,
    0x03,
    0x07,
    0x0F,
    0x1F,
    0x3E,
    0x3D,
    0x3B,
    0x37,
    0x2F,
    0x1E,
    0x3C,
    0x39,
    0x33,
    0x27,
    0x0E,
    0x1D,
    0x3A,
    0x35,
    0x2B,
    0x16,
    0x2C,
    0x18,
    0x30,
    0x21,
    0x02,
    0x05,
    0x0B,
)
_MASK128 = (1 << 128) - 1

GIFT64_SBOX = _SBOX
GIFT64_PERMUTATION = tuple(_GIFT64_PERM)


def _sub_cells(state: int) -> int:
    out = 0
    for i in range(16):
        out |= _SBOX[(state >> (4 * i)) & 0xF] << (4 * i)
    return out


def _perm_bits(state: int) -> int:
    out = 0
    for source, target in enumerate(_GIFT64_PERM):
        out |= ((state >> source) & 1) << target
    return out


def _inverse_perm_bits(state: int) -> int:
    out = 0
    for source, target in enumerate(_GIFT64_PERM):
        out |= ((state >> target) & 1) << source
    return out


def _key_nibbles_from_int(key: int) -> list[int]:
    return [(key >> (4 * i)) & 0xF for i in range(32)]


def _update_key_nibbles(key: list[int]) -> list[int]:
    temp = [key[(i + 8) % 32] for i in range(32)]
    updated = temp[:24] + [0] * 8
    updated[24] = temp[27]
    updated[25] = temp[24]
    updated[26] = temp[25]
    updated[27] = temp[26]
    updated[28] = ((temp[28] & 0xC) >> 2) ^ ((temp[29] & 0x3) << 2)
    updated[29] = ((temp[29] & 0xC) >> 2) ^ ((temp[30] & 0x3) << 2)
    updated[30] = ((temp[30] & 0xC) >> 2) ^ ((temp[31] & 0x3) << 2)
    updated[31] = ((temp[31] & 0xC) >> 2) ^ ((temp[28] & 0x3) << 2)
    return updated


@dataclass(frozen=True)
class Gift64:
    rounds: int = 28
    key: int = 0
    name: str = "GIFT-64"
    structure: str = "SPN"
    block_bits: int = 64
    key_bits: int = 128

    @staticmethod
    def permutation_layer(state: int) -> int:
        if state < 0 or state >= (1 << 64):
            raise ValueError("GIFT-64 state must fit in 64 bits")
        return _perm_bits(state)

    @staticmethod
    def inverse_permutation_layer(state: int) -> int:
        if state < 0 or state >= (1 << 64):
            raise ValueError("GIFT-64 state must fit in 64 bits")
        return _inverse_perm_bits(state)

    def __post_init__(self) -> None:
        if self.rounds < 1 or self.rounds > 28:
            raise ValueError("GIFT-64 supports 1..28 rounds")
        if self.key < 0 or self.key >= (1 << 128):
            raise ValueError("GIFT-64 key must fit in 128 bits")

    def encrypt(self, plaintext: int) -> int:
        if plaintext < 0 or plaintext >= (1 << 64):
            raise ValueError("GIFT-64 plaintext must fit in 64 bits")

        state = plaintext
        key = _key_nibbles_from_int(self.key)
        for r in range(self.rounds):
            state = _sub_cells(state)
            state = _perm_bits(state)
            key_bits = [((key[i] >> j) & 1) for i in range(32) for j in range(4)]
            for i in range(16):
                state ^= key_bits[i] << (4 * i)
                state ^= key_bits[i + 16] << (4 * i + 1)
            rc = _ROUND_CONSTANTS[r]
            state ^= 1 << 63
            state ^= ((rc >> 0) & 1) << 3
            state ^= ((rc >> 1) & 1) << 7
            state ^= ((rc >> 2) & 1) << 11
            state ^= ((rc >> 3) & 1) << 15
            state ^= ((rc >> 4) & 1) << 19
            state ^= ((rc >> 5) & 1) << 23
            key = _update_key_nibbles(key)
        return state
