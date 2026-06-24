"""LEA block cipher implementations.

The implementation follows the 128-bit block LEA family with 128/192/256-bit
keys.  It exposes the same small integer-oriented interface used by the rest of
the project so reduced-round neural distinguishers can sample it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


_MASK32 = 0xFFFFFFFF
_DELTA = (
    0xC3EFE9DB,
    0x44626B02,
    0x79E27C8A,
    0x78DF30EC,
    0x715EA49E,
    0xC785DA0A,
    0xE04EF22A,
    0xE5C40957,
)


def _rol32(value: int, amount: int) -> int:
    amount &= 31
    return ((value << amount) | (value >> (32 - amount))) & _MASK32


def _ror32(value: int, amount: int) -> int:
    amount &= 31
    return ((value >> amount) | (value << (32 - amount))) & _MASK32


def _words_from_int(value: int, bits: int) -> list[int]:
    return [(value >> shift) & _MASK32 for shift in range(0, bits, 32)]


def _int_from_words(words: Sequence[int]) -> int:
    value = 0
    for index, word in enumerate(words):
        value |= (word & _MASK32) << (32 * index)
    return value


def _round_count_for_key_bits(key_bits: int) -> int:
    if key_bits == 128:
        return 24
    if key_bits == 192:
        return 28
    if key_bits == 256:
        return 32
    raise ValueError(f"unsupported LEA key size: {key_bits}")


def _expand_key_128(key: int) -> list[tuple[int, int, int, int, int, int]]:
    t = _words_from_int(key, 128)
    round_keys: list[tuple[int, int, int, int, int, int]] = []
    for i in range(24):
        delta = _rol32(_DELTA[i % 4], i)
        t[0] = _rol32((t[0] + _rol32(delta, 0)) & _MASK32, 1)
        t[1] = _rol32((t[1] + _rol32(delta, 1)) & _MASK32, 3)
        t[2] = _rol32((t[2] + _rol32(delta, 2)) & _MASK32, 6)
        t[3] = _rol32((t[3] + _rol32(delta, 3)) & _MASK32, 11)
        round_keys.append((t[0], t[1], t[2], t[1], t[3], t[1]))
    return round_keys


def _expand_key_192(key: int) -> list[tuple[int, int, int, int, int, int]]:
    t = _words_from_int(key, 192)
    round_keys: list[tuple[int, int, int, int, int, int]] = []
    rotations = (1, 3, 6, 11, 13, 17)
    for i in range(28):
        delta = _rol32(_DELTA[i % 6], i)
        for j, rot in enumerate(rotations):
            t[j] = _rol32((t[j] + _rol32(delta, j)) & _MASK32, rot)
        round_keys.append(tuple(t))  # type: ignore[arg-type]
    return round_keys


def _expand_key_256(key: int) -> list[tuple[int, int, int, int, int, int]]:
    t = _words_from_int(key, 256)
    round_keys: list[tuple[int, int, int, int, int, int]] = []
    rotations = (1, 3, 6, 11, 13, 17)
    for i in range(32):
        delta = _rol32(_DELTA[i % 8], i)
        selected: list[int] = []
        for j, rot in enumerate(rotations):
            index = (6 * i + j) % 8
            t[index] = _rol32((t[index] + _rol32(delta, j)) & _MASK32, rot)
            selected.append(t[index])
        round_keys.append(tuple(selected))  # type: ignore[arg-type]
    return round_keys


def _expand_key(key: int, key_bits: int) -> list[tuple[int, int, int, int, int, int]]:
    if key < 0 or key >= (1 << key_bits):
        raise ValueError(f"LEA key must fit in {key_bits} bits")
    if key_bits == 128:
        return _expand_key_128(key)
    if key_bits == 192:
        return _expand_key_192(key)
    if key_bits == 256:
        return _expand_key_256(key)
    raise ValueError(f"unsupported LEA key size: {key_bits}")


@dataclass(frozen=True)
class Lea:
    """LEA reduced/full-round encryptor with 128-bit block size."""

    rounds: int
    key: int
    key_bits: int = 128
    name: str = "LEA"
    structure: str = "ARX"
    block_bits: int = 128

    def __post_init__(self) -> None:
        full_rounds = _round_count_for_key_bits(self.key_bits)
        if self.rounds < 1 or self.rounds > full_rounds:
            raise ValueError(f"LEA-{self.key_bits} supports 1..{full_rounds} rounds")
        if self.key < 0 or self.key >= (1 << self.key_bits):
            raise ValueError(f"LEA key must fit in {self.key_bits} bits")

    def encrypt(self, plaintext: int) -> int:
        if plaintext < 0 or plaintext >= (1 << self.block_bits):
            raise ValueError("LEA plaintext must fit in 128 bits")

        x0, x1, x2, x3 = _words_from_int(plaintext, 128)
        round_keys = _expand_key(self.key, self.key_bits)
        for i in range(self.rounds):
            k0, k1, k2, k3, k4, k5 = round_keys[i]
            y0 = _rol32(((x0 ^ k0) + (x1 ^ k1)) & _MASK32, 9)
            y1 = _ror32(((x1 ^ k2) + (x2 ^ k3)) & _MASK32, 5)
            y2 = _ror32(((x2 ^ k4) + (x3 ^ k5)) & _MASK32, 3)
            x0, x1, x2, x3 = y0, y1, y2, x0
        return _int_from_words((x0, x1, x2, x3))


class Lea128(Lea):
    def __init__(self, rounds: int = 24, key: int = 0) -> None:
        super().__init__(rounds=rounds, key=key, key_bits=128, name="LEA-128")


class Lea192(Lea):
    def __init__(self, rounds: int = 28, key: int = 0) -> None:
        super().__init__(rounds=rounds, key=key, key_bits=192, name="LEA-192")


class Lea256(Lea):
    def __init__(self, rounds: int = 32, key: int = 0) -> None:
        super().__init__(rounds=rounds, key=key, key_bits=256, name="LEA-256")
