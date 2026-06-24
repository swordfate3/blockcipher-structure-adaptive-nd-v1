from __future__ import annotations

from typing import Protocol


class ReducedRoundCipher(Protocol):
    name: str
    structure: str
    block_bits: int
    key_bits: int
    rounds: int

    def encrypt(self, plaintext: int) -> int:
        ...


def rol(value: int, shift: int, width: int) -> int:
    mask = (1 << width) - 1
    shift %= width
    return ((value << shift) & mask) | (value >> (width - shift))


def ror(value: int, shift: int, width: int) -> int:
    mask = (1 << width) - 1
    shift %= width
    return (value >> shift) | ((value << (width - shift)) & mask)
