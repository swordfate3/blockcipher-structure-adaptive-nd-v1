"""Simeck block cipher implementations.

Simeck64/128 follows the CHES 2015 paper test vector and exposes the reduced
round integer API used by the experiment matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

from blockcipher_nd.ciphers.base import rol


def _z1_bits() -> tuple[int, ...]:
    state = [1, 1, 1, 1, 1, 1]
    bits: list[int] = []
    for _ in range(63):
        bits.append(state[0])
        state = state[1:] + [state[0] ^ state[1]]
    return tuple(bits)


_Z1 = _z1_bits()


def _round_keys(key: int, rounds: int) -> list[int]:
    mask = 0xFFFFFFFF
    words = [(key >> (32 * i)) & mask for i in range(4)]
    keys = [words[0]]
    c = mask ^ 3
    for i in range(rounds - 1):
        tmp = words[0] ^ (rol(words[1], 5, 32) & words[1]) ^ rol(words[1], 1, 32)
        tmp ^= c ^ _Z1[i % 63]
        words = [words[1], words[2], words[3], tmp & mask]
        keys.append(words[0])
    return keys


@dataclass(frozen=True)
class Simeck64_128:
    rounds: int = 44
    key: int = 0
    name: str = "Simeck64/128"
    structure: str = "Feistel-like"
    block_bits: int = 64
    key_bits: int = 128

    def __post_init__(self) -> None:
        if self.rounds < 1 or self.rounds > 44:
            raise ValueError("Simeck64/128 supports 1..44 rounds")
        if self.key < 0 or self.key >= (1 << 128):
            raise ValueError("Simeck64/128 key must fit in 128 bits")

    def encrypt(self, plaintext: int) -> int:
        if plaintext < 0 or plaintext >= (1 << 64):
            raise ValueError("Simeck64/128 plaintext must fit in 64 bits")
        mask = 0xFFFFFFFF
        left = (plaintext >> 32) & mask
        right = plaintext & mask
        for round_key in _round_keys(self.key, self.rounds):
            feedback = (rol(left, 5, 32) & left) ^ rol(left, 1, 32)
            left, right = (right ^ feedback ^ round_key) & mask, left
        return ((left & mask) << 32) | (right & mask)
