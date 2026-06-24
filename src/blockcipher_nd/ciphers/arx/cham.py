"""CHAM block cipher implementations.

CHAM64/128 follows the CHAM paper Appendix A test vector as distributed in
Crypto++ test vectors.  It exposes the reduced-round integer API used by the
experiment matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

from blockcipher_nd.ciphers.base import rol


def _expand_key_64_128(key: int) -> list[int]:
    words = []
    for offset in range(0, 16, 2):
        shift = (14 - offset) * 8
        words.append((key >> shift) & 0xFFFF)
    round_keys = [0] * 16
    for i, word in enumerate(words):
        round_keys[i] = word ^ rol(word, 1, 16) ^ rol(word, 8, 16)
        round_keys[(i + 8) ^ 1] = word ^ rol(word, 1, 16) ^ rol(word, 11, 16)
    return [k & 0xFFFF for k in round_keys]


@dataclass(frozen=True)
class Cham64_128:
    rounds: int = 80
    key: int = 0
    name: str = "CHAM-64/128"
    structure: str = "ARX"
    block_bits: int = 64
    key_bits: int = 128

    def __post_init__(self) -> None:
        if self.rounds < 1 or self.rounds > 80:
            raise ValueError("CHAM-64/128 supports 1..80 rounds")
        if self.key < 0 or self.key >= (1 << 128):
            raise ValueError("CHAM-64/128 key must fit in 128 bits")

    def encrypt(self, plaintext: int) -> int:
        if plaintext < 0 or plaintext >= (1 << 64):
            raise ValueError("CHAM-64/128 plaintext must fit in 64 bits")
        state = [
            (plaintext >> 48) & 0xFFFF,
            (plaintext >> 32) & 0xFFFF,
            (plaintext >> 16) & 0xFFFF,
            plaintext & 0xFFFF,
        ]
        round_keys = _expand_key_64_128(self.key)
        for i in range(self.rounds):
            idx0 = i % 4
            idx1 = (i + 1) % 4
            idx3 = (i + 4) % 4
            r1 = 1 if i % 2 == 0 else 8
            r2 = 8 if i % 2 == 0 else 1
            aa = state[idx0] ^ i
            bb = rol(state[idx1], r1, 16) ^ round_keys[i % 16]
            state[idx3] = rol((aa + bb) & 0xFFFF, r2, 16)
        return (state[0] << 48) | (state[1] << 32) | (state[2] << 16) | state[3]
