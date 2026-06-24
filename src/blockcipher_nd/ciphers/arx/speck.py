from __future__ import annotations

from dataclasses import dataclass

from blockcipher_nd.ciphers.base import rol, ror


@dataclass(frozen=True)
class Speck32_64:
    rounds: int
    key: int

    name: str = "SPECK32/64"
    structure: str = "ARX"
    block_bits: int = 32
    key_bits: int = 64

    def encrypt(self, plaintext: int) -> int:
        mask = 0xFFFF
        x = (plaintext >> 16) & mask
        y = plaintext & mask
        for round_key in self._round_keys():
            x = ror(x, 7, 16)
            x = (x + y) & mask
            x ^= round_key
            y = rol(y, 2, 16) ^ x
        return (x << 16) | y

    def _round_keys(self) -> list[int]:
        mask = 0xFFFF
        words = [(self.key >> (16 * i)) & mask for i in range(4)]
        l_words = [words[1], words[2], words[3]]
        round_keys = [words[0]]
        for i in range(max(0, self.rounds - 1)):
            l_value = ror(l_words[i], 7, 16)
            l_value = (l_value + round_keys[i]) & mask
            l_value ^= i
            next_key = rol(round_keys[i], 2, 16) ^ l_value
            l_words.append(l_value)
            round_keys.append(next_key)
        return round_keys[: self.rounds]
