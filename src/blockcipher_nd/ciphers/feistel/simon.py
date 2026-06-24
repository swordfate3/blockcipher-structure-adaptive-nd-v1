from __future__ import annotations

from dataclasses import dataclass

from blockcipher_nd.ciphers.base import rol, ror


SIMON_Z3 = "11011011101011000110010111100000010010001010011100110100001111"


@dataclass(frozen=True)
class Simon64_128:
    rounds: int
    key: int

    name: str = "SIMON64/128"
    structure: str = "Feistel-like"
    block_bits: int = 64
    key_bits: int = 128

    def encrypt(self, plaintext: int) -> int:
        mask = 0xFFFFFFFF
        x = (plaintext >> 32) & mask
        y = plaintext & mask
        for round_key in self._round_keys():
            x, y = (
                y ^ (rol(x, 1, 32) & rol(x, 8, 32)) ^ rol(x, 2, 32) ^ round_key
            ) & mask, x
        return (x << 32) | y

    def _round_keys(self) -> list[int]:
        if self.rounds < 0 or self.rounds > 44:
            raise ValueError("SIMON64/128 rounds must be between 0 and 44")
        mask = 0xFFFFFFFF
        c = mask ^ 0x3
        round_keys = [(self.key >> (32 * index)) & mask for index in range(4)]
        for index in range(4, self.rounds):
            tmp = ror(round_keys[index - 1], 3, 32) ^ round_keys[index - 3]
            tmp ^= ror(tmp, 1, 32)
            z_bit = int(SIMON_Z3[(index - 4) % 62])
            round_keys.append((c ^ z_bit ^ round_keys[index - 4] ^ tmp) & mask)
        return round_keys[: self.rounds]
