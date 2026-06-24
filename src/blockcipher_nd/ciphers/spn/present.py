from __future__ import annotations

from dataclasses import dataclass

from blockcipher_nd.ciphers.base import rol


PRESENT_SBOX = (
    0xC,
    0x5,
    0x6,
    0xB,
    0x9,
    0x0,
    0xA,
    0xD,
    0x3,
    0xE,
    0xF,
    0x8,
    0x4,
    0x7,
    0x1,
    0x2,
)

PRESENT_INV_SBOX = tuple(PRESENT_SBOX.index(value) for value in range(16))


@dataclass(frozen=True)
class Present80:
    rounds: int
    key: int

    name: str = "PRESENT-80"
    structure: str = "SPN"
    block_bits: int = 64
    key_bits: int = 80

    def encrypt(self, plaintext: int) -> int:
        state = plaintext & ((1 << 64) - 1)
        key_register = self.key & ((1 << 80) - 1)
        for round_counter in range(1, self.rounds + 1):
            state ^= key_register >> 16
            state = self._sbox_layer(state)
            state = self._permutation_layer(state)
            key_register = self._update_key(key_register, round_counter)
        state ^= key_register >> 16
        return state

    @staticmethod
    def _sbox_layer(state: int) -> int:
        out = 0
        for nibble_index in range(16):
            nibble = (state >> (4 * nibble_index)) & 0xF
            out |= PRESENT_SBOX[nibble] << (4 * nibble_index)
        return out

    @staticmethod
    def inverse_sbox_layer(state: int) -> int:
        out = 0
        for nibble_index in range(16):
            nibble = (state >> (4 * nibble_index)) & 0xF
            out |= PRESENT_INV_SBOX[nibble] << (4 * nibble_index)
        return out

    @staticmethod
    def _permutation_layer(state: int) -> int:
        return Present80.permutation_layer(state)

    @staticmethod
    def permutation_layer(state: int) -> int:
        out = 0
        for bit_index in range(63):
            bit = (state >> bit_index) & 1
            out |= bit << ((16 * bit_index) % 63)
        out |= ((state >> 63) & 1) << 63
        return out

    @staticmethod
    def inverse_permutation_layer(state: int) -> int:
        out = 0
        for bit_index in range(63):
            target_index = (16 * bit_index) % 63
            bit = (state >> target_index) & 1
            out |= bit << bit_index
        out |= ((state >> 63) & 1) << 63
        return out

    @staticmethod
    def _update_key(key_register: int, round_counter: int) -> int:
        key_register = rol(key_register, 61, 80)
        top_nibble = (key_register >> 76) & 0xF
        key_register &= ~(0xF << 76)
        key_register |= PRESENT_SBOX[top_nibble] << 76
        key_register ^= (round_counter & 0x1F) << 15
        return key_register & ((1 << 80) - 1)
