from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CipherProfile:
    """Structure-level cipher description used by architecture matching."""

    name: str
    structure: str
    block_bits: int
    key_bits: int
    traits: tuple[str, ...]

    @staticmethod
    def speck32_64() -> "CipherProfile":
        return CipherProfile(
            name="SPECK32/64",
            structure="ARX",
            block_bits=32,
            key_bits=64,
            traits=(
                "modular_addition",
                "xor",
                "rotation",
                "carry_propagation",
                "word_parallelism",
            ),
        )

    @staticmethod
    def present80() -> "CipherProfile":
        return CipherProfile(
            name="PRESENT-80",
            structure="SPN",
            block_bits=64,
            key_bits=80,
            traits=(
                "sbox_layer",
                "permutation_layer",
                "sbox_locality",
                "bit_permutation",
                "lightweight_spn",
            ),
        )

    @staticmethod
    def gift64() -> "CipherProfile":
        return CipherProfile(
            name="GIFT-64",
            structure="SPN",
            block_bits=64,
            key_bits=128,
            traits=(
                "sbox_layer",
                "permutation_layer",
                "sbox_locality",
                "bit_permutation",
                "lightweight_spn",
            ),
        )

    @staticmethod
    def sm4() -> "CipherProfile":
        return CipherProfile(
            name="SM4",
            structure="Feistel-like",
            block_bits=128,
            key_bits=128,
            traits=(
                "unbalanced_round_update",
                "sbox_layer",
                "linear_diffusion",
                "word_parallelism",
                "round_recurrence",
            ),
        )


__all__ = ["CipherProfile"]
