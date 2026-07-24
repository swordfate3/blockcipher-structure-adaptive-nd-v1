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
    def rectangle80() -> "CipherProfile":
        return CipherProfile(
            name="RECTANGLE-80",
            structure="SPN",
            block_bits=64,
            key_bits=80,
            traits=(
                "sbox_layer",
                "permutation_layer",
                "sbox_locality",
                "bit_permutation",
                "rotation",
                "non_contiguous_sbox_cells",
                "bitsliced_columns",
                "row_rotation",
                "lightweight_spn",
            ),
        )

    @staticmethod
    def uknit64() -> "CipherProfile":
        return CipherProfile(
            name="uKNIT-BC",
            structure="SPN",
            block_bits=64,
            key_bits=128,
            traits=(
                "sbox_layer",
                "cell_specific_sboxes",
                "round_specific_sboxes",
                "general_gf2_diffusion",
                "round_specific_diffusion",
                "non_aligned_spn",
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

    @staticmethod
    def des() -> "CipherProfile":
        return CipherProfile(
            name="DES",
            structure="Feistel-like",
            block_bits=64,
            key_bits=64,
            traits=(
                "balanced_feistel",
                "left_right_branch_exchange",
                "sbox_layer",
                "expansion_permutation",
                "round_recurrence",
            ),
        )

    @staticmethod
    def simon64_128() -> "CipherProfile":
        return CipherProfile(
            name="SIMON64/128",
            structure="Feistel-like",
            block_bits=64,
            key_bits=128,
            traits=(
                "balanced_feistel",
                "left_right_branch_exchange",
                "nonlinear_and",
                "xor",
                "rotation",
                "word_parallelism",
                "round_recurrence",
            ),
        )

    @staticmethod
    def simeck64_128() -> "CipherProfile":
        return CipherProfile(
            name="Simeck64/128",
            structure="Feistel-like",
            block_bits=64,
            key_bits=128,
            traits=(
                "balanced_feistel",
                "left_right_branch_exchange",
                "nonlinear_and",
                "xor",
                "rotation",
                "word_parallelism",
                "round_recurrence",
            ),
        )


__all__ = ["CipherProfile"]
