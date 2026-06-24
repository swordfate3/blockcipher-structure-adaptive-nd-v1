from __future__ import annotations

import numpy as np

from blockcipher_nd.registry.cipher_profiles import CipherProfile


STRUCTURE_FEATURE_NAMES = (
    "is_arx",
    "is_spn",
    "is_feistel_like",
    "has_modular_addition",
    "has_xor",
    "has_rotation",
    "has_carry_propagation",
    "has_word_parallelism",
    "has_sbox_layer",
    "has_permutation_layer",
    "has_sbox_locality",
    "has_bit_permutation",
    "has_lightweight_spn",
    "has_unbalanced_round_update",
    "has_linear_diffusion",
    "has_round_recurrence",
    "block_bits_div_128",
    "key_bits_div_128",
    "rounds_div_32",
)


def structure_feature_vector(cipher: CipherProfile, rounds: int) -> np.ndarray:
    traits = set(cipher.traits)
    values = {
        "is_arx": cipher.structure == "ARX",
        "is_spn": cipher.structure == "SPN",
        "is_feistel_like": cipher.structure == "Feistel-like",
        "has_modular_addition": "modular_addition" in traits,
        "has_xor": "xor" in traits,
        "has_rotation": "rotation" in traits,
        "has_carry_propagation": "carry_propagation" in traits,
        "has_word_parallelism": "word_parallelism" in traits,
        "has_sbox_layer": "sbox_layer" in traits,
        "has_permutation_layer": "permutation_layer" in traits,
        "has_sbox_locality": "sbox_locality" in traits,
        "has_bit_permutation": "bit_permutation" in traits,
        "has_lightweight_spn": "lightweight_spn" in traits,
        "has_unbalanced_round_update": "unbalanced_round_update" in traits,
        "has_linear_diffusion": "linear_diffusion" in traits,
        "has_round_recurrence": "round_recurrence" in traits,
        "block_bits_div_128": cipher.block_bits / 128.0,
        "key_bits_div_128": cipher.key_bits / 128.0,
        "rounds_div_32": rounds / 32.0,
    }
    return np.array(
        [float(values[name]) for name in STRUCTURE_FEATURE_NAMES],
        dtype=np.float32,
    )
