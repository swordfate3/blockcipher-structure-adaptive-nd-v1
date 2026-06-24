from __future__ import annotations

from dataclasses import is_dataclass, replace

import numpy as np

from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.data.differential.random import random_int


def cipher_for_row(config: DifferentialDatasetConfig, rng: np.random.Generator, row_index: int):
    if config.key_rotation_interval == 0:
        return config.cipher
    key_block_index = row_index // config.key_rotation_interval
    row_key = _key_for_block(config, rng, key_block_index)
    return _cipher_with_key(config.cipher, row_key)


def _key_for_block(config: DifferentialDatasetConfig, rng: np.random.Generator, block_index: int) -> int:
    if not hasattr(config.cipher, "key_bits"):
        raise ValueError("rotating key schedule requires cipher.key_bits")
    key_bits = int(config.cipher.key_bits)
    state = rng.bit_generator.state
    try:
        key_rng = np.random.default_rng(config.seed + 1_000_003 * (block_index + 1))
        return random_int(key_rng, key_bits)
    finally:
        rng.bit_generator.state = state


def _cipher_with_key(cipher, key: int):
    if not hasattr(cipher, "rounds"):
        raise ValueError("rotating key schedule requires cipher.rounds")
    if is_dataclass(cipher) and hasattr(cipher, "key"):
        return replace(cipher, key=key)
    try:
        return type(cipher)(rounds=int(cipher.rounds), key=key)
    except TypeError:
        pass
    raise ValueError(f"rotating key schedule is not supported for cipher {type(cipher).__name__}")
