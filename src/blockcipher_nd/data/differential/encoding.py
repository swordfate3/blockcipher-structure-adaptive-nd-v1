from __future__ import annotations

from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.features.pair_features import encode_ciphertext_pair


def encode_pair(left: int, right: int, block_bits: int, config: DifferentialDatasetConfig, cipher) -> list[int]:
    encoded = encode_ciphertext_pair(
        left,
        right,
        width=block_bits,
        feature_encoding=config.feature_encoding,
        cipher=cipher,
    )
    if not config.selected_bit_indices:
        return encoded
    return [encoded[index] for index in config.selected_bit_indices]
