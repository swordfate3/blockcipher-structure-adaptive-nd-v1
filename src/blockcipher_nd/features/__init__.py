from blockcipher_nd.features.pair_features import (
    encode_ciphertext_pair,
    int_to_bits,
    pair_bits_for_encoding,
)
from blockcipher_nd.features.profile import (
    STRUCTURE_FEATURE_NAMES,
    structure_feature_vector,
)
from blockcipher_nd.features.spn_aligned import (
    aligned_difference_bits,
    inverse_permutation_difference,
)

__all__ = [
    "STRUCTURE_FEATURE_NAMES",
    "encode_ciphertext_pair",
    "int_to_bits",
    "pair_bits_for_encoding",
    "FEATURE_ENCODINGS",
    "is_supported_feature_encoding",
    "aligned_difference_bits",
    "inverse_permutation_difference",
    "structure_feature_vector",
]

from blockcipher_nd.features.registry import FEATURE_ENCODINGS, is_supported_feature_encoding
