from __future__ import annotations

from blockcipher_nd.features.pair_features import (
    encode_ciphertext_pair,
    is_parameterized_present_sboxddt_encoding,
    pair_bits_for_encoding,
)

FEATURE_ENCODINGS = {
    "ciphertext_pair_bits",
    "present_mcnd_cell_matrix_bits",
    "present_pair_xor_paligned_cell_matrix_bits",
    "present_pair_xor_paligned_sinv_cell_matrix_bits",
    "present_pair_xor_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits",
    "present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits",
    "present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits",
    "present_pair_xor_paligned_sboxddt_cell_matrix_bits",
    "present_pair_xor_paligned_sboxddt_top2_cell_matrix_bits",
    "present_pair_xor_paligned_sboxddt_back2_cell_matrix_bits",
    "present_pair_xor_paligned_sboxddt_beam2_cell_matrix_bits",
    "present_pair_xor_paligned_sboxddt_beam4deep3_cell_matrix_bits",
    "present_pair_xor_cell_matrix_bits",
    "present_xor_paligned_cell_matrix_bits",
    "present_nibble_paligned_view",
    "ciphertext_xor_bits",
    "ciphertext_xor_spn_aligned_bits",
    "ciphertext_xor_spn_paligned_bits",
    "ciphertext_pair_xor_bits",
    "ciphertext_pair_xor_spn_aligned_bits",
    "ciphertext_pair_xor_arx_aligned_bits",
    "ciphertext_pair_xor_arx_partial_inverse_bits",
    "ciphertext_pair_xor_arx_partial_inverse_rx_bits",
    "ciphertext_pair_xor_arx_partial_inverse_rx_carrychain_bits",
    "ciphertext_pair_xor_arx_partial_inverse_rx_carrychain_plus_bits",
}


def is_supported_feature_encoding(feature_encoding: str) -> bool:
    return feature_encoding in FEATURE_ENCODINGS or is_parameterized_present_sboxddt_encoding(feature_encoding)


__all__ = [
    "FEATURE_ENCODINGS",
    "encode_ciphertext_pair",
    "is_supported_feature_encoding",
    "pair_bits_for_encoding",
]
