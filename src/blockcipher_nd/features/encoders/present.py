from __future__ import annotations

from blockcipher_nd.features.encoders.present_matrix import (
    parameterized_present_sboxddt_cell_matrix_bits,
    present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits,
    present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits,
    present_mcnd_cell_matrix_bits,
    present_pair_xor_cell_matrix_bits,
    present_pair_xor_paligned_cell_matrix_bits,
    present_pair_xor_paligned_sboxddt_back2_cell_matrix_bits,
    present_pair_xor_paligned_sboxddt_beam2_cell_matrix_bits,
    present_pair_xor_paligned_sboxddt_beam4deep3_cell_matrix_bits,
    present_pair_xor_paligned_sboxddt_cell_matrix_bits,
    present_pair_xor_paligned_sboxddt_top2_cell_matrix_bits,
    present_pair_xor_paligned_sinv_cell_matrix_bits,
    present_pair_xor_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits,
    present_xor_paligned_cell_matrix_bits,
)
from blockcipher_nd.features.encoders.present_sbox_ddt import (
    is_parameterized_present_sboxddt_encoding,
    parse_parameterized_present_sboxddt_encoding,
    present_active_nibble_count,
    present_sbox_ddt_back2_words,
    present_sbox_ddt_beam_statistics_words,
    present_sbox_ddt_beam_words,
    present_sbox_ddt_score_nibble,
    present_sbox_ddt_top2_margin_words,
    present_sbox_ddt_top2_words,
    present_sbox_ddt_topk_words,
    present_sbox_ddt_words,
    present_structural_inverse_sbox_difference,
)
