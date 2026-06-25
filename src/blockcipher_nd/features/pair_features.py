from __future__ import annotations

from blockcipher_nd.ciphers import ReducedRoundCipher
from blockcipher_nd.features.encoders.arx import arx_aligned_pair_xor_bits, arx_partial_inverse_bits
from blockcipher_nd.features.encoders.bitwise import int_to_bits, pair_to_bits, pair_xor_bits, xor_bits
from blockcipher_nd.features.encoders.present import (
    is_parameterized_present_sboxddt_encoding,
    parameterized_present_sboxddt_cell_matrix_bits,
    parse_parameterized_present_sboxddt_encoding,
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
    present_nibble_paligned_view_bits,
    present_sbox_ddt_beam_statistics_words,
    present_sbox_ddt_score_nibble,
    present_structural_inverse_sbox_difference,
    present_xor_paligned_cell_matrix_bits,
)
from blockcipher_nd.features.encoders.spn import spn_aligned_pair_xor_bits, spn_aligned_xor_bits


def encode_ciphertext_pair(
    left: int,
    right: int,
    *,
    width: int,
    feature_encoding: str,
    cipher: ReducedRoundCipher,
) -> list[int]:
    if feature_encoding == "ciphertext_pair_bits":
        return pair_to_bits(left, right, width)
    if feature_encoding == "ciphertext_xor_bits":
        return xor_bits(left, right, width)
    if feature_encoding == "ciphertext_pair_xor_bits":
        left_bits, right_bits, difference_bits = pair_xor_bits(left, right, width)
        return left_bits + right_bits + difference_bits

    if feature_encoding in {"ciphertext_xor_spn_aligned_bits", "ciphertext_xor_spn_paligned_bits"}:
        return spn_aligned_xor_bits(left, right, width, cipher)
    if feature_encoding == "ciphertext_pair_xor_spn_aligned_bits":
        return spn_aligned_pair_xor_bits(left, right, width, cipher)
    if feature_encoding == "ciphertext_pair_xor_arx_aligned_bits":
        return arx_aligned_pair_xor_bits(left, right, width, cipher)

    arx_partial_inverse_variants = {
        "ciphertext_pair_xor_arx_partial_inverse_bits": "plain",
        "ciphertext_pair_xor_arx_partial_inverse_rx_bits": "rx",
        "ciphertext_pair_xor_arx_partial_inverse_rx_carrychain_bits": "rx_carrychain",
        "ciphertext_pair_xor_arx_partial_inverse_rx_carrychain_plus_bits": "rx_carrychain_plus",
    }
    if feature_encoding in arx_partial_inverse_variants:
        return arx_partial_inverse_bits(
            left,
            right,
            width,
            cipher,
            variant=arx_partial_inverse_variants[feature_encoding],
        )

    present_encoders = {
        "present_mcnd_cell_matrix_bits": lambda: present_mcnd_cell_matrix_bits(left, right, width),
        "present_pair_xor_cell_matrix_bits": lambda: present_pair_xor_cell_matrix_bits(left, right, width),
        "present_pair_xor_paligned_cell_matrix_bits": lambda: present_pair_xor_paligned_cell_matrix_bits(left, right, width, cipher),
        "present_pair_xor_paligned_sinv_cell_matrix_bits": lambda: present_pair_xor_paligned_sinv_cell_matrix_bits(left, right, width, cipher),
        "present_pair_xor_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits": lambda: present_pair_xor_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits(left, right, width, cipher),
        "present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits": lambda: present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits(left, right, width, cipher),
        "present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits": lambda: present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits(left, right, width, cipher),
        "present_pair_xor_paligned_sboxddt_cell_matrix_bits": lambda: present_pair_xor_paligned_sboxddt_cell_matrix_bits(left, right, width, cipher),
        "present_pair_xor_paligned_sboxddt_top2_cell_matrix_bits": lambda: present_pair_xor_paligned_sboxddt_top2_cell_matrix_bits(left, right, width, cipher),
        "present_pair_xor_paligned_sboxddt_back2_cell_matrix_bits": lambda: present_pair_xor_paligned_sboxddt_back2_cell_matrix_bits(left, right, width, cipher),
        "present_pair_xor_paligned_sboxddt_beam2_cell_matrix_bits": lambda: present_pair_xor_paligned_sboxddt_beam2_cell_matrix_bits(left, right, width, cipher),
        "present_pair_xor_paligned_sboxddt_beam4deep3_cell_matrix_bits": lambda: present_pair_xor_paligned_sboxddt_beam4deep3_cell_matrix_bits(left, right, width, cipher),
        "present_xor_paligned_cell_matrix_bits": lambda: present_xor_paligned_cell_matrix_bits(left, right, width, cipher),
        "present_nibble_paligned_view": lambda: present_nibble_paligned_view_bits(left, right, width, cipher),
    }
    if feature_encoding in present_encoders:
        return present_encoders[feature_encoding]()

    present_trail_config = parse_parameterized_present_sboxddt_encoding(feature_encoding)
    if present_trail_config is not None:
        return parameterized_present_sboxddt_cell_matrix_bits(
            left,
            right,
            width,
            cipher,
            feature_encoding=feature_encoding,
            **present_trail_config,
        )
    raise ValueError(f"unsupported feature encoding: {feature_encoding}")


def pair_bits_for_encoding(block_bits: int, feature_encoding: str) -> int:
    present_trail_config = parse_parameterized_present_sboxddt_encoding(feature_encoding)
    if present_trail_config is not None:
        prefix_words = 2
        if present_trail_config["include_pair"]:
            prefix_words += 2
        if present_trail_config["include_sinv"]:
            prefix_words += 1
        if present_trail_config["use_statistics"]:
            trail_words = present_trail_config["depth"] * 9
        else:
            trail_words = present_trail_config["depth"] * (3 * present_trail_config["beam_width"] + 3)
        return block_bits * (prefix_words + trail_words)

    widths = {
        "ciphertext_pair_bits": 2,
        "present_mcnd_cell_matrix_bits": 2,
        "present_pair_xor_cell_matrix_bits": 3,
        "present_pair_xor_paligned_cell_matrix_bits": 4,
        "present_pair_xor_paligned_sinv_cell_matrix_bits": 5,
        "present_pair_xor_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits": 50,
        "present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits": 48,
        "present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits": 30,
        "present_pair_xor_paligned_sboxddt_cell_matrix_bits": 6,
        "present_pair_xor_paligned_sboxddt_top2_cell_matrix_bits": 8,
        "present_pair_xor_paligned_sboxddt_back2_cell_matrix_bits": 9,
        "present_pair_xor_paligned_sboxddt_beam2_cell_matrix_bits": 12,
        "present_pair_xor_paligned_sboxddt_beam4deep3_cell_matrix_bits": 49,
        "present_xor_paligned_cell_matrix_bits": 2,
        "present_nibble_paligned_view": 2,
        "ciphertext_xor_bits": 1,
        "ciphertext_xor_spn_aligned_bits": 2,
        "ciphertext_xor_spn_paligned_bits": 2,
        "ciphertext_pair_xor_bits": 3,
        "ciphertext_pair_xor_spn_aligned_bits": 4,
        "ciphertext_pair_xor_arx_aligned_bits": 4,
        "ciphertext_pair_xor_arx_partial_inverse_bits": 7,
        "ciphertext_pair_xor_arx_partial_inverse_rx_bits": 11,
        "ciphertext_pair_xor_arx_partial_inverse_rx_carrychain_bits": 17,
        "ciphertext_pair_xor_arx_partial_inverse_rx_carrychain_plus_bits": 23,
    }
    try:
        return block_bits * widths[feature_encoding]
    except KeyError as exc:
        raise ValueError(f"unsupported feature encoding: {feature_encoding}") from exc
