from __future__ import annotations

from blockcipher_nd.ciphers import ReducedRoundCipher
from blockcipher_nd.features.encoders.bitwise import int_to_bits
from blockcipher_nd.features.encoders.present_sbox_ddt import (
    present_sbox_ddt_beam_statistics_words,
    present_sbox_ddt_beam_words,
    present_sbox_ddt_top2_margin_words,
    present_sbox_ddt_top2_words,
    present_sbox_ddt_words,
    present_structural_inverse_sbox_difference,
)
from blockcipher_nd.features.spn_aligned import inverse_permutation_difference


def present_mcnd_cell_matrix_bits(left: int, right: int, width: int) -> list[int]:
    return words_to_present_cell_matrix_bits([left, right], width, "present_mcnd_cell_matrix_bits")


def present_pair_xor_cell_matrix_bits(left: int, right: int, width: int) -> list[int]:
    return words_to_present_cell_matrix_bits([left, right, left ^ right], width, "present_pair_xor_cell_matrix_bits")


def present_pair_xor_paligned_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    return words_to_present_cell_matrix_bits(
        [left, right, difference, aligned_difference],
        width,
        "present_pair_xor_paligned_cell_matrix_bits",
    )


def present_pair_xor_paligned_sinv_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    structural_inverse_difference = present_structural_inverse_sbox_difference(left, right, width, cipher)
    return words_to_present_cell_matrix_bits(
        [left, right, difference, aligned_difference, structural_inverse_difference],
        width,
        "present_pair_xor_paligned_sinv_cell_matrix_bits",
    )


def present_pair_xor_paligned_sboxddt_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    best_input_difference, ddt_confidence = present_sbox_ddt_words(aligned_difference, width)
    return words_to_present_cell_matrix_bits(
        [left, right, difference, aligned_difference, best_input_difference, ddt_confidence],
        width,
        "present_pair_xor_paligned_sboxddt_cell_matrix_bits",
    )


def present_pair_xor_paligned_sboxddt_top2_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    top1, top2, confidence1, confidence2 = present_sbox_ddt_top2_words(aligned_difference, width)
    return words_to_present_cell_matrix_bits(
        [left, right, difference, aligned_difference, top1, top2, confidence1, confidence2],
        width,
        "present_pair_xor_paligned_sboxddt_top2_cell_matrix_bits",
    )



def present_pair_xor_paligned_sboxddt_back2_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    layer1, confidence1 = present_sbox_ddt_words(aligned_difference, width)
    layer1_paligned = inverse_permutation_difference(layer1, width, cipher)
    layer2, confidence2 = present_sbox_ddt_words(layer1_paligned, width)
    return words_to_present_cell_matrix_bits(
        [
            left,
            right,
            difference,
            aligned_difference,
            layer1,
            confidence1,
            layer1_paligned,
            layer2,
            confidence2,
        ],
        width,
        "present_pair_xor_paligned_sboxddt_back2_cell_matrix_bits",
    )


def present_pair_xor_paligned_sboxddt_beam2_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    layer1_top1, layer1_top2, confidence1, confidence2, margin1 = present_sbox_ddt_top2_margin_words(
        aligned_difference,
        width,
    )
    layer1_top1_paligned = inverse_permutation_difference(layer1_top1, width, cipher)
    layer1_top2_paligned = inverse_permutation_difference(layer1_top2, width, cipher)
    layer2_from_top1, _layer2_confidence1 = present_sbox_ddt_words(layer1_top1_paligned, width)
    layer2_from_top2, _layer2_confidence2 = present_sbox_ddt_words(layer1_top2_paligned, width)
    beam_disagreement = layer2_from_top1 ^ layer2_from_top2
    return words_to_present_cell_matrix_bits(
        [
            left,
            right,
            difference,
            aligned_difference,
            layer1_top1,
            layer1_top2,
            confidence1,
            confidence2,
            margin1,
            layer2_from_top1,
            layer2_from_top2,
            beam_disagreement,
        ],
        width,
        "present_pair_xor_paligned_sboxddt_beam2_cell_matrix_bits",
    )


def present_pair_xor_paligned_sboxddt_beam4deep3_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    trail_words = present_sbox_ddt_beam_words(
        aligned_difference,
        width,
        cipher,
        beam_width=4,
        depth=3,
    )
    return words_to_present_cell_matrix_bits(
        [
            left,
            right,
            difference,
            aligned_difference,
            *trail_words,
        ],
        width,
        "present_pair_xor_paligned_sboxddt_beam4deep3_cell_matrix_bits",
    )


def present_pair_xor_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    structural_inverse_difference = present_structural_inverse_sbox_difference(left, right, width, cipher)
    trail_words = present_sbox_ddt_beam_words(
        structural_inverse_difference,
        width,
        cipher,
        beam_width=4,
        depth=3,
    )
    return words_to_present_cell_matrix_bits(
        [
            left,
            right,
            difference,
            aligned_difference,
            structural_inverse_difference,
            *trail_words,
        ],
        width,
        "present_pair_xor_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits",
    )


def present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    structural_inverse_difference = present_structural_inverse_sbox_difference(left, right, width, cipher)
    trail_words = present_sbox_ddt_beam_words(
        structural_inverse_difference,
        width,
        cipher,
        beam_width=4,
        depth=3,
    )
    return words_to_present_cell_matrix_bits(
        [
            difference,
            aligned_difference,
            structural_inverse_difference,
            *trail_words,
        ],
        width,
        "present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits",
    )


def present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    structural_inverse_difference = present_structural_inverse_sbox_difference(left, right, width, cipher)
    trail_stats = present_sbox_ddt_beam_statistics_words(
        structural_inverse_difference,
        width,
        cipher,
        beam_width=4,
        depth=3,
    )
    return words_to_present_cell_matrix_bits(
        [
            difference,
            aligned_difference,
            structural_inverse_difference,
            *trail_stats,
        ],
        width,
        "present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits",
    )


def parameterized_present_sboxddt_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
    *,
    include_pair: bool,
    include_sinv: bool,
    use_statistics: bool,
    beam_width: int,
    depth: int,
    feature_encoding: str,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    source_difference = aligned_difference
    prefix_words = [difference, aligned_difference]
    if include_pair:
        prefix_words = [left, right, *prefix_words]
    if include_sinv:
        source_difference = present_structural_inverse_sbox_difference(left, right, width, cipher)
        prefix_words.append(source_difference)
    if use_statistics:
        trail_words = present_sbox_ddt_beam_statistics_words(
            source_difference,
            width,
            cipher,
            beam_width=beam_width,
            depth=depth,
        )
    else:
        trail_words = present_sbox_ddt_beam_words(
            source_difference,
            width,
            cipher,
            beam_width=beam_width,
            depth=depth,
        )
    return words_to_present_cell_matrix_bits(
        [*prefix_words, *trail_words],
        width,
        feature_encoding,
    )


def present_xor_paligned_cell_matrix_bits(
    left: int,
    right: int,
    width: int,
    cipher: ReducedRoundCipher,
) -> list[int]:
    difference = left ^ right
    aligned_difference = inverse_permutation_difference(difference, width, cipher)
    return words_to_present_cell_matrix_bits(
        [difference, aligned_difference],
        width,
        "present_xor_paligned_cell_matrix_bits",
    )


def words_to_present_cell_matrix_bits(words: list[int], width: int, feature_encoding: str) -> list[int]:
    if width % 4 != 0:
        raise ValueError(f"{feature_encoding} requires a 4-bit cell block size")
    bits = []
    for word in words:
        bits.extend(int_to_bits(word, width))
    cells = [bits[index : index + 4] for index in range(0, len(bits), 4)]
    return [cell[bit_index] for bit_index in range(4) for cell in cells]
