import pytest

from blockcipher_nd.ciphers.spn.skinny import (
    SKINNY64_SBOX,
    Skinny64,
    add_constants,
    cells_to_int,
    generate_round_constants,
    int_to_cells,
    mix_columns,
    shift_rows,
    skinny64_round,
)


def test_skinny64_state_mapping_is_row_major_msb_first() -> None:
    cells = int_to_cells(0x06034F957724D19D)

    assert cells == (
        0x0,
        0x6,
        0x0,
        0x3,
        0x4,
        0xF,
        0x9,
        0x5,
        0x7,
        0x7,
        0x2,
        0x4,
        0xD,
        0x1,
        0x9,
        0xD,
    )
    assert cells_to_int(cells) == 0x06034F957724D19D


def test_skinny64_round_components_match_public_specification() -> None:
    assert SKINNY64_SBOX == (
        0xC,
        0x6,
        0x9,
        0x0,
        0x1,
        0xA,
        0x2,
        0xB,
        0x3,
        0x8,
        0x5,
        0xD,
        0x4,
        0xE,
        0x7,
        0xF,
    )
    assert generate_round_constants(8) == (
        0x01,
        0x03,
        0x07,
        0x0F,
        0x1F,
        0x3E,
        0x3D,
        0x3B,
    )
    assert shift_rows(tuple(range(16))) == (
        0,
        1,
        2,
        3,
        7,
        4,
        5,
        6,
        10,
        11,
        8,
        9,
        13,
        14,
        15,
        12,
    )
    assert mix_columns(tuple(range(16))) == (
        4,
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        12,
        12,
        12,
        12,
        8,
        8,
        8,
        8,
    )
    assert add_constants((0,) * 16, 0x01) == (
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_skinny64_first_round_zero_state_is_auditable() -> None:
    assert skinny64_round((0,) * 16, (0,) * 16, 0x01) == (
        0xD,
        0xC,
        0xE,
        0xC,
        0xD,
        0xC,
        0xC,
        0xC,
        0x0,
        0x0,
        0x2,
        0x0,
        0x1,
        0x0,
        0x2,
        0x0,
    )


def test_skinny64_matches_public_appendix_b_vector() -> None:
    cipher = Skinny64(rounds=32, key=0xF5269826FC681238)

    assert cipher.encrypt(0x06034F957724D19D) == 0xBB39DFB2429B8AC7


def test_skinny64_rejects_out_of_range_parameters() -> None:
    with pytest.raises(ValueError, match="rounds"):
        Skinny64(rounds=33)
    with pytest.raises(ValueError, match="key"):
        Skinny64(key=1 << 64)
    with pytest.raises(ValueError, match="plaintext|block"):
        Skinny64().encrypt(1 << 64)
