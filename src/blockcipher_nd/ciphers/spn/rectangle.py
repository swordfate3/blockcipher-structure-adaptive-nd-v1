from __future__ import annotations

from dataclasses import dataclass


RECTANGLE_SBOX = (
    0x6,
    0x5,
    0xC,
    0xA,
    0x1,
    0xE,
    0x7,
    0x9,
    0xB,
    0x0,
    0x3,
    0xD,
    0x8,
    0xF,
    0x4,
    0x2,
)

RECTANGLE_ROUND_CONSTANTS = (
    0x01,
    0x02,
    0x04,
    0x09,
    0x12,
    0x05,
    0x0B,
    0x16,
    0x0C,
    0x19,
    0x13,
    0x07,
    0x0F,
    0x1F,
    0x1E,
    0x1C,
    0x18,
    0x11,
    0x03,
    0x06,
    0x0D,
    0x1B,
    0x17,
    0x0E,
    0x1D,
)

RECTANGLE_ROW_ROTATIONS = (0, 1, 12, 13)
_MASK16 = (1 << 16) - 1
_MASK64 = (1 << 64) - 1
_MASK80 = (1 << 80) - 1


def _rol16(value: int, amount: int) -> int:
    amount %= 16
    return ((value << amount) | (value >> (16 - amount))) & _MASK16


def _rows_from_int(value: int, count: int) -> list[int]:
    return [(value >> (16 * row)) & _MASK16 for row in range(count)]


def _rows_to_int(rows: list[int]) -> int:
    return sum((row & _MASK16) << (16 * index) for index, row in enumerate(rows))


def rectangle_sub_columns(state: int) -> int:
    rows = _rows_from_int(state, 4)
    output = [0, 0, 0, 0]
    for column in range(16):
        value = sum(((rows[row] >> column) & 1) << row for row in range(4))
        substituted = RECTANGLE_SBOX[value]
        for row in range(4):
            output[row] |= ((substituted >> row) & 1) << column
    return _rows_to_int(output)


def rectangle_shift_rows(state: int) -> int:
    rows = _rows_from_int(state, 4)
    return _rows_to_int(
        [
            _rol16(row, RECTANGLE_ROW_ROTATIONS[index])
            for index, row in enumerate(rows)
        ]
    )


def rectangle_player() -> tuple[int, ...]:
    return tuple(
        16 * row + ((column + RECTANGLE_ROW_ROTATIONS[row]) % 16)
        for row in range(4)
        for column in range(16)
    )


def update_rectangle80_key(key: int, round_index: int) -> int:
    if key < 0 or key > _MASK80:
        raise ValueError("RECTANGLE-80 key must fit in 80 bits")
    if round_index not in range(25):
        raise ValueError("round_index must be in [0, 24]")
    rows = _rows_from_int(key, 5)
    for column in range(4):
        value = sum(((rows[row] >> column) & 1) << row for row in range(4))
        substituted = RECTANGLE_SBOX[value]
        for row in range(4):
            rows[row] &= ~(1 << column)
            rows[row] |= ((substituted >> row) & 1) << column
    old = rows
    rows = [
        _rol16(old[0], 8) ^ old[1],
        old[2],
        old[3],
        _rol16(old[3], 12) ^ old[4],
        old[0],
    ]
    rows[0] ^= RECTANGLE_ROUND_CONSTANTS[round_index]
    return _rows_to_int(rows) & _MASK80


@dataclass(frozen=True)
class Rectangle80:
    rounds: int = 25
    key: int = 0

    name: str = "RECTANGLE-80"
    structure: str = "SPN"
    block_bits: int = 64
    key_bits: int = 80

    def __post_init__(self) -> None:
        if self.rounds < 1 or self.rounds > 25:
            raise ValueError("RECTANGLE-80 supports 1..25 rounds")
        if self.key < 0 or self.key > _MASK80:
            raise ValueError("RECTANGLE-80 key must fit in 80 bits")

    def encrypt(self, plaintext: int) -> int:
        if plaintext < 0 or plaintext > _MASK64:
            raise ValueError("RECTANGLE-80 plaintext must fit in 64 bits")
        state = plaintext
        key = self.key
        for round_index in range(self.rounds):
            state ^= key & _MASK64
            state = rectangle_sub_columns(state)
            state = rectangle_shift_rows(state)
            key = update_rectangle80_key(key, round_index)
        return (state ^ (key & _MASK64)) & _MASK64


__all__ = [
    "RECTANGLE_ROUND_CONSTANTS",
    "RECTANGLE_ROW_ROTATIONS",
    "RECTANGLE_SBOX",
    "Rectangle80",
    "rectangle_player",
    "rectangle_shift_rows",
    "rectangle_sub_columns",
    "update_rectangle80_key",
]
