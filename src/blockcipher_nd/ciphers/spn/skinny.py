from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


SKINNY64_SBOX = (
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

SKINNY64_TK_PERMUTATION = (
    9,
    15,
    8,
    13,
    10,
    14,
    12,
    11,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
)

SKINNY64_ROUNDS = 32
_BLOCK_MASK = (1 << 64) - 1


def int_to_cells(block: int) -> tuple[int, ...]:
    if not isinstance(block, int):
        raise TypeError("block must be an integer")
    if block < 0 or block > _BLOCK_MASK:
        raise ValueError("block must fit in 64 bits")
    return tuple((block >> (4 * (15 - index))) & 0xF for index in range(16))


def cells_to_int(cells: Sequence[int]) -> int:
    state = _validate_cells(cells, label="cells")
    block = 0
    for cell in state:
        block = (block << 4) | cell
    return block


def generate_round_constants(rounds: int = SKINNY64_ROUNDS) -> tuple[int, ...]:
    if not isinstance(rounds, int):
        raise TypeError("rounds must be an integer")
    if rounds < 0 or rounds > SKINNY64_ROUNDS:
        raise ValueError("rounds must be between 0 and 32")
    constants: list[int] = []
    constant = 0x01
    for _ in range(rounds):
        constants.append(constant)
        feedback = ((constant >> 5) ^ (constant >> 4) ^ 0x01) & 0x01
        constant = ((constant << 1) & 0x3F) | feedback
    return tuple(constants)


def sub_cells(cells: Sequence[int]) -> tuple[int, ...]:
    return tuple(SKINNY64_SBOX[cell] for cell in _validate_cells(cells, label="cells"))


def add_constants(cells: Sequence[int], constant: int) -> tuple[int, ...]:
    if not isinstance(constant, int):
        raise TypeError("constant must be an integer")
    if constant < 0 or constant >= (1 << 6):
        raise ValueError("constant must fit in 6 bits")
    state = list(_validate_cells(cells, label="cells"))
    state[0] ^= constant & 0xF
    state[4] ^= (constant >> 4) & 0x3
    state[8] ^= 0x2
    return tuple(state)


def add_round_tweakey(
    cells: Sequence[int], tweakey: Sequence[int]
) -> tuple[int, ...]:
    state = list(_validate_cells(cells, label="cells"))
    tk = _validate_cells(tweakey, label="tweakey")
    for index in range(8):
        state[index] ^= tk[index]
    return tuple(state)


def shift_rows(cells: Sequence[int]) -> tuple[int, ...]:
    state = _validate_cells(cells, label="cells")
    return (
        state[0],
        state[1],
        state[2],
        state[3],
        state[7],
        state[4],
        state[5],
        state[6],
        state[10],
        state[11],
        state[8],
        state[9],
        state[13],
        state[14],
        state[15],
        state[12],
    )


def mix_columns(cells: Sequence[int]) -> tuple[int, ...]:
    state = _validate_cells(cells, label="cells")
    mixed = [0] * 16
    for column in range(4):
        s0 = state[column]
        s1 = state[4 + column]
        s2 = state[8 + column]
        s3 = state[12 + column]
        mixed[column] = s0 ^ s2 ^ s3
        mixed[4 + column] = s0
        mixed[8 + column] = s1 ^ s2
        mixed[12 + column] = s0 ^ s2
    return tuple(mixed)


def permute_tweakey(tweakey: Sequence[int]) -> tuple[int, ...]:
    tk = _validate_cells(tweakey, label="tweakey")
    return tuple(tk[index] for index in SKINNY64_TK_PERMUTATION)


def skinny64_round(
    cells: Sequence[int], tweakey: Sequence[int], constant: int
) -> tuple[int, ...]:
    state = sub_cells(cells)
    state = add_constants(state, constant)
    state = add_round_tweakey(state, tweakey)
    state = shift_rows(state)
    return mix_columns(state)


@dataclass(frozen=True)
class Skinny64:
    rounds: int = SKINNY64_ROUNDS
    key: int = 0

    name: str = "SKINNY-64/64"
    structure: str = "SPN"
    block_bits: int = 64
    key_bits: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.rounds, int):
            raise TypeError("rounds must be an integer")
        if self.rounds < 0 or self.rounds > SKINNY64_ROUNDS:
            raise ValueError("rounds must be between 0 and 32")
        if not isinstance(self.key, int):
            raise TypeError("key must be an integer")
        if self.key < 0 or self.key > _BLOCK_MASK:
            raise ValueError("key must fit in 64 bits")

    def encrypt(self, plaintext: int) -> int:
        state = int_to_cells(plaintext)
        tweakey = int_to_cells(self.key)
        for constant in generate_round_constants(self.rounds):
            state = skinny64_round(state, tweakey, constant)
            tweakey = permute_tweakey(tweakey)
        return cells_to_int(state)


def _validate_cells(cells: Sequence[int], *, label: str) -> tuple[int, ...]:
    if len(cells) != 16:
        raise ValueError(f"{label} must contain exactly 16 cells")
    state = tuple(cells)
    for cell in state:
        if not isinstance(cell, int):
            raise TypeError(f"{label} must contain integers")
        if cell < 0 or cell > 0xF:
            raise ValueError(f"{label} must contain only 4-bit cells")
    return state
