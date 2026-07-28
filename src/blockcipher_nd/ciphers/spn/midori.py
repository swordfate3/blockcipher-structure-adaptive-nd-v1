"""Specification-faithful Midori64 with prefix-reduced round semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


MIDORI64_SBOX = (
    0xC,
    0xA,
    0xD,
    0x3,
    0xE,
    0xB,
    0xF,
    0x7,
    0x8,
    0x9,
    0x1,
    0x5,
    0x0,
    0x2,
    0x4,
    0x6,
)
MIDORI64_SHUFFLE = (0, 10, 5, 15, 14, 4, 11, 1, 9, 3, 12, 6, 7, 13, 2, 8)
MIDORI64_BETA = (
    0x15B3,
    0x78C0,
    0xA435,
    0x6213,
    0x104F,
    0xD170,
    0x0266,
    0x0BCC,
    0x9481,
    0x40B8,
    0x7197,
    0x228E,
    0x5130,
    0xF8CA,
    0xDF90,
)
MIDORI64_ROUNDS = 16
MIDORI64_FULL_ROUNDS = 15
_BLOCK_MASK = (1 << 64) - 1
_KEY_MASK = (1 << 128) - 1


def _expand_beta(beta: int) -> int:
    expanded = 0
    for index in range(16):
        expanded = (expanded << 4) | ((beta >> (15 - index)) & 1)
    return expanded


MIDORI64_ROUND_CONSTANTS = tuple(_expand_beta(beta) for beta in MIDORI64_BETA)


def _int_to_cells(block: int) -> tuple[int, ...]:
    _validate_block(block, label="block")
    return tuple((block >> (4 * (15 - index))) & 0xF for index in range(16))


def _cells_to_int(cells: Sequence[int]) -> int:
    values = tuple(cells)
    if len(values) != 16:
        raise ValueError("Midori64 state must contain 16 cells")
    if not all(type(value) is int and 0 <= value < 16 for value in values):
        raise ValueError("Midori64 cells must be four-bit integers")
    state = 0
    for value in values:
        state = (state << 4) | value
    return state


def midori64_sub_cells(state: int) -> int:
    return _cells_to_int(tuple(MIDORI64_SBOX[value] for value in _int_to_cells(state)))


def midori64_shuffle_cells(state: int) -> int:
    cells = _int_to_cells(state)
    return _cells_to_int(tuple(cells[source] for source in MIDORI64_SHUFFLE))


def midori64_inverse_shuffle_cells(state: int) -> int:
    cells = _int_to_cells(state)
    inverse = [0] * 16
    for target, source in enumerate(MIDORI64_SHUFFLE):
        inverse[source] = cells[target]
    return _cells_to_int(inverse)


def midori64_linear_layer(state: int) -> int:
    """Apply the canonical MIDORI MixColumn primitive without ShuffleCell."""

    cells = _int_to_cells(state)
    mixed: list[int] = []
    for offset in range(0, 16, 4):
        column = cells[offset : offset + 4]
        total = column[0] ^ column[1] ^ column[2] ^ column[3]
        mixed.extend(total ^ value for value in column)
    return _cells_to_int(mixed)


def midori64_round_linear_layer(state: int) -> int:
    """Apply the complete post-S-box Midori64 linear transition."""

    return midori64_linear_layer(midori64_shuffle_cells(state))


def midori64_round_keys(key: int) -> tuple[int, ...]:
    _validate_key(key)
    halves = (key >> 64, key & _BLOCK_MASK)
    return tuple(
        halves[round_index % 2] ^ MIDORI64_ROUND_CONSTANTS[round_index]
        for round_index in range(MIDORI64_FULL_ROUNDS)
    )


def midori64_round_trace(plaintext: int, key: int) -> tuple[int, ...]:
    """Return the state after each of the 16 Midori64 data rounds."""

    plaintext = _validate_block(plaintext, label="plaintext")
    key = _validate_key(key)
    key0, key1 = key >> 64, key & _BLOCK_MASK
    whitening_key = key0 ^ key1
    round_keys = midori64_round_keys(key)
    state = plaintext ^ whitening_key
    trace: list[int] = []
    for round_key in round_keys:
        state = midori64_sub_cells(state)
        state = midori64_round_linear_layer(state)
        state ^= round_key
        trace.append(state)
    state = midori64_sub_cells(state) ^ whitening_key
    trace.append(state)
    if len(trace) != MIDORI64_ROUNDS:
        raise AssertionError("Midori64 trace length must equal 16 rounds")
    return tuple(trace)


@dataclass(frozen=True)
class Midori64:
    rounds: int = MIDORI64_ROUNDS
    key: int = 0

    name: str = "Midori64"
    structure: str = "SPN"
    block_bits: int = 64
    key_bits: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.rounds, int):
            raise TypeError("rounds must be an integer")
        if self.rounds < 1 or self.rounds > MIDORI64_ROUNDS:
            raise ValueError("Midori64 supports 1..16 rounds")
        _validate_key(self.key)

    def encrypt(self, plaintext: int) -> int:
        return midori64_round_trace(plaintext, self.key)[self.rounds - 1]


def _validate_block(value: int, *, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0 or value > _BLOCK_MASK:
        raise ValueError(f"{label} must fit in 64 bits")
    return value


def _validate_key(value: int) -> int:
    if not isinstance(value, int):
        raise TypeError("key must be an integer")
    if value < 0 or value > _KEY_MASK:
        raise ValueError("key must fit in 128 bits")
    return value


__all__ = [
    "MIDORI64_BETA",
    "MIDORI64_ROUNDS",
    "MIDORI64_ROUND_CONSTANTS",
    "MIDORI64_SBOX",
    "MIDORI64_SHUFFLE",
    "Midori64",
    "midori64_inverse_shuffle_cells",
    "midori64_linear_layer",
    "midori64_round_keys",
    "midori64_round_linear_layer",
    "midori64_round_trace",
    "midori64_shuffle_cells",
    "midori64_sub_cells",
]
