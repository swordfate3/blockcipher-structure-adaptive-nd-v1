"""Dialga-128 from the published CC BY 4.0 specification.

The tables and algorithm below transcribe Banik et al., "Dialga: A Family of
Low-Latency Tweakable Block Ciphers Using Multiple Linear Layers", ToSC 2025(4),
Tables 1--4 and 18 and Algorithm 1.  The implementation exposes the 16- and
20-round variants plus their data-state traces so reduced-round experiments can
use the paper's non-round-aligned schedule without inventing an iterative round.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


DIALGA128_SUPPORTED_ROUNDS = (16, 20)
DIALGA_SBOX = (
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
DIALGA_BIT_PERMUTATIONS = (
    (4, 1, 6, 3, 0, 5, 2, 7),
    (1, 6, 7, 0, 5, 2, 3, 4),
    (2, 3, 4, 1, 6, 7, 0, 5),
    (7, 4, 1, 2, 3, 0, 5, 6),
)
DIALGA_BYTE_PERMUTATIONS = (
    (7, 0, 13, 10, 5, 2, 15, 8, 4, 3, 14, 9, 6, 1, 12, 11),
    (13, 0, 10, 7, 11, 6, 12, 1, 2, 15, 5, 8, 4, 9, 3, 14),
    (7, 13, 10, 0, 6, 12, 11, 1, 5, 15, 8, 2, 4, 14, 9, 3),
    (13, 8, 6, 3, 14, 11, 5, 0, 12, 9, 7, 2, 15, 10, 4, 1),
)
DIALGA_MIDDLE_PERMUTATION = (
    0,
    10,
    5,
    15,
    14,
    4,
    11,
    1,
    9,
    3,
    12,
    6,
    7,
    13,
    2,
    8,
)
DIALGA_FORWARD_CONSTANTS = (
    0x243F6A8885A308D313198A2E03707344,
    0xA4093822299F31D0082EFA98EC4E6C89,
    0x452821E638D01377BE5466CF34E90C6C,
    0xC0AC29B7C97C50DD3F84D5B5B5470917,
    0x9216D5D98979FB1BD1310BA698DFB5AC,
    0x2FFD72DBD01ADFB7B8E1AFED6A267E96,
    0xBA7C9045F12C7F9924A19947B3916CF7,
    0x0801F2E2858EFC16636920D871574E69,
    0xA458FEA3F4933D7E0D95748F728EB658,
    0x718BCD5882154AEE7B54A41DC25A59B5,
)
DIALGA_MIDDLE_CONSTANTS = (
    0x9C30D5392AF26013C5D1B023286085F0,
    0xCA417918B8DB38EF8E79DCB0603A180E,
)
DIALGA_BACKWARD_CONSTANTS = (
    0x6C9E0E8BB01E8A3ED71577C1BD314B27,
    0x78AF2FDA55605C60E65525F3AA55AB94,
    0x5748986263E8144055CA396A2AAB10B6,
    0xB4CC5C341141E8CEA15486AF7C72E993,
    0xB3EE1411636FBC2A2BA9C55D741831F6,
    0xCE5C3E169B87931EAFD6BA336C24CF5C,
    0x7A325381289586773B8F48986B4BB9AF,
    0xC4BFE81B6628219361D809CCFB21A991,
)

_BLOCK_BITS = 128
_KEY_BITS = 256
_BLOCK_MASK = (1 << _BLOCK_BITS) - 1
_KEY_MASK = (1 << _KEY_BITS) - 1


def _inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for target, source in enumerate(permutation):
        inverse[source] = target
    return tuple(inverse)


def _permute_byte(value: int, permutation: Sequence[int]) -> int:
    output = 0
    for target, source in enumerate(permutation):
        source_bit = (value >> (7 - source)) & 1
        output |= source_bit << (7 - target)
    return output


def _conjugated_sbox_table(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = _inverse_permutation(permutation)
    table: list[int] = []
    for value in range(256):
        permuted = _permute_byte(value, permutation)
        substituted = (DIALGA_SBOX[permuted >> 4] << 4) | DIALGA_SBOX[
            permuted & 0xF
        ]
        table.append(_permute_byte(substituted, inverse))
    return tuple(table)


DIALGA_BYTE_SBOXES = tuple(
    _conjugated_sbox_table(permutation)
    for permutation in DIALGA_BIT_PERMUTATIONS
)
DIALGA_BYTE_INVERSE_SBOXES = tuple(
    tuple(_inverse_permutation(table)) for table in DIALGA_BYTE_SBOXES
)


def _validate_block(value: int, *, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0 or value > _BLOCK_MASK:
        raise ValueError(f"{label} must fit in 128 bits")
    return value


def _validate_key(key: int) -> int:
    if not isinstance(key, int):
        raise TypeError("key must be an integer")
    if key < 0 or key > _KEY_MASK:
        raise ValueError("key must fit in 256 bits")
    return key


def _to_bytes(state: int) -> tuple[int, ...]:
    return tuple(state.to_bytes(16, byteorder="big"))


def _from_bytes(state: Sequence[int]) -> int:
    if len(state) != 16:
        raise ValueError("Dialga state must contain exactly 16 bytes")
    return int.from_bytes(bytes(state), byteorder="big")


def dialga_sub_cells(state: int) -> int:
    values = _to_bytes(_validate_block(state, label="state"))
    return _from_bytes(
        tuple(
            DIALGA_BYTE_SBOXES[index % 4][value]
            for index, value in enumerate(values)
        )
    )


def dialga_inverse_sub_cells(state: int) -> int:
    values = _to_bytes(_validate_block(state, label="state"))
    return _from_bytes(
        tuple(
            DIALGA_BYTE_INVERSE_SBOXES[index % 4][value]
            for index, value in enumerate(values)
        )
    )


def _permute_state(state: int, permutation: Sequence[int]) -> int:
    values = _to_bytes(_validate_block(state, label="state"))
    return _from_bytes(tuple(values[source] for source in permutation))


def dialga_mix_columns(state: int) -> int:
    values = _to_bytes(_validate_block(state, label="state"))
    mixed = [0] * 16
    for column in range(4):
        offset = 4 * column
        s0, s1, s2, s3 = values[offset : offset + 4]
        mixed[offset] = s1 ^ s2 ^ s3
        mixed[offset + 1] = s0 ^ s2 ^ s3
        mixed[offset + 2] = s0 ^ s1 ^ s3
        mixed[offset + 3] = s0 ^ s1 ^ s2
    return _from_bytes(mixed)


def dialga_linear_layer(state: int, round_type: int) -> int:
    if not isinstance(round_type, int):
        raise TypeError("round_type must be an integer")
    if round_type < 0 or round_type >= 4:
        raise ValueError("round_type must be in [0, 3]")
    state = _permute_state(state, DIALGA_BYTE_PERMUTATIONS[round_type])
    return dialga_mix_columns(state)


def dialga_inverse_linear_layer(state: int, round_type: int) -> int:
    if not isinstance(round_type, int):
        raise TypeError("round_type must be an integer")
    if round_type < 0 or round_type >= 4:
        raise ValueError("round_type must be in [0, 3]")
    state = dialga_mix_columns(state)
    return _permute_state(
        state, _inverse_permutation(DIALGA_BYTE_PERMUTATIONS[round_type])
    )


def dialga_round_function(state: int, round_type: int) -> int:
    return dialga_linear_layer(dialga_sub_cells(state), round_type)


def dialga_inverse_round_function(state: int, round_type: int) -> int:
    return dialga_inverse_sub_cells(dialga_inverse_linear_layer(state, round_type))


def dialga_middle_shuffle(state: int) -> int:
    return _permute_state(state, DIALGA_MIDDLE_PERMUTATION)


def dialga128_round_trace(
    plaintext: int,
    key: int,
    tweak: int,
    *,
    total_rounds: int = 16,
) -> tuple[int, ...]:
    """Return the data state after each of the 16 or 20 data rounds."""

    plaintext = _validate_block(plaintext, label="plaintext")
    key = _validate_key(key)
    tweak = _validate_block(tweak, label="tweak")
    if total_rounds not in DIALGA128_SUPPORTED_ROUNDS:
        raise ValueError("Dialga-128 total_rounds must be 16 or 20")

    alpha = 4 if total_rounds == 16 else 5
    beta = 3 if total_rounds == 16 else 4
    key_parts = (key >> 128, key & _BLOCK_MASK)
    data_state = plaintext ^ tweak ^ key_parts[0] ^ key_parts[1]
    tweak_state = tweak
    trace: list[int] = []

    for iteration in range(1, alpha + 1):
        key_index = (iteration - 1) % 2
        if iteration == 1:
            tweak_state ^= key_parts[key_index]
        else:
            tweak_state = dialga_round_function(
                tweak_state, (iteration - 1) % 4
            ) ^ key_parts[key_index]

        data_state = (
            dialga_round_function(data_state, (2 * iteration - 2) % 4)
            ^ key_parts[iteration % 2]
            ^ DIALGA_FORWARD_CONSTANTS[2 * (iteration - 1)]
        )
        trace.append(data_state)
        data_state = (
            dialga_round_function(data_state, (2 * iteration - 1) % 4)
            ^ tweak_state
            ^ DIALGA_FORWARD_CONSTANTS[2 * iteration - 1]
        )
        trace.append(data_state)

    tweak_state = dialga_sub_cells(tweak_state)
    data_state = (
        dialga_round_function(data_state, (2 * alpha) % 4)
        ^ key_parts[(alpha + 1) % 2]
        ^ DIALGA_MIDDLE_CONSTANTS[0]
        ^ tweak_state
    )
    trace.append(data_state)
    tweak_state = dialga_inverse_sub_cells(tweak_state) ^ key_parts[
        (alpha - 1) % 2
    ]
    tweak_state = dialga_inverse_round_function(
        tweak_state, (alpha - 1) % 4
    )
    data_state = (
        dialga_round_function(data_state, (2 * alpha + 1) % 4)
        ^ dialga_middle_shuffle(tweak_state)
        ^ DIALGA_MIDDLE_CONSTANTS[1]
    )
    trace.append(data_state)

    for iteration in range(1, beta + 1):
        key_index = (alpha - iteration - 1) % 2
        if iteration == beta:
            tweak_state ^= key_parts[key_index]
        else:
            tweak_state = dialga_inverse_round_function(
                tweak_state ^ key_parts[key_index],
                (alpha - iteration - 1) % 4,
            )

        data_state = (
            dialga_round_function(data_state, (2 * (alpha + iteration)) % 4)
            ^ key_parts[(alpha + iteration + 1) % 2]
            ^ DIALGA_BACKWARD_CONSTANTS[2 * (iteration - 1)]
        )
        trace.append(data_state)
        data_state = (
            dialga_round_function(
                data_state, (2 * (alpha + iteration) + 1) % 4
            )
            ^ dialga_middle_shuffle(tweak_state)
            ^ DIALGA_BACKWARD_CONSTANTS[2 * iteration - 1]
        )
        trace.append(data_state)

    if len(trace) != total_rounds:
        raise AssertionError("Dialga-128 trace length does not match total rounds")
    return tuple(trace)


def dialga128_encrypt(
    plaintext: int,
    key: int,
    tweak: int,
    *,
    total_rounds: int = 16,
) -> int:
    trace = dialga128_round_trace(
        plaintext,
        key,
        tweak,
        total_rounds=total_rounds,
    )
    key = _validate_key(key)
    key0, key1 = key >> 128, key & _BLOCK_MASK
    return dialga_sub_cells(trace[-1]) ^ key0 ^ key1


@dataclass(frozen=True)
class Dialga128:
    rounds: int = 16
    key: int = 0
    tweak: int = 0
    variant_rounds: int = 16

    name: str = "Dialga-128"
    structure: str = "SPN"
    block_bits: int = 128
    key_bits: int = 256
    tweak_bits: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.variant_rounds, int):
            raise TypeError("variant_rounds must be an integer")
        if self.variant_rounds not in DIALGA128_SUPPORTED_ROUNDS:
            raise ValueError("Dialga-128 variant_rounds must be 16 or 20")
        if not isinstance(self.rounds, int):
            raise TypeError("rounds must be an integer")
        if self.rounds < 1 or self.rounds > self.variant_rounds:
            raise ValueError("rounds must be between 1 and variant_rounds")
        _validate_key(self.key)
        _validate_block(self.tweak, label="tweak")

    def encrypt(self, plaintext: int) -> int:
        trace = dialga128_round_trace(
            plaintext,
            self.key,
            self.tweak,
            total_rounds=self.variant_rounds,
        )
        if self.rounds < self.variant_rounds:
            return trace[self.rounds - 1]
        key0, key1 = self.key >> 128, self.key & _BLOCK_MASK
        return dialga_sub_cells(trace[-1]) ^ key0 ^ key1


__all__ = [
    "DIALGA128_SUPPORTED_ROUNDS",
    "DIALGA_BACKWARD_CONSTANTS",
    "DIALGA_BIT_PERMUTATIONS",
    "DIALGA_BYTE_PERMUTATIONS",
    "DIALGA_BYTE_SBOXES",
    "DIALGA_FORWARD_CONSTANTS",
    "DIALGA_MIDDLE_CONSTANTS",
    "DIALGA_MIDDLE_PERMUTATION",
    "DIALGA_SBOX",
    "Dialga128",
    "dialga128_encrypt",
    "dialga128_round_trace",
    "dialga_inverse_linear_layer",
    "dialga_inverse_round_function",
    "dialga_inverse_sub_cells",
    "dialga_middle_shuffle",
    "dialga_linear_layer",
    "dialga_mix_columns",
    "dialga_round_function",
    "dialga_sub_cells",
]
