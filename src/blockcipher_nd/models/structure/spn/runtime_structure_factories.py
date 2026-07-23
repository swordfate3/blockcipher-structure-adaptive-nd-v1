from __future__ import annotations

import torch

from blockcipher_nd.ciphers.spn.gift import GIFT64_SBOX, Gift64
from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, Present80
from blockcipher_nd.ciphers.spn.skinny import (
    SKINNY64_SBOX,
    cells_to_int,
    int_to_cells,
    mix_columns,
    shift_rows,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    linear_matrix_from_callable,
    runtime_spn_structure,
)


def standard_four_bit_cells(block_bits: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if block_bits <= 0 or block_bits % 4:
        raise ValueError("block_bits must be a positive multiple of 4")
    return (
        tuple(index // 4 for index in range(block_bits)),
        tuple(3 - index % 4 for index in range(block_bits)),
    )


def present_runtime_structure(rounds: int = 1) -> RuntimeSpnStructure:
    return _repeated_structure(
        block_bits=64,
        rounds=rounds,
        sbox=PRESENT_SBOX,
        linear=linear_matrix_from_callable(64, Present80.permutation_layer),
    )


def gift64_runtime_structure(rounds: int = 1) -> RuntimeSpnStructure:
    return _repeated_structure(
        block_bits=64,
        rounds=rounds,
        sbox=GIFT64_SBOX,
        linear=linear_matrix_from_callable(64, Gift64.permutation_layer),
    )


def skinny64_runtime_structure(rounds: int = 1) -> RuntimeSpnStructure:
    def skinny_linear_layer(state: int) -> int:
        return cells_to_int(mix_columns(shift_rows(int_to_cells(state))))

    return _repeated_structure(
        block_bits=64,
        rounds=rounds,
        sbox=SKINNY64_SBOX,
        linear=linear_matrix_from_callable(64, skinny_linear_layer),
    )


def _repeated_structure(
    *,
    block_bits: int,
    rounds: int,
    sbox: tuple[int, ...],
    linear: torch.Tensor,
) -> RuntimeSpnStructure:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    membership, roles = standard_four_bit_cells(block_bits)
    matrices = linear.unsqueeze(0).repeat(rounds, 1, 1)
    return runtime_spn_structure(
        cell_membership=membership,
        bit_role=roles,
        sbox_tables=sbox,
        linear_matrices=matrices,
    )


__all__ = [
    "gift64_runtime_structure",
    "present_runtime_structure",
    "skinny64_runtime_structure",
    "standard_four_bit_cells",
]
