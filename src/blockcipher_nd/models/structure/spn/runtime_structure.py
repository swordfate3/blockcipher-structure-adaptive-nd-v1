from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


def gf2_inverse(matrix: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(matrix, dtype=torch.uint8, device="cpu")
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("GF(2) matrix must be square")
    if not torch.all((values == 0) | (values == 1)):
        raise ValueError("GF(2) matrix must be binary")

    size = values.shape[0]
    reduced = values.clone()
    inverse = torch.eye(size, dtype=torch.uint8)
    for column in range(size):
        candidates = torch.nonzero(reduced[column:, column], as_tuple=False)
        if candidates.numel() == 0:
            raise ValueError("GF(2) matrix must be invertible")
        pivot = column + int(candidates[0, 0])
        if pivot != column:
            reduced[[column, pivot]] = reduced[[pivot, column]]
            inverse[[column, pivot]] = inverse[[pivot, column]]
        for row in range(size):
            if row != column and int(reduced[row, column]):
                reduced[row] ^= reduced[column]
                inverse[row] ^= inverse[column]

    if not torch.equal(reduced, torch.eye(size, dtype=torch.uint8)):
        raise ValueError("GF(2) matrix inversion failed")
    return inverse


def apply_gf2(matrix: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    relation = torch.as_tensor(matrix, device=values.device, dtype=values.dtype)
    if relation.ndim != 2 or relation.shape[0] != relation.shape[1]:
        raise ValueError("GF(2) matrix must be square")
    if values.shape[-1] != relation.shape[1]:
        raise ValueError("GF(2) values do not match matrix width")
    transformed = torch.einsum("ts,...s->...t", relation, values)
    return torch.remainder(transformed, 2.0)


def permutation_matrix(permutation: Sequence[int]) -> torch.Tensor:
    mapping = tuple(int(value) for value in permutation)
    size = len(mapping)
    if sorted(mapping) != list(range(size)):
        raise ValueError("permutation must contain every bit index exactly once")
    matrix = torch.zeros(size, size, dtype=torch.uint8)
    for source, target in enumerate(mapping):
        matrix[target, source] = 1
    return matrix


def linear_matrix_from_callable(
    block_bits: int,
    linear_layer: Callable[[int], int],
) -> torch.Tensor:
    if block_bits <= 0:
        raise ValueError("block_bits must be positive")
    matrix = torch.zeros(block_bits, block_bits, dtype=torch.uint8)
    for source in range(block_bits):
        output = int(linear_layer(1 << source))
        if output < 0 or output >= 1 << block_bits:
            raise ValueError("linear layer output does not fit block width")
        for target in range(block_bits):
            matrix[target, source] = (output >> target) & 1
    return matrix


def _truth_bits(sbox_tables: torch.Tensor) -> torch.Tensor:
    tables = torch.as_tensor(sbox_tables, dtype=torch.long, device="cpu")
    if tables.ndim != 3 or tables.shape[-1] != 16:
        raise ValueError("sbox tables must have shape rounds x cells x 16")
    if not torch.all((tables >= 0) & (tables < 16)):
        raise ValueError("4-bit S-box outputs must be in [0, 15]")
    if not all(
        torch.equal(torch.sort(table).values, torch.arange(16))
        for table in tables.reshape(-1, 16)
    ):
        raise ValueError("each S-box table must be a 4-bit permutation")
    shifts = torch.arange(4, dtype=torch.long)
    return (
        ((tables[..., None] >> shifts) & 1)
        .reshape(*tables.shape[:2], 64)
        .to(torch.uint8)
    )


@dataclass(frozen=True)
class RuntimeSpnStructure:
    cell_membership: torch.Tensor
    bit_role: torch.Tensor
    sbox_truth_bits: torch.Tensor
    linear_matrices: torch.Tensor
    inverse_linear_matrices: torch.Tensor

    def __post_init__(self) -> None:
        membership = torch.as_tensor(
            self.cell_membership, dtype=torch.long, device="cpu"
        ).clone()
        roles = torch.as_tensor(self.bit_role, dtype=torch.long, device="cpu").clone()
        truth = torch.as_tensor(
            self.sbox_truth_bits, dtype=torch.uint8, device="cpu"
        ).clone()
        linear = torch.as_tensor(
            self.linear_matrices, dtype=torch.uint8, device="cpu"
        ).clone()
        inverse = torch.as_tensor(
            self.inverse_linear_matrices, dtype=torch.uint8, device="cpu"
        ).clone()
        object.__setattr__(self, "cell_membership", membership)
        object.__setattr__(self, "bit_role", roles)
        object.__setattr__(self, "sbox_truth_bits", truth)
        object.__setattr__(self, "linear_matrices", linear)
        object.__setattr__(self, "inverse_linear_matrices", inverse)
        self._validate()

    @property
    def block_bits(self) -> int:
        return int(self.cell_membership.numel())

    @property
    def cells(self) -> int:
        return int(torch.max(self.cell_membership)) + 1

    @property
    def rounds(self) -> int:
        return int(self.linear_matrices.shape[0])

    def exact_inverse(
        self, values: torch.Tensor, round_index: int = -1
    ) -> torch.Tensor:
        index = round_index % self.rounds
        return apply_gf2(self.inverse_linear_matrices[index], values)

    def corrupted(self, cell_shift: int = 1) -> RuntimeSpnStructure:
        if self.cells < 2:
            raise ValueError("topology corruption requires at least two cells")
        shift = cell_shift % self.cells
        if shift == 0:
            raise ValueError("topology corruption must change source cells")
        lookup = self._cell_role_lookup()
        source_permutation = torch.empty(self.block_bits, dtype=torch.long)
        for source in range(self.block_bits):
            cell = int(self.cell_membership[source])
            role = int(self.bit_role[source])
            source_permutation[source] = lookup[((cell + shift) % self.cells, role)]
        corrupted = torch.empty_like(self.linear_matrices)
        corrupted[:, :, source_permutation] = self.linear_matrices
        return runtime_spn_structure_from_truth_bits(
            self.cell_membership,
            self.bit_role,
            self.sbox_truth_bits,
            corrupted,
        )

    def relabel_cells(
        self, cell_permutation: Sequence[int]
    ) -> tuple[RuntimeSpnStructure, torch.Tensor]:
        permutation = tuple(int(value) for value in cell_permutation)
        if sorted(permutation) != list(range(self.cells)):
            raise ValueError("cell permutation must contain every cell exactly once")
        lookup = self._cell_role_lookup()
        bit_permutation = torch.empty(self.block_bits, dtype=torch.long)
        for source in range(self.block_bits):
            old_cell = int(self.cell_membership[source])
            role = int(self.bit_role[source])
            bit_permutation[source] = lookup[(permutation[old_cell], role)]

        linear = torch.zeros_like(self.linear_matrices)
        for old_target in range(self.block_bits):
            new_target = int(bit_permutation[old_target])
            for old_source in range(self.block_bits):
                new_source = int(bit_permutation[old_source])
                linear[:, new_target, new_source] = self.linear_matrices[
                    :, old_target, old_source
                ]
        truth = torch.empty_like(self.sbox_truth_bits)
        for old_cell, new_cell in enumerate(permutation):
            truth[:, new_cell] = self.sbox_truth_bits[:, old_cell]
        return (
            runtime_spn_structure_from_truth_bits(
                self.cell_membership,
                self.bit_role,
                truth,
                linear,
            ),
            bit_permutation,
        )

    def _cell_role_lookup(self) -> dict[tuple[int, int], int]:
        return {
            (int(cell), int(role)): index
            for index, (cell, role) in enumerate(
                zip(self.cell_membership, self.bit_role, strict=True)
            )
        }

    def _validate(self) -> None:
        if (
            self.cell_membership.ndim != 1
            or self.bit_role.shape != self.cell_membership.shape
        ):
            raise ValueError("cell membership and bit roles must be one-dimensional")
        if self.block_bits <= 0 or self.block_bits % 4:
            raise ValueError("runtime SPN block width must be a positive multiple of 4")
        if torch.min(self.cell_membership) != 0:
            raise ValueError("cell ids must start at zero")
        if not torch.equal(
            torch.unique(self.cell_membership), torch.arange(self.cells)
        ):
            raise ValueError("cell ids must be contiguous")
        for cell in range(self.cells):
            roles = self.bit_role[self.cell_membership == cell]
            if not torch.equal(torch.sort(roles).values, torch.arange(4)):
                raise ValueError(
                    "every SPN cell must contain bit roles 0..3 exactly once"
                )
        expected_matrix_shape = (self.rounds, self.block_bits, self.block_bits)
        if self.rounds <= 0 or self.linear_matrices.shape != expected_matrix_shape:
            raise ValueError("linear matrices must have shape rounds x bits x bits")
        if self.inverse_linear_matrices.shape != expected_matrix_shape:
            raise ValueError("inverse linear matrices do not match linear matrices")
        if not torch.all((self.linear_matrices == 0) | (self.linear_matrices == 1)):
            raise ValueError("linear matrices must be binary")
        if self.sbox_truth_bits.shape != (self.rounds, self.cells, 64):
            raise ValueError("S-box truth bits must have shape rounds x cells x 64")
        if not torch.all((self.sbox_truth_bits == 0) | (self.sbox_truth_bits == 1)):
            raise ValueError("S-box truth descriptors must be binary")
        identity = torch.eye(self.block_bits, dtype=torch.int64)
        for round_index in range(self.rounds):
            product = torch.remainder(
                self.linear_matrices[round_index].to(torch.int64)
                @ self.inverse_linear_matrices[round_index].to(torch.int64),
                2,
            )
            if not torch.equal(product, identity):
                raise ValueError(
                    "linear and inverse matrices are inconsistent over GF(2)"
                )


def runtime_spn_structure(
    *,
    cell_membership: Sequence[int] | torch.Tensor,
    bit_role: Sequence[int] | torch.Tensor,
    sbox_tables: Sequence[int] | torch.Tensor,
    linear_matrices: torch.Tensor,
) -> RuntimeSpnStructure:
    membership = torch.as_tensor(cell_membership, dtype=torch.long)
    roles = torch.as_tensor(bit_role, dtype=torch.long)
    linear = torch.as_tensor(linear_matrices, dtype=torch.uint8)
    if linear.ndim == 2:
        linear = linear.unsqueeze(0)
    if linear.ndim != 3:
        raise ValueError("linear matrices must be two- or three-dimensional")
    rounds = linear.shape[0]
    cells = int(torch.max(membership)) + 1
    tables = torch.as_tensor(sbox_tables, dtype=torch.long)
    if tables.ndim == 1:
        if tables.shape != (16,):
            raise ValueError("a shared 4-bit S-box table must contain 16 values")
        tables = tables[None, None, :].expand(rounds, cells, -1).clone()
    elif tables.shape != (rounds, cells, 16):
        raise ValueError("per-cell S-box tables must have shape rounds x cells x 16")
    return runtime_spn_structure_from_truth_bits(
        membership,
        roles,
        _truth_bits(tables),
        linear,
    )


def runtime_spn_structure_from_truth_bits(
    cell_membership: Sequence[int] | torch.Tensor,
    bit_role: Sequence[int] | torch.Tensor,
    sbox_truth_bits: torch.Tensor,
    linear_matrices: torch.Tensor,
) -> RuntimeSpnStructure:
    linear = torch.as_tensor(linear_matrices, dtype=torch.uint8, device="cpu")
    if linear.ndim == 2:
        linear = linear.unsqueeze(0)
    inverses = torch.stack([gf2_inverse(matrix) for matrix in linear])
    return RuntimeSpnStructure(
        cell_membership=torch.as_tensor(cell_membership, dtype=torch.long),
        bit_role=torch.as_tensor(bit_role, dtype=torch.long),
        sbox_truth_bits=torch.as_tensor(sbox_truth_bits, dtype=torch.uint8),
        linear_matrices=linear,
        inverse_linear_matrices=inverses,
    )


__all__ = [
    "RuntimeSpnStructure",
    "apply_gf2",
    "gf2_inverse",
    "linear_matrix_from_callable",
    "permutation_matrix",
    "runtime_spn_structure",
    "runtime_spn_structure_from_truth_bits",
]
