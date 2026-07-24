from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

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

    def corrupted(self, seed: int = 20260724) -> RuntimeSpnStructure:
        if self.block_bits < 2:
            raise ValueError("topology corruption requires at least two bits")
        generator = torch.Generator().manual_seed(seed)
        source_permutation = torch.randperm(
            self.block_bits,
            generator=generator,
        )
        if torch.equal(source_permutation, torch.arange(self.block_bits)):
            raise ValueError("topology corruption must change source bits")
        corrupted = torch.empty_like(self.linear_matrices)
        corrupted[:, :, source_permutation] = self.linear_matrices
        return runtime_spn_structure_from_truth_bits(
            self.cell_membership,
            self.bit_role,
            self.sbox_truth_bits,
            corrupted,
        )

    def shuffled_sbox_assignments(self, seed: int = 20260724) -> RuntimeSpnStructure:
        if self.cells < 2:
            raise ValueError("S-box assignment shuffle requires at least two cells")
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(self.cells, generator=generator)
        shuffled = self.sbox_truth_bits[:, permutation]
        if torch.equal(shuffled, self.sbox_truth_bits):
            distinct_cell = next(
                (
                    cell
                    for cell in range(1, self.cells)
                    if not torch.equal(
                        self.sbox_truth_bits[:, 0], self.sbox_truth_bits[:, cell]
                    )
                ),
                None,
            )
            if distinct_cell is None:
                raise ValueError("S-box assignments are identical across all cells")
            permutation = torch.arange(self.cells)
            permutation[[0, distinct_cell]] = permutation[[distinct_cell, 0]]
            shuffled = self.sbox_truth_bits[:, permutation]
        return runtime_spn_structure_from_truth_bits(
            self.cell_membership,
            self.bit_role,
            shuffled,
            self.linear_matrices,
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


@dataclass(frozen=True)
class LoadedRuntimeSpnDescriptor:
    name: str
    path: Path
    sha256: str
    round_start: int
    available_rounds: int
    structure: RuntimeSpnStructure


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


def load_runtime_spn_descriptor(
    path: str | Path,
    *,
    rounds: int | None = None,
    round_start: int = 0,
) -> LoadedRuntimeSpnDescriptor:
    descriptor_path = Path(path)
    try:
        raw = descriptor_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"runtime SPN descriptor unreadable: {descriptor_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("runtime SPN descriptor must be a JSON object")
    allowed = {
        "schema_version",
        "name",
        "cell_membership",
        "bit_role",
        "sbox_tables",
        "linear_layers",
        "repeat_single_round",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"runtime SPN descriptor has unknown fields: {unknown}")
    if payload.get("schema_version") != 1:
        raise ValueError("runtime SPN descriptor schema_version must be 1")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("runtime SPN descriptor name must be a non-empty string")
    membership = _descriptor_int_list(payload.get("cell_membership"), "cell_membership")
    roles = _descriptor_int_list(payload.get("bit_role"), "bit_role")
    if len(membership) != len(roles):
        raise ValueError("runtime SPN descriptor cell and bit-role widths differ")

    layer_payloads = payload.get("linear_layers")
    if not isinstance(layer_payloads, list) or not layer_payloads:
        raise ValueError(
            "runtime SPN descriptor linear_layers must be a non-empty list"
        )
    matrices = torch.stack(
        [
            _descriptor_linear_matrix(layer, len(membership), index)
            for index, layer in enumerate(layer_payloads)
        ]
    )
    if rounds is not None and type(rounds) is not int:
        raise ValueError("runtime SPN descriptor rounds must be an integer")
    if type(round_start) is not int:
        raise ValueError("runtime SPN descriptor round_start must be an integer")
    if round_start < 0:
        raise ValueError("runtime SPN descriptor round_start must be non-negative")
    available_rounds = len(layer_payloads)
    requested_rounds = available_rounds - round_start if rounds is None else rounds
    if requested_rounds <= 0:
        raise ValueError("runtime SPN descriptor rounds must be positive")
    repeat_single = payload.get("repeat_single_round", False)
    if not isinstance(repeat_single, bool):
        raise ValueError("runtime SPN descriptor repeat_single_round must be boolean")
    if repeat_single:
        if available_rounds != 1:
            raise ValueError(
                "runtime SPN descriptor repeat_single_round requires one linear layer"
            )
        if round_start != 0:
            raise ValueError(
                "runtime SPN repeated single round does not support round_start"
            )
        matrices = matrices.repeat(requested_rounds, 1, 1)
        tables = _descriptor_sbox_tables(
            payload.get("sbox_tables"),
            rounds=requested_rounds,
            cells=max(membership) + 1,
            repeat_single_round=True,
        )
    else:
        round_stop = round_start + requested_rounds
        if round_start >= available_rounds or round_stop > available_rounds:
            raise ValueError(
                "runtime SPN descriptor round count/window exceeds available rounds"
            )
        matrices = matrices[round_start:round_stop]
        tables = _descriptor_sbox_tables(
            payload.get("sbox_tables"),
            rounds=available_rounds,
            cells=max(membership) + 1,
            repeat_single_round=False,
        )
        if tables.ndim == 3:
            tables = tables[round_start:round_stop]
    structure = runtime_spn_structure(
        cell_membership=membership,
        bit_role=roles,
        sbox_tables=tables,
        linear_matrices=matrices,
    )
    return LoadedRuntimeSpnDescriptor(
        name=name.strip(),
        path=descriptor_path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        round_start=round_start,
        available_rounds=available_rounds,
        structure=structure,
    )


def _descriptor_int_list(value: object, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"runtime SPN descriptor {field} must be a non-empty list")
    if not all(type(item) is int for item in value):
        raise ValueError(f"runtime SPN descriptor {field} must contain integers")
    return list(value)


def _descriptor_linear_matrix(
    value: object,
    block_bits: int,
    index: int,
) -> torch.Tensor:
    if not isinstance(value, dict):
        raise ValueError(
            f"runtime SPN descriptor linear layer {index} must be an object"
        )
    kind = value.get("kind")
    if kind == "permutation":
        unknown = sorted(set(value) - {"kind", "source_to_target"})
        if unknown:
            raise ValueError(
                f"runtime SPN descriptor permutation layer has unknown fields: {unknown}"
            )
        permutation = _descriptor_int_list(
            value.get("source_to_target"),
            f"linear_layers[{index}].source_to_target",
        )
        if len(permutation) != block_bits:
            raise ValueError("runtime SPN permutation width does not match cell layout")
        return permutation_matrix(permutation)
    if kind == "gf2":
        unknown = sorted(set(value) - {"kind", "target_sources"})
        if unknown:
            raise ValueError(
                f"runtime SPN descriptor GF(2) layer has unknown fields: {unknown}"
            )
        target_sources = value.get("target_sources")
        if not isinstance(target_sources, list) or len(target_sources) != block_bits:
            raise ValueError(
                "runtime SPN GF(2) target_sources must contain one list per target bit"
            )
        matrix = torch.zeros(block_bits, block_bits, dtype=torch.uint8)
        for target, sources in enumerate(target_sources):
            source_list = _descriptor_int_list(
                sources,
                f"linear_layers[{index}].target_sources[{target}]",
            )
            if len(set(source_list)) != len(source_list):
                raise ValueError("runtime SPN GF(2) target source lists must be unique")
            if any(source < 0 or source >= block_bits for source in source_list):
                raise ValueError("runtime SPN GF(2) source index is out of range")
            matrix[target, source_list] = 1
        return matrix
    raise ValueError(
        f"runtime SPN descriptor linear layer {index} kind must be permutation or gf2"
    )


def _descriptor_sbox_tables(
    value: object,
    *,
    rounds: int,
    cells: int,
    repeat_single_round: bool,
) -> torch.Tensor:
    if not _descriptor_contains_only_integers(value):
        raise ValueError("runtime SPN descriptor sbox_tables must be integer arrays")
    try:
        tables = torch.as_tensor(value, dtype=torch.long)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime SPN descriptor sbox_tables must be integer arrays"
        ) from exc
    if tables.ndim == 1:
        if tables.shape != (16,):
            raise ValueError("runtime SPN shared S-box must contain 16 values")
        return tables
    if tables.ndim == 2:
        if tables.shape != (cells, 16):
            raise ValueError(
                "runtime SPN per-cell S-box tables must have shape cells x 16"
            )
        return tables[None, :, :].expand(rounds, -1, -1).clone()
    if tables.ndim == 3:
        if tables.shape[1:] != (cells, 16):
            raise ValueError(
                "runtime SPN S-box tables must have shape rounds x cells x 16"
            )
        if tables.shape[0] == rounds:
            return tables
        if tables.shape[0] == 1 and repeat_single_round:
            return tables.expand(rounds, -1, -1).clone()
        raise ValueError(
            "runtime SPN S-box round count does not match requested rounds"
        )
    raise ValueError("runtime SPN descriptor sbox_tables has unsupported dimensions")


def _descriptor_contains_only_integers(value: object) -> bool:
    if isinstance(value, list):
        return bool(value) and all(
            _descriptor_contains_only_integers(item) for item in value
        )
    return type(value) is int


__all__ = [
    "LoadedRuntimeSpnDescriptor",
    "RuntimeSpnStructure",
    "apply_gf2",
    "gf2_inverse",
    "linear_matrix_from_callable",
    "load_runtime_spn_descriptor",
    "permutation_matrix",
    "runtime_spn_structure",
    "runtime_spn_structure_from_truth_bits",
]
