from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Mapping, Sequence

import torch

from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    gf2_inverse,
    runtime_spn_structure_from_truth_bits,
)


SBOX_EXPERT = "sbox4_table"
PERMUTATION_EXPERT = "linear_permutation"
GF2_EXPERT = "linear_gf2"
EXPERT_CONTRACT: Mapping[str, Mapping[str, int | bool]] = {
    SBOX_EXPERT: {
        "descriptor_width": 64,
        "state_width": 4,
        "uses_cipher_identity": False,
    },
    PERMUTATION_EXPERT: {
        "edge_token_width": 10,
        "coefficient_width": 1,
        "uses_cipher_identity": False,
    },
    GF2_EXPERT: {
        "edge_token_width": 10,
        "coefficient_width": 1,
        "uses_cipher_identity": False,
    },
}


@dataclass(frozen=True)
class CompiledSboxCell:
    cell: int
    truth_bits: tuple[int, ...]
    expert: str = SBOX_EXPERT

    def __post_init__(self) -> None:
        if self.cell < 0:
            raise ValueError("compiled S-box cell must be non-negative")
        if len(self.truth_bits) != 64 or set(self.truth_bits) - {0, 1}:
            raise ValueError("compiled S-box descriptor must contain 64 binary bits")
        if self.expert != SBOX_EXPERT:
            raise ValueError("compiled S-box must route to the shared 4-bit expert")


@dataclass(frozen=True)
class CompiledLinearCell:
    target_cell: int
    edges: tuple[tuple[int, int, int], ...]
    expert: str

    def __post_init__(self) -> None:
        if self.target_cell < 0 or not self.edges:
            raise ValueError("compiled linear cell requires a target and edges")
        if self.expert not in {PERMUTATION_EXPERT, GF2_EXPERT}:
            raise ValueError("compiled linear cell has an unsupported expert")
        if tuple(sorted(set(self.edges))) != self.edges:
            raise ValueError("compiled linear edges must be sorted and unique")
        if any(
            target_role not in range(4)
            or source_cell < 0
            or source_role not in range(4)
            for target_role, source_cell, source_role in self.edges
        ):
            raise ValueError("compiled linear edge endpoint is invalid")
        fan_in = [0, 0, 0, 0]
        for target_role, _source_cell, _source_role in self.edges:
            fan_in[target_role] += 1
        expected = PERMUTATION_EXPERT if fan_in == [1, 1, 1, 1] else GF2_EXPERT
        if self.expert != expected:
            raise ValueError("compiled linear expert does not match target fan-in")


@dataclass(frozen=True)
class CompiledSpnStage:
    source_stage: int
    sboxes: tuple[CompiledSboxCell, ...]
    linear_cells: tuple[CompiledLinearCell, ...]

    def __post_init__(self) -> None:
        if self.source_stage < 0:
            raise ValueError("compiled source stage must be non-negative")
        sbox_cells = tuple(item.cell for item in self.sboxes)
        linear_cells = tuple(item.target_cell for item in self.linear_cells)
        expected = tuple(range(len(self.sboxes)))
        if sbox_cells != expected or linear_cells != expected:
            raise ValueError("compiled stage must contain every cell in order")

    @property
    def content_sha256(self) -> str:
        return _sha256(
            {
                "sboxes": [
                    {
                        "cell": item.cell,
                        "truth_bits": item.truth_bits,
                        "expert": item.expert,
                    }
                    for item in self.sboxes
                ],
                "linear_cells": [
                    {
                        "target_cell": item.target_cell,
                        "edges": item.edges,
                        "expert": item.expert,
                    }
                    for item in self.linear_cells
                ],
            }
        )


@dataclass(frozen=True)
class CompiledSpnProgram:
    block_bits: int
    cells: int
    native_cell_membership: tuple[int, ...]
    native_bit_role: tuple[int, ...]
    semantic_cell_to_native_cell: tuple[int, ...]
    semantic_cell_to_native_bits: tuple[tuple[int, ...], ...]
    stages: tuple[CompiledSpnStage, ...]
    control: str = "ordered"

    def __post_init__(self) -> None:
        if self.block_bits <= 0 or self.cells <= 0 or not self.stages:
            raise ValueError("compiled SPN program geometry must be positive")
        if len(self.native_cell_membership) != self.block_bits:
            raise ValueError("compiled native cell membership width drifted")
        if len(self.native_bit_role) != self.block_bits:
            raise ValueError("compiled native bit-role width drifted")
        if sorted(self.semantic_cell_to_native_cell) != list(range(self.cells)):
            raise ValueError("compiled semantic-to-native cell map is not bijective")
        if len(self.semantic_cell_to_native_bits) != self.cells or any(
            len(bits) != 4 for bits in self.semantic_cell_to_native_bits
        ):
            raise ValueError("compiled semantic cells must each contain four bits")
        flattened = sorted(
            bit for bits in self.semantic_cell_to_native_bits for bit in bits
        )
        if flattened != list(range(self.block_bits)):
            raise ValueError("compiled semantic-to-native bit map is not bijective")

    @property
    def rounds(self) -> int:
        return len(self.stages)

    @property
    def stage_content_sha256s(self) -> tuple[str, ...]:
        return tuple(stage.content_sha256 for stage in self.stages)

    @property
    def semantic_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": 1,
                "block_bits": self.block_bits,
                "cells": self.cells,
                "stage_content_sha256s": self.stage_content_sha256s,
                "expert_contract": EXPERT_CONTRACT,
            }
        )

    @property
    def expert_usage(self) -> dict[str, int]:
        usage = {name: 0 for name in EXPERT_CONTRACT}
        for stage in self.stages:
            for item in stage.sboxes:
                usage[item.expert] += 1
            for item in stage.linear_cells:
                usage[item.expert] += 1
        return usage


def compile_ordered_primitive_program(
    structure: RuntimeSpnStructure,
    *,
    semantic_cell_ids: Sequence[int] | torch.Tensor | None = None,
) -> CompiledSpnProgram:
    semantic_ids = _semantic_cell_ids(structure, semantic_cell_ids)
    semantic_to_native_cell = [0] * structure.cells
    for native_cell, semantic_cell in enumerate(semantic_ids.tolist()):
        semantic_to_native_cell[semantic_cell] = native_cell

    cell_role_bits = _native_cell_role_bits(structure)
    semantic_to_native_bits = tuple(
        tuple(int(bit) for bit in cell_role_bits[native_cell])
        for native_cell in semantic_to_native_cell
    )
    stages: list[CompiledSpnStage] = []
    for stage_index in range(structure.rounds):
        sboxes = tuple(
            CompiledSboxCell(
                cell=semantic_cell,
                truth_bits=tuple(
                    int(value)
                    for value in structure.sbox_truth_bits[
                        stage_index,
                        semantic_to_native_cell[semantic_cell],
                    ].tolist()
                ),
            )
            for semantic_cell in range(structure.cells)
        )
        inverse = structure.inverse_linear_matrices[stage_index]
        edges_by_target: list[list[tuple[int, int, int]]] = [
            [] for _ in range(structure.cells)
        ]
        for native_target, native_source in torch.nonzero(
            inverse, as_tuple=False
        ).tolist():
            target_cell = int(
                semantic_ids[int(structure.cell_membership[native_target])]
            )
            source_cell = int(
                semantic_ids[int(structure.cell_membership[native_source])]
            )
            edges_by_target[target_cell].append(
                (
                    int(structure.bit_role[native_target]),
                    source_cell,
                    int(structure.bit_role[native_source]),
                )
            )
        linear_cells = []
        for target_cell, unsorted_edges in enumerate(edges_by_target):
            edges = tuple(sorted(unsorted_edges))
            fan_in = [0, 0, 0, 0]
            for target_role, _source_cell, _source_role in edges:
                fan_in[target_role] += 1
            expert = (
                PERMUTATION_EXPERT
                if fan_in == [1, 1, 1, 1]
                else GF2_EXPERT
            )
            linear_cells.append(
                CompiledLinearCell(
                    target_cell=target_cell,
                    edges=edges,
                    expert=expert,
                )
            )
        stages.append(
            CompiledSpnStage(
                source_stage=stage_index,
                sboxes=sboxes,
                linear_cells=tuple(linear_cells),
            )
        )
    return CompiledSpnProgram(
        block_bits=structure.block_bits,
        cells=structure.cells,
        native_cell_membership=tuple(int(value) for value in structure.cell_membership),
        native_bit_role=tuple(int(value) for value in structure.bit_role),
        semantic_cell_to_native_cell=tuple(semantic_to_native_cell),
        semantic_cell_to_native_bits=semantic_to_native_bits,
        stages=tuple(stages),
    )


def replay_ordered_primitive_program(
    program: CompiledSpnProgram,
) -> RuntimeSpnStructure:
    truth, inverse = materialize_ordered_primitive_payload(program)
    forward = torch.stack([gf2_inverse(value) for value in inverse])
    return runtime_spn_structure_from_truth_bits(
        program.native_cell_membership,
        program.native_bit_role,
        truth,
        forward,
    )


def materialize_ordered_primitive_payload(
    program: CompiledSpnProgram,
) -> tuple[torch.Tensor, torch.Tensor]:
    truth = torch.zeros(program.rounds, program.cells, 64, dtype=torch.uint8)
    inverse = torch.zeros(
        program.rounds,
        program.block_bits,
        program.block_bits,
        dtype=torch.uint8,
    )
    for stage_index, stage in enumerate(program.stages):
        for item in stage.sboxes:
            native_cell = program.semantic_cell_to_native_cell[item.cell]
            truth[stage_index, native_cell] = torch.tensor(
                item.truth_bits, dtype=torch.uint8
            )
        for item in stage.linear_cells:
            target_bits = program.semantic_cell_to_native_bits[item.target_cell]
            for target_role, source_cell, source_role in item.edges:
                inverse[
                    stage_index,
                    target_bits[target_role],
                    program.semantic_cell_to_native_bits[source_cell][source_role],
                ] = 1
    return truth, inverse


def rotate_program_stages(program: CompiledSpnProgram) -> CompiledSpnProgram:
    if program.rounds < 2:
        raise ValueError("compiled stage rotation requires at least two stages")
    return replace(
        program,
        stages=(*program.stages[1:], program.stages[0]),
        control="wrong_order",
    )


def permute_program_target_bindings(
    program: CompiledSpnProgram,
    *,
    seed: int,
) -> CompiledSpnProgram:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(program.cells, generator=generator)
    if torch.equal(permutation, torch.arange(program.cells)):
        permutation = torch.roll(permutation, shifts=1)
    stages = []
    for stage in program.stages:
        moved = tuple(
            sorted(
                (
                    replace(item, target_cell=int(permutation[item.target_cell]))
                    for item in stage.linear_cells
                ),
                key=lambda item: item.target_cell,
            )
        )
        stages.append(replace(stage, linear_cells=moved))
    return replace(
        program,
        stages=tuple(stages),
        control=f"wrong_target_binding_seed{seed}",
    )


def permute_program_source_roles(
    program: CompiledSpnProgram,
    *,
    role_permutation: Sequence[int],
) -> CompiledSpnProgram:
    """Corrupt source roles without changing target fan-in or expert routing."""
    roles = tuple(int(value) for value in role_permutation)
    if sorted(roles) != list(range(4)) or roles == tuple(range(4)):
        raise ValueError("source-role control requires a non-identity 4-role permutation")
    stages = []
    for stage in program.stages:
        linear_cells = tuple(
            replace(
                item,
                edges=tuple(
                    sorted(
                        (target_role, source_cell, roles[source_role])
                        for target_role, source_cell, source_role in item.edges
                    )
                ),
            )
            for item in stage.linear_cells
        )
        stages.append(replace(stage, linear_cells=linear_cells))
    role_label = "_".join(str(value) for value in roles)
    return replace(
        program,
        stages=tuple(stages),
        control=f"source_role_permutation_{role_label}",
    )


def permute_program_source_endpoints_affine(
    program: CompiledSpnProgram,
    *,
    multiplier: int,
    offset: int,
) -> CompiledSpnProgram:
    """Apply one global affine bijection to compiled source-bit endpoints."""
    if program.block_bits != program.cells * 4:
        raise ValueError("affine endpoint control requires complete 4-bit cells")
    modulus = program.block_bits
    multiplier = int(multiplier)
    offset = int(offset) % modulus
    if math.gcd(multiplier, modulus) != 1:
        raise ValueError("affine endpoint multiplier must be invertible")
    mapped = tuple((multiplier * endpoint + offset) % modulus for endpoint in range(modulus))
    if mapped == tuple(range(modulus)):
        raise ValueError("affine endpoint control must be non-identity")
    stages = []
    for stage in program.stages:
        linear_cells = tuple(
            replace(
                item,
                edges=tuple(
                    sorted(
                        (
                            target_role,
                            mapped[4 * source_cell + source_role] // 4,
                            mapped[4 * source_cell + source_role] % 4,
                        )
                        for target_role, source_cell, source_role in item.edges
                    )
                ),
            )
            for item in stage.linear_cells
        )
        stages.append(replace(stage, linear_cells=linear_cells))
    return replace(
        program,
        stages=tuple(stages),
        control=(
            f"source_endpoint_affine_m{multiplier}_b{offset}_mod{modulus}"
        ),
    )


def program_exactly_replays(
    program: CompiledSpnProgram,
    structure: RuntimeSpnStructure,
) -> bool:
    truth, inverse = materialize_ordered_primitive_payload(program)
    return all(
        torch.equal(left, right)
        for left, right in (
            (
                torch.tensor(program.native_cell_membership),
                structure.cell_membership,
            ),
            (torch.tensor(program.native_bit_role), structure.bit_role),
            (truth, structure.sbox_truth_bits),
            (inverse, structure.inverse_linear_matrices),
        )
    )


def _semantic_cell_ids(
    structure: RuntimeSpnStructure,
    values: Sequence[int] | torch.Tensor | None,
) -> torch.Tensor:
    if values is None:
        return torch.arange(structure.cells)
    semantic_ids = torch.as_tensor(values, dtype=torch.long, device="cpu")
    if semantic_ids.shape != (structure.cells,) or not torch.equal(
        torch.sort(semantic_ids).values,
        torch.arange(structure.cells),
    ):
        raise ValueError("semantic cell IDs must be a permutation")
    return semantic_ids


def _native_cell_role_bits(structure: RuntimeSpnStructure) -> torch.Tensor:
    indices = torch.empty(structure.cells, 4, dtype=torch.long)
    bit_indices = torch.arange(structure.block_bits)
    indices[structure.cell_membership, structure.bit_role] = bit_indices
    return indices


def _sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "EXPERT_CONTRACT",
    "GF2_EXPERT",
    "PERMUTATION_EXPERT",
    "SBOX_EXPERT",
    "CompiledLinearCell",
    "CompiledSboxCell",
    "CompiledSpnProgram",
    "CompiledSpnStage",
    "compile_ordered_primitive_program",
    "materialize_ordered_primitive_payload",
    "permute_program_source_endpoints_affine",
    "permute_program_source_roles",
    "permute_program_target_bindings",
    "program_exactly_replays",
    "replay_ordered_primitive_program",
    "rotate_program_stages",
]
