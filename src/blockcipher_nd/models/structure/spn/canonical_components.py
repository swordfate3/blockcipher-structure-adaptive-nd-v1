from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json

import networkx as nx
import torch

from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    apply_gf2,
    gf2_inverse,
    linear_matrix_from_callable,
)


MIDORI_LINEAR_SOURCE_IMAGES_MSB = (
    0x0888000000000000,
    0x0444000000000000,
    0x0222000000000000,
    0x0111000000000000,
    0x8088000000000000,
    0x4044000000000000,
    0x2022000000000000,
    0x1011000000000000,
    0x8808000000000000,
    0x4404000000000000,
    0x2202000000000000,
    0x1101000000000000,
    0x8880000000000000,
    0x4440000000000000,
    0x2220000000000000,
    0x1110000000000000,
    0x0000088800000000,
    0x0000044400000000,
    0x0000022200000000,
    0x0000011100000000,
    0x0000808800000000,
    0x0000404400000000,
    0x0000202200000000,
    0x0000101100000000,
    0x0000880800000000,
    0x0000440400000000,
    0x0000220200000000,
    0x0000110100000000,
    0x0000888000000000,
    0x0000444000000000,
    0x0000222000000000,
    0x0000111000000000,
    0x0000000008880000,
    0x0000000004440000,
    0x0000000002220000,
    0x0000000001110000,
    0x0000000080880000,
    0x0000000040440000,
    0x0000000020220000,
    0x0000000010110000,
    0x0000000088080000,
    0x0000000044040000,
    0x0000000022020000,
    0x0000000011010000,
    0x0000000088800000,
    0x0000000044400000,
    0x0000000022200000,
    0x0000000011100000,
    0x0000000000000888,
    0x0000000000000444,
    0x0000000000000222,
    0x0000000000000111,
    0x0000000000008088,
    0x0000000000004044,
    0x0000000000002022,
    0x0000000000001011,
    0x0000000000008808,
    0x0000000000004404,
    0x0000000000002202,
    0x0000000000001101,
    0x0000000000008880,
    0x0000000000004440,
    0x0000000000002220,
    0x0000000000001110,
)

BitPermutation = tuple[int, ...]
LinearFactor = tuple[BitPermutation, BitPermutation]


@dataclass(frozen=True)
class CanonicalLinearSchedule:
    primitive: str
    native_to_canonical_output: torch.Tensor
    canonical_inverse_matrices: torch.Tensor
    canonical_input_to_native: torch.Tensor
    canonical_edge_index: torch.Tensor
    factors: tuple[LinearFactor, ...]
    manifest_sha256: str
    control: str = "ordered"

    @property
    def rounds(self) -> int:
        return int(self.native_to_canonical_output.shape[0])

    @property
    def block_bits(self) -> int:
        return int(self.native_to_canonical_output.shape[-1])

    def transition(
        self,
        values: torch.Tensor,
        round_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        canonical_output = apply_gf2(
            self.native_to_canonical_output[round_index], values
        )
        canonical_input = apply_gf2(
            self.canonical_inverse_matrices[round_index], canonical_output
        )
        native_input = apply_gf2(
            self.canonical_input_to_native[round_index], canonical_input
        )
        return canonical_output, canonical_input, native_input

    def rotate_transitions(self) -> CanonicalLinearSchedule:
        indices = torch.roll(torch.arange(self.rounds), shifts=-1)
        factors = (*self.factors[1:], self.factors[0])
        return _schedule_from_tensors(
            primitive=self.primitive,
            native_to_canonical_output=self.native_to_canonical_output[indices],
            canonical_inverse_matrices=self.canonical_inverse_matrices[indices],
            canonical_input_to_native=self.canonical_input_to_native[indices],
            factors=factors,
            control="rotated",
        )


def compile_canonical_linear_schedule(
    structure: RuntimeSpnStructure,
    *,
    control: str = "ordered",
) -> CanonicalLinearSchedule:
    if control not in {"ordered", "rotated"}:
        raise ValueError("canonical schedule control must be ordered or rotated")
    primitive, canonical_matrix = canonical_primitive_matrix(structure.block_bits)
    canonical_inverse = gf2_inverse(canonical_matrix)
    factors = tuple(
        recover_linear_factor(canonical_matrix, native)
        for native in structure.linear_matrices
    )
    native_to_canonical_output: list[torch.Tensor] = []
    canonical_input_to_native: list[torch.Tensor] = []
    for canonical_input_native, canonical_output_native in factors:
        output_matrix = torch.zeros(
            structure.block_bits, structure.block_bits, dtype=torch.uint8
        )
        for canonical_bit, native_bit in enumerate(canonical_output_native):
            output_matrix[canonical_bit, native_bit] = 1
        input_matrix = torch.zeros_like(output_matrix)
        for canonical_bit, native_bit in enumerate(canonical_input_native):
            input_matrix[native_bit, canonical_bit] = 1
        native_to_canonical_output.append(output_matrix)
        canonical_input_to_native.append(input_matrix)
    schedule = _schedule_from_tensors(
        primitive=primitive,
        native_to_canonical_output=torch.stack(native_to_canonical_output),
        canonical_inverse_matrices=canonical_inverse[None].repeat(
            structure.rounds, 1, 1
        ),
        canonical_input_to_native=torch.stack(canonical_input_to_native),
        factors=factors,
        control="ordered",
    )
    return schedule.rotate_transitions() if control == "rotated" else schedule


def canonical_primitive_matrix(block_bits: int) -> tuple[str, torch.Tensor]:
    if block_bits == 64:
        return "midori64", linear_matrix_from_callable(64, midori_linear_layer)
    if block_bits == 128:
        return "dialga_midori_mix128", linear_matrix_from_callable(
            128, dialga_midori_mix_columns
        )
    raise ValueError(
        "CT-SPN K1 supports only the K0-verified 64- and 128-bit canonical primitives"
    )


def recover_linear_factor(
    canonical_matrix: torch.Tensor,
    native_matrix: torch.Tensor,
) -> LinearFactor:
    canonical = torch.as_tensor(canonical_matrix, dtype=torch.uint8, device="cpu")
    native = torch.as_tensor(native_matrix, dtype=torch.uint8, device="cpu")
    if canonical.shape != native.shape or canonical.ndim != 2:
        raise ValueError("canonical and native matrices must be equal-size squares")
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        _linear_bipartite_graph(canonical),
        _linear_bipartite_graph(native),
        node_match=lambda left, right: left["kind"] == right["kind"],
    )
    mapping = next(matcher.isomorphisms_iter(), None)
    if mapping is None:
        raise ValueError("native linear layer is not permutation-equivalent to primitive")
    width = int(canonical.shape[0])
    return (
        tuple(mapping[("input", bit)][1] for bit in range(width)),
        tuple(mapping[("output", bit)][1] for bit in range(width)),
    )


def apply_linear_factor(
    state: int,
    factor: LinearFactor,
    primitive: Callable[[int], int],
) -> int:
    canonical_input_native, canonical_output_native = factor
    canonical_state = 0
    for canonical_bit, native_bit in enumerate(canonical_input_native):
        canonical_state |= ((state >> native_bit) & 1) << canonical_bit
    canonical_output = int(primitive(canonical_state))
    native_output = 0
    for canonical_bit, native_bit in enumerate(canonical_output_native):
        native_output |= ((canonical_output >> canonical_bit) & 1) << native_bit
    return native_output


def midori_linear_layer(state: int) -> int:
    output = 0
    for source_lsb in range(64):
        if (state >> source_lsb) & 1:
            output ^= MIDORI_LINEAR_SOURCE_IMAGES_MSB[63 - source_lsb]
    return output


def dialga_midori_mix_columns(state: int) -> int:
    values = state.to_bytes(16, byteorder="big")
    mixed = bytearray(16)
    for column in range(4):
        offset = 4 * column
        s0, s1, s2, s3 = values[offset : offset + 4]
        mixed[offset] = s1 ^ s2 ^ s3
        mixed[offset + 1] = s0 ^ s2 ^ s3
        mixed[offset + 2] = s0 ^ s1 ^ s3
        mixed[offset + 3] = s0 ^ s1 ^ s2
    return int.from_bytes(mixed, byteorder="big")


def _linear_bipartite_graph(matrix: torch.Tensor) -> nx.Graph:
    width = int(matrix.shape[0])
    graph = nx.Graph()
    graph.add_nodes_from((("input", bit) for bit in range(width)), kind="input")
    graph.add_nodes_from((("output", bit) for bit in range(width)), kind="output")
    for target, source in torch.nonzero(matrix, as_tuple=False).tolist():
        graph.add_edge(("input", source), ("output", target))
    return graph


def _schedule_from_tensors(
    *,
    primitive: str,
    native_to_canonical_output: torch.Tensor,
    canonical_inverse_matrices: torch.Tensor,
    canonical_input_to_native: torch.Tensor,
    factors: tuple[LinearFactor, ...],
    control: str,
) -> CanonicalLinearSchedule:
    canonical_forward = gf2_inverse(canonical_inverse_matrices[0])
    canonical_edge_index = torch.nonzero(
        canonical_forward, as_tuple=False
    ).T.contiguous()
    payload = {
        "schema_version": 1,
        "primitive": primitive,
        "control": control,
        "factors": factors,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CanonicalLinearSchedule(
        primitive=primitive,
        native_to_canonical_output=native_to_canonical_output.clone(),
        canonical_inverse_matrices=canonical_inverse_matrices.clone(),
        canonical_input_to_native=canonical_input_to_native.clone(),
        canonical_edge_index=canonical_edge_index,
        factors=factors,
        manifest_sha256=digest,
        control=control,
    )


__all__ = [
    "CanonicalLinearSchedule",
    "MIDORI_LINEAR_SOURCE_IMAGES_MSB",
    "apply_linear_factor",
    "canonical_primitive_matrix",
    "compile_canonical_linear_schedule",
    "dialga_midori_mix_columns",
    "midori_linear_layer",
    "recover_linear_factor",
]
