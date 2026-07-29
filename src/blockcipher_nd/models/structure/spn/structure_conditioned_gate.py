from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


SBOX_SUMMARY_DIM = 16
LINEAR_SUMMARY_DIM = 18
STRUCTURE_SUMMARY_DIM = SBOX_SUMMARY_DIM + LINEAR_SUMMARY_DIM


def runtime_structure_summary(structure: RuntimeSpnStructure) -> torch.Tensor:
    """Return a fixed-width, cell-relabeling-invariant SPN summary."""

    sbox = sbox_structure_summary(structure)
    linear = linear_structure_summary(structure)
    summary = torch.cat((sbox, linear)).to(torch.float32)
    if summary.shape != (STRUCTURE_SUMMARY_DIM,):
        raise RuntimeError("runtime SPN structure summary width drifted")
    if not bool(torch.isfinite(summary).all()):
        raise ValueError("runtime SPN structure summary must be finite")
    if not bool(torch.all((summary >= 0.0) & (summary <= 1.0))):
        raise ValueError("runtime SPN structure summary must remain in [0, 1]")
    return summary


def sbox_structure_summary(structure: RuntimeSpnStructure) -> torch.Tensor:
    ddt_values: list[torch.Tensor] = []
    lat_values: list[torch.Tensor] = []
    truth_hashes: set[str] = set()
    round_signatures: set[tuple[str, ...]] = set()
    for round_index in range(structure.rounds):
        round_hashes: list[str] = []
        for table, truth in zip(
            structure.sbox_tables(round_index),
            structure.sbox_truth_bits[round_index],
            strict=True,
        ):
            ddt, lat = _ddt_and_lat(table)
            ddt_values.append(ddt[1:].reshape(-1).to(torch.float64) / 16.0)
            flat_lat = lat.abs().reshape(-1).to(torch.float64)
            lat_values.append(flat_lat[1:] / 16.0)
            digest = hashlib.sha256(truth.numpy().tobytes()).hexdigest()
            truth_hashes.add(digest)
            round_hashes.append(digest)
        round_signatures.add(tuple(sorted(round_hashes)))
    total_sboxes = structure.rounds * structure.cells
    return torch.cat(
        (
            _distribution_summary(torch.cat(ddt_values)),
            _distribution_summary(torch.cat(lat_values)),
            torch.tensor(
                (
                    len(truth_hashes) / total_sboxes,
                    len(round_signatures) / structure.rounds,
                ),
                dtype=torch.float64,
            ),
        )
    ).to(torch.float32)


def linear_structure_summary(structure: RuntimeSpnStructure) -> torch.Tensor:
    matrices = structure.linear_matrices.to(torch.float64)
    width = structure.block_bits
    row_weights = matrices.sum(dim=2).reshape(-1) / width
    column_weights = matrices.sum(dim=1).reshape(-1) / width
    ranks = torch.tensor(
        [_gf2_rank(matrix) / width for matrix in structure.linear_matrices],
        dtype=torch.float64,
    )
    matrix_hashes = {
        hashlib.sha256(matrix.numpy().tobytes()).hexdigest()
        for matrix in structure.linear_matrices
    }
    return torch.cat(
        (
            _distribution_summary(row_weights),
            _distribution_summary(column_weights),
            torch.tensor(
                (
                    float(matrices.mean()),
                    float(ranks.mean()),
                    float(ranks.min()),
                    len(matrix_hashes) / structure.rounds,
                ),
                dtype=torch.float64,
            ),
        )
    ).to(torch.float32)


def hybrid_structure_summary(
    *,
    sbox_structure: RuntimeSpnStructure,
    linear_structure: RuntimeSpnStructure,
) -> torch.Tensor:
    return torch.cat(
        (
            sbox_structure_summary(sbox_structure),
            linear_structure_summary(linear_structure),
        )
    )


class SharedStructureTransitionGate(nn.Module):
    """One bounded transition gate shared by every runtime SPN descriptor."""

    def __init__(self, hidden_dim: int = 12) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("structure transition gate hidden_dim must be positive")
        self.summary_dim = STRUCTURE_SUMMARY_DIM
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.summary_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        nn.init.normal_(self.network[2].weight, mean=0.0, std=0.02)

    def forward(
        self,
        global_bias: torch.Tensor,
        summary: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return torch.tanh(global_bias)
        values = torch.as_tensor(
            summary,
            dtype=global_bias.dtype,
            device=global_bias.device,
        )
        if values.shape != (self.summary_dim,):
            raise ValueError("structure transition gate summary shape drifted")
        return torch.tanh(global_bias + self.network(values).squeeze(-1))


def _distribution_summary(values: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(-1)
    if values.numel() == 0:
        raise ValueError("structure distribution summary cannot be empty")
    quantiles = torch.quantile(
        values,
        torch.tensor((0.25, 0.5, 0.75), dtype=torch.float64),
    )
    return torch.stack(
        (
            values.mean(),
            values.std(unbiased=False),
            quantiles[0],
            quantiles[1],
            quantiles[2],
            values.max(),
            (values > 0.0).to(torch.float64).mean(),
        )
    )


def _ddt_and_lat(table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(table, dtype=torch.long, device="cpu")
    inputs = torch.arange(16, dtype=torch.long)
    masks = torch.arange(16, dtype=torch.long)
    parity = torch.tensor([int(index).bit_count() & 1 for index in range(16)])
    output_differences = values[None, :] ^ values[
        inputs[None, :] ^ masks[:, None]
    ]
    ddt = F.one_hot(output_differences, num_classes=16).sum(dim=1)
    input_signs = 1 - 2 * parity[masks[:, None] & inputs[None, :]]
    output_signs = 1 - 2 * parity[masks[:, None] & values[None, :]]
    lat = input_signs @ output_signs.transpose(0, 1)
    return ddt, lat


def _gf2_rank(matrix: torch.Tensor) -> int:
    reduced = torch.as_tensor(matrix, dtype=torch.uint8, device="cpu").clone()
    rows, columns = reduced.shape
    rank = 0
    for column in range(columns):
        candidates = torch.nonzero(reduced[rank:, column], as_tuple=False)
        if candidates.numel() == 0:
            continue
        pivot = rank + int(candidates[0, 0])
        if pivot != rank:
            reduced[[rank, pivot]] = reduced[[pivot, rank]]
        for row in range(rows):
            if row != rank and int(reduced[row, column]):
                reduced[row] ^= reduced[rank]
        rank += 1
        if rank == rows:
            break
    return rank


__all__ = [
    "LINEAR_SUMMARY_DIM",
    "SBOX_SUMMARY_DIM",
    "STRUCTURE_SUMMARY_DIM",
    "SharedStructureTransitionGate",
    "hybrid_structure_summary",
    "linear_structure_summary",
    "runtime_structure_summary",
    "sbox_structure_summary",
]
