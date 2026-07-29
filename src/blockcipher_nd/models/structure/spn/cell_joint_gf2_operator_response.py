from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from blockcipher_nd.models.structure.spn.exact_gf2_operator_response import (
    RAW_CHANNEL_NAMES,
    RESPONSE_VIEW_NAMES,
    exact_gf2_operator_response,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


CELL_VALUE_COUNT = 16


def response_bits_to_cell_values(
    response: np.ndarray,
    structure: RuntimeSpnStructure,
) -> np.ndarray:
    """Reconstruct native four-bit cell values for every response channel."""
    values = np.asarray(response, dtype=np.uint8)
    if values.ndim < 2 or values.shape[-2] != structure.block_bits:
        raise ValueError("response geometry does not match the runtime structure")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("response values must be binary")

    membership = structure.cell_membership.numpy()
    roles = structure.bit_role.numpy()
    output = np.zeros((*values.shape[:-2], structure.cells, values.shape[-1]), dtype=np.uint8)
    for bit in range(structure.block_bits):
        output[..., int(membership[bit]), :] |= values[..., bit, :] << int(roles[bit])
    return output


def extract_cell_joint_gf2_operator_features(
    flat_features: np.ndarray,
    structure: RuntimeSpnStructure,
    *,
    pairs_per_sample: int = 4,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract pair histograms of native cell values for exact GF(2) views."""
    values = np.asarray(flat_features)
    if pairs_per_sample <= 0 or batch_size <= 0:
        raise ValueError("pair count and batch size must be positive")
    expected_width = pairs_per_sample * 2 * structure.block_bits
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError("flattened feature geometry does not match the structure")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("flattened ciphertext features must be binary")

    rows = int(values.shape[0])
    feature_dim = cell_joint_response_feature_dim(structure.cells)
    output = np.empty((rows, feature_dim), dtype=np.float32)
    categories = np.arange(CELL_VALUE_COUNT, dtype=np.uint8)
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        runtime_pairs = np.asarray(values[start:stop], dtype=np.uint8).reshape(
            stop - start,
            pairs_per_sample,
            2,
            structure.block_bits,
        )[..., ::-1]
        response = exact_gf2_operator_response(runtime_pairs, structure)
        cell_values = response_bits_to_cell_values(response, structure)
        one_hot = cell_values[..., None] == categories
        output[start:stop] = one_hot.mean(axis=1, dtype=np.float32).reshape(
            stop - start,
            feature_dim,
        )
    return output


def cell_joint_response_feature_dim(cells: int) -> int:
    if cells <= 0:
        raise ValueError("cell count must be positive")
    return (
        cells
        * len(RESPONSE_VIEW_NAMES)
        * len(RAW_CHANNEL_NAMES)
        * CELL_VALUE_COUNT
    )


__all__: Sequence[str] = (
    "CELL_VALUE_COUNT",
    "cell_joint_response_feature_dim",
    "extract_cell_joint_gf2_operator_features",
    "response_bits_to_cell_values",
)
