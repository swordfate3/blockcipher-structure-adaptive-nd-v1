from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure


RESPONSE_VIEW_NAMES = (
    "raw",
    "inverse_linear_0",
    "inverse_linear_1",
    "composed_1_then_0",
)
RAW_CHANNEL_NAMES = ("left", "right", "left_xor_right")


def apply_numpy_gf2_operator(
    values: np.ndarray,
    operator: np.ndarray,
) -> np.ndarray:
    """Apply one binary operator with XOR reductions over source positions."""
    source = np.asarray(values, dtype=np.uint8)
    matrix = np.asarray(operator, dtype=np.uint8)
    if source.ndim < 2:
        raise ValueError("GF(2) values must include bit and channel dimensions")
    bits = source.shape[-2]
    if matrix.shape != (bits, bits):
        raise ValueError("GF(2) operator dimensions do not match values")
    if not np.all((source == 0) | (source == 1)):
        raise ValueError("GF(2) response values must be binary")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("GF(2) operator must be binary")

    transformed = np.empty_like(source, dtype=np.uint8)
    for target, row in enumerate(matrix):
        sources = np.flatnonzero(row)
        if sources.size == 0:
            transformed[..., target, :] = 0
        else:
            transformed[..., target, :] = np.bitwise_xor.reduce(
                source[..., sources, :],
                axis=-2,
            )
    return transformed


def exact_gf2_operator_response(
    ciphertext_pairs: np.ndarray,
    structure: RuntimeSpnStructure,
) -> np.ndarray:
    """Return ordered bit x view x channel responses without spatial pooling."""
    pairs = np.asarray(ciphertext_pairs, dtype=np.uint8)
    if pairs.ndim != 4 or pairs.shape[2] != 2:
        raise ValueError("ciphertext pairs must have shape [rows, pairs, 2, bits]")
    if pairs.shape[-1] != structure.block_bits:
        raise ValueError("ciphertext pair width does not match runtime structure")
    if structure.rounds != 2:
        raise ValueError("exact GF(2) response requires exactly two transitions")
    if not np.all((pairs == 0) | (pairs == 1)):
        raise ValueError("ciphertext pairs must be binary")

    left = pairs[:, :, 0]
    right = pairs[:, :, 1]
    raw = np.stack((left, right, np.bitwise_xor(left, right)), axis=-1)
    matrices = np.asarray(structure.inverse_linear_matrices, dtype=np.uint8)
    first = apply_numpy_gf2_operator(raw, matrices[0])
    second = apply_numpy_gf2_operator(raw, matrices[1])
    composed = apply_numpy_gf2_operator(second, matrices[0])
    return np.concatenate((raw, first, second, composed), axis=-1)


def extract_exact_gf2_operator_features(
    flat_features: np.ndarray,
    structure: RuntimeSpnStructure,
    *,
    pairs_per_sample: int = 4,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract pair-mean responses while retaining native bit/view coordinates."""
    values = np.asarray(flat_features)
    if pairs_per_sample <= 0 or batch_size <= 0:
        raise ValueError("pair count and batch size must be positive")
    expected_width = pairs_per_sample * 2 * structure.block_bits
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError("flattened feature geometry does not match the structure")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("flattened ciphertext features must be binary")

    rows = int(values.shape[0])
    feature_dim = structure.block_bits * len(RESPONSE_VIEW_NAMES) * len(
        RAW_CHANNEL_NAMES
    )
    output = np.empty((rows, feature_dim), dtype=np.float32)
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        # Project storage MSB order into the runtime structure's LSB bit order.
        runtime_pairs = np.asarray(values[start:stop], dtype=np.uint8).reshape(
            stop - start,
            pairs_per_sample,
            2,
            structure.block_bits,
        )[..., ::-1]
        response = exact_gf2_operator_response(runtime_pairs, structure)
        output[start:stop] = response.mean(axis=1, dtype=np.float32).reshape(
            stop - start,
            feature_dim,
        )
    return output


def response_feature_dim(block_bits: int) -> int:
    if block_bits <= 0:
        raise ValueError("block_bits must be positive")
    return block_bits * len(RESPONSE_VIEW_NAMES) * len(RAW_CHANNEL_NAMES)


__all__: Sequence[str] = (
    "RAW_CHANNEL_NAMES",
    "RESPONSE_VIEW_NAMES",
    "apply_numpy_gf2_operator",
    "exact_gf2_operator_response",
    "extract_exact_gf2_operator_features",
    "response_feature_dim",
)
