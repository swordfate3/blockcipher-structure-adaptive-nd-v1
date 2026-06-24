from __future__ import annotations

import numpy as np

from blockcipher_nd.data.differential.config import (
    DifferentialDataset,
    DifferentialDatasetConfig,
)
from blockcipher_nd.data.differential.metadata import dataset_metadata
from blockcipher_nd.data.differential.rows import generate_negative_row, generate_positive_row
from blockcipher_nd.data.differential.validation import validate_differential_config


def make_differential_dataset(config: DifferentialDatasetConfig) -> DifferentialDataset:
    validate_differential_config(config)
    rng = np.random.default_rng(config.seed)
    block_bits = config.cipher.block_bits
    mask = (1 << block_bits) - 1
    rows: list[list[int]] = []
    labels: list[int] = []

    for row_index in range(config.samples_per_class):
        rows.append(generate_positive_row(config, rng, block_bits, mask, row_index=row_index))
        labels.append(1)

    for row_index in range(config.samples_per_class):
        rows.append(generate_negative_row(config, rng, block_bits, row_index=row_index))
        labels.append(0)

    features = np.array(rows, dtype=np.uint8)
    label_array = np.array(labels, dtype=np.uint8)
    if config.shuffle:
        order = rng.permutation(len(label_array))
        features = features[order]
        label_array = label_array[order]

    return DifferentialDataset(
        features=features,
        labels=label_array,
        metadata=dataset_metadata(config),
    )
