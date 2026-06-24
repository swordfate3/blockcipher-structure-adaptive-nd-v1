from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blockcipher_nd.ciphers import ReducedRoundCipher


@dataclass(frozen=True)
class DifferentialDatasetConfig:
    cipher: ReducedRoundCipher
    input_difference: int
    samples_per_class: int
    seed: int
    shuffle: bool = True
    feature_encoding: str = "ciphertext_pair_bits"
    pairs_per_sample: int = 1
    negative_mode: str = "random_ciphertext"
    key_rotation_interval: int = 0
    sample_structure: str = "independent_pairs"
    integral_active_nibble: int = 0
    selected_bit_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class DifferentialDataset:
    features: NDArray[np.uint8]
    labels: NDArray[np.uint8]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DiskDifferentialDataset(DifferentialDataset):
    cache_dir: Path
