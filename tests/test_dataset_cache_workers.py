from __future__ import annotations

import json

import numpy as np

from blockcipher_nd.data.cache import make_chunked_differential_dataset
from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.registry.cipher_factory import build_cipher


def test_chunked_dataset_cache_can_generate_with_multiple_workers(tmp_path) -> None:
    events: list[str] = []
    cipher = build_cipher("speck32", rounds=3, key=0)
    config = DifferentialDatasetConfig(
        cipher=cipher,
        input_difference=0x0040,
        samples_per_class=16,
        seed=9,
        shuffle=False,
        feature_encoding="ciphertext_pair_xor_bits",
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="independent_pairs",
    )

    dataset = make_chunked_differential_dataset(
        config,
        cache_dir=tmp_path / "cache",
        chunk_size=4,
        workers=2,
        progress_callback=lambda event, _payload: events.append(event),
    )

    assert dataset.features.shape == (32, 192)
    assert dataset.labels.shape == (32,)
    assert set(np.unique(dataset.labels).tolist()) == {0, 1}
    assert events.count("cache_positive_chunk") == 4
    assert events.count("cache_negative_chunk") == 4
    assert dataset.metadata["generation_workers"] == 2

    metadata = json.loads((tmp_path / "cache" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["generation_workers"] == 2

    reused = make_chunked_differential_dataset(
        config,
        cache_dir=tmp_path / "cache",
        chunk_size=4,
        workers=2,
    )

    assert reused.metadata["cache_status"] == "reused"
    assert np.array_equal(np.asarray(dataset.features), np.asarray(reused.features))
    assert np.array_equal(np.asarray(dataset.labels), np.asarray(reused.labels))


def test_single_worker_cache_preserves_historical_generation_stream(tmp_path) -> None:
    cipher = build_cipher("present80", rounds=7, key=0)
    config = DifferentialDatasetConfig(
        cipher=cipher,
        input_difference=0x1111111111111111,
        samples_per_class=8,
        seed=3,
        shuffle=False,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_official_mcnd",
    )

    memory_dataset = make_differential_dataset(config)
    cache_dataset = make_chunked_differential_dataset(
        config,
        cache_dir=tmp_path / "single_worker_cache",
        chunk_size=3,
        workers=1,
        reuse=False,
    )

    assert cache_dataset.metadata["generation_workers"] == 1
    assert np.array_equal(np.asarray(memory_dataset.features), np.asarray(cache_dataset.features))
    assert np.array_equal(np.asarray(memory_dataset.labels), np.asarray(cache_dataset.labels))
