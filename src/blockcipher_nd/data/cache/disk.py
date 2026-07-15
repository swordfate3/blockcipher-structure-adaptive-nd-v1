from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.data.differential import (
    DifferentialDatasetConfig,
    DiskDifferentialDataset,
)
from blockcipher_nd.data.differential.metadata import dataset_metadata
from blockcipher_nd.data.differential.rows import generate_negative_row, generate_positive_row
from blockcipher_nd.data.differential.validation import validate_differential_config


ProgressCallback = Callable[[str, dict[str, Any]], None]


def make_chunked_differential_dataset(
    config: DifferentialDatasetConfig,
    *,
    cache_dir: str | Path,
    chunk_size: int = 8192,
    workers: int = 1,
    reuse: bool = True,
    progress_callback: ProgressCallback | None = None,
    progress_context: dict[str, Any] | None = None,
) -> DiskDifferentialDataset:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    validate_differential_config(config)

    cache_path = Path(cache_dir)
    features_path = cache_path / "features.npy"
    labels_path = cache_path / "labels.npy"
    metadata_path = cache_path / "metadata.json"
    expected_metadata = dataset_metadata(config)
    total_rows = int(expected_metadata["samples_total"])
    input_bits = (
        expected_metadata["pair_bits"] * config.pairs_per_sample
        + expected_metadata["row_metadata_bits"]
    )
    context = dict(progress_context or {})

    if reuse and features_path.exists() and labels_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _cache_matches(metadata, expected_metadata, total_rows, input_bits):
            metadata = {**metadata, "cache_status": "reused"}
            _emit_progress(
                progress_callback,
                "cache_reuse",
                context,
                cache_path=cache_path,
                total_rows=total_rows,
                input_bits=input_bits,
                chunk_size=chunk_size,
                workers=workers,
            )
            features = np.load(features_path, mmap_mode="r")
            labels = np.load(labels_path, mmap_mode="r")
            return DiskDifferentialDataset(
                features=features,
                labels=labels,
                metadata=metadata,
                cache_dir=cache_path,
            )

    cache_path.mkdir(parents=True, exist_ok=True)
    _emit_progress(
        progress_callback,
        "cache_start",
        context,
        cache_path=cache_path,
        samples_per_class=config.samples_per_class,
        total_rows=total_rows,
        input_bits=input_bits,
        chunk_size=chunk_size,
        workers=workers,
        requested_shuffle=config.shuffle,
        physical_shuffle=False,
    )
    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_rows, input_bits),
    )
    labels = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_rows,),
    )
    block_bits = config.cipher.block_bits
    mask = (1 << block_bits) - 1
    rng = np.random.default_rng(config.seed)

    if config.dataset_label_mode == "random_labels_total":
        sampled_labels = rng.integers(0, 2, size=total_rows, dtype=np.uint8)
        labels[:] = sampled_labels
        for start, count, chunk in _iter_mixed_label_chunks(
            config=config,
            labels=sampled_labels,
            input_bits=input_bits,
            block_bits=block_bits,
            mask=mask,
            chunk_size=chunk_size,
            workers=workers,
        ):
            features[start : start + count] = chunk
            _emit_progress(
                progress_callback,
                "cache_mixed_label_chunk",
                context,
                cache_path=cache_path,
                rows_done=start + count,
                total_rows=total_rows,
                chunk_rows=count,
            )
    else:
        row_index = 0
        for start, count, chunk in _iter_generated_chunks(
            config=config,
            rng=rng,
            input_bits=input_bits,
            block_bits=block_bits,
            mask=mask,
            chunk_size=chunk_size,
            workers=workers,
            label=1,
        ):
            features[row_index : row_index + count] = chunk
            labels[row_index : row_index + count] = 1
            row_index += count
            _emit_progress(
                progress_callback,
                "cache_positive_chunk",
                context,
                cache_path=cache_path,
                rows_done=row_index,
                class_rows_done=start + count,
                class_total=config.samples_per_class,
                total_rows=total_rows,
                chunk_rows=count,
            )

        for start, count, chunk in _iter_generated_chunks(
            config=config,
            rng=rng,
            input_bits=input_bits,
            block_bits=block_bits,
            mask=mask,
            chunk_size=chunk_size,
            workers=workers,
            label=0,
        ):
            features[row_index : row_index + count] = chunk
            labels[row_index : row_index + count] = 0
            row_index += count
            _emit_progress(
                progress_callback,
                "cache_negative_chunk",
                context,
                cache_path=cache_path,
                rows_done=row_index,
                class_rows_done=start + count,
                class_total=config.samples_per_class,
                total_rows=total_rows,
                chunk_rows=count,
            )

    _emit_progress(
        progress_callback,
        "cache_flush_start",
        context,
        cache_path=cache_path,
        total_rows=total_rows,
    )
    features.flush()
    labels.flush()
    metadata = {
        **expected_metadata,
        "positive_rows": int(labels.sum()),
        "negative_rows": int(total_rows - labels.sum()),
        "total_rows": total_rows,
        "input_bits": input_bits,
        "generation_chunk_size": chunk_size,
        "generation_workers": workers,
        "cache_status": "created",
        "requested_shuffle": config.shuffle,
        "physical_shuffle": False,
        "training_shuffle": True,
    }
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
    _emit_progress(
        progress_callback,
        "cache_done",
        context,
        cache_path=cache_path,
        total_rows=total_rows,
        input_bits=input_bits,
        metadata_path=metadata_path,
    )
    return DiskDifferentialDataset(
        features=np.load(features_path, mmap_mode="r"),
        labels=np.load(labels_path, mmap_mode="r"),
        metadata=metadata,
        cache_dir=cache_path,
    )


def _iter_mixed_label_chunks(
    *,
    config: DifferentialDatasetConfig,
    labels: np.ndarray,
    input_bits: int,
    block_bits: int,
    mask: int,
    chunk_size: int,
    workers: int,
):
    specs = [
        (start, min(chunk_size, len(labels) - start))
        for start in range(0, len(labels), chunk_size)
    ]
    if workers == 1:
        for start, count in specs:
            yield start, count, _generate_mixed_chunk_worker(
                config,
                input_bits,
                block_bits,
                mask,
                start,
                labels[start : start + count],
            )
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _generate_mixed_chunk_worker,
                config,
                input_bits,
                block_bits,
                mask,
                start,
                labels[start : start + count],
            )
            for start, count in specs
        ]
        for (start, count), future in zip(specs, futures):
            yield start, count, future.result()


def _generate_mixed_chunk_worker(
    config: DifferentialDatasetConfig,
    input_bits: int,
    block_bits: int,
    mask: int,
    start: int,
    labels: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(_chunk_seed(config.seed, start, 2))
    chunk = np.empty((len(labels), input_bits), dtype=np.uint8)
    for offset, label in enumerate(labels):
        row_index = start + offset
        if label == 1:
            row = generate_positive_row(
                config,
                rng,
                block_bits,
                mask,
                row_index=row_index,
            )
        else:
            row = generate_negative_row(
                config,
                rng,
                block_bits,
                row_index=row_index,
            )
        if len(row) != input_bits:
            raise ValueError(f"generated row has {len(row)} bits, expected {input_bits}")
        chunk[offset] = row
    return chunk


def _iter_generated_chunks(
    *,
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    input_bits: int,
    block_bits: int,
    mask: int,
    chunk_size: int,
    workers: int,
    label: int,
):
    specs = [
        (start, min(chunk_size, config.samples_per_class - start))
        for start in range(0, config.samples_per_class, chunk_size)
    ]
    if workers == 1:
        for start, count in specs:
            yield start, count, _generate_chunk(
                count=count,
                input_bits=input_bits,
                row_factory=_row_factory(config, rng, block_bits, mask, start, label),
            )
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _generate_chunk_worker,
                config,
                input_bits,
                block_bits,
                mask,
                start,
                count,
                label,
            )
            for start, count in specs
        ]
        for (start, count), future in zip(specs, futures):
            yield start, count, future.result()


def _generate_chunk_worker(
    config: DifferentialDatasetConfig,
    input_bits: int,
    block_bits: int,
    mask: int,
    start: int,
    count: int,
    label: int,
) -> np.ndarray:
    rng = np.random.default_rng(_chunk_seed(config.seed, start, label))
    return _generate_chunk(
        count=count,
        input_bits=input_bits,
        row_factory=_row_factory(config, rng, block_bits, mask, start, label),
    )


def _row_factory(
    config: DifferentialDatasetConfig,
    rng: np.random.Generator,
    block_bits: int,
    mask: int,
    start: int,
    label: int,
) -> Callable[[int], list[int]]:
    if label == 1:
        return lambda offset: generate_positive_row(
            config,
            rng,
            block_bits,
            mask,
            row_index=start + offset,
        )
    return lambda offset: generate_negative_row(
        config,
        rng,
        block_bits,
        row_index=config.samples_per_class + start + offset,
    )


def _chunk_seed(seed: int, start: int, label: int) -> int:
    return seed + 1_000_003 * (start + 1) + 97_531 * label


def _generate_chunk(
    *,
    count: int,
    input_bits: int,
    row_factory: Callable[[int], list[int]],
) -> np.ndarray:
    chunk = np.empty((count, input_bits), dtype=np.uint8)
    for offset in range(count):
        row = row_factory(offset)
        if len(row) != input_bits:
            raise ValueError(f"generated row has {len(row)} bits, expected {input_bits}")
        chunk[offset] = row
    return chunk


def _cache_matches(
    metadata: dict[str, object],
    expected_metadata: dict[str, object],
    total_rows: int,
    input_bits: int,
) -> bool:
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            return False
    return metadata.get("total_rows") == total_rows and metadata.get("input_bits") == input_bits


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    context: dict[str, Any],
    **payload: Any,
) -> None:
    if callback is None:
        return
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in {**context, **payload}.items()
    }
    callback(event, serializable)
