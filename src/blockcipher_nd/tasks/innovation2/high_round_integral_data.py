from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.models.structure.spn.present_integral_multiset import (
    STATE_BITS,
    TEXTS_PER_MULTISET,
    VIEWS_PER_MULTISET,
    integral_input_bits,
)


PROTOCOL_VERSION = "wu_guo_2024_integral_multiset_v1"
NEGATIVE_MODE = "encrypted_unrestricted_random_plaintext_multisets"
BIT_ORDER = "multiset_view_text_state_lsb_first"
SPLIT_KEY_OFFSETS = {
    "train": 0,
    "validation": 1 << 32,
    "test": 2 << 32,
}

ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class IntegralMultisetCacheConfig:
    split: str
    rounds: int
    total_rows: int
    multiset_count: int
    seed: int
    cache_root: Path
    chunk_size: int = 256

    def __post_init__(self) -> None:
        if self.split not in SPLIT_KEY_OFFSETS:
            raise ValueError(f"unsupported split: {self.split}")
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if self.total_rows < 2 or self.total_rows % 2:
            raise ValueError("total_rows must be a positive even number")
        if self.total_rows >= 1 << 32:
            raise ValueError("total_rows exceeds the reserved split key range")
        if self.multiset_count < 1:
            raise ValueError("multiset_count must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")


@dataclass(frozen=True)
class IntegralMultisetSample:
    label: int
    key: int
    plaintexts: np.ndarray
    ciphertexts: np.ndarray
    views: np.ndarray
    features: np.ndarray


def build_integral_multiset_sample(
    *,
    rounds: int,
    multiset_count: int,
    label: int,
    seed: int,
    split: str,
    row_index: int,
) -> IntegralMultisetSample:
    if split not in SPLIT_KEY_OFFSETS:
        raise ValueError(f"unsupported split: {split}")
    if label not in {0, 1}:
        raise ValueError("label must be 0 or 1")
    if row_index < 0 or row_index >= 1 << 32:
        raise ValueError("row_index is outside the reserved split range")
    rng = np.random.default_rng(_sample_seed(seed, split, row_index))
    plaintexts = np.empty(
        (multiset_count, TEXTS_PER_MULTISET),
        dtype=np.uint64,
    )
    for multiset_index in range(multiset_count):
        if label == 1:
            base = int(rng.integers(0, 1 << 64, dtype=np.uint64)) & ~0xF
            plaintexts[multiset_index] = np.array(
                [base | value for value in range(TEXTS_PER_MULTISET)],
                dtype=np.uint64,
            )
        else:
            plaintexts[multiset_index] = rng.integers(
                0,
                1 << 64,
                size=TEXTS_PER_MULTISET,
                dtype=np.uint64,
            )

    key_index = SPLIT_KEY_OFFSETS[split] + row_index
    key = _derive_unique_key(seed=seed, index=key_index)
    cipher = Present80(rounds=rounds, key=key)
    ciphertexts = np.empty_like(plaintexts)
    views = np.empty(
        (multiset_count, VIEWS_PER_MULTISET, TEXTS_PER_MULTISET),
        dtype=np.uint64,
    )
    for multiset_index in range(multiset_count):
        for text_index in range(TEXTS_PER_MULTISET):
            ciphertexts[multiset_index, text_index] = cipher.encrypt(
                int(plaintexts[multiset_index, text_index])
            )
        reference = int(ciphertexts[multiset_index, 0])
        for text_index in range(TEXTS_PER_MULTISET):
            difference = int(ciphertexts[multiset_index, text_index]) ^ reference
            invp = Present80.inverse_permutation_layer(difference)
            views[multiset_index, 0, text_index] = invp
            views[multiset_index, 1, text_index] = Present80.inverse_sbox_layer(invp)

    shifts = np.arange(STATE_BITS, dtype=np.uint64)
    features = ((views[..., None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return IntegralMultisetSample(
        label=label,
        key=key,
        plaintexts=plaintexts,
        ciphertexts=ciphertexts,
        views=views,
        features=features,
    )


def make_integral_multiset_cache(
    config: IntegralMultisetCacheConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> DiskDifferentialDataset:
    identity = cache_identity(config)
    cache_dir = cache_directory(config)
    features_path = cache_dir / "features.npy"
    labels_path = cache_dir / "labels.npy"
    metadata_path = cache_dir / "metadata.json"
    input_bits = integral_input_bits(config.multiset_count)
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any]
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_identity") != identity:
            raise ValueError(f"cache identity mismatch: {cache_dir}")
        _validate_cache_arrays(
            features_path,
            labels_path,
            total_rows=config.total_rows,
            input_bits=input_bits,
        )
        rows_generated = int(metadata.get("rows_generated", 0))
        if metadata.get("status") == "complete":
            if rows_generated != config.total_rows:
                raise ValueError("complete cache has an invalid rows_generated value")
            _emit(
                progress_callback,
                "integral_cache_reuse",
                split=config.split,
                cache_dir=str(cache_dir),
                rows=config.total_rows,
                input_bits=input_bits,
            )
            return _load_disk_dataset(cache_dir, metadata, cache_status="reused")
        if not 0 <= rows_generated < config.total_rows:
            raise ValueError("in-progress cache has an invalid rows_generated value")
        features = np.load(features_path, mmap_mode="r+")
        labels = np.load(labels_path, mmap_mode="r+")
        cache_event = "integral_cache_resume"
    else:
        if features_path.exists() or labels_path.exists():
            raise ValueError(f"cache arrays exist without metadata: {cache_dir}")
        features = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=np.uint8,
            shape=(config.total_rows, input_bits),
        )
        labels = np.lib.format.open_memmap(
            labels_path,
            mode="w+",
            dtype=np.uint8,
            shape=(config.total_rows,),
        )
        rows_generated = 0
        metadata = {
            **cache_metadata(config),
            "cache_identity": identity,
            "cache_dir": str(cache_dir),
            "status": "in_progress",
            "rows_generated": 0,
            "positive_rows": 0,
            "negative_rows": 0,
        }
        _write_metadata(metadata_path, metadata)
        cache_event = "integral_cache_start"

    _emit(
        progress_callback,
        cache_event,
        split=config.split,
        cache_dir=str(cache_dir),
        rows_done=rows_generated,
        total_rows=config.total_rows,
        input_bits=input_bits,
        chunk_size=config.chunk_size,
    )
    for start in range(rows_generated, config.total_rows, config.chunk_size):
        end = min(config.total_rows, start + config.chunk_size)
        for row_index in range(start, end):
            label = row_index & 1
            sample = build_integral_multiset_sample(
                rounds=config.rounds,
                multiset_count=config.multiset_count,
                label=label,
                seed=config.seed,
                split=config.split,
                row_index=row_index,
            )
            features[row_index] = sample.features
            labels[row_index] = label
        features.flush()
        labels.flush()
        metadata["rows_generated"] = end
        metadata["positive_rows"] = end // 2
        metadata["negative_rows"] = end - end // 2
        _write_metadata(metadata_path, metadata)
        _emit(
            progress_callback,
            "integral_cache_chunk",
            split=config.split,
            cache_dir=str(cache_dir),
            rows_done=end,
            total_rows=config.total_rows,
            chunk_rows=end - start,
            input_bits=input_bits,
        )

    metadata.update(
        {
            "status": "complete",
            "rows_generated": config.total_rows,
            "positive_rows": config.total_rows // 2,
            "negative_rows": config.total_rows // 2,
            "cache_status": "created" if rows_generated == 0 else "resumed",
        }
    )
    _write_metadata(metadata_path, metadata)
    _emit(
        progress_callback,
        "integral_cache_done",
        split=config.split,
        cache_dir=str(cache_dir),
        rows_done=config.total_rows,
        total_rows=config.total_rows,
        input_bits=input_bits,
    )
    del features
    del labels
    return _load_disk_dataset(
        cache_dir,
        metadata,
        cache_status=str(metadata["cache_status"]),
    )


def cache_metadata(config: IntegralMultisetCacheConfig) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task": "innovation2_present_high_round_integral_multiset",
        "cipher": "PRESENT-80",
        "split": config.split,
        "rounds": config.rounds,
        "total_rows": config.total_rows,
        "samples_per_class": config.total_rows // 2,
        "multisets_per_sample": config.multiset_count,
        "texts_per_multiset": TEXTS_PER_MULTISET,
        "views": ["invp_cj_xor_c0", "invs_invp_cj_xor_c0"],
        "input_bits": integral_input_bits(config.multiset_count),
        "feature_dtype": "uint8",
        "label_dtype": "uint8",
        "bit_order": BIT_ORDER,
        "positive_definition": "low_nibble_0_to_15_high_60_bits_fixed",
        "negative_definition": NEGATIVE_MODE,
        "key_sampling": "one_unique_present80_master_key_per_sample",
        "key_derivation": "six_round_sha256_feistel80",
        "key_index_offset": SPLIT_KEY_OFFSETS[config.split],
        "seed": config.seed,
        "chunk_size": config.chunk_size,
        "paper_tensor_concat_assumption": "spatial_axis_1",
    }


def cache_identity(config: IntegralMultisetCacheConfig) -> str:
    payload = json.dumps(
        cache_metadata(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_directory(config: IntegralMultisetCacheConfig) -> Path:
    return config.cache_root / f"{config.split}_{cache_identity(config)[:16]}"


def fixed_parity_weight_scores(
    features: np.ndarray,
    *,
    multiset_count: int,
) -> dict[str, np.ndarray]:
    shaped = np.asarray(features).reshape(
        len(features),
        multiset_count,
        VIEWS_PER_MULTISET,
        TEXTS_PER_MULTISET,
        STATE_BITS,
    )
    parity = shaped.sum(axis=3, dtype=np.uint16) & 1
    invp_weight = parity[:, :, 0, :].sum(axis=(1, 2), dtype=np.uint16)
    invs_weight = parity[:, :, 1, :].sum(axis=(1, 2), dtype=np.uint16)
    total_weight = parity.sum(axis=(1, 2, 3), dtype=np.uint16)
    return {
        "negative_total_parity_weight": -total_weight.astype(np.float32),
        "negative_invp_parity_weight": -invp_weight.astype(np.float32),
        "negative_invs_parity_weight": -invs_weight.astype(np.float32),
    }


def _load_disk_dataset(
    cache_dir: Path,
    metadata: dict[str, Any],
    *,
    cache_status: str,
) -> DiskDifferentialDataset:
    return DiskDifferentialDataset(
        features=np.load(cache_dir / "features.npy", mmap_mode="r"),
        labels=np.load(cache_dir / "labels.npy", mmap_mode="r"),
        metadata={**metadata, "cache_status": cache_status},
        cache_dir=cache_dir,
    )


def _validate_cache_arrays(
    features_path: Path,
    labels_path: Path,
    *,
    total_rows: int,
    input_bits: int,
) -> None:
    if not features_path.is_file() or not labels_path.is_file():
        raise ValueError("cache metadata exists without both arrays")
    features = np.load(features_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    if features.shape != (total_rows, input_bits) or features.dtype != np.uint8:
        raise ValueError("cached features have the wrong shape or dtype")
    if labels.shape != (total_rows,) or labels.dtype != np.uint8:
        raise ValueError("cached labels have the wrong shape or dtype")


def _sample_seed(seed: int, split: str, row_index: int) -> int:
    digest = hashlib.sha256(
        f"{PROTOCOL_VERSION}:{seed}:{split}:{row_index}:plaintext".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _derive_unique_key(*, seed: int, index: int) -> int:
    mask = (1 << 40) - 1
    left = (index >> 40) & mask
    right = index & mask
    seed_bytes = int(seed).to_bytes(16, "big", signed=True)
    for round_index in range(6):
        digest = hashlib.sha256(
            seed_bytes + round_index.to_bytes(1, "big") + right.to_bytes(5, "big")
        ).digest()
        function_value = int.from_bytes(digest[:5], "big")
        left, right = right, (left ^ function_value) & mask
    return (left << 40) | right


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)


__all__ = [
    "BIT_ORDER",
    "IntegralMultisetCacheConfig",
    "IntegralMultisetSample",
    "NEGATIVE_MODE",
    "PROTOCOL_VERSION",
    "SPLIT_KEY_OFFSETS",
    "build_integral_multiset_sample",
    "cache_directory",
    "cache_identity",
    "fixed_parity_weight_scores",
    "make_integral_multiset_cache",
]
