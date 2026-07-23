from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.lib.format import open_memmap

from blockcipher_nd.ciphers.spn.gift import Gift64
from blockcipher_nd.tasks.innovation2.gift64_unit_balance_profile_readiness import (
    encrypt_gift_words,
    gift_round_injections,
)


SOURCE_CACHE_VERSION = 1
FRESH_CACHE_VERSION = 1
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class Gift64OutputPredictionDataConfig:
    run_id: str = "i2_output_prediction_gx1_gift64_r3_full64_readiness_seed11_20260723"
    mode: str = "readiness"
    rounds: int = 3
    seed: int = 11
    key_seed: int = 11
    train_rows: int = 64
    discovery_rows: int = 64
    fresh_rows: int = 64
    chunk_rows: int = 32

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"readiness", "formal"}:
            raise ValueError("mode must be readiness or formal")
        if self.rounds < 1 or self.rounds > 28:
            raise ValueError("GIFT-64 rounds must be in 1..28")
        if min(
            self.train_rows,
            self.discovery_rows,
            self.fresh_rows,
            self.chunk_rows,
        ) <= 0:
            raise ValueError("row and chunk values must be positive")
        if self.mode == "formal" and (
            self.rounds != 3
            or self.seed != 11
            or self.key_seed != 11
            or self.train_rows != 1 << 17
            or self.discovery_rows != 1 << 16
            or self.fresh_rows != 1 << 16
            or self.chunk_rows != 4096
        ):
            raise ValueError("formal GX1 data protocol is frozen")

    @classmethod
    def formal(
        cls,
        *,
        run_id: str = "i2_output_prediction_gx1_gift64_r3_full64_seed11_20260723",
    ) -> Gift64OutputPredictionDataConfig:
        return cls(
            run_id=run_id,
            mode="formal",
            train_rows=1 << 17,
            discovery_rows=1 << 16,
            fresh_rows=1 << 16,
            chunk_rows=4096,
        )


def prepare_gift64_source_data(
    config: Gift64OutputPredictionDataConfig,
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    secret_key = _secret_key(config.key_seed)
    total_rows = config.train_rows + config.discovery_rows
    expected = {
        "cache_version": SOURCE_CACHE_VERSION,
        "cipher": "GIFT-64",
        "rounds": config.rounds,
        "seed": config.seed,
        "key_seed": config.key_seed,
        "train_rows": config.train_rows,
        "discovery_rows": config.discovery_rows,
        "total_rows": total_rows,
        "bit_order": "msb_first",
        "secret_key_hex": f"{secret_key:032x}",
        "split": "train_then_discovery",
        "plaintext_rng": f"numpy_default_rng({1_120_000 + config.seed})",
    }
    data = _prepare_cache(
        data_root=output_root / "data",
        expected=expected,
        rows=total_rows,
        chunk_rows=config.chunk_rows,
        rng_seed=1_120_000 + config.seed,
        secret_key=secret_key,
        rounds=config.rounds,
        excluded_plaintexts=set(),
        progress_event="source_cache_chunk",
        progress=progress,
    )
    return {
        **data,
        "secret_key": secret_key,
        "train_features": data["features"][: config.train_rows],
        "train_targets": data["full_targets"][: config.train_rows],
        "discovery_features": data["features"][config.train_rows :],
        "discovery_targets": data["full_targets"][config.train_rows :],
    }


def prepare_gift64_fresh_data(
    config: Gift64OutputPredictionDataConfig,
    source: dict[str, Any],
    output_root: Path,
    *,
    candidate_sha256: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if len(candidate_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in candidate_sha256
    ):
        raise ValueError("candidate_sha256 must be a lowercase SHA256 hex digest")
    source_plaintexts = {int(value) for value in source["plaintexts"]}
    source_metadata = source["metadata"]
    expected = {
        "cache_version": FRESH_CACHE_VERSION,
        "cipher": "GIFT-64",
        "rounds": config.rounds,
        "seed": config.seed,
        "key_seed": config.key_seed,
        "fresh_rows": config.fresh_rows,
        "bit_order": "msb_first",
        "secret_key_hex": f"{int(source['secret_key']):032x}",
        "source_rows": len(source_plaintexts),
        "source_plaintexts_sha256": _sha256_array(source["plaintexts"]),
        "source_cache_version": source_metadata["cache_version"],
        "candidate_sha256": candidate_sha256,
        "split": "fresh_after_candidate_freeze",
        "plaintext_rng": f"numpy_default_rng({1_130_000 + config.seed})",
    }
    data = _prepare_cache(
        data_root=output_root / "fresh_data",
        expected=expected,
        rows=config.fresh_rows,
        chunk_rows=config.chunk_rows,
        rng_seed=1_130_000 + config.seed,
        secret_key=int(source["secret_key"]),
        rounds=config.rounds,
        excluded_plaintexts=source_plaintexts,
        progress_event="fresh_cache_chunk",
        progress=progress,
    )
    return {**data, "secret_key": int(source["secret_key"])}


def validate_gift64_source_data(
    config: Gift64OutputPredictionDataConfig,
    data: dict[str, Any],
) -> dict[str, bool]:
    total_rows = config.train_rows + config.discovery_rows
    plaintexts = data["plaintexts"]
    features = data["features"]
    targets = data["full_targets"]
    train_plaintexts = {int(value) for value in plaintexts[: config.train_rows]}
    discovery_plaintexts = {
        int(value) for value in plaintexts[config.train_rows :]
    }
    sample_indices = sorted(
        {0, config.train_rows - 1, config.train_rows, total_rows - 1}
    )
    cipher = Gift64(rounds=config.rounds, key=int(data["secret_key"]))
    return {
        "official_gift64_zero_vector_matches": Gift64(rounds=28, key=0).encrypt(0)
        == 0xF62BC3EF34F775AC,
        "cache_is_complete": data["metadata"].get("status") == "complete"
        and int(data["metadata"].get("completed_rows", -1)) == total_rows,
        "cache_arrays_have_expected_shapes": plaintexts.shape == (total_rows,)
        and features.shape == (total_rows, 64)
        and targets.shape == (total_rows, 64),
        "plaintexts_are_unique": len(train_plaintexts | discovery_plaintexts)
        == total_rows,
        "train_and_discovery_plaintexts_are_disjoint": train_plaintexts.isdisjoint(
            discovery_plaintexts
        ),
        "features_are_msb_first_plaintext_bits": all(
            _bits_to_word(features[index]) == int(plaintexts[index])
            for index in sample_indices
        ),
        "targets_are_msb_first_true_gift64_ciphertext_bits": all(
            _bits_to_word(targets[index]) == cipher.encrypt(int(plaintexts[index]))
            for index in sample_indices
        ),
        "secret_key_is_128_bit": 0 <= int(data["secret_key"]) < 1 << 128,
        "metadata_declares_true_output_task": data["metadata"].get("cipher")
        == "GIFT-64"
        and data["metadata"].get("bit_order") == "msb_first",
        "labels_are_outputs_not_sample_classes": True,
    }


def validate_gift64_fresh_data(
    config: Gift64OutputPredictionDataConfig,
    source: dict[str, Any],
    fresh: dict[str, Any],
    *,
    candidate_sha256: str,
) -> dict[str, bool]:
    plaintexts = fresh["plaintexts"]
    features = fresh["features"]
    targets = fresh["full_targets"]
    source_plaintexts = {int(value) for value in source["plaintexts"]}
    fresh_plaintexts = {int(value) for value in plaintexts}
    cipher = Gift64(rounds=config.rounds, key=int(source["secret_key"]))
    sample_indices = sorted({0, config.fresh_rows - 1})
    return {
        "fresh_cache_is_complete": fresh["metadata"].get("status") == "complete"
        and int(fresh["metadata"].get("completed_rows", -1)) == config.fresh_rows,
        "fresh_arrays_have_expected_shapes": plaintexts.shape
        == (config.fresh_rows,)
        and features.shape == (config.fresh_rows, 64)
        and targets.shape == (config.fresh_rows, 64),
        "fresh_plaintexts_are_unique": len(fresh_plaintexts) == config.fresh_rows,
        "fresh_plaintexts_are_disjoint_from_source": source_plaintexts.isdisjoint(
            fresh_plaintexts
        ),
        "fresh_features_are_msb_first_plaintext_bits": all(
            _bits_to_word(features[index]) == int(plaintexts[index])
            for index in sample_indices
        ),
        "fresh_targets_are_true_gift64_ciphertext_bits": all(
            _bits_to_word(targets[index]) == cipher.encrypt(int(plaintexts[index]))
            for index in sample_indices
        ),
        "fresh_uses_source_secret_key": int(fresh["secret_key"])
        == int(source["secret_key"]),
        "candidate_hash_is_frozen_in_cache": fresh["metadata"].get(
            "candidate_sha256"
        )
        == candidate_sha256,
        "fresh_split_declares_post_freeze_generation": fresh["metadata"].get(
            "split"
        )
        == "fresh_after_candidate_freeze",
        "labels_are_outputs_not_sample_classes": True,
    }


def serializable_config(
    config: Gift64OutputPredictionDataConfig,
) -> dict[str, Any]:
    return asdict(config)


def _prepare_cache(
    *,
    data_root: Path,
    expected: dict[str, Any],
    rows: int,
    chunk_rows: int,
    rng_seed: int,
    secret_key: int,
    rounds: int,
    excluded_plaintexts: set[int],
    progress_event: str,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    data_root.mkdir(parents=True, exist_ok=True)
    metadata_path = data_root / "cache_metadata.json"
    paths = {
        "plaintexts": data_root / "plaintexts.npy",
        "features": data_root / "features.npy",
        "full_targets": data_root / "full_targets.npy",
    }
    metadata = _read_json(metadata_path) if metadata_path.exists() else None
    if metadata is not None and {
        key: metadata.get(key) for key in expected
    } != expected:
        raise ValueError("existing GIFT-64 output cache parameters do not match")
    completed_rows = int(metadata.get("completed_rows", 0)) if metadata else 0
    arrays_exist = all(path.exists() for path in paths.values())
    if completed_rows and not arrays_exist:
        raise ValueError("GIFT-64 cache reports progress but arrays are missing")
    if arrays_exist:
        plaintexts = np.load(paths["plaintexts"], mmap_mode="r+")
        features = np.load(paths["features"], mmap_mode="r+")
        full_targets = np.load(paths["full_targets"], mmap_mode="r+")
        if (
            plaintexts.shape != (rows,)
            or features.shape != (rows, 64)
            or full_targets.shape != (rows, 64)
        ):
            raise ValueError("existing GIFT-64 output cache has invalid shapes")
    else:
        plaintexts = open_memmap(
            paths["plaintexts"], mode="w+", dtype=np.uint64, shape=(rows,)
        )
        features = open_memmap(
            paths["features"], mode="w+", dtype=np.float32, shape=(rows, 64)
        )
        full_targets = open_memmap(
            paths["full_targets"],
            mode="w+",
            dtype=np.float32,
            shape=(rows, 64),
        )
        completed_rows = 0
    rng = np.random.default_rng(rng_seed)
    if metadata and metadata.get("rng_state"):
        rng.bit_generator.state = metadata["rng_state"]
    seen = excluded_plaintexts | {
        int(value) for value in plaintexts[:completed_rows]
    }
    injections = gift_round_injections((secret_key,), rounds)
    shifts = np.arange(63, -1, -1, dtype=np.uint64)
    _write_json(
        metadata_path,
        {
            **expected,
            "status": "generating" if completed_rows < rows else "complete",
            "completed_rows": completed_rows,
            "rng_state": rng.bit_generator.state,
        },
    )
    while completed_rows < rows:
        stop = min(rows, completed_rows + chunk_rows)
        values: list[int] = []
        while len(values) < stop - completed_rows:
            low = int(rng.integers(0, 1 << 32, dtype=np.uint64))
            high = int(rng.integers(0, 1 << 32, dtype=np.uint64))
            value = low | (high << 32)
            if value not in seen:
                seen.add(value)
                values.append(value)
        words = np.asarray(values, dtype=np.uint64)
        ciphertexts = encrypt_gift_words(words, injections)[0]
        plaintexts[completed_rows:stop] = words
        features[completed_rows:stop] = (
            (words[:, None] >> shifts[None, :]) & np.uint64(1)
        ).astype(np.float32)
        full_targets[completed_rows:stop] = (
            (ciphertexts[:, None] >> shifts[None, :]) & np.uint64(1)
        ).astype(np.float32)
        plaintexts.flush()
        features.flush()
        full_targets.flush()
        completed_rows = stop
        _write_json(
            metadata_path,
            {
                **expected,
                "status": "complete" if stop == rows else "generating",
                "completed_rows": completed_rows,
                "rng_state": rng.bit_generator.state,
            },
        )
        if progress is not None:
            progress(
                progress_event,
                {"completed_rows": completed_rows, "total_rows": rows},
            )
    return {
        "plaintexts": plaintexts,
        "features": features,
        "full_targets": full_targets,
        "metadata": _read_json(metadata_path),
        "data_root": data_root,
        "cache_reused": bool(metadata and metadata.get("status") == "complete"),
    }


def _secret_key(key_seed: int) -> int:
    return random.Random(1_110_000 + key_seed).getrandbits(128)


def _bits_to_word(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(round(float(bit)))
    return value


def _sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    values = np.asarray(array)
    for start in range(0, len(values), 1 << 15):
        digest.update(np.ascontiguousarray(values[start : start + (1 << 15)]).tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "Gift64OutputPredictionDataConfig",
    "prepare_gift64_fresh_data",
    "prepare_gift64_source_data",
    "serializable_config",
    "validate_gift64_fresh_data",
    "validate_gift64_source_data",
]
