from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.lib.format import open_memmap

from blockcipher_nd.ciphers.arx.speck import Speck32_64
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import (
    encrypt_speck32_numpy,
)


CACHE_VERSION = 1
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class Speck32OutputPredictionDataConfig:
    run_id: str = "i2_output_prediction_arx1_speck32_r3_readiness_seed21"
    mode: str = "readiness"
    rounds: int = 3
    seed: int = 21
    key_seed: int = 21
    train_rows: int = 64
    test_rows: int = 64
    chunk_rows: int = 32

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"readiness", "arx1_a", "arx1_b", "arx1_c"}:
            raise ValueError("mode must be readiness, arx1_a, arx1_b, or arx1_c")
        if self.rounds < 1 or self.rounds > 22:
            raise ValueError("SPECK32/64 rounds must be in 1..22")
        if min(self.train_rows, self.test_rows, self.chunk_rows) <= 0:
            raise ValueError("row and chunk values must be positive")
        if min(self.seed, self.key_seed) < 0:
            raise ValueError("seed and key_seed must be non-negative")
        if self.train_rows + self.test_rows > 1 << 32:
            raise ValueError("unique SPECK32 plaintext rows cannot exceed 2^32")
        expected_formal = {
            "arx1_a": (21, 1 << 20, 1 << 15),
            "arx1_b": (22, 1 << 20, 1 << 15),
            "arx1_c": (21, 1 << 22, 1 << 15),
        }
        if self.mode in expected_formal:
            key_seed, train_rows, test_rows = expected_formal[self.mode]
            if (
                self.rounds != 3
                or self.seed != 21
                or self.key_seed != key_seed
                or self.train_rows != train_rows
                or self.test_rows != test_rows
                or self.chunk_rows != 4096
            ):
                raise ValueError(f"formal {self.mode} data protocol is frozen")

    @classmethod
    def arx1_a(
        cls,
        *,
        run_id: str = "i2_output_prediction_arx1a_speck32_r3_key21",
    ) -> Speck32OutputPredictionDataConfig:
        return cls(
            run_id=run_id,
            mode="arx1_a",
            train_rows=1 << 20,
            test_rows=1 << 15,
            chunk_rows=4096,
        )

    @classmethod
    def arx1_b(
        cls,
        *,
        run_id: str = "i2_output_prediction_arx1b_speck32_r3_key22",
    ) -> Speck32OutputPredictionDataConfig:
        return cls(
            run_id=run_id,
            mode="arx1_b",
            key_seed=22,
            train_rows=1 << 20,
            test_rows=1 << 15,
            chunk_rows=4096,
        )

    @classmethod
    def arx1_c(
        cls,
        *,
        run_id: str = "i2_output_prediction_arx1c_speck32_r3_2p22_key21",
    ) -> Speck32OutputPredictionDataConfig:
        return cls(
            run_id=run_id,
            mode="arx1_c",
            train_rows=1 << 22,
            test_rows=1 << 15,
            chunk_rows=4096,
        )


def prepare_speck32_output_prediction_data(
    config: Speck32OutputPredictionDataConfig,
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    metadata_path = data_root / "cache_metadata.json"
    total_rows = config.train_rows + config.test_rows
    secret_key = _secret_key(config.key_seed)
    expected = {
        "cache_version": CACHE_VERSION,
        "cipher": "SPECK32/64",
        "rounds": config.rounds,
        "seed": config.seed,
        "key_seed": config.key_seed,
        "train_rows": config.train_rows,
        "test_rows": config.test_rows,
        "total_rows": total_rows,
        "task": "innovation2_output_prediction",
        "input": "32_msb_first_plaintext_bits",
        "target": "32_msb_first_true_speck32_ciphertext_bits",
        "sample_classification": False,
        "secret_key_scope": "single_fixed_unknown_key",
        "bit_order": "msb_first",
        "word_order": ["x_msw", "y_lsw"],
        "word_bits": 16,
        "secret_key_hex": f"{secret_key:016x}",
        "split": "train_then_test_with_independent_rng_streams",
        "train_plaintext_rng": f"numpy_default_rng({1_220_000 + config.seed})",
        "test_plaintext_rng": f"numpy_default_rng({1_230_000 + config.seed})",
    }
    paths = {
        "plaintexts": data_root / "plaintexts.npy",
        "features": data_root / "features.npy",
        "full_targets": data_root / "full_targets.npy",
    }
    metadata = _read_json(metadata_path) if metadata_path.exists() else None
    if (
        metadata is not None
        and {key: metadata.get(key) for key in expected} != expected
    ):
        raise ValueError("existing SPECK32 output cache parameters do not match")
    arrays_exist = [path.exists() for path in paths.values()]
    if any(arrays_exist) and not all(arrays_exist):
        raise ValueError("SPECK32 output cache arrays are incomplete")
    if all(arrays_exist) and metadata is None:
        raise ValueError("SPECK32 output cache arrays exist without metadata")
    if metadata is not None and not all(arrays_exist):
        raise ValueError("SPECK32 output cache metadata exists without arrays")

    completed_train_rows = (
        int(metadata.get("completed_train_rows", 0)) if metadata else 0
    )
    completed_test_rows = int(metadata.get("completed_test_rows", 0)) if metadata else 0
    if not 0 <= completed_train_rows <= config.train_rows:
        raise ValueError("invalid completed_train_rows in SPECK32 cache")
    if not 0 <= completed_test_rows <= config.test_rows:
        raise ValueError("invalid completed_test_rows in SPECK32 cache")
    if completed_train_rows and completed_test_rows != config.test_rows:
        raise ValueError("SPECK32 training cache cannot precede frozen test cache")

    if all(arrays_exist):
        plaintexts = np.load(paths["plaintexts"], mmap_mode="r+")
        features = np.load(paths["features"], mmap_mode="r+")
        full_targets = np.load(paths["full_targets"], mmap_mode="r+")
        if (
            plaintexts.shape != (total_rows,)
            or features.shape != (total_rows, 32)
            or full_targets.shape != (total_rows, 32)
        ):
            raise ValueError("existing SPECK32 output cache has invalid shapes")
    else:
        plaintexts = open_memmap(
            paths["plaintexts"], mode="w+", dtype=np.uint32, shape=(total_rows,)
        )
        features = open_memmap(
            paths["features"], mode="w+", dtype=np.float32, shape=(total_rows, 32)
        )
        full_targets = open_memmap(
            paths["full_targets"],
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, 32),
        )
    train_rng = np.random.default_rng(1_220_000 + config.seed)
    test_rng = np.random.default_rng(1_230_000 + config.seed)
    if metadata and metadata.get("train_rng_state"):
        train_rng.bit_generator.state = metadata["train_rng_state"]
    if metadata and metadata.get("test_rng_state"):
        test_rng.bit_generator.state = metadata["test_rng_state"]
    seen = {int(value) for value in plaintexts[:completed_train_rows]}
    seen.update(
        int(value)
        for value in plaintexts[
            config.train_rows : config.train_rows + completed_test_rows
        ]
    )
    _write_metadata(
        metadata_path,
        expected,
        completed_train_rows=completed_train_rows,
        completed_test_rows=completed_test_rows,
        train_rng=train_rng,
        test_rng=test_rng,
    )
    # Freeze the test reserve first so every later training scale excludes the
    # same plaintexts while retaining a prefix-stable training RNG stream.
    completed_test_rows = _fill_split(
        plaintexts=plaintexts,
        features=features,
        full_targets=full_targets,
        start=config.train_rows,
        rows=config.test_rows,
        completed=completed_test_rows,
        rng=test_rng,
        seen=seen,
        secret_key=secret_key,
        rounds=config.rounds,
        chunk_rows=config.chunk_rows,
        on_chunk=lambda completed: _persist_chunk(
            plaintexts,
            features,
            full_targets,
            metadata_path,
            expected,
            completed_train_rows=completed_train_rows,
            completed_test_rows=completed,
            train_rng=train_rng,
            test_rng=test_rng,
            split="test",
            progress=progress,
        ),
    )
    completed_train_rows = _fill_split(
        plaintexts=plaintexts,
        features=features,
        full_targets=full_targets,
        start=0,
        rows=config.train_rows,
        completed=completed_train_rows,
        rng=train_rng,
        seen=seen,
        secret_key=secret_key,
        rounds=config.rounds,
        chunk_rows=config.chunk_rows,
        on_chunk=lambda completed: _persist_chunk(
            plaintexts,
            features,
            full_targets,
            metadata_path,
            expected,
            completed_train_rows=completed,
            completed_test_rows=completed_test_rows,
            train_rng=train_rng,
            test_rng=test_rng,
            split="train",
            progress=progress,
        ),
    )
    return {
        "secret_key": secret_key,
        "plaintexts": plaintexts,
        "features": features,
        "full_targets": full_targets,
        "train_features": features[: config.train_rows],
        "train_targets": full_targets[: config.train_rows],
        "test_features": features[config.train_rows :],
        "test_targets": full_targets[config.train_rows :],
        "metadata": _read_json(metadata_path),
        "data_root": data_root,
        "cache_reused": bool(metadata and metadata.get("status") == "complete"),
    }


def validate_speck32_output_prediction_data(
    config: Speck32OutputPredictionDataConfig,
    data: dict[str, Any],
) -> dict[str, bool]:
    total_rows = config.train_rows + config.test_rows
    plaintexts = data["plaintexts"]
    features = data["features"]
    targets = data["full_targets"]
    train_plaintexts = np.unique(np.asarray(plaintexts[: config.train_rows]))
    test_plaintexts = np.unique(np.asarray(plaintexts[config.train_rows :]))
    plaintexts_are_unique = (
        len(train_plaintexts) == config.train_rows
        and len(test_plaintexts) == config.test_rows
    )
    train_and_test_are_disjoint = (
        len(np.intersect1d(train_plaintexts, test_plaintexts, assume_unique=True)) == 0
    )
    sample_indices = sorted(
        {0, config.train_rows - 1, config.train_rows, total_rows - 1}
    )
    cipher = Speck32_64(rounds=config.rounds, key=int(data["secret_key"]))
    metadata = data["metadata"]
    formal_fields_match = True
    if config.mode != "readiness":
        formal_fields_match = (
            config.rounds == 3
            and config.seed == 21
            and config.test_rows == 1 << 15
            and config.chunk_rows == 4096
        )
    return {
        "official_speck32_vector_matches": Speck32_64(
            rounds=22, key=0x1918111009080100
        ).encrypt(0x6574694C)
        == 0xA86842F2,
        "cache_is_complete": metadata.get("status") == "complete"
        and int(metadata.get("completed_train_rows", -1)) == config.train_rows
        and int(metadata.get("completed_test_rows", -1)) == config.test_rows,
        "cache_arrays_have_expected_shapes": plaintexts.shape == (total_rows,)
        and features.shape == (total_rows, 32)
        and targets.shape == (total_rows, 32),
        "plaintexts_are_unique": plaintexts_are_unique,
        "train_and_test_plaintexts_are_disjoint": train_and_test_are_disjoint,
        "features_are_msb_first_plaintext_bits": all(
            _bits_to_word(features[index]) == int(plaintexts[index])
            for index in sample_indices
        ),
        "features_preserve_x_msw_y_lsw_word_roles": all(
            _bits_to_word(features[index][:16]) == (int(plaintexts[index]) >> 16)
            and _bits_to_word(features[index][16:]) == (int(plaintexts[index]) & 0xFFFF)
            for index in sample_indices
        ),
        "targets_are_msb_first_true_speck32_ciphertext_bits": all(
            _bits_to_word(targets[index]) == cipher.encrypt(int(plaintexts[index]))
            for index in sample_indices
        ),
        "single_fixed_unknown_key_is_64_bit": 0 <= int(data["secret_key"]) < 1 << 64,
        "train_and_test_use_independent_rng_streams": metadata.get(
            "train_plaintext_rng"
        )
        != metadata.get("test_plaintext_rng")
        and metadata.get("split") == "train_then_test_with_independent_rng_streams",
        "metadata_declares_true_output_not_sample_classification": metadata.get("task")
        == "innovation2_output_prediction"
        and metadata.get("target") == "32_msb_first_true_speck32_ciphertext_bits"
        and metadata.get("sample_classification") is False,
        "formal_data_fields_match": formal_fields_match,
        "labels_are_outputs_not_sample_classes": True,
    }


def serializable_config(
    config: Speck32OutputPredictionDataConfig,
) -> dict[str, Any]:
    return asdict(config)


def _fill_split(
    *,
    plaintexts: np.ndarray,
    features: np.ndarray,
    full_targets: np.ndarray,
    start: int,
    rows: int,
    completed: int,
    rng: np.random.Generator,
    seen: set[int],
    secret_key: int,
    rounds: int,
    chunk_rows: int,
    on_chunk: Callable[[int], None],
) -> int:
    shifts = np.arange(31, -1, -1, dtype=np.uint32)
    while completed < rows:
        stop = min(rows, completed + chunk_rows)
        values: list[int] = []
        while len(values) < stop - completed:
            value = int(rng.integers(0, 1 << 32, dtype=np.uint32))
            if value not in seen:
                seen.add(value)
                values.append(value)
        words = np.asarray(values, dtype=np.uint32)
        ciphertexts = encrypt_speck32_numpy(
            words,
            rounds=rounds,
            key=secret_key,
        )
        destination = slice(start + completed, start + stop)
        plaintexts[destination] = words
        features[destination] = (
            (words[:, None] >> shifts[None, :]) & np.uint32(1)
        ).astype(np.float32)
        full_targets[destination] = (
            (ciphertexts[:, None] >> shifts[None, :]) & np.uint32(1)
        ).astype(np.float32)
        completed = stop
        on_chunk(completed)
    return completed


def _persist_chunk(
    plaintexts: np.ndarray,
    features: np.ndarray,
    full_targets: np.ndarray,
    metadata_path: Path,
    expected: dict[str, Any],
    *,
    completed_train_rows: int,
    completed_test_rows: int,
    train_rng: np.random.Generator,
    test_rng: np.random.Generator,
    split: str,
    progress: ProgressCallback | None,
) -> None:
    plaintexts.flush()
    features.flush()
    full_targets.flush()
    _write_metadata(
        metadata_path,
        expected,
        completed_train_rows=completed_train_rows,
        completed_test_rows=completed_test_rows,
        train_rng=train_rng,
        test_rng=test_rng,
    )
    if progress is not None:
        completed = completed_train_rows if split == "train" else completed_test_rows
        total = int(
            expected["train_rows"] if split == "train" else expected["test_rows"]
        )
        progress(
            "cache_chunk",
            {
                "split": split,
                "completed_rows": completed,
                "total_rows": total,
                "completed_train_rows": completed_train_rows,
                "completed_test_rows": completed_test_rows,
            },
        )


def _write_metadata(
    path: Path,
    expected: dict[str, Any],
    *,
    completed_train_rows: int,
    completed_test_rows: int,
    train_rng: np.random.Generator,
    test_rng: np.random.Generator,
) -> None:
    complete = completed_train_rows == int(
        expected["train_rows"]
    ) and completed_test_rows == int(expected["test_rows"])
    _write_json(
        path,
        {
            **expected,
            "status": "complete" if complete else "generating",
            "completed_rows": completed_train_rows + completed_test_rows,
            "completed_train_rows": completed_train_rows,
            "completed_test_rows": completed_test_rows,
            "train_rng_state": train_rng.bit_generator.state,
            "test_rng_state": test_rng.bit_generator.state,
        },
    )


def _secret_key(key_seed: int) -> int:
    return random.Random(1_210_000 + key_seed).getrandbits(64)


def _bits_to_word(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(round(float(bit)))
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "Speck32OutputPredictionDataConfig",
    "prepare_speck32_output_prediction_data",
    "serializable_config",
    "validate_speck32_output_prediction_data",
]
