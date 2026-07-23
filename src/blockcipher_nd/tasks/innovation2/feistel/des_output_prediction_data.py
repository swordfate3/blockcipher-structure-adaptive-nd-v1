from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.lib.format import open_memmap

from blockcipher_nd.ciphers.feistel.des import Des
from blockcipher_nd.tasks.innovation2.feistel.des_numpy import encrypt_des_numpy
from blockcipher_nd.tasks.innovation2.feistel.des_output_prediction_protocol import (
    DesOutputPredictionDataConfig,
    secret_key_for_seed,
    serializable_config,
)


CACHE_VERSION = 1
ProgressCallback = Callable[[str, dict[str, Any]], None]


def prepare_des_output_prediction_data(
    config: DesOutputPredictionDataConfig,
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    metadata_path = data_root / "cache_metadata.json"
    total_rows = config.train_rows + config.test_rows
    secret_key = secret_key_for_seed(config.key_seed)
    expected = {
        "cache_version": CACHE_VERSION,
        "cipher": "DES",
        "rounds": config.rounds,
        "seed": config.seed,
        "key_seed": config.key_seed,
        "train_rows": config.train_rows,
        "test_rows": config.test_rows,
        "total_rows": total_rows,
        "task": "innovation2_output_prediction",
        "input": "64_msb_first_plaintext_bits",
        "target": "64_msb_first_true_round_reduced_des_ciphertext_bits",
        "sample_classification": False,
        "secret_key_scope": "single_fixed_unknown_key",
        "bit_order": "msb_first",
        "serialized_half_order": ["plaintext_msw32", "plaintext_lsw32"],
        "secret_key_hex": f"{secret_key:016x}",
        "des_key_serialization": "64_bits_including_8_parity_positions",
        "round_reduced_semantics": "DES_IP_then_r_Feistel_rounds_then_swap_then_DES_FP",
        "public_input_permutation": "DES_IP",
        "public_output_permutation": "swap_then_DES_FP",
        "split": "train_then_test_with_independent_rng_streams",
        "train_plaintext_rng": f"numpy_default_rng({1_320_000 + config.seed})",
        "test_plaintext_rng": f"numpy_default_rng({1_330_000 + config.seed})",
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
        raise ValueError("existing DES output cache parameters do not match")
    arrays_exist = [path.exists() for path in paths.values()]
    if any(arrays_exist) and not all(arrays_exist):
        raise ValueError("DES output cache arrays are incomplete")
    if all(arrays_exist) and metadata is None:
        raise ValueError("DES output cache arrays exist without metadata")
    if metadata is not None and not all(arrays_exist):
        raise ValueError("DES output cache metadata exists without arrays")

    completed_train_rows = (
        int(metadata.get("completed_train_rows", 0)) if metadata else 0
    )
    completed_test_rows = int(metadata.get("completed_test_rows", 0)) if metadata else 0
    if not 0 <= completed_train_rows <= config.train_rows:
        raise ValueError("invalid completed_train_rows in DES cache")
    if not 0 <= completed_test_rows <= config.test_rows:
        raise ValueError("invalid completed_test_rows in DES cache")
    if completed_train_rows and completed_test_rows != config.test_rows:
        raise ValueError("DES training cache cannot precede frozen test cache")

    if all(arrays_exist):
        plaintexts = np.load(paths["plaintexts"], mmap_mode="r+")
        features = np.load(paths["features"], mmap_mode="r+")
        full_targets = np.load(paths["full_targets"], mmap_mode="r+")
        if (
            plaintexts.shape != (total_rows,)
            or features.shape != (total_rows, 64)
            or full_targets.shape != (total_rows, 64)
        ):
            raise ValueError("existing DES output cache has invalid shapes")
    else:
        plaintexts = open_memmap(
            paths["plaintexts"],
            mode="w+",
            dtype=np.uint64,
            shape=(total_rows,),
        )
        features = open_memmap(
            paths["features"],
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, 64),
        )
        full_targets = open_memmap(
            paths["full_targets"],
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, 64),
        )
    train_rng = np.random.default_rng(1_320_000 + config.seed)
    test_rng = np.random.default_rng(1_330_000 + config.seed)
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


def validate_des_output_prediction_data(
    config: DesOutputPredictionDataConfig,
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
    cipher = Des(rounds=config.rounds, key=int(data["secret_key"]))
    metadata = data["metadata"]
    replay_matches = all(
        _bits_to_word(targets[index]) == cipher.encrypt(int(plaintexts[index]))
        for index in sample_indices
    )
    vector_replay = encrypt_des_numpy(
        np.asarray([plaintexts[index] for index in sample_indices], dtype=np.uint64),
        rounds=config.rounds,
        key=int(data["secret_key"]),
    )
    formal_fields_match = True
    if config.mode != "readiness":
        formal_fields_match = (
            config.seed == 31
            and config.test_rows == 1 << 15
            and config.chunk_rows == 4096
            and config.mode in {"f1_a", "f1_b", "f1_c", "f1_r_a", "f1_r_b"}
        )
    return {
        "official_des_vector_matches_scalar": Des(
            rounds=16,
            key=0x133457799BBCDFF1,
        ).encrypt(0x0123456789ABCDEF)
        == 0x85E813540F0AB405,
        "official_des_vector_matches_numpy": int(
            encrypt_des_numpy(
                np.asarray([0x0123456789ABCDEF], dtype=np.uint64),
                rounds=16,
                key=0x133457799BBCDFF1,
            )[0]
        )
        == 0x85E813540F0AB405,
        "cache_is_complete": metadata.get("status") == "complete"
        and int(metadata.get("completed_train_rows", -1)) == config.train_rows
        and int(metadata.get("completed_test_rows", -1)) == config.test_rows,
        "cache_arrays_have_expected_shapes": plaintexts.shape == (total_rows,)
        and features.shape == (total_rows, 64)
        and targets.shape == (total_rows, 64),
        "plaintexts_are_unique": plaintexts_are_unique,
        "train_and_test_plaintexts_are_disjoint": train_and_test_are_disjoint,
        "features_are_msb_first_plaintext_bits": all(
            _bits_to_word(features[index]) == int(plaintexts[index])
            for index in sample_indices
        ),
        "features_preserve_serialized_32_bit_halves": all(
            _bits_to_word(features[index][:32]) == (int(plaintexts[index]) >> 32)
            and _bits_to_word(features[index][32:])
            == (int(plaintexts[index]) & 0xFFFFFFFF)
            for index in sample_indices
        ),
        "targets_are_msb_first_true_des_ciphertext_bits": replay_matches,
        "vectorized_targets_match_scalar_replay": all(
            int(vector_replay[offset]) == cipher.encrypt(int(plaintexts[index]))
            for offset, index in enumerate(sample_indices)
        ),
        "single_fixed_unknown_key_is_64_bit": 0 <= int(data["secret_key"]) < 1 << 64,
        "round_reduced_ip_swap_fp_semantics_are_declared": metadata.get(
            "round_reduced_semantics"
        )
        == "DES_IP_then_r_Feistel_rounds_then_swap_then_DES_FP"
        and metadata.get("public_input_permutation") == "DES_IP"
        and metadata.get("public_output_permutation") == "swap_then_DES_FP",
        "train_and_test_use_independent_rng_streams": metadata.get(
            "train_plaintext_rng"
        )
        != metadata.get("test_plaintext_rng")
        and metadata.get("split") == "train_then_test_with_independent_rng_streams",
        "metadata_declares_true_output_not_sample_classification": metadata.get("task")
        == "innovation2_output_prediction"
        and metadata.get("target")
        == "64_msb_first_true_round_reduced_des_ciphertext_bits"
        and metadata.get("sample_classification") is False,
        "formal_data_fields_match": formal_fields_match,
        "labels_are_outputs_not_sample_classes": True,
    }


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
    shifts = np.arange(63, -1, -1, dtype=np.uint64)
    while completed < rows:
        stop = min(rows, completed + chunk_rows)
        words = _draw_unique_uint64(rng, stop - completed, seen)
        ciphertexts = encrypt_des_numpy(words, rounds=rounds, key=secret_key)
        destination = slice(start + completed, start + stop)
        plaintexts[destination] = words
        features[destination] = (
            (words[:, None] >> shifts[None, :]) & np.uint64(1)
        ).astype(np.float32)
        full_targets[destination] = (
            (ciphertexts[:, None] >> shifts[None, :]) & np.uint64(1)
        ).astype(np.float32)
        completed = stop
        on_chunk(completed)
    return completed


def _draw_unique_uint64(
    rng: np.random.Generator,
    rows: int,
    seen: set[int],
) -> np.ndarray:
    values: list[int] = []
    while len(values) < rows:
        draw_rows = max(16, 2 * (rows - len(values)))
        high = rng.integers(0, 1 << 32, size=draw_rows, dtype=np.uint64)
        low = rng.integers(0, 1 << 32, size=draw_rows, dtype=np.uint64)
        for value in (high << np.uint64(32)) | low:
            integer = int(value)
            if integer not in seen:
                seen.add(integer)
                values.append(integer)
                if len(values) == rows:
                    break
    return np.asarray(values, dtype=np.uint64)


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
    "DesOutputPredictionDataConfig",
    "encrypt_des_numpy",
    "prepare_des_output_prediction_data",
    "serializable_config",
    "validate_des_output_prediction_data",
]
