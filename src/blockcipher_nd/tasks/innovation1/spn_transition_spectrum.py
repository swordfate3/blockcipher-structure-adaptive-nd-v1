from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.features.spn_transition_spectrum import present_bit_transition_spectrum_features
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.difference_profiles import difference_for_profile
from blockcipher_nd.tasks.innovation1.protocols import OFFICIAL_ZHANG_WANG_CASE2_MCND
from blockcipher_nd.tasks.innovation1.spn_candidate.baseline import (
    binary_accuracy,
    binary_auc,
    calibrated_binary_accuracy,
    train_model,
)
from blockcipher_nd.tasks.innovation1.spn_candidate.dataset import _cipher_for_row, _negative_pairs, _positive_pairs


DEFAULT_DIFFERENCE_PROFILE = "present_zhang_wang2022_mcnd"
DEFAULT_VALIDATION_KEY = 0x11111111111111111111
MODEL_ROUTE_KEYS = {
    "linear": "bit_transition_spectrum_linear",
    "mlp": "bit_transition_spectrum_mlp",
    "shuffled_p": "bit_transition_spectrum_shuffled_p",
}
MODEL_TRAINING_KEYS = {
    "linear": "linear",
    "mlp": "mlp",
    "shuffled_p": "mlp",
}
DEFAULTS = {
    "output": None,
    "rounds": 7,
    "seed": 0,
    "samples_per_class": 4096,
    "pairs_per_sample": 16,
    "negative_mode": "encrypted_random_plaintexts",
    "sample_structure": OFFICIAL_ZHANG_WANG_CASE2_MCND,
    "difference_profile": DEFAULT_DIFFERENCE_PROFILE,
    "difference_member": 0,
    "train_key": 0,
    "validation_key": DEFAULT_VALIDATION_KEY,
    "key_rotation_interval": 0,
    "feature_cache_root": None,
    "feature_cache_chunk_size": 4096,
    "feature_cache_workers": 1,
    "progress_output": None,
    "epochs": 30,
    "learning_rate": 1e-2,
    "model": "linear",
    "device": "cpu",
}
PATH_FIELDS = {"output", "feature_cache_root", "progress_output"}
INT_FIELDS = {
    "rounds",
    "seed",
    "samples_per_class",
    "pairs_per_sample",
    "difference_member",
    "train_key",
    "validation_key",
    "key_rotation_interval",
    "feature_cache_chunk_size",
    "feature_cache_workers",
    "epochs",
}
FLOAT_FIELDS = {"learning_rate"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--pairs-per-sample", type=int, default=None)
    parser.add_argument("--negative-mode", default=None)
    parser.add_argument("--sample-structure", default=None)
    parser.add_argument("--difference-profile", default=None)
    parser.add_argument("--difference-member", type=int, default=None)
    parser.add_argument("--train-key", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--validation-key", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--key-rotation-interval", type=int, default=None)
    parser.add_argument("--feature-cache-root", type=Path, default=None)
    parser.add_argument("--feature-cache-chunk-size", type=int, default=None)
    parser.add_argument("--feature-cache-workers", type=int, default=None)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--model", choices=list(MODEL_ROUTE_KEYS), default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    return _apply_config_defaults(args)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_transition_spectrum(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")


def run_transition_spectrum(args: argparse.Namespace) -> dict[str, object]:
    input_difference = difference_for_profile(args.difference_profile, args.difference_member)
    train_features, train_labels = make_transition_spectrum_dataset(
        rounds=args.rounds,
        key=args.train_key,
        input_difference=input_difference,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        pairs_per_sample=args.pairs_per_sample,
        negative_mode=args.negative_mode,
        sample_structure=args.sample_structure,
        key_rotation_interval=args.key_rotation_interval,
        shuffled=args.model == "shuffled_p",
        feature_cache_root=args.feature_cache_root,
        feature_cache_chunk_size=args.feature_cache_chunk_size,
        feature_cache_workers=args.feature_cache_workers,
        progress_output=args.progress_output,
        split="train",
    )
    validation_samples_per_class = max(1024, args.samples_per_class // 4)
    validation_features, validation_labels = make_transition_spectrum_dataset(
        rounds=args.rounds,
        key=args.validation_key,
        input_difference=input_difference,
        seed=args.seed + 10_000,
        samples_per_class=validation_samples_per_class,
        pairs_per_sample=args.pairs_per_sample,
        negative_mode=args.negative_mode,
        sample_structure=args.sample_structure,
        key_rotation_interval=args.key_rotation_interval,
        shuffled=args.model == "shuffled_p",
        feature_cache_root=args.feature_cache_root,
        feature_cache_chunk_size=args.feature_cache_chunk_size,
        feature_cache_workers=args.feature_cache_workers,
        progress_output=args.progress_output,
        split="validation",
    )
    device = torch.device(args.device)
    model = train_model(
        train_features,
        train_labels,
        model_name=MODEL_TRAINING_KEYS[args.model],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    with torch.no_grad():
        logits = (
            model(torch.from_numpy(validation_features.astype(np.float32)).to(device))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    val_accuracy = binary_accuracy(validation_labels, probabilities)
    val_auc = binary_auc(validation_labels, probabilities)
    calibrated_accuracy, calibrated_threshold = calibrated_binary_accuracy(validation_labels, probabilities)
    route = MODEL_ROUTE_KEYS[args.model]
    metrics = {
        "accuracy": val_accuracy,
        "auc": val_auc,
        "best_accuracy": calibrated_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_advantage": 2.0 * calibrated_accuracy - 1.0,
        "calibrated_threshold": calibrated_threshold,
    }
    return {
        "route": route,
        "model": route,
        "selected_model": route,
        "training_model": args.model,
        "training_model_family": MODEL_TRAINING_KEYS[args.model],
        "rounds": args.rounds,
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "validation_samples_per_class": validation_samples_per_class,
        "pairs_per_sample": args.pairs_per_sample,
        "negative_mode": args.negative_mode,
        "sample_structure": args.sample_structure,
        "difference_profile": args.difference_profile,
        "difference_member": args.difference_member,
        "input_difference": input_difference,
        "key_rotation_interval": args.key_rotation_interval,
        "device": args.device,
        "feature_route": "bit_transition_spectrum",
        "feature_cache_enabled": args.feature_cache_root is not None,
        "feature_cache_root": str(args.feature_cache_root) if args.feature_cache_root is not None else None,
        "feature_cache_chunk_size": args.feature_cache_chunk_size,
        "feature_cache_workers": args.feature_cache_workers,
        "progress_output": str(args.progress_output) if args.progress_output is not None else None,
        "shuffled_p": args.model == "shuffled_p",
        "feature_dim": int(train_features.shape[1]),
        "metrics": metrics,
        "accuracy": val_accuracy,
        "auc": val_auc,
        "best_accuracy": calibrated_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_advantage": 2.0 * calibrated_accuracy - 1.0,
        "calibrated_threshold": calibrated_threshold,
        "val_accuracy": val_accuracy,
        "val_auc": val_auc,
        "val_best_accuracy": calibrated_accuracy,
        "val_calibrated_accuracy": calibrated_accuracy,
    }


def args_from_config(config: dict[str, object]) -> argparse.Namespace:
    args = argparse.Namespace(config=None)
    for field in DEFAULTS:
        setattr(args, field, config.get(field))
    return _apply_config_defaults(args, require_output=False)


def make_transition_spectrum_dataset(
    *,
    rounds: int,
    key: int,
    input_difference: int,
    seed: int,
    samples_per_class: int,
    pairs_per_sample: int,
    negative_mode: str,
    sample_structure: str,
    key_rotation_interval: int,
    shuffled: bool = False,
    feature_cache_root: Path | None = None,
    feature_cache_chunk_size: int = 4096,
    feature_cache_workers: int = 1,
    progress_output: Path | None = None,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    cipher = build_cipher("present80", rounds, key=key)
    config = DifferentialDatasetConfig(
        cipher=cipher,
        input_difference=input_difference,
        samples_per_class=samples_per_class,
        seed=seed,
        pairs_per_sample=pairs_per_sample,
        negative_mode=negative_mode,
        key_rotation_interval=key_rotation_interval,
        sample_structure=sample_structure,
    )
    metadata = _cache_metadata(
        split=split,
        rounds=rounds,
        key=key,
        input_difference=input_difference,
        seed=seed,
        samples_per_class=samples_per_class,
        pairs_per_sample=pairs_per_sample,
        negative_mode=negative_mode,
        sample_structure=sample_structure,
        key_rotation_interval=key_rotation_interval,
        shuffled=shuffled,
        feature_cache_workers=feature_cache_workers,
        width=cipher.block_bits,
    )
    if feature_cache_root is None:
        _write_progress(progress_output, "transition_spectrum_cache_disabled", metadata)
        return _generate_dataset(config=config, metadata=metadata, chunk_size=samples_per_class, workers=1)
    return _cached_dataset(
        config=config,
        metadata=metadata,
        cache_root=feature_cache_root,
        chunk_size=feature_cache_chunk_size,
        workers=feature_cache_workers,
        progress_output=progress_output,
    )


def _cached_dataset(
    *,
    config: DifferentialDatasetConfig,
    metadata: dict[str, Any],
    cache_root: Path,
    chunk_size: int,
    workers: int,
    progress_output: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if chunk_size < 1:
        raise ValueError("feature_cache_chunk_size must be at least 1")
    if workers < 1:
        raise ValueError("feature_cache_workers must be at least 1")
    cache_dir = _cache_dir(cache_root, metadata)
    features_path = cache_dir / "features.npy"
    labels_path = cache_dir / "labels.npy"
    metadata_path = cache_dir / "metadata.json"
    if features_path.exists() and labels_path.exists() and metadata_path.exists():
        observed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed_metadata == metadata:
            _write_progress(progress_output, "transition_spectrum_cache_reuse", {**metadata, "cache_dir": str(cache_dir)})
            return np.load(features_path, mmap_mode="r"), np.load(labels_path, mmap_mode="r")

    cache_dir.mkdir(parents=True, exist_ok=True)
    total_rows = int(metadata["total_rows"])
    feature_dim = int(metadata["feature_dim"])
    _write_progress(
        progress_output,
        "transition_spectrum_cache_start",
        {**metadata, "cache_dir": str(cache_dir), "chunk_size": chunk_size, "workers": workers},
    )
    features = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float32, shape=(total_rows, feature_dim))
    labels = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.uint8, shape=(total_rows,))
    _fill_dataset(
        config=config,
        features=features,
        labels=labels,
        metadata=metadata,
        chunk_size=chunk_size,
        workers=workers,
        progress_output=progress_output,
        cache_dir=cache_dir,
    )
    _write_progress(progress_output, "transition_spectrum_cache_flush_start", {**metadata, "cache_dir": str(cache_dir)})
    features.flush()
    labels.flush()
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _write_progress(
        progress_output,
        "transition_spectrum_cache_done",
        {
            **metadata,
            "cache_dir": str(cache_dir),
            "features_path": str(features_path),
            "labels_path": str(labels_path),
            "metadata_path": str(metadata_path),
        },
    )
    return np.load(features_path, mmap_mode="r"), np.load(labels_path, mmap_mode="r")


def _generate_dataset(
    *,
    config: DifferentialDatasetConfig,
    metadata: dict[str, Any],
    chunk_size: int,
    workers: int,
    progress_output: Path | None = None,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    total_rows = int(metadata["total_rows"])
    feature_dim = int(metadata["feature_dim"])
    features = np.empty((total_rows, feature_dim), dtype=np.float32)
    labels = np.empty((total_rows,), dtype=np.uint8)
    _fill_dataset(
        config=config,
        features=features,
        labels=labels,
        metadata=metadata,
        chunk_size=chunk_size,
        workers=workers,
        progress_output=progress_output,
        cache_dir=cache_dir,
    )
    return features, labels


def _fill_dataset(
    *,
    config: DifferentialDatasetConfig,
    features: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
    chunk_size: int,
    workers: int,
    progress_output: Path | None = None,
    cache_dir: Path | None = None,
) -> None:
    row_index = 0
    for start, count, chunk in _iter_chunks(config=config, metadata=metadata, chunk_size=chunk_size, workers=workers, label=1):
        features[row_index : row_index + count] = chunk
        labels[row_index : row_index + count] = 1
        row_index += count
        _write_progress(
            progress_output,
            "transition_spectrum_positive_chunk",
            {
                **metadata,
                "cache_dir": str(cache_dir) if cache_dir is not None else None,
                "rows_done": row_index,
                "class_rows_done": start + count,
                "class_total": config.samples_per_class,
                "chunk_rows": count,
                "workers": workers,
            },
        )
    for start, count, chunk in _iter_chunks(config=config, metadata=metadata, chunk_size=chunk_size, workers=workers, label=0):
        features[row_index : row_index + count] = chunk
        labels[row_index : row_index + count] = 0
        row_index += count
        _write_progress(
            progress_output,
            "transition_spectrum_negative_chunk",
            {
                **metadata,
                "cache_dir": str(cache_dir) if cache_dir is not None else None,
                "rows_done": row_index,
                "class_rows_done": start + count,
                "class_total": config.samples_per_class,
                "chunk_rows": count,
                "workers": workers,
            },
        )
    order = np.random.default_rng(config.seed + 2_147_483_647).permutation(labels.size)
    features[:] = features[order]
    labels[:] = labels[order]


def _iter_chunks(
    *,
    config: DifferentialDatasetConfig,
    metadata: dict[str, Any],
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
            yield start, count, _generate_chunk(config, metadata, start, count, label)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_generate_chunk, config, metadata, start, count, label) for start, count in specs]
        for (start, count), future in zip(specs, futures):
            yield start, count, future.result()


def _generate_chunk(
    config: DifferentialDatasetConfig,
    metadata: dict[str, Any],
    start: int,
    count: int,
    label: int,
) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 1_000_003 * (start + 1) + 97_531 * label)
    mask = (1 << config.cipher.block_bits) - 1
    chunk = np.empty((count, int(metadata["feature_dim"])), dtype=np.float32)
    for offset in range(count):
        source_row = start + offset
        row_cipher = _cipher_for_row(config, rng, source_row)
        if label == 1:
            pairs = _positive_pairs(config, rng, row_cipher, mask)
        else:
            pairs = _negative_pairs(config, rng, row_cipher, mask)
        compact_pairs = [(int(pair[0]), int(pair[1])) for pair in pairs]
        chunk[offset] = present_bit_transition_spectrum_features(
            compact_pairs,
            width=config.cipher.block_bits,
            cipher=row_cipher,
            shuffled=bool(metadata["shuffled_p"]),
        )
    return chunk


def _cache_metadata(
    *,
    split: str,
    rounds: int,
    key: int,
    input_difference: int,
    seed: int,
    samples_per_class: int,
    pairs_per_sample: int,
    negative_mode: str,
    sample_structure: str,
    key_rotation_interval: int,
    shuffled: bool,
    feature_cache_workers: int,
    width: int,
) -> dict[str, Any]:
    return {
        "cache_type": "spn_transition_spectrum",
        "cipher": "present80",
        "split": split,
        "rounds": rounds,
        "key": key,
        "input_difference": input_difference,
        "seed": seed,
        "samples_per_class": samples_per_class,
        "total_rows": samples_per_class * 2,
        "pairs_per_sample": pairs_per_sample,
        "negative_mode": negative_mode,
        "sample_structure": sample_structure,
        "key_rotation_interval": key_rotation_interval,
        "feature_route": "bit_transition_spectrum",
        "shuffled_p": shuffled,
        "feature_cache_workers": feature_cache_workers,
        "width": width,
        "feature_dim": _feature_dim(width),
        "feature_dtype": "float32",
        "label_dtype": "uint8",
        "shuffle": True,
        "cache_version": 1,
    }


def _feature_dim(width: int) -> int:
    cells = width // 4
    pair_dim = 18 + 4 * cells + cells * cells
    return pair_dim * 3 + 6


def _cache_dir(cache_root: Path, metadata: dict[str, Any]) -> Path:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return cache_root / str(metadata["split"]) / digest


def _write_progress(path: Path | None, event: str, payload: dict[str, Any] | None = None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "time": time.time(), **(payload or {})}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _apply_config_defaults(args: argparse.Namespace, *, require_output: bool = True) -> argparse.Namespace:
    config = _load_config(args.config)
    for field, default in DEFAULTS.items():
        value = getattr(args, field)
        if value is None:
            value = config.get(field, default)
        if field in PATH_FIELDS and value is not None:
            value = Path(value)
        elif field in INT_FIELDS and value is not None:
            value = int(value, 0) if isinstance(value, str) else int(value)
        elif field in FLOAT_FIELDS and value is not None:
            value = float(value)
        setattr(args, field, value)
    if require_output and args.output is None:
        raise SystemExit("--output is required unless provided by --config")
    if args.model not in MODEL_ROUTE_KEYS:
        raise SystemExit(f"unsupported model in config: {args.model}")
    return args


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"config must be a JSON object: {path}")
    return data
