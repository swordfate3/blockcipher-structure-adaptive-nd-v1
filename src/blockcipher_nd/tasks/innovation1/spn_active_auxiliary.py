from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.data.differential.rows import generate_negative_row, generate_positive_row
from blockcipher_nd.features.spn_active_auxiliary import (
    present_invp_active_mask_targets,
    shuffled_active_mask_targets,
)
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.difference_profiles import difference_for_profile
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.protocols import OFFICIAL_ZHANG_WANG_CASE2_MCND
from blockcipher_nd.tasks.innovation1.spn_candidate.baseline import (
    binary_accuracy,
    binary_auc,
    calibrated_binary_accuracy,
)
from blockcipher_nd.training.trainer import should_report_step


DEFAULT_DIFFERENCE_PROFILE = "present_zhang_wang2022_mcnd"
DEFAULT_VALIDATION_KEY = 0x11111111111111111111
ACTIVE_AUX_MODEL = "present_nibble_invp_active_aux_spn_only"
SHUFFLED_ACTIVE_AUX_MODEL = "present_nibble_invp_active_aux_shuffled_targets"
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
    "epochs": 5,
    "learning_rate": 1e-3,
    "lambda_aux": 0.1,
    "batch_size": 512,
    "hidden_bits": 32,
    "spn_mixer_depth": 2,
    "dataset_cache_root": None,
    "dataset_cache_chunk_size": 4096,
    "dataset_cache_workers": 1,
    "progress_output": None,
    "device": "cpu",
    "model": ACTIVE_AUX_MODEL,
}
PATH_FIELDS = {"output", "dataset_cache_root", "progress_output"}
INT_FIELDS = {
    "rounds",
    "seed",
    "samples_per_class",
    "pairs_per_sample",
    "difference_member",
    "train_key",
    "validation_key",
    "key_rotation_interval",
    "epochs",
    "batch_size",
    "hidden_bits",
    "spn_mixer_depth",
    "dataset_cache_chunk_size",
    "dataset_cache_workers",
}
FLOAT_FIELDS = {"learning_rate", "lambda_aux"}


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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lambda-aux", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-bits", type=int, default=None)
    parser.add_argument("--spn-mixer-depth", type=int, default=None)
    parser.add_argument("--dataset-cache-root", type=Path, default=None)
    parser.add_argument("--dataset-cache-chunk-size", type=int, default=None)
    parser.add_argument("--dataset-cache-workers", type=int, default=None)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", choices=[ACTIVE_AUX_MODEL, SHUFFLED_ACTIVE_AUX_MODEL], default=None)
    args = parser.parse_args(argv)
    return _apply_config_defaults(args)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_active_auxiliary(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")


def run_active_auxiliary(args: argparse.Namespace) -> dict[str, object]:
    input_difference = difference_for_profile(args.difference_profile, args.difference_member)
    train_features, train_labels = make_active_auxiliary_dataset(
        rounds=args.rounds,
        key=args.train_key,
        input_difference=input_difference,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        pairs_per_sample=args.pairs_per_sample,
        negative_mode=args.negative_mode,
        sample_structure=args.sample_structure,
        key_rotation_interval=args.key_rotation_interval,
        dataset_cache_root=args.dataset_cache_root,
        dataset_cache_chunk_size=args.dataset_cache_chunk_size,
        dataset_cache_workers=args.dataset_cache_workers,
        progress_output=args.progress_output,
        split="train",
    )
    validation_samples_per_class = max(2, args.samples_per_class // 4)
    validation_features, validation_labels = make_active_auxiliary_dataset(
        rounds=args.rounds,
        key=args.validation_key,
        input_difference=input_difference,
        seed=args.seed + 10_000,
        samples_per_class=validation_samples_per_class,
        pairs_per_sample=args.pairs_per_sample,
        negative_mode=args.negative_mode,
        sample_structure=args.sample_structure,
        key_rotation_interval=args.key_rotation_interval,
        dataset_cache_root=args.dataset_cache_root,
        dataset_cache_chunk_size=args.dataset_cache_chunk_size,
        dataset_cache_workers=args.dataset_cache_workers,
        progress_output=args.progress_output,
        split="validation",
    )
    device = torch.device(args.device)
    shuffled_targets = args.model == SHUFFLED_ACTIVE_AUX_MODEL
    model, train_aux_loss = _train_active_aux_model(
        train_features,
        train_labels,
        model_name=ACTIVE_AUX_MODEL,
        hidden_bits=args.hidden_bits,
        spn_mixer_depth=args.spn_mixer_depth,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_aux=args.lambda_aux,
        batch_size=args.batch_size,
        shuffled_targets=shuffled_targets,
        seed=args.seed,
        device=device,
        progress_output=args.progress_output,
    )
    logits, validation_aux_loss = _evaluate_active_aux_model(
        model,
        validation_features,
        validation_labels,
        lambda_aux=args.lambda_aux,
        batch_size=args.batch_size,
        shuffled_targets=shuffled_targets,
        seed=args.seed + 10_000,
        device=device,
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    val_accuracy = binary_accuracy(validation_labels, probabilities)
    val_auc = binary_auc(validation_labels, probabilities)
    calibrated_accuracy, calibrated_threshold = calibrated_binary_accuracy(validation_labels, probabilities)
    aux_target = "shuffled_present_invp_active_mask" if shuffled_targets else "present_invp_active_mask"
    metrics = {
        "accuracy": val_accuracy,
        "auc": val_auc,
        "best_accuracy": calibrated_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_advantage": 2.0 * calibrated_accuracy - 1.0,
        "calibrated_threshold": calibrated_threshold,
        "auxiliary_loss": validation_aux_loss,
        "train_auxiliary_loss": train_aux_loss,
    }
    return {
        "route": args.model,
        "model": args.model,
        "selected_model": args.model,
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
        "feature_route": "active_pattern_auxiliary_head",
        "dataset_cache_enabled": args.dataset_cache_root is not None,
        "dataset_cache_root": str(args.dataset_cache_root) if args.dataset_cache_root is not None else None,
        "dataset_cache_chunk_size": args.dataset_cache_chunk_size,
        "dataset_cache_workers": args.dataset_cache_workers,
        "progress_output": str(args.progress_output) if args.progress_output is not None else None,
        "auxiliary_target": aux_target,
        "lambda_aux": args.lambda_aux,
        "shuffled_auxiliary_targets": shuffled_targets,
        "input_bits": int(train_features.shape[1]),
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


def make_active_auxiliary_dataset(
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
    dataset_cache_root: Path | None = None,
    dataset_cache_chunk_size: int = 4096,
    dataset_cache_workers: int = 1,
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
        sample_structure=sample_structure,
        key_rotation_interval=key_rotation_interval,
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
        width=cipher.block_bits,
    )
    if dataset_cache_root is None:
        _write_progress(progress_output, "active_auxiliary_cache_disabled", metadata)
        dataset = make_differential_dataset(config)
        return dataset.features.astype(np.float32), dataset.labels.astype(np.uint8)
    return _cached_active_auxiliary_dataset(
        config=config,
        metadata=metadata,
        cache_root=dataset_cache_root,
        chunk_size=dataset_cache_chunk_size,
        workers=dataset_cache_workers,
        progress_output=progress_output,
    )


def _cached_active_auxiliary_dataset(
    *,
    config: DifferentialDatasetConfig,
    metadata: dict[str, Any],
    cache_root: Path,
    chunk_size: int,
    workers: int,
    progress_output: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if chunk_size < 1:
        raise ValueError("dataset_cache_chunk_size must be at least 1")
    if workers < 1:
        raise ValueError("dataset_cache_workers must be at least 1")
    cache_dir = _cache_dir(cache_root, metadata)
    features_path = cache_dir / "features.npy"
    labels_path = cache_dir / "labels.npy"
    metadata_path = cache_dir / "metadata.json"
    if features_path.exists() and labels_path.exists() and metadata_path.exists():
        observed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed_metadata == metadata:
            _write_progress(
                progress_output,
                "active_auxiliary_cache_reuse",
                {**metadata, "cache_dir": str(cache_dir), "features_path": str(features_path), "labels_path": str(labels_path)},
            )
            return np.load(features_path, mmap_mode="r"), np.load(labels_path, mmap_mode="r")

    cache_dir.mkdir(parents=True, exist_ok=True)
    total_rows = int(metadata["total_rows"])
    feature_dim = int(metadata["feature_dim"])
    _write_progress(
        progress_output,
        "active_auxiliary_cache_start",
        {**metadata, "cache_dir": str(cache_dir), "chunk_size": chunk_size, "workers": workers},
    )
    features = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float32, shape=(total_rows, feature_dim))
    labels = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.uint8, shape=(total_rows,))
    _fill_active_auxiliary_dataset(
        config=config,
        features=features,
        labels=labels,
        metadata=metadata,
        chunk_size=chunk_size,
        progress_output=progress_output,
        cache_dir=cache_dir,
    )
    _write_progress(progress_output, "active_auxiliary_cache_flush_start", {**metadata, "cache_dir": str(cache_dir)})
    features.flush()
    labels.flush()
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _write_progress(
        progress_output,
        "active_auxiliary_cache_done",
        {
            **metadata,
            "cache_dir": str(cache_dir),
            "features_path": str(features_path),
            "labels_path": str(labels_path),
            "metadata_path": str(metadata_path),
        },
    )
    return np.load(features_path, mmap_mode="r"), np.load(labels_path, mmap_mode="r")


def _fill_active_auxiliary_dataset(
    *,
    config: DifferentialDatasetConfig,
    features: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
    chunk_size: int,
    progress_output: Path | None,
    cache_dir: Path | None,
) -> None:
    block_bits = config.cipher.block_bits
    mask = (1 << block_bits) - 1
    row_index = 0
    for start in range(0, config.samples_per_class, chunk_size):
        count = min(chunk_size, config.samples_per_class - start)
        chunk = _generate_active_auxiliary_chunk(config, start, count, label=1, block_bits=block_bits, mask=mask)
        features[row_index : row_index + count] = chunk
        labels[row_index : row_index + count] = 1
        row_index += count
        _write_progress(
            progress_output,
            "active_auxiliary_positive_chunk",
            {
                **metadata,
                "cache_dir": str(cache_dir) if cache_dir is not None else None,
                "rows_done": row_index,
                "class_rows_done": start + count,
                "class_total": config.samples_per_class,
                "chunk_rows": count,
            },
        )
    for start in range(0, config.samples_per_class, chunk_size):
        count = min(chunk_size, config.samples_per_class - start)
        chunk = _generate_active_auxiliary_chunk(config, start, count, label=0, block_bits=block_bits, mask=mask)
        features[row_index : row_index + count] = chunk
        labels[row_index : row_index + count] = 0
        row_index += count
        _write_progress(
            progress_output,
            "active_auxiliary_negative_chunk",
            {
                **metadata,
                "cache_dir": str(cache_dir) if cache_dir is not None else None,
                "rows_done": row_index,
                "class_rows_done": start + count,
                "class_total": config.samples_per_class,
                "chunk_rows": count,
            },
        )
    order = np.random.default_rng(config.seed + 2_147_483_647).permutation(labels.size)
    features[:] = features[order]
    labels[:] = labels[order]


def _generate_active_auxiliary_chunk(
    config: DifferentialDatasetConfig,
    start: int,
    count: int,
    *,
    label: int,
    block_bits: int,
    mask: int,
) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 1_000_003 * (start + 1) + 97_531 * label)
    chunk = np.empty((count, config.pairs_per_sample * 128), dtype=np.float32)
    for offset in range(count):
        source_row = start + offset
        if label == 1:
            row = generate_positive_row(config, rng, block_bits, mask, row_index=source_row)
        else:
            row = generate_negative_row(config, rng, block_bits, row_index=source_row)
        chunk[offset] = np.asarray(row, dtype=np.float32)
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
    width: int,
) -> dict[str, Any]:
    return {
        "cache_type": "spn_active_auxiliary_raw",
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
        "feature_route": "active_pattern_auxiliary_head",
        "width": width,
        "feature_dim": pairs_per_sample * 128,
        "feature_dtype": "float32",
        "label_dtype": "uint8",
        "shuffle": True,
        "cache_version": 1,
    }


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


def _train_active_aux_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    model_name: str,
    hidden_bits: int,
    spn_mixer_depth: int,
    epochs: int,
    learning_rate: float,
    lambda_aux: float,
    batch_size: int,
    shuffled_targets: bool,
    seed: int,
    device: torch.device,
    progress_output: Path | None = None,
) -> tuple[torch.nn.Module, float]:
    torch.manual_seed(seed)
    model = build_model(
        model_name,
        input_bits=features.shape[1],
        hidden_bits=hidden_bits,
        pair_bits=128,
        structure="spn",
        model_options={"spn_mixer_depth": spn_mixer_depth, "activation": "relu", "norm": "layernorm"},
    ).to(device)
    x = torch.from_numpy(features.astype(np.float32)).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1).to(device)
    aux_targets = _active_targets(x, shuffled=shuffled_targets, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    main_loss_fn = torch.nn.BCEWithLogitsLoss()
    aux_loss_fn = torch.nn.BCEWithLogitsLoss()
    last_aux_loss = 0.0
    max_batch = max(1, batch_size)
    steps_per_epoch = max(1, (int(x.shape[0]) + max_batch - 1) // max_batch)
    train_rows = int(x.shape[0])
    _write_progress(
        progress_output,
        "active_auxiliary_train_start",
        {
            "model": model_name,
            "epochs": int(epochs),
            "steps_per_epoch": steps_per_epoch,
            "train_rows": train_rows,
            "batch_size": max_batch,
        },
    )
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_aux_loss = 0.0
        total_seen = 0
        for step, batch_start in enumerate(range(0, x.shape[0], max_batch), start=1):
            batch_end = min(x.shape[0], batch_start + max_batch)
            batch_x = x[batch_start:batch_end]
            batch_y = y[batch_start:batch_end]
            batch_aux = aux_targets[batch_start:batch_end]
            optimizer.zero_grad()
            main_loss = main_loss_fn(model(batch_x), batch_y)
            aux_loss = aux_loss_fn(model.active_mask_logits(batch_x), batch_aux)
            loss = main_loss + float(lambda_aux) * aux_loss
            loss.backward()
            optimizer.step()
            last_aux_loss = float(aux_loss.detach().cpu().item())
            batch_rows = int(batch_end - batch_start)
            total_loss += float(loss.detach().cpu().item()) * batch_rows
            total_aux_loss += last_aux_loss * batch_rows
            total_seen += batch_rows
            if should_report_step(step, steps_per_epoch):
                _write_progress(
                    progress_output,
                    "train_batch",
                    {
                        "stage": "training",
                        "model": model_name,
                        "epoch": epoch,
                        "epochs": int(epochs),
                        "step": step,
                        "steps_per_epoch": steps_per_epoch,
                        "train_rows_seen": total_seen,
                        "train_rows": train_rows,
                        "train_loss": total_loss / max(1, total_seen),
                        "auxiliary_loss": total_aux_loss / max(1, total_seen),
                        "train_rows_progress_percent": 100.0 * total_seen / max(1, train_rows),
                    },
                )
        _write_progress(
            progress_output,
            "active_auxiliary_epoch_end",
            {
                "model": model_name,
                "epoch": epoch,
                "epochs": int(epochs),
                "train_rows": train_rows,
                "train_loss": total_loss / max(1, total_seen),
                "auxiliary_loss": total_aux_loss / max(1, total_seen),
            },
        )
    return model, last_aux_loss


def _evaluate_active_aux_model(
    model: torch.nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    lambda_aux: float,
    batch_size: int,
    shuffled_targets: bool,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    del labels, lambda_aux
    max_batch = max(1, int(batch_size))
    aux_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    logits_parts: list[np.ndarray] = []
    aux_loss_sum = 0.0
    aux_element_count = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, features.shape[0], max_batch):
            batch_end = min(features.shape[0], batch_start + max_batch)
            batch_features = np.array(features[batch_start:batch_end], dtype=np.float32, copy=True)
            x = torch.from_numpy(batch_features).to(device)
            aux_targets = _active_targets(x, shuffled=shuffled_targets, seed=seed + batch_start)
            logits_parts.append(model(x).detach().cpu().numpy().reshape(-1))
            aux_logits = model.active_mask_logits(x)
            aux_loss_sum += float(aux_loss_fn(aux_logits, aux_targets).detach().cpu().item())
            aux_element_count += int(aux_targets.numel())
    if was_training:
        model.train()
    logits = np.concatenate(logits_parts) if logits_parts else np.empty((0,), dtype=np.float32)
    aux_loss = aux_loss_sum / max(1, aux_element_count)
    return logits, aux_loss


def _active_targets(features: torch.Tensor, *, shuffled: bool, seed: int) -> torch.Tensor:
    targets = present_invp_active_mask_targets(features, pair_bits=128)
    if shuffled:
        targets = shuffled_active_mask_targets(targets, seed=seed)
    return targets


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"active-auxiliary config must be a JSON object: {path}")
    return data


def _apply_config_defaults(args: argparse.Namespace, *, require_output: bool = True) -> argparse.Namespace:
    config = _load_config(args.config)
    for field, default in DEFAULTS.items():
        value = getattr(args, field)
        if value is None:
            value = config.get(field, default)
        if field in PATH_FIELDS and value is not None:
            value = Path(str(value))
        elif field in INT_FIELDS:
            value = int(str(value), 0) if isinstance(value, str) else int(value)
        elif field in FLOAT_FIELDS:
            value = float(value)
        setattr(args, field, value)
    if require_output and args.output is None:
        raise SystemExit("active-auxiliary output path is required")
    if args.model not in {ACTIVE_AUX_MODEL, SHUFFLED_ACTIVE_AUX_MODEL}:
        raise SystemExit(f"unsupported active-auxiliary model: {args.model}")
    return args


__all__ = ["args_from_config", "make_active_auxiliary_dataset", "run_active_auxiliary"]
