from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.data.cache import make_chunked_differential_dataset
from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.engine.progress import progress_callback


def make_task_dataset(
    config: DifferentialDatasetConfig,
    args: argparse.Namespace,
    task: dict[str, Any],
    *,
    split: str,
    progress_path: str | None = None,
    index: int | None = None,
    total: int | None = None,
):
    if not args.dataset_cache_root:
        return make_differential_dataset(config)
    return make_chunked_differential_dataset(
        config,
        cache_dir=dataset_cache_dir(Path(args.dataset_cache_root), task, config, split),
        chunk_size=args.dataset_cache_chunk_size,
        workers=args.dataset_cache_workers,
        progress_callback=progress_callback(
            progress_path,
            "dataset_cache",
            task,
            index=index,
            total=total,
            split=split,
        ),
        progress_context={"split": split},
    )


def dataset_cache_dir(
    root: Path,
    task: dict[str, Any],
    config: DifferentialDatasetConfig,
    split: str,
) -> Path:
    cache_identity = {
        "cipher_key": task["cipher_key"],
        "rounds": task["rounds"],
        "split": split,
        "seed": config.seed,
        "samples_per_class": config.samples_per_class,
        "pairs_per_sample": config.pairs_per_sample,
        "input_difference": config.input_difference,
        "feature_encoding": config.feature_encoding,
        "negative_mode": config.negative_mode,
        "key_rotation_interval": config.key_rotation_interval,
        "sample_structure": config.sample_structure,
        "integral_active_nibble": config.integral_active_nibble,
        "selected_bit_indices": config.selected_bit_indices,
        "key": task.get("train_key") if split in {"train", "pretrain_train"} else task.get("validation_key"),
    }
    digest = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return root / task["cipher_key"] / f"r{task['rounds']}" / split / f"seed-{config.seed}_{digest}"
