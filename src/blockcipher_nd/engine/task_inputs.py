from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from blockcipher_nd.ciphers import ReducedRoundCipher
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.engine.datasets import make_task_dataset
from blockcipher_nd.engine.modeling import infer_pair_bits, select_model_key
from blockcipher_nd.engine.progress import task_progress_payload, write_progress
from blockcipher_nd.engine.task_config import (
    build_dataset_config,
    resolve_final_test_key,
    resolve_task_keys,
    validation_samples_per_class,
)
from blockcipher_nd.registry.cipher_factory import build_cipher


@dataclass(frozen=True)
class TaskInputs:
    train_key: int | None
    validation_key: int | None
    final_test_key: int | None
    train_cipher: ReducedRoundCipher
    validation_cipher: ReducedRoundCipher
    final_test_cipher: ReducedRoundCipher
    train_dataset: DifferentialDataset
    validation_dataset: DifferentialDataset
    model_key: str
    pair_bits: int


def prepare_task_inputs(
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    progress_path: str | None = None,
    index: int | None = None,
    total: int | None = None,
) -> TaskInputs:
    train_key, validation_key = resolve_task_keys(task)
    final_test_key = resolve_final_test_key(task)
    train_cipher = build_cipher(task["cipher_key"], task["rounds"], key=train_key)
    validation_cipher = build_cipher(
        task["cipher_key"], task["rounds"], key=validation_key
    )
    final_test_cipher = build_cipher(
        task["cipher_key"], task["rounds"], key=final_test_key
    )
    model_key = select_model_key(
        task["model_key"], train_cipher.structure, task["pairs_per_sample"]
    )
    pair_bits = infer_pair_bits(train_cipher.block_bits, task["feature_encoding"])

    train_dataset = make_task_dataset(
        build_dataset_config(
            task,
            cipher=train_cipher,
            samples_per_class=task["samples_per_class"],
            samples_total=task.get("train_samples_total"),
            seed=task["seed"],
            split="train",
        ),
        args,
        task,
        split="train",
        progress_path=progress_path,
        index=index,
        total=total,
    )
    validation_dataset = make_task_dataset(
        build_dataset_config(
            task,
            cipher=validation_cipher,
            samples_per_class=validation_samples_per_class(task),
            samples_total=task.get("validation_samples_total"),
            seed=task["seed"] + 10_000,
            split="validation",
        ),
        args,
        task,
        split="validation",
        progress_path=progress_path,
        index=index,
        total=total,
    )
    write_progress(
        progress_path,
        "cache_ready",
        {
            "index": index,
            "total": total,
            "dataset_cache_enabled": bool(args.dataset_cache_root),
            "train_rows": int(train_dataset.features.shape[0]),
            "validation_rows": int(validation_dataset.features.shape[0]),
            "input_bits": int(train_dataset.features.shape[1]),
            **task_progress_payload(task),
        },
    )
    return TaskInputs(
        train_key=train_key,
        validation_key=validation_key,
        final_test_key=final_test_key,
        train_cipher=train_cipher,
        validation_cipher=validation_cipher,
        final_test_cipher=final_test_cipher,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        model_key=model_key,
        pair_bits=pair_bits,
    )


__all__ = ["TaskInputs", "prepare_task_inputs"]
