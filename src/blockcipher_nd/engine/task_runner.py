from __future__ import annotations

import argparse
from typing import Any

from blockcipher_nd.engine.datasets import make_task_dataset
from blockcipher_nd.engine.modeling import (
    configure_structure_aware_model,
    infer_pair_bits,
    select_model_key,
)
from blockcipher_nd.engine.pretraining import run_optional_pretraining
from blockcipher_nd.engine.progress import progress_callback, task_progress_payload, write_progress
from blockcipher_nd.engine.results import build_task_result
from blockcipher_nd.engine.task_config import build_dataset_config, build_training_config, resolve_task_keys
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.training import train_binary_classifier


def run_task(
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    progress_path: str | None = None,
    index: int | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    train_key, validation_key = resolve_task_keys(task)
    train_cipher = build_cipher(task["cipher_key"], task["rounds"], key=train_key)
    validation_cipher = build_cipher(task["cipher_key"], task["rounds"], key=validation_key)
    model_key = select_model_key(
        task["model_key"],
        train_cipher.structure,
        task["pairs_per_sample"],
    )
    pair_bits = infer_pair_bits(train_cipher.block_bits, task["feature_encoding"])

    train_dataset = make_task_dataset(
        build_dataset_config(
            task,
            cipher=train_cipher,
            samples_per_class=task["samples_per_class"],
            seed=task["seed"],
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
            samples_per_class=max(8, task["samples_per_class"] // 2),
            seed=task["seed"] + 10_000,
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

    model = build_model(
        model_key,
        input_bits=train_dataset.features.shape[1],
        hidden_bits=args.hidden_bits,
        pair_bits=pair_bits,
        structure=train_cipher.structure,
        model_options=task.get("model_options"),
    )
    configure_structure_aware_model(model, task["cipher_key"], task["rounds"])
    pretrain_result = run_optional_pretraining(
        model,
        task,
        args,
        pair_bits=pair_bits,
        progress_path=progress_path,
        index=index,
        total=total,
    )
    configure_structure_aware_model(model, task["cipher_key"], task["rounds"])
    training_result = train_binary_classifier(
        model,
        train_dataset,
        validation_dataset,
        build_training_config(task, args, epochs=args.epochs, seed=task["seed"]),
        progress_callback=progress_callback(
            progress_path,
            "training",
            task,
            index=index,
            total=total,
        ),
    )
    return build_task_result(
        task=task,
        args=args,
        train_cipher=train_cipher,
        validation_cipher=validation_cipher,
        train_key=train_key,
        validation_key=validation_key,
        model=model,
        model_key=model_key,
        pair_bits=pair_bits,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        training_result=training_result,
        pretrain_result=pretrain_result,
    )
