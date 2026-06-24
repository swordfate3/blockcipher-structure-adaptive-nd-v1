from __future__ import annotations

import argparse
from typing import Any

from blockcipher_nd.engine.datasets import make_task_dataset
from blockcipher_nd.engine.progress import progress_callback, task_progress_payload, write_progress
from blockcipher_nd.engine.task_config import build_dataset_config, build_training_config, resolve_task_keys
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.training import TrainingResult, train_binary_classifier


def run_optional_pretraining(
    model,
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    pair_bits: int | None,
    progress_path: str | None,
    index: int | None,
    total: int | None,
) -> TrainingResult | None:
    pretrain_epochs = int(
        task.get("pretrain_epochs")
        if task.get("pretrain_epochs") is not None
        else args.pretrain_epochs
    )
    pretrain_rounds = (
        int(task["pretrain_rounds"])
        if task.get("pretrain_rounds") is not None
        else args.pretrain_rounds
    )
    if pretrain_epochs <= 0 or pretrain_rounds is None:
        return None
    if pretrain_rounds == task["rounds"]:
        raise ValueError("pretrain_rounds must differ from target rounds")

    pretrain_task = {**task, "rounds": pretrain_rounds}
    pretrain_train_key, pretrain_validation_key = resolve_task_keys(pretrain_task)
    pretrain_cipher = build_cipher(
        pretrain_task["cipher_key"],
        pretrain_rounds,
        key=pretrain_train_key,
    )
    pretrain_validation_cipher = build_cipher(
        pretrain_task["cipher_key"],
        pretrain_rounds,
        key=pretrain_validation_key,
    )
    pretrain_dataset = make_task_dataset(
        build_dataset_config(
            pretrain_task,
            cipher=pretrain_cipher,
            samples_per_class=pretrain_task["samples_per_class"],
            seed=pretrain_task["seed"] + 20_000,
        ),
        args,
        pretrain_task,
        split="pretrain_train",
        progress_path=progress_path,
        index=index,
        total=total,
    )
    pretrain_validation_dataset = make_task_dataset(
        build_dataset_config(
            pretrain_task,
            cipher=pretrain_validation_cipher,
            samples_per_class=max(8, pretrain_task["samples_per_class"] // 2),
            seed=pretrain_task["seed"] + 30_000,
        ),
        args,
        pretrain_task,
        split="pretrain_validation",
        progress_path=progress_path,
        index=index,
        total=total,
    )
    expected_input_bits = int(pretrain_dataset.features.shape[1])
    if pair_bits is not None and expected_input_bits % pair_bits != 0:
        raise ValueError("pretraining feature width is incompatible with target pair_bits")
    write_progress(
        progress_path,
        "pretrain_cache_ready",
        {
            "index": index,
            "total": total,
            "target_rounds": task["rounds"],
            "pretrain_rounds": pretrain_rounds,
            "pretrain_epochs": pretrain_epochs,
            "train_rows": int(pretrain_dataset.features.shape[0]),
            "validation_rows": int(pretrain_validation_dataset.features.shape[0]),
            "input_bits": expected_input_bits,
            **task_progress_payload(pretrain_task),
        },
    )
    return train_binary_classifier(
        model,
        pretrain_dataset,
        pretrain_validation_dataset,
        build_training_config(
            pretrain_task,
            args,
            epochs=pretrain_epochs,
            seed=pretrain_task["seed"] + 40_000,
        ),
        progress_callback=progress_callback(
            progress_path,
            "pretraining",
            pretrain_task,
            index=index,
            total=total,
        ),
    )


def pretraining_metadata(result: TrainingResult | None) -> dict[str, Any]:
    if result is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "metrics": result.final_metrics,
        "epochs_ran": result.metadata.get("epochs_ran"),
        "best_epoch": result.metadata.get("best_epoch"),
        "best_checkpoint_metric": result.metadata.get("best_checkpoint_metric"),
        "selected_checkpoint": result.metadata.get("selected_checkpoint"),
    }
