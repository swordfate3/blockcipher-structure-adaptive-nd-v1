from __future__ import annotations

import argparse
from typing import Any

from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.training import TrainingConfig


def resolve_task_keys(task: dict[str, Any]) -> tuple[int | None, int | None]:
    train_key = task.get("train_key")
    validation_key = task.get("validation_key")
    if validation_key is None:
        validation_key = train_key
    return train_key, validation_key


def build_dataset_config(
    task: dict[str, Any],
    *,
    cipher,
    samples_per_class: int,
    seed: int,
) -> DifferentialDatasetConfig:
    return DifferentialDatasetConfig(
        cipher=cipher,
        input_difference=task["input_difference"],
        samples_per_class=samples_per_class,
        seed=seed,
        feature_encoding=task["feature_encoding"],
        pairs_per_sample=task["pairs_per_sample"],
        negative_mode=task["negative_mode"],
        key_rotation_interval=task["key_rotation_interval"],
        sample_structure=task["sample_structure"],
        integral_active_nibble=task["integral_active_nibble"],
        selected_bit_indices=task["selected_bit_indices"],
    )


def build_training_config(
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    epochs: int,
    seed: int,
) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=float(task.get("learning_rate") or args.learning_rate),
        optimizer=str(task.get("optimizer") or args.optimizer),
        amsgrad=args.amsgrad,
        weight_decay=float(task.get("weight_decay") if task.get("weight_decay") is not None else args.weight_decay),
        lr_scheduler=str(task.get("lr_scheduler") or args.lr_scheduler),
        max_learning_rate=task.get("max_learning_rate")
        if task.get("max_learning_rate") is not None
        else args.max_learning_rate,
        checkpoint_metric=str(task.get("checkpoint_metric") or args.checkpoint_metric),
        restore_best_checkpoint=bool(
            task.get("restore_best_checkpoint")
            if task.get("restore_best_checkpoint") is not None
            else args.restore_best_checkpoint
        ),
        early_stopping_patience=int(
            task.get("early_stopping_patience")
            if task.get("early_stopping_patience") is not None
            else args.early_stopping_patience
        ),
        early_stopping_min_delta=float(
            task.get("early_stopping_min_delta")
            if task.get("early_stopping_min_delta") is not None
            else args.early_stopping_min_delta
        ),
        loss=str(task.get("loss") or args.loss),
        train_eval_interval=int(args.train_eval_interval),
        seed=seed,
        device=args.device,
    )
