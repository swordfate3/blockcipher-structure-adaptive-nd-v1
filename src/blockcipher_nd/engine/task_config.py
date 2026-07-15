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


def resolve_final_test_key(task: dict[str, Any]) -> int | None:
    final_test_key = task.get("final_test_key")
    if final_test_key is not None:
        return final_test_key
    return resolve_task_keys(task)[1]


def validation_samples_per_class(task: dict[str, Any]) -> int:
    total = task.get("validation_samples_total")
    if total is not None:
        return max(1, int(total) // 2)
    return max(8, int(task["samples_per_class"]) // 2)


def build_dataset_config(
    task: dict[str, Any],
    *,
    cipher,
    samples_per_class: int,
    seed: int,
    samples_total: int | None = None,
    split: str = "train",
) -> DifferentialDatasetConfig:
    active_nibbles = task.get("integral_active_nibbles", ())
    dataset_label_mode = str(
        task.get("dataset_label_mode") or "balanced_per_class"
    )
    if (
        split == "validation" or split.startswith("final_test_")
    ) and task.get("validation_integral_active_nibbles"):
        active_nibbles = task["validation_integral_active_nibbles"]
    return DifferentialDatasetConfig(
        cipher=cipher,
        input_difference=task["input_difference"],
        samples_per_class=samples_per_class,
        samples_total=(
            samples_total if dataset_label_mode == "random_labels_total" else None
        ),
        dataset_label_mode=dataset_label_mode,
        seed=seed,
        feature_encoding=task["feature_encoding"],
        pairs_per_sample=task["pairs_per_sample"],
        negative_mode=task["negative_mode"],
        key_rotation_interval=task["key_rotation_interval"],
        sample_structure=task["sample_structure"],
        integral_active_nibble=task["integral_active_nibble"],
        integral_active_nibbles=tuple(active_nibbles),
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
        checkpoint_output=args.checkpoint_output,
        seed=seed,
        device=args.device,
    )
