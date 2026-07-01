from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn

from blockcipher_nd.data.differential import DifferentialDataset, DiskDifferentialDataset
from blockcipher_nd.training.data import select_device
from blockcipher_nd.training.metrics import evaluate_binary_classifier
from blockcipher_nd.training.optim import (
    OfficialEpochCyclicLR,
    compute_loss,
    current_learning_rate,
    make_loss,
    make_optimizer,
    make_scheduler,
)
from blockcipher_nd.training.trainer import (
    clone_state_dict_to_cpu,
    emit_progress,
    is_checkpoint_improved,
    should_evaluate_train,
    should_report_step,
    skipped_train_metrics,
    validate_checkpoint_metric,
    write_checkpoint_if_requested,
)
from blockcipher_nd.training.types import ProgressCallback, TrainingConfig, TrainingResult

from blockcipher_training_accelerator.data import make_accelerated_loader
from blockcipher_training_accelerator.profiles import SpeedProfile


def train_binary_classifier_accelerated(
    model: nn.Module,
    train_dataset: DifferentialDataset,
    validation_dataset: DifferentialDataset,
    config: TrainingConfig,
    *,
    profile: SpeedProfile,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    selected_device = select_device(config.device)
    model = model.to(selected_device)
    compile_enabled = bool(profile.compile_model and selected_device.type == "cuda")
    if compile_enabled:
        model = torch.compile(model)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config, len(train_dataset.labels))
    loss_fn = make_loss(config.loss)
    train_loader = make_accelerated_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        profile=profile,
        device=selected_device,
    )
    steps_per_epoch = len(train_loader)
    emit_progress(
        progress_callback,
        "accelerated_train_start",
        speed_profile=profile.name,
        epochs=config.epochs,
        batch_size=config.batch_size,
        train_rows=int(len(train_dataset.labels)),
        validation_rows=int(len(validation_dataset.labels)),
        steps_per_epoch=steps_per_epoch,
        device=str(selected_device),
    )

    history: list[dict[str, float]] = []
    validate_checkpoint_metric(config.checkpoint_metric)
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metric_value: float | None = None
    epochs_without_improvement = 0
    stopped_epoch = 0
    for epoch in range(1, config.epochs + 1):
        if isinstance(scheduler, OfficialEpochCyclicLR):
            scheduler.step_epoch(epoch)
        emit_progress(
            progress_callback,
            "epoch_start",
            epoch=epoch,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
        )
        model.train()
        total_loss = 0.0
        total_seen = 0
        for step, (features, labels) in enumerate(train_loader, start=1):
            features = move_to_device(features, selected_device, profile)
            labels = move_to_device(labels, selected_device, profile)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(selected_device, profile):
                logits = model(features).squeeze(1)
                loss = compute_loss(loss_fn, logits, labels, config.loss)
            loss.backward()
            optimizer.step()
            if scheduler is not None and not isinstance(scheduler, OfficialEpochCyclicLR):
                scheduler.step()
            total_loss += float(loss.detach().cpu()) * len(labels)
            total_seen += len(labels)
            if should_report_step(step, steps_per_epoch):
                emit_progress(
                    progress_callback,
                    "train_batch",
                    epoch=epoch,
                    epochs=config.epochs,
                    step=step,
                    steps_per_epoch=steps_per_epoch,
                    train_rows_seen=total_seen,
                    train_rows=int(len(train_dataset.labels)),
                    train_loss=total_loss / max(1, total_seen),
                    learning_rate=current_learning_rate(optimizer),
                )

        emit_progress(
            progress_callback,
            "validation_start",
            epoch=epoch,
            epochs=config.epochs,
            validation_rows=int(len(validation_dataset.labels)),
        )
        validation_metrics = evaluate_binary_classifier(
            model,
            validation_dataset,
            batch_size=config.batch_size,
            device=str(selected_device),
        )
        train_metrics = (
            evaluate_binary_classifier(
                model,
                train_dataset,
                batch_size=config.batch_size,
                device=str(selected_device),
            )
            if should_evaluate_train(epoch, config.train_eval_interval)
            else skipped_train_metrics()
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(1, total_seen),
                "train_eval_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_auc": train_metrics["auc"],
                "train_best_accuracy": train_metrics["best_accuracy"],
                "train_calibrated_accuracy": train_metrics["calibrated_accuracy"],
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["accuracy"],
                "val_auc": validation_metrics["auc"],
                "val_best_accuracy": validation_metrics["best_accuracy"],
                "val_calibrated_accuracy": validation_metrics["calibrated_accuracy"],
                "learning_rate": current_learning_rate(optimizer),
            }
        )
        current_metric_value = history[-1][config.checkpoint_metric]
        if is_checkpoint_improved(
            current=current_metric_value,
            best=best_metric_value,
            metric=config.checkpoint_metric,
            min_delta=config.early_stopping_min_delta,
        ):
            best_metric_value = current_metric_value
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = clone_state_dict_to_cpu(model)
            emit_progress(
                progress_callback,
                "checkpoint_improved",
                epoch=epoch,
                metric=config.checkpoint_metric,
                value=current_metric_value,
            )
        else:
            epochs_without_improvement += 1
        emit_progress(
            progress_callback,
            "epoch_end",
            epoch=epoch,
            epochs=config.epochs,
            train_loss=history[-1]["train_loss"],
            train_eval_loss=history[-1]["train_eval_loss"],
            train_accuracy=history[-1]["train_accuracy"],
            train_auc=history[-1]["train_auc"],
            val_loss=history[-1]["val_loss"],
            val_accuracy=history[-1]["val_accuracy"],
            val_auc=history[-1]["val_auc"],
            learning_rate=history[-1]["learning_rate"],
            best_epoch=best_epoch,
            best_checkpoint_metric=best_metric_value,
        )
        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_epoch = epoch
            emit_progress(
                progress_callback,
                "early_stopping",
                epoch=epoch,
                patience=config.early_stopping_patience,
                best_epoch=best_epoch,
                metric=config.checkpoint_metric,
                best_value=best_metric_value,
            )
            break

    selected_checkpoint = "last"
    if config.restore_best_checkpoint and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model = model.to(selected_device)
        selected_checkpoint = "best"
        emit_progress(
            progress_callback,
            "checkpoint_restored",
            best_epoch=best_epoch,
            metric=config.checkpoint_metric,
            best_value=best_metric_value,
        )
    emit_progress(progress_callback, "final_evaluation_start")
    final_metrics = evaluate_binary_classifier(
        model,
        validation_dataset,
        batch_size=config.batch_size,
        device=str(selected_device),
    )
    metadata: dict[str, Any] = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "train_dataset_storage": "disk" if isinstance(train_dataset, DiskDifferentialDataset) else "memory",
        "validation_dataset_storage": "disk" if isinstance(validation_dataset, DiskDifferentialDataset) else "memory",
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "amsgrad": config.amsgrad,
        "weight_decay": config.weight_decay,
        "lr_scheduler": config.lr_scheduler,
        "max_learning_rate": config.max_learning_rate,
        "checkpoint_metric": config.checkpoint_metric,
        "restore_best_checkpoint": config.restore_best_checkpoint,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "train_eval_interval": config.train_eval_interval,
        "loss": config.loss,
        "best_epoch": best_epoch,
        "best_checkpoint_metric": best_metric_value,
        "selected_checkpoint": selected_checkpoint,
        "stopped_epoch": stopped_epoch,
        "epochs_ran": len(history),
        "seed": config.seed,
        "device": str(selected_device),
        "accelerator": {
            **profile.to_json_dict(),
            "compile_effective": compile_enabled,
            "amp_effective": bool(profile.amp_dtype and selected_device.type == "cuda"),
            "duration_seconds": round(time.perf_counter() - started, 6),
        },
    }
    checkpoint_output = write_checkpoint_if_requested(
        model,
        config,
        history=history,
        final_metrics=final_metrics,
        metadata=metadata,
    )
    if checkpoint_output is not None:
        emit_progress(
            progress_callback,
            "checkpoint_written",
            path=checkpoint_output,
            selected_checkpoint=selected_checkpoint,
        )
    emit_progress(
        progress_callback,
        "train_done",
        epochs=config.epochs,
        epochs_ran=len(history),
        accuracy=final_metrics["accuracy"],
        auc=final_metrics["auc"],
        calibrated_accuracy=final_metrics["calibrated_accuracy"],
    )
    return TrainingResult(history=history, final_metrics=final_metrics, metadata=metadata)


def move_to_device(tensor: torch.Tensor, device: torch.device, profile: SpeedProfile) -> torch.Tensor:
    return tensor.to(
        device,
        non_blocking=bool(profile.non_blocking_transfer and device.type == "cuda"),
    )


def autocast_context(device: torch.device, profile: SpeedProfile):
    if device.type != "cuda" or profile.amp_dtype is None:
        return nullcontext()
    if profile.amp_dtype == "bf16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if profile.amp_dtype == "fp16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"unsupported amp dtype: {profile.amp_dtype}")
