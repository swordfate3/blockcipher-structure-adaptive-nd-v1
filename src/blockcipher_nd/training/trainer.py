from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from blockcipher_nd.data.differential import DifferentialDataset, DiskDifferentialDataset
from blockcipher_nd.training.data import make_loader, select_device
from blockcipher_nd.training.metrics import evaluate_binary_classifier
from blockcipher_nd.training.optim import (
    OfficialEpochCyclicLR,
    compute_loss,
    current_learning_rate,
    make_loss,
    make_optimizer,
    make_scheduler,
)
from blockcipher_nd.training.types import ProgressCallback, TrainingConfig, TrainingResult


@dataclass
class OptimizerSession:
    optimizer: torch.optim.Optimizer | None = None
    config_signature: tuple[Any, ...] | None = None
    training_calls: int = 0


def train_binary_classifier(
    model: nn.Module,
    train_dataset: DifferentialDataset,
    validation_dataset: DifferentialDataset,
    config: TrainingConfig,
    progress_callback: ProgressCallback | None = None,
    optimizer_session: OptimizerSession | None = None,
) -> TrainingResult:
    torch.manual_seed(config.seed)
    selected_device = select_device(config.device)
    model = model.to(selected_device)
    optimizer, optimizer_state_reused, optimizer_session_call = resolve_optimizer(
        model,
        config,
        optimizer_session,
    )
    optimizer_state_step_before = optimizer_state_step(optimizer)
    scheduler = make_scheduler(optimizer, config, len(train_dataset.labels))
    loss_fn = make_loss(config.loss)
    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    steps_per_epoch = len(train_loader)
    emit_progress(
        progress_callback,
        "train_start",
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
            features = features.to(selected_device)
            labels = labels.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features).squeeze(1)
            loss = compute_loss(loss_fn, logits, labels, config.loss)
            auxiliary_loss = getattr(model, "last_auxiliary_loss", None)
            if auxiliary_loss is not None:
                loss = loss + auxiliary_loss
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
    metadata = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "train_dataset_storage": "disk" if isinstance(train_dataset, DiskDifferentialDataset) else "memory",
        "validation_dataset_storage": "disk" if isinstance(validation_dataset, DiskDifferentialDataset) else "memory",
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "optimizer_state_reused": optimizer_state_reused,
        "optimizer_state_step_before": optimizer_state_step_before,
        "optimizer_state_step_after": optimizer_state_step(optimizer),
        "optimizer_session_call": optimizer_session_call,
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


def resolve_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    session: OptimizerSession | None,
) -> tuple[torch.optim.Optimizer, bool, int]:
    if session is None:
        return make_optimizer(model, config), False, 1

    signature = optimizer_config_signature(config)
    reused = session.optimizer is not None
    if session.optimizer is None:
        session.optimizer = make_optimizer(model, config)
        session.config_signature = signature
    else:
        if session.config_signature != signature:
            raise ValueError("optimizer session configuration changed between stages")
        optimizer_parameter_ids = {
            id(parameter)
            for group in session.optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_parameter_ids != {id(parameter) for parameter in model.parameters()}:
            raise ValueError("optimizer session parameters do not match the model")
        if config.lr_scheduler != "none":
            raise ValueError("optimizer state carry currently requires lr_scheduler=none")
    session.training_calls += 1
    return session.optimizer, reused, session.training_calls


def optimizer_config_signature(config: TrainingConfig) -> tuple[Any, ...]:
    return (
        config.optimizer,
        config.learning_rate,
        config.amsgrad,
        config.weight_decay,
        config.lr_scheduler,
        config.max_learning_rate,
    )


def optimizer_state_step(optimizer: torch.optim.Optimizer) -> int:
    steps: list[int] = []
    for state in optimizer.state.values():
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            step = step.item()
        if step is not None:
            steps.append(int(step))
    return max(steps, default=0)


def write_checkpoint_if_requested(
    model: nn.Module,
    config: TrainingConfig,
    *,
    history: list[dict[str, float]],
    final_metrics: dict[str, float],
    metadata: dict[str, Any],
) -> str | None:
    if config.checkpoint_output is None:
        return None
    path = Path(config.checkpoint_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata["checkpoint_output"] = str(path)
    payload = {
        "state_dict": clone_state_dict_to_cpu(model),
        "history": history,
        "final_metrics": final_metrics,
        "metadata": metadata,
    }
    torch.save(payload, path)
    return str(path)


def validate_checkpoint_metric(metric: str) -> None:
    if metric not in {"val_accuracy", "val_auc", "val_loss"}:
        raise ValueError(f"unsupported checkpoint metric: {metric}")


def should_evaluate_train(epoch: int, interval: int) -> bool:
    if interval < 0:
        raise ValueError("train_eval_interval must be non-negative")
    return interval > 0 and epoch % interval == 0


def skipped_train_metrics() -> dict[str, float | None]:
    return {
        "loss": None,
        "accuracy": None,
        "auc": None,
        "best_accuracy": None,
        "calibrated_accuracy": None,
    }


def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def is_checkpoint_improved(
    *,
    current: float,
    best: float | None,
    metric: str,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    if metric == "val_loss":
        return current < best - min_delta
    return current > best + min_delta


def should_report_step(step: int, steps_per_epoch: int) -> bool:
    if steps_per_epoch <= 10:
        return True
    interval = max(1, steps_per_epoch // 10)
    return step == 1 or step == steps_per_epoch or step % interval == 0


def emit_progress(callback: ProgressCallback | None, event: str, **payload: Any) -> None:
    if callback is None:
        return
    callback(event, payload)
