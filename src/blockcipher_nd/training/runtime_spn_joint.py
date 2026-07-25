from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.training.data import make_loader, select_device
from blockcipher_nd.training.metrics import (
    best_threshold_accuracy_and_threshold,
    binary_auc,
)
from blockcipher_nd.training.optim import compute_loss, make_loss
from blockcipher_nd.training.types import ProgressCallback, TrainingConfig


@dataclass(frozen=True)
class RuntimeSpnJointTask:
    name: str
    group: str
    structure: RuntimeSpnStructure
    train_dataset: DifferentialDataset
    validation_dataset: DifferentialDataset


@dataclass(frozen=True)
class RuntimeSpnJointTrainingResult:
    history: list[dict[str, Any]]
    train_metrics: dict[str, dict[str, float]]
    validation_metrics: dict[str, dict[str, float]]
    metadata: dict[str, Any]
    router_traffic: dict[str, dict[str, float]]
    gradient_diagnostics: dict[str, Any]


def train_runtime_spn_joint(
    model: nn.Module,
    tasks: list[RuntimeSpnJointTask],
    config: TrainingConfig,
    progress_callback: ProgressCallback | None = None,
) -> RuntimeSpnJointTrainingResult:
    if not tasks:
        raise ValueError("joint Runtime-SPN training requires at least one task")
    if len({task.name for task in tasks}) != len(tasks):
        raise ValueError("joint Runtime-SPN task names must be unique")
    if config.checkpoint_metric != "val_macro_auc":
        raise ValueError("joint Runtime-SPN checkpoint metric must be val_macro_auc")
    if config.optimizer != "adam":
        raise ValueError("joint Runtime-SPN training currently requires Adam")
    if config.lr_scheduler != "none":
        raise ValueError("joint Runtime-SPN training currently requires no scheduler")

    torch.manual_seed(config.seed)
    device = select_device(config.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
    )
    loss_fn = make_loss(config.loss)
    task_weight = 1.0 / len(tasks)
    task_weights = {task.name: task_weight for task in tasks}
    best_state: dict[str, torch.Tensor] | None = None
    best_macro_auc: float | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    total_wraps = {task.name: 0 for task in tasks}
    total_task_batches = {task.name: 0 for task in tasks}
    total_router_traffic: dict[str, dict[str, float]] = {
        task.name: {} for task in tasks
    }
    gradient_sums: dict[str, float] = {}
    gradient_observations: dict[str, int] = {}
    all_gradients_finite = True
    optimizer_steps = 0

    _emit(
        progress_callback,
        "joint_train_start",
        tasks=[task.name for task in tasks],
        task_weights=task_weights,
        epochs=config.epochs,
        device=str(device),
    )
    for epoch in range(1, config.epochs + 1):
        loaders = {
            task.name: make_loader(
                task.train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.seed + epoch * 1009 + index,
            )
            for index, task in enumerate(tasks)
        }
        iterators = {name: iter(loader) for name, loader in loaders.items()}
        steps = max(len(loader) for loader in loaders.values())
        epoch_loss_sums = {task.name: 0.0 for task in tasks}
        epoch_rows = {task.name: 0 for task in tasks}
        model.train()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []
            for task in tasks:
                iterator = iterators[task.name]
                try:
                    features, labels = next(iterator)
                except StopIteration:
                    total_wraps[task.name] += 1
                    iterator = iter(loaders[task.name])
                    iterators[task.name] = iterator
                    features, labels = next(iterator)
                features = _to_runtime_coordinates(
                    features.to(device),
                    task.structure.block_bits,
                )
                labels = labels.to(device)
                logits = model(features, task.structure).squeeze(1)
                loss = compute_loss(loss_fn, logits, labels, config.loss)
                losses.append(loss)
                epoch_loss_sums[task.name] += float(loss.detach().cpu()) * len(labels)
                epoch_rows[task.name] += len(labels)
                total_task_batches[task.name] += 1
                _accumulate_router_traffic(
                    total_router_traffic[task.name],
                    getattr(model, "last_primitive_adapter_traffic", {}),
                )
            joint_loss = torch.stack(losses).mean()
            joint_loss.backward()
            finite = _accumulate_gradient_diagnostics(
                model,
                gradient_sums,
                gradient_observations,
            )
            all_gradients_finite = all_gradients_finite and finite
            optimizer.step()
            optimizer_steps += 1
            _emit(
                progress_callback,
                "joint_train_batch",
                epoch=epoch,
                step=step,
                steps=steps,
                joint_loss=float(joint_loss.detach().cpu()),
            )

        validation_metrics = evaluate_runtime_spn_joint(
            model,
            tasks,
            split="validation",
            batch_size=config.batch_size,
            device=device,
            loss=config.loss,
        )
        macro_auc = float(
            np.mean([metrics["auc"] for metrics in validation_metrics.values()])
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_joint_loss": float(
                np.mean(
                    [
                        epoch_loss_sums[task.name] / max(1, epoch_rows[task.name])
                        for task in tasks
                    ]
                )
            ),
            "val_macro_auc": macro_auc,
        }
        for task in tasks:
            row[f"train_loss_{task.name}"] = epoch_loss_sums[task.name] / max(
                1, epoch_rows[task.name]
            )
            row[f"val_auc_{task.name}"] = validation_metrics[task.name]["auc"]
        history.append(row)
        if best_macro_auc is None or macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        _emit(
            progress_callback,
            "joint_epoch_end",
            epoch=epoch,
            val_macro_auc=macro_auc,
            best_epoch=best_epoch,
            best_val_macro_auc=best_macro_auc,
        )

    if config.restore_best_checkpoint and best_state is not None:
        model.load_state_dict(best_state, strict=True)
        model.to(device)
    train_metrics = evaluate_runtime_spn_joint(
        model,
        tasks,
        split="train",
        batch_size=config.batch_size,
        device=device,
        loss=config.loss,
    )
    validation_metrics = evaluate_runtime_spn_joint(
        model,
        tasks,
        split="validation",
        batch_size=config.batch_size,
        device=device,
        loss=config.loss,
    )
    adapter_gradient_means = {
        name: gradient_sums[name] / max(1, gradient_observations[name])
        for name in gradient_sums
    }
    return RuntimeSpnJointTrainingResult(
        history=history,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        metadata={
            "task_names": [task.name for task in tasks],
            "task_groups": {task.name: task.group for task in tasks},
            "task_weights": task_weights,
            "optimizer_steps": optimizer_steps,
            "task_batch_counts": total_task_batches,
            "iterator_wrap_counts": total_wraps,
            "best_epoch": best_epoch,
            "best_val_macro_auc": best_macro_auc,
            "selected_checkpoint": (
                "best" if config.restore_best_checkpoint else "last"
            ),
            "shared_state_dict_count": 1,
            "task_specific_trainable_state": False,
            "loss": config.loss,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "device": str(device),
        },
        router_traffic=total_router_traffic,
        gradient_diagnostics={
            "all_gradients_finite": all_gradients_finite,
            "adapter_gradient_mean_abs_sum": adapter_gradient_means,
            "adapter_gradient_observations": gradient_observations,
        },
    )


def evaluate_runtime_spn_joint(
    model: nn.Module,
    tasks: list[RuntimeSpnJointTask],
    *,
    split: str,
    batch_size: int,
    device: torch.device,
    loss: str,
) -> dict[str, dict[str, float]]:
    if split not in {"train", "validation"}:
        raise ValueError(
            "joint Runtime-SPN evaluation split must be train or validation"
        )
    loss_fn = make_loss(loss)
    model.eval()
    result: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for task in tasks:
            dataset = (
                task.train_dataset if split == "train" else task.validation_dataset
            )
            labels_all: list[float] = []
            probabilities: list[float] = []
            loss_sum = 0.0
            for features, labels in make_loader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            ):
                features = _to_runtime_coordinates(
                    features.to(device),
                    task.structure.block_bits,
                )
                labels = labels.to(device)
                logits = model(features, task.structure).squeeze(1)
                batch_loss = compute_loss(loss_fn, logits, labels, loss)
                loss_sum += float(batch_loss.detach().cpu()) * len(labels)
                labels_all.extend(float(value) for value in labels.detach().cpu())
                probabilities.extend(
                    float(value) for value in torch.sigmoid(logits).detach().cpu()
                )
            label_array = np.asarray(labels_all, dtype=np.float32)
            probability_array = np.asarray(probabilities, dtype=np.float32)
            predictions = (probability_array >= 0.5).astype(np.float32)
            best_accuracy, best_threshold = best_threshold_accuracy_and_threshold(
                label_array,
                probability_array,
            )
            result[task.name] = {
                "loss": loss_sum / max(1, len(label_array)),
                "accuracy": float(np.mean(predictions == label_array)),
                "auc": binary_auc(label_array, probability_array),
                "best_accuracy": best_accuracy,
                "calibrated_threshold": best_threshold,
                "rows": float(len(label_array)),
            }
    return result


def _to_runtime_coordinates(features: torch.Tensor, block_bits: int) -> torch.Tensor:
    pair_bits = 2 * block_bits
    if features.ndim != 2 or features.shape[1] <= 0 or features.shape[1] % pair_bits:
        raise ValueError("joint Runtime-SPN features must contain complete pairs")
    return features.reshape(features.shape[0], -1, 2, block_bits).flip(-1)


def _accumulate_router_traffic(
    destination: dict[str, float],
    current: dict[str, float],
) -> None:
    for name, value in current.items():
        destination[name] = destination.get(name, 0.0) + float(value)


def _accumulate_gradient_diagnostics(
    model: nn.Module,
    sums: dict[str, float],
    observations: dict[str, int],
) -> bool:
    finite = True
    for name, parameter in model.named_parameters():
        if (
            "primitive_adapter" not in name
            and "primitive_film" not in name
            and "typed_relation" not in name
        ) or parameter.grad is None:
            continue
        group = ".".join(name.split(".")[:2])
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all())
        sums[group] = sums.get(group, 0.0) + float(gradient.abs().sum().cpu())
        observations[group] = observations.get(group, 0) + 1
    return finite


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)


__all__ = [
    "RuntimeSpnJointTask",
    "RuntimeSpnJointTrainingResult",
    "evaluate_runtime_spn_joint",
    "train_runtime_spn_joint",
]
