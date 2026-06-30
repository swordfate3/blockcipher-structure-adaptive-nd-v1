from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "auto"
    optimizer: str = "adam"
    amsgrad: bool = False
    weight_decay: float = 0.0
    lr_scheduler: str = "none"
    max_learning_rate: float | None = None
    checkpoint_metric: str = "val_accuracy"
    restore_best_checkpoint: bool = False
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    loss: str = "bce"
    train_eval_interval: int = 1
    checkpoint_output: Path | str | None = None


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict[str, float]]
    final_metrics: dict[str, float]
    metadata: dict[str, Any]
