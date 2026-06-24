from __future__ import annotations

import math

import torch
from torch import nn

from blockcipher_nd.training.types import TrainingConfig


class Lion(torch.optim.Optimizer):
    """Small local Lion optimizer implementation for HPO experiments."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)
                state = self.state[parameter]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exp_avg = state["exp_avg"]
                update = exp_avg * beta1 + grad * (1.0 - beta1)
                parameter.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


class OfficialEpochCyclicLR:
    """Zhang/Wang-style epoch LR cycle: high to low over a fixed epoch window."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_lr: float,
        max_lr: float,
        cycle_epochs: int = 10,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.max_lr = float(max_lr)
        self.cycle_epochs = max(1, int(cycle_epochs))

    def step_epoch(self, epoch: int) -> None:
        position = (max(1, epoch) - 1) % self.cycle_epochs
        if self.cycle_epochs == 1:
            lr = self.base_lr
        else:
            fraction = position / float(self.cycle_epochs - 1)
            lr = self.max_lr - fraction * (self.max_lr - self.base_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> None:
        return None


def make_loss(loss: str) -> nn.Module:
    if loss == "bce":
        return nn.BCEWithLogitsLoss()
    if loss == "mse":
        return nn.MSELoss()
    raise ValueError(f"unsupported loss: {loss}")


def compute_loss(loss_fn: nn.Module, logits: torch.Tensor, labels: torch.Tensor, loss: str) -> torch.Tensor:
    if loss == "mse":
        return loss_fn(torch.sigmoid(logits), labels)
    return loss_fn(logits, labels)


def make_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
        )
    if config.optimizer == "lion":
        return Lion(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {config.optimizer}")


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    train_size: int,
) -> torch.optim.lr_scheduler.LRScheduler | OfficialEpochCyclicLR | None:
    if config.lr_scheduler == "none":
        return None
    if config.lr_scheduler == "official_cyclic":
        return OfficialEpochCyclicLR(
            optimizer,
            base_lr=config.learning_rate,
            max_lr=config.max_learning_rate or config.learning_rate * 20.0,
            cycle_epochs=10,
        )
    if config.lr_scheduler == "cyclic":
        steps_per_epoch = max(1, (train_size + config.batch_size - 1) // config.batch_size)
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.learning_rate,
            max_lr=config.max_learning_rate or config.learning_rate * 10.0,
            step_size_up=max(1, steps_per_epoch // 2),
            cycle_momentum=False,
        )
    if config.lr_scheduler == "cosine_warmup":
        steps_per_epoch = max(1, (train_size + config.batch_size - 1) // config.batch_size)
        total_steps = max(1, config.epochs * steps_per_epoch)
        warmup_steps = max(1, min(total_steps // 10, steps_per_epoch))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"unsupported lr scheduler: {config.lr_scheduler}")


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
