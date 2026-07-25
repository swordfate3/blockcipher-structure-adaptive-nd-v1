from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.training.runtime_spn_joint import (
    RuntimeSpnJointTask,
    train_runtime_spn_joint,
)
from blockcipher_nd.training.types import TrainingConfig


class _TinySharedDistinguisher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.representation = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 1)

    def forward(self, pairs: torch.Tensor, _structure: object) -> torch.Tensor:
        cells = pairs.mean(dim=(1, 2))
        return self.classifier(torch.relu(self.representation(cells)))


def test_representation_l2_equalized_joint_training_is_finite_and_recorded() -> None:
    structure = SimpleNamespace(block_bits=4)
    tasks = [
        RuntimeSpnJointTask(
            name="small",
            group="source",
            structure=structure,
            train_dataset=_dataset(scale=1.0),
            validation_dataset=_dataset(scale=1.0),
        ),
        RuntimeSpnJointTask(
            name="large",
            group="source",
            structure=structure,
            train_dataset=_dataset(scale=8.0),
            validation_dataset=_dataset(scale=8.0),
        ),
    ]
    result = train_runtime_spn_joint(
        _TinySharedDistinguisher(),
        tasks,
        _training_config(),
        gradient_combination="representation_l2_equalized",
    )

    assert result.metadata["gradient_combination"] == "representation_l2_equalized"
    assert result.metadata["optimizer_steps"] == 2
    assert result.gradient_diagnostics["all_gradients_finite"] is True
    assert set(result.gradient_diagnostics["task_gradient_scale_mean"]) == {
        "small",
        "large",
    }
    assert result.gradient_diagnostics["task_gradient_scale_mean"]["small"] != (
        result.gradient_diagnostics["task_gradient_scale_mean"]["large"]
    )
    assert set(result.validation_metrics) == {"small", "large"}


def test_mean_loss_remains_the_default_gradient_combination() -> None:
    structure = SimpleNamespace(block_bits=4)
    task = RuntimeSpnJointTask(
        name="only",
        group="source",
        structure=structure,
        train_dataset=_dataset(scale=1.0),
        validation_dataset=_dataset(scale=1.0),
    )

    result = train_runtime_spn_joint(
        _TinySharedDistinguisher(),
        [task],
        _training_config(),
    )

    assert result.metadata["gradient_combination"] == "mean_loss"
    assert result.gradient_diagnostics["task_gradient_scale_observations"] == {
        "only": 0
    }


def _dataset(*, scale: float) -> DifferentialDataset:
    features = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [scale, 0, 0, 0, scale, 0, 0, 0],
            [0, scale, 0, 0, 0, scale, 0, 0],
            [scale, scale, 0, 0, 0, 0, scale, scale],
        ],
        dtype=np.float32,
    )
    return DifferentialDataset(
        features=features,
        labels=np.asarray([0, 1, 1, 0], dtype=np.float32),
        metadata={"negative_mode": "encrypted_random_plaintexts"},
    )


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        seed=0,
        device="cpu",
        optimizer="adam",
        weight_decay=1e-5,
        lr_scheduler="none",
        checkpoint_metric="val_macro_auc",
        restore_best_checkpoint=True,
        loss="mse",
    )
