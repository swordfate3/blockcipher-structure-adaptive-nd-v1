from __future__ import annotations

from typing import Any

import torch
from torch import nn


FCNN_MODEL_NAME = "speck32_full32_fcnn_true_output"
BILSTM_MODEL_NAME = "speck32_full32_bilstm_true_output"
JEONG_2024_DOI = "10.3390/math12131936"


class Speck32JeongFcnn(nn.Module):
    """Jeong-2024-family FCNN for 32-bit SPECK output prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_flat_features(features)
        return self.network(features.float())


class Speck32JeongBiLstm(nn.Module):
    """Jeong-2024-family three-layer BiLSTM over two SPECK words."""

    def __init__(self, hidden_size: int = 256, layers: int = 3) -> None:
        super().__init__()
        if hidden_size <= 0 or layers <= 0:
            raise ValueError("hidden_size and layers must be positive")
        self.hidden_size = hidden_size
        self.layers = layers
        self.encoder = nn.LSTM(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        words = split_speck32_words(features)
        _, (hidden, _) = self.encoder(words)
        top_forward = hidden[-2]
        top_backward = hidden[-1]
        return self.output(torch.cat((top_forward, top_backward), dim=1))


def split_speck32_words(features: torch.Tensor) -> torch.Tensor:
    _validate_flat_features(features)
    return features.float().reshape(features.shape[0], 2, 16)


def jeong_anchor_protocols() -> dict[str, dict[str, Any]]:
    shared = {
        "paper": (
            "Jeong et al., Comprehensive Neural Cryptanalysis on Block Ciphers "
            "Using Different Encryption Methods"
        ),
        "doi": JEONG_2024_DOI,
        "paper_family": "Jeong 2024 encryption emulation",
        "paper_exact_reproduction": False,
        "task": "innovation2_output_prediction",
        "cipher": "SPECK32/64",
        "input": "32_msb_first_plaintext_bits",
        "target": "32_msb_first_true_speck32_ciphertext_bits",
        "sample_classification": False,
        "output_activation": "sigmoid",
        "planned_loss": "binary_cross_entropy",
        "planned_optimizer": "adamw",
        "planned_learning_rate": 1e-3,
        "paper_unspecified_shared_details": [
            "weight_initialization",
            "checkpoint_selection",
        ],
    }
    return {
        FCNN_MODEL_NAME: {
            **shared,
            "architecture": "fully_connected",
            "input_shape": ["batch", 32],
            "hidden_widths": [512, 1024, 512],
            "hidden_activation": "relu",
            "batch_normalization_after_hidden_layers": True,
            "dropout": "not_reported_and_omitted",
            "implementation_approximation": (
                "paper architecture fields are implemented; unreported shared "
                "training details remain explicit"
            ),
        },
        BILSTM_MODEL_NAME: {
            **shared,
            "architecture": "bidirectional_lstm",
            "input_shape": ["batch", 2, 16],
            "word_order": ["x_msw", "y_lsw"],
            "layers": 3,
            "hidden_size_per_direction": 256,
            "sequence_reduction": "top_layer_final_forward_backward_hidden_concat",
            "dropout": "not_reported_and_omitted",
            "paper_unspecified_model_details": [
                "bidirectional_sequence_reduction_before_output_layer",
            ],
            "implementation_approximation": (
                "paper fields are implemented; final top-layer forward/backward "
                "hidden concatenation resolves an unreported reduction detail"
            ),
        },
    }


def parameter_counts() -> dict[str, int]:
    models = {
        FCNN_MODEL_NAME: Speck32JeongFcnn(),
        BILSTM_MODEL_NAME: Speck32JeongBiLstm(),
    }
    return {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }


def _validate_flat_features(features: torch.Tensor) -> None:
    if features.ndim != 2 or features.shape[1] != 32:
        raise ValueError(f"expected [batch, 32] input, got {tuple(features.shape)}")


__all__ = [
    "BILSTM_MODEL_NAME",
    "FCNN_MODEL_NAME",
    "JEONG_2024_DOI",
    "Speck32JeongBiLstm",
    "Speck32JeongFcnn",
    "jeong_anchor_protocols",
    "parameter_counts",
    "split_speck32_words",
]
