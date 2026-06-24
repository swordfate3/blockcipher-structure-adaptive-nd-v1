from __future__ import annotations

import torch
from torch import nn


class MlpDistinguisher(nn.Module):
    def __init__(self, input_bits: int, hidden_bits: int = 128) -> None:
        super().__init__()
        self.input_bits = input_bits
        self.network = nn.Sequential(
            nn.Linear(input_bits, hidden_bits),
            nn.ReLU(),
            nn.Linear(hidden_bits, hidden_bits),
            nn.ReLU(),
            nn.Linear(hidden_bits, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        return self.network(features.float())
