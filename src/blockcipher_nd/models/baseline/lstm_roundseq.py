from __future__ import annotations

import torch
from torch import nn


class LstmRoundSeqDistinguisher(nn.Module):
    """Sequence baseline over paired ciphertext bit positions."""

    def __init__(self, input_bits: int, hidden_bits: int = 32) -> None:
        super().__init__()
        if input_bits % 2 != 0:
            raise ValueError(
                "LstmRoundSeqDistinguisher requires an even number of input bits"
            )
        self.input_bits = input_bits
        self.sequence_bits = input_bits // 2
        self.encoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden_bits,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(2 * hidden_bits, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        batch = features.float().reshape(features.shape[0], 2, self.sequence_bits)
        sequence = batch.transpose(1, 2)
        _, (hidden, _) = self.encoder(sequence)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(last_hidden)
