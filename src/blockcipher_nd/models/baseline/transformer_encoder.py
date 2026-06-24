from __future__ import annotations

import torch
from torch import nn


class TransformerEncoderDistinguisher(nn.Module):
    """High-cost global-attention ablation over bit-pair sequences."""

    def __init__(self, input_bits: int, hidden_bits: int = 32, heads: int = 2) -> None:
        super().__init__()
        if input_bits % 2 != 0:
            raise ValueError(
                "TransformerEncoderDistinguisher requires an even number of input bits"
            )
        if hidden_bits % heads != 0:
            hidden_bits += heads - (hidden_bits % heads)
        self.input_bits = input_bits
        self.sequence_bits = input_bits // 2
        self.embedding = nn.Linear(2, hidden_bits)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_bits,
            nhead=heads,
            dim_feedforward=2 * hidden_bits,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(hidden_bits, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        batch = features.float().reshape(features.shape[0], 2, self.sequence_bits)
        sequence = batch.transpose(1, 2)
        hidden = self.encoder(self.embedding(sequence))
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)
