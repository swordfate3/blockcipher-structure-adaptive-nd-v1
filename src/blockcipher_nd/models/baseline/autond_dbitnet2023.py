from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def autond_dbitnet_dilations(input_bits: int) -> list[int]:
    if input_bits < 8:
        raise ValueError("AutoND DBitNet requires at least 8 input bits")
    dilations: list[int] = []
    width = input_bits
    while width >= 8:
        dilations.append(width // 2 - 1)
        width //= 2
    return dilations


class AutoNDDBitNet2023Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.wide = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=2,
            dilation=dilation,
        )
        self.wide_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.narrow = nn.Conv1d(out_channels, out_channels, kernel_size=2)
        self.output_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        wide = self.wide_norm(F.relu(self.wide(features)))
        narrow = F.relu(self.narrow(F.pad(wide, (1, 0))))
        return self.output_norm(narrow + wide)


class AutoNDDBitNet2023Distinguisher(nn.Module):
    """PyTorch reimplementation of the public AutoND DBitNet architecture."""

    def __init__(
        self,
        input_bits: int,
        initial_channels: int = 32,
        channel_increment: int = 16,
        l2_coefficient: float = 1e-5,
    ) -> None:
        super().__init__()
        self.input_bits = input_bits
        self.dilations = autond_dbitnet_dilations(input_bits)
        self.output_width = input_bits - sum(self.dilations)
        self.output_channels = initial_channels + channel_increment * (len(self.dilations) - 1)
        self.flattened_width = self.output_width * self.output_channels
        self.l2_coefficient = l2_coefficient
        self.last_auxiliary_loss = torch.tensor(0.0)

        blocks: list[nn.Module] = []
        in_channels = 1
        for index, dilation in enumerate(self.dilations):
            out_channels = initial_channels + channel_increment * index
            blocks.append(AutoNDDBitNet2023Block(in_channels, out_channels, dilation))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_width, 256),
            nn.BatchNorm1d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._reset_keras_parameters()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        hidden = ((features.float() - 0.5) / 0.5).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.classifier(hidden.flatten(start_dim=1))
        dense_kernels = [
            layer.weight.square().sum()
            for layer in self.classifier
            if isinstance(layer, nn.Linear)
        ]
        self.last_auxiliary_loss = self.l2_coefficient * torch.stack(dense_kernels).sum()
        return logits

    def _reset_keras_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
