from __future__ import annotations

import torch
from torch import nn


STATE_BITS = 64
TEXTS_PER_MULTISET = 16
VIEWS_PER_MULTISET = 2


def integral_input_bits(multiset_count: int) -> int:
    if multiset_count < 1:
        raise ValueError("multiset_count must be positive")
    return multiset_count * VIEWS_PER_MULTISET * TEXTS_PER_MULTISET * STATE_BITS


class IntegralMbConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        expanded = channels * 2
        self.expand = nn.Sequential(
            nn.Conv2d(channels, expanded, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU(),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded,
                expanded,
                kernel_size=3,
                padding=1,
                groups=expanded,
                bias=False,
            ),
            nn.BatchNorm2d(expanded),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.project = nn.Sequential(
            nn.Conv2d(expanded, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.project(self.depthwise(self.expand(features)))
        return self.activation(features + hidden)


class PresentIntegralPaperMbconvAnchor(nn.Module):
    """Auditable PyTorch approximation of the Wu/Guo paper-family network."""

    def __init__(
        self,
        *,
        multiset_count: int,
        base_channels: int = 16,
        head_bits: int = 64,
        block_count: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if base_channels < 1 or head_bits < 1 or block_count < 1:
            raise ValueError("network widths and block_count must be positive")
        self.multiset_count = multiset_count
        self.input_bits = integral_input_bits(multiset_count)
        self.paper_tensor_concat_assumption = "spatial_axis_1"
        self.stem = nn.Sequential(
            nn.Conv2d(8, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[
                IntegralMbConvBlock(base_channels, dropout=dropout)
                for _ in range(block_count)
            ]
        )
        flattened = (
            base_channels * TEXTS_PER_MULTISET * (TEXTS_PER_MULTISET * multiset_count)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, head_bits),
            nn.BatchNorm1d(head_bits),
            nn.ReLU(),
            nn.Linear(head_bits, head_bits),
            nn.BatchNorm1d(head_bits),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_bits, head_bits),
            nn.BatchNorm1d(head_bits),
            nn.ReLU(),
            nn.Linear(head_bits, 1),
        )

    def paper_tensor_view(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected [batch,{self.input_bits}], got {tuple(features.shape)}"
            )
        # The paper specifies [16,16,8] for one multiset but not the two-
        # multiset join. Preserve Eq. 6 order and extend the second spatial axis.
        return (
            features.float()
            .reshape(
                features.shape[0],
                TEXTS_PER_MULTISET,
                TEXTS_PER_MULTISET * self.multiset_count,
                8,
            )
            .permute(0, 3, 1, 2)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.blocks(self.stem(self.paper_tensor_view(features))))


class IntegralGridResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


class PresentIntegralStructuredResidualCandidate(nn.Module):
    """Preserves multiset, view, text, nibble, and bit axes explicitly."""

    def __init__(
        self,
        *,
        multiset_count: int,
        base_channels: int = 16,
        head_bits: int = 64,
        block_count: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if base_channels < 1 or head_bits < 1 or block_count < 1:
            raise ValueError("network widths and block_count must be positive")
        self.multiset_count = multiset_count
        self.input_bits = integral_input_bits(multiset_count)
        semantic_channels = multiset_count * VIEWS_PER_MULTISET * 4
        self.stem = nn.Sequential(
            nn.Conv2d(semantic_channels, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[
                IntegralGridResidualBlock(base_channels, dropout=dropout)
                for _ in range(block_count)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 2, head_bits),
            nn.LayerNorm(head_bits),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_bits, 1),
        )

    def structured_tensor_view(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected [batch,{self.input_bits}], got {tuple(features.shape)}"
            )
        semantic = features.float().reshape(
            features.shape[0],
            self.multiset_count,
            VIEWS_PER_MULTISET,
            TEXTS_PER_MULTISET,
            16,
            4,
        )
        return semantic.permute(0, 1, 2, 5, 3, 4).reshape(
            features.shape[0],
            self.multiset_count * VIEWS_PER_MULTISET * 4,
            TEXTS_PER_MULTISET,
            16,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.stem(self.structured_tensor_view(features)))
        pooled = torch.cat(
            [hidden.mean(dim=(2, 3)), hidden.amax(dim=(2, 3))],
            dim=1,
        )
        return self.classifier(pooled)


class PresentIntegralFlatLinear(nn.Module):
    def __init__(self, *, multiset_count: int) -> None:
        super().__init__()
        self.input_bits = integral_input_bits(multiset_count)
        self.linear = nn.Linear(self.input_bits, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected [batch,{self.input_bits}], got {tuple(features.shape)}"
            )
        return self.linear(features.float())


__all__ = [
    "PresentIntegralFlatLinear",
    "PresentIntegralPaperMbconvAnchor",
    "PresentIntegralStructuredResidualCandidate",
    "integral_input_bits",
]
