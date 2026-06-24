from __future__ import annotations

from typing import Any

import torch
from torch import nn

from blockcipher_nd.models.structure.adaptive_dbitnet import (
    AdaptiveDBitNetDistinguisher,
    PairwiseAdaptiveDBitNetDistinguisher,
)
from blockcipher_nd.models.structure.spn import SpnTokenMixerPairSetDistinguisher
from blockcipher_nd.models.baseline.cnn import CnnDistinguisher
from blockcipher_nd.models.common.components import build_activation
from blockcipher_nd.models.baseline.dbitnet import DBitNetDistinguisher
from blockcipher_nd.models.baseline.mlp import MlpDistinguisher
from blockcipher_nd.models.baseline.multiscale_dense_resnet import (
    MultiScaleDenseResNetDistinguisher,
)
from blockcipher_nd.models.baseline.resnet_bitslice import ResNetBitSliceDistinguisher
from blockcipher_nd.models.baseline.senet_resnext import SeResNeXtDistinguisher


EXPERT_KEYS = (
    "resnet_bitslice",
    "dbitnet_dilated_cnn",
    "cnn",
    "mlp",
    "senet_resnext",
    "multiscale_dense_resnet",
)

V2_EXPERT_KEYS = (
    "resnet_bitslice",
    "adaptive_dbitnet",
    "cnn",
    "mlp",
    "senet_resnext",
    "multiscale_dense_resnet",
)

V3_EXPERT_KEYS = (
    "resnet_bitslice",
    "adaptive_dbitnet_pairwise",
    "cnn",
    "mlp",
    "senet_resnext",
    "multiscale_dense_resnet",
)

V4_EXPERT_KEYS = V3_EXPERT_KEYS

V5_EXPERT_KEYS = (
    "adaptive_dbitnet_pairwise",
    "spn_token_mixer_pairset",
    "resnet_bitslice",
    "senet_resnext",
    "multiscale_dense_resnet",
)

HARD_GATE_WEIGHTS = {
    "ARX": (0.35, 0.20, 0.05, 0.05, 0.10, 0.25),
    "SPN": (0.10, 0.30, 0.30, 0.05, 0.20, 0.05),
    "Feistel-like": (0.20, 0.35, 0.10, 0.05, 0.10, 0.20),
}

V5_HARD_GATE_WEIGHTS = {
    "ARX": (0.45, 0.05, 0.20, 0.10, 0.20),
    "SPN": (0.30, 0.45, 0.05, 0.15, 0.05),
    "Feistel-like": (0.40, 0.10, 0.20, 0.10, 0.20),
}


class IdentityStructureAdapter(nn.Module):
    name = "identity"

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.float()


class ArxWordMixAdapter(nn.Module):
    name = "arx_word_mix"

    def __init__(self, input_bits: int) -> None:
        super().__init__()
        self.rotation_a = max(1, input_bits // 4)
        self.rotation_b = max(1, input_bits // 8)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.float()
        return (
            x
            + torch.roll(x, shifts=self.rotation_a, dims=1)
            + torch.roll(x, shifts=self.rotation_b, dims=1)
        ) / 3.0


class SpnCellMixAdapter(nn.Module):
    name = "spn_cell_mix"

    def __init__(self, input_bits: int, cell_bits: int = 4) -> None:
        super().__init__()
        self.input_bits = input_bits
        self.cell_bits = cell_bits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.float()
        if self.input_bits % self.cell_bits != 0:
            return x
        cells = x.reshape(x.shape[0], self.input_bits // self.cell_bits, self.cell_bits)
        cell_context = cells.mean(dim=2, keepdim=True).expand_as(cells)
        return (0.75 * cells + 0.25 * cell_context).reshape_as(x)


class FeistelBranchMixAdapter(nn.Module):
    name = "feistel_branch_mix"

    def __init__(self, input_bits: int) -> None:
        super().__init__()
        self.input_bits = input_bits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.float()
        if self.input_bits % 2 != 0:
            return x
        left, right = x.chunk(2, dim=1)
        mixed_left = 0.75 * left + 0.25 * right
        mixed_right = 0.75 * right + 0.25 * left
        return torch.cat([mixed_left, mixed_right], dim=1)


class StructureAwareMoEDistinguisher(nn.Module):
    def __init__(
        self,
        input_bits: int,
        hidden_bits: int,
        structure_feature_bits: int,
        gate_mode: str,
        expert_set: str = "legacy",
        pair_bits: int | None = None,
        gate_hidden_bits: int | None = None,
        gate_activation: str = "relu",
        gate_dropout: float = 0.0,
        gate_temperature: float = 1.0,
        pairwise_pooling: str = "mean_max",
        spn_token_dim: int | None = None,
        spn_mixer_depth: int = 3,
        spn_token_mlp_ratio: int = 2,
        expert_activation: str = "gelu",
        expert_norm: str = "layernorm",
        spn_pooling: str = "attention_mean_max",
        expert_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if gate_mode not in {"uniform", "hard", "soft"}:
            raise ValueError(f"unsupported gate_mode: {gate_mode}")
        if expert_set not in {
            "legacy",
            "v2_adaptive",
            "v3_pairwise",
            "v4_structure_adapter",
            "v5_structure_experts",
        }:
            raise ValueError(f"unsupported expert_set: {expert_set}")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be > 0")
        if gate_dropout < 0 or expert_dropout < 0:
            raise ValueError("dropout values must be non-negative")
        self.input_bits = input_bits
        self.hidden_bits = hidden_bits
        self.structure_feature_bits = structure_feature_bits
        self.gate_mode = gate_mode
        self.expert_set = expert_set
        self.pair_bits = pair_bits
        self.gate_hidden_bits = gate_hidden_bits or hidden_bits
        self.gate_activation = gate_activation
        self.gate_dropout = gate_dropout
        self.gate_temperature = gate_temperature
        self.pairwise_pooling = pairwise_pooling
        self.spn_token_dim = spn_token_dim
        self.spn_mixer_depth = spn_mixer_depth
        self.spn_token_mlp_ratio = spn_token_mlp_ratio
        self.expert_activation = expert_activation
        self.expert_norm = expert_norm
        self.spn_pooling = spn_pooling
        self.expert_dropout = expert_dropout
        self.expert_keys = _expert_keys(expert_set)
        self.experts = nn.ModuleList(self._build_experts(input_bits, hidden_bits))
        self.adapters = nn.ModuleDict(
            {
                "identity": IdentityStructureAdapter(),
                "arx_word_mix": ArxWordMixAdapter(input_bits),
                "spn_cell_mix": SpnCellMixAdapter(input_bits),
                "feistel_branch_mix": FeistelBranchMixAdapter(input_bits),
            }
        )
        self.soft_gate = nn.Sequential(
            nn.Linear(structure_feature_bits, self.gate_hidden_bits),
            build_activation(gate_activation),
            nn.Dropout(gate_dropout),
            nn.Linear(self.gate_hidden_bits, len(self.expert_keys)),
        )
        self.register_buffer(
            "_structure_features",
            torch.zeros(structure_feature_bits, dtype=torch.float32),
        )

    def set_structure_features(self, structure_features: torch.Tensor) -> None:
        if structure_features.shape != (self.structure_feature_bits,):
            raise ValueError(
                "structure_features must have shape "
                f"({self.structure_feature_bits},), got {tuple(structure_features.shape)}"
            )
        self._structure_features.copy_(structure_features.detach().to(self._structure_features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted_features = self._adapt_features(features)
        expert_logits = torch.cat([expert(adapted_features) for expert in self.experts], dim=1)
        weights = self.current_gate_weights(batch_size=features.shape[0]).to(features.device)
        return (expert_logits * weights).sum(dim=1, keepdim=True)

    def current_gate_weights(self, batch_size: int) -> torch.Tensor:
        structure = self._structure_features.unsqueeze(0).expand(batch_size, -1)
        if self.gate_mode == "uniform":
            return torch.full(
                (batch_size, len(self.expert_keys)),
                1.0 / len(self.expert_keys),
                dtype=structure.dtype,
                device=structure.device,
            )
        if self.gate_mode == "hard":
            weights = torch.tensor(
                self._hard_weights_from_structure(),
                dtype=structure.dtype,
                device=structure.device,
            )
            return weights.unsqueeze(0).expand(batch_size, -1)
        return torch.softmax(self.soft_gate(structure) / self.gate_temperature, dim=1)

    def gate_summary(self) -> dict[str, Any]:
        weights = self.current_gate_weights(batch_size=1).detach().cpu()[0]
        return {
            "gate_mode": self.gate_mode,
            "expert_set": self.expert_set,
            "adapter_mode": "structure" if self.expert_set == "v4_structure_adapter" else "none",
            "adapter_name": self._adapter_name_from_structure(),
            "gate_hidden_bits": self.gate_hidden_bits,
            "gate_activation": self.gate_activation,
            "gate_dropout": round(float(self.gate_dropout), 6),
            "gate_temperature": round(float(self.gate_temperature), 6),
            "pairwise_pooling": self.pairwise_pooling,
            "spn_token_dim": self.spn_token_dim,
            "spn_mixer_depth": self.spn_mixer_depth,
            "spn_token_mlp_ratio": self.spn_token_mlp_ratio,
            "expert_activation": self.expert_activation,
            "expert_norm": self.expert_norm,
            "spn_pooling": self.spn_pooling,
            "expert_dropout": round(float(self.expert_dropout), 6),
            **{
                f"gate_weight_{key}": round(float(weight), 6)
                for key, weight in zip(self.expert_keys, weights)
            },
        }

    def _build_experts(self, input_bits: int, hidden_bits: int) -> list[nn.Module]:
        if self.expert_set == "v5_structure_experts":
            return [
                PairwiseAdaptiveDBitNetDistinguisher(
                    input_bits=input_bits,
                    pair_bits=self.pair_bits or 96,
                    base_channels=hidden_bits,
                    pooling=self.pairwise_pooling,
                ),
                SpnTokenMixerPairSetDistinguisher(
                    input_bits=input_bits,
                    pair_bits=self.pair_bits or 96,
                    base_channels=hidden_bits,
                    token_dim=self.spn_token_dim,
                    mixer_depth=self.spn_mixer_depth,
                    token_mlp_ratio=self.spn_token_mlp_ratio,
                    activation=self.expert_activation,
                    norm=self.expert_norm,
                    pooling=self.spn_pooling,
                    dropout=self.expert_dropout,
                ),
                ResNetBitSliceDistinguisher(input_bits=input_bits, channels=hidden_bits),
                SeResNeXtDistinguisher(input_bits=input_bits, channels=hidden_bits),
                MultiScaleDenseResNetDistinguisher(
                    input_bits=input_bits,
                    channels=hidden_bits,
                ),
            ]
        return [
            ResNetBitSliceDistinguisher(input_bits=input_bits, channels=hidden_bits),
            self._build_dbitnet_expert(input_bits, hidden_bits),
            CnnDistinguisher(input_bits=input_bits, channels=hidden_bits),
            MlpDistinguisher(input_bits=input_bits, hidden_bits=hidden_bits),
            SeResNeXtDistinguisher(input_bits=input_bits, channels=hidden_bits),
            MultiScaleDenseResNetDistinguisher(
                input_bits=input_bits,
                channels=hidden_bits,
            ),
        ]

    def _build_dbitnet_expert(self, input_bits: int, hidden_bits: int) -> nn.Module:
        if self.expert_set in {"v3_pairwise", "v4_structure_adapter"}:
            return PairwiseAdaptiveDBitNetDistinguisher(
                input_bits=input_bits,
                pair_bits=self.pair_bits or 96,
                base_channels=hidden_bits,
            )
        if self.expert_set == "v2_adaptive":
            return AdaptiveDBitNetDistinguisher(
                input_bits=input_bits,
                base_channels=hidden_bits,
            )
        return DBitNetDistinguisher(input_bits=input_bits, channels=hidden_bits)

    def _adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.expert_set != "v4_structure_adapter":
            return features
        return self.adapters[self._adapter_name_from_structure()](features)

    def _adapter_name_from_structure(self) -> str:
        if self.expert_set != "v4_structure_adapter":
            return "identity"
        is_arx = bool(self._structure_features[0].item())
        is_spn = bool(self._structure_features[1].item())
        is_feistel_like = bool(self._structure_features[2].item())
        if is_arx:
            return "arx_word_mix"
        if is_spn:
            return "spn_cell_mix"
        if is_feistel_like:
            return "feistel_branch_mix"
        return "identity"

    def _hard_weights_from_structure(self) -> tuple[float, ...]:
        is_arx = bool(self._structure_features[0].item())
        is_spn = bool(self._structure_features[1].item())
        is_feistel_like = bool(self._structure_features[2].item())
        if self.expert_set == "v5_structure_experts":
            if is_arx:
                return V5_HARD_GATE_WEIGHTS["ARX"]
            if is_spn:
                return V5_HARD_GATE_WEIGHTS["SPN"]
            if is_feistel_like:
                return V5_HARD_GATE_WEIGHTS["Feistel-like"]
            return tuple(1.0 / len(self.expert_keys) for _ in self.expert_keys)
        if is_arx:
            return HARD_GATE_WEIGHTS["ARX"]
        if is_spn:
            return HARD_GATE_WEIGHTS["SPN"]
        if is_feistel_like:
            return HARD_GATE_WEIGHTS["Feistel-like"]
        return tuple(1.0 / len(self.expert_keys) for _ in self.expert_keys)


def _expert_keys(expert_set: str) -> tuple[str, ...]:
    if expert_set == "v2_adaptive":
        return V2_EXPERT_KEYS
    if expert_set == "v3_pairwise":
        return V3_EXPERT_KEYS
    if expert_set == "v4_structure_adapter":
        return V4_EXPERT_KEYS
    if expert_set == "v5_structure_experts":
        return V5_EXPERT_KEYS
    return EXPERT_KEYS
