from __future__ import annotations

from typing import Any

import torch
from torch import nn

from blockcipher_nd.ciphers.feistel.des import DES_E, DES_FP, DES_IP, DES_P


FCNN_MODEL_NAME = "des_full64_fcnn_true_output"
BILSTM_MODEL_NAME = "des_full64_bilstm_true_output"
FEISTEL_MODEL_NAME = "des_full64_feistel_recurrent_true_output"
WRONG_F_MODEL_NAME = "des_full64_wrong_f_branch_recurrent_true_output"
JEONG_2024_DOI = "10.3390/math12131936"
FCNN_PARAMETER_COUNT = 1_120_320
BILSTM_PARAMETER_COUNT = 3_780_672
FEISTEL_CHANNELS = 276
FEISTEL_R2_PARAMETER_COUNT = 3_877_312
FEISTEL_R3_PARAMETER_COUNT = 3_890_560


class DesJeongFcnn(nn.Module):
    """Jeong-2024-family FCNN for full DES ciphertext prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_flat_features(features)
        return self.network(features.float())


class DesJeongBiLstm(nn.Module):
    """Jeong-2024-family three-layer BiLSTM over two DES halves."""

    def __init__(self, hidden_size: int = 256, layers: int = 3) -> None:
        super().__init__()
        if hidden_size <= 0 or layers <= 0:
            raise ValueError("hidden_size and layers must be positive")
        self.hidden_size = hidden_size
        self.layers = layers
        self.encoder = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        halves = split_des_halves(features)
        _, (hidden, _) = self.encoder(halves)
        top_forward = hidden[-2]
        top_backward = hidden[-1]
        return self.output(torch.cat((top_forward, top_backward), dim=1))


class _SharedDesFBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        cell_input = channels * 6
        cell_output = channels * 4
        self.register_buffer(
            "expanded_source_for_destination",
            torch.tensor([position - 1 for position in DES_E], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "p_source_for_destination",
            torch.tensor([position - 1 for position in DES_P], dtype=torch.long),
            persistent=False,
        )
        self.local_norm = nn.LayerNorm(channels)
        self.local_mlp = nn.Sequential(
            nn.Linear(cell_input, cell_output),
            nn.GELU(),
            nn.Linear(cell_output, cell_output),
        )
        self.global_norm = nn.LayerNorm(channels)
        self.global_mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(
        self,
        branch: torch.Tensor,
        round_context: torch.Tensor,
    ) -> torch.Tensor:
        expanded = self.local_norm(branch).index_select(
            1,
            self.expanded_source_for_destination,
        )
        expanded = expanded + round_context
        batch, _, channels = expanded.shape
        substituted = self.local_mlp(expanded.reshape(batch, 8, channels * 6))
        substituted = substituted.reshape(batch, 32, channels)
        routed = substituted.index_select(1, self.p_source_for_destination)
        return routed + self.global_mlp(self.global_norm(routed))


class _SharedDesXorMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels * 2)
        self.network = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, left: torch.Tensor, f_output: torch.Tensor) -> torch.Tensor:
        operands = torch.cat((left, f_output), dim=2)
        return self.network(self.norm(operands))


class DesFeistelRecurrent(nn.Module):
    """Shared round-recurrent DES predictor with a controlled F-input branch."""

    def __init__(
        self,
        *,
        rounds: int = 2,
        channels: int = FEISTEL_CHANNELS,
        f_input_branch: str = "right",
    ) -> None:
        super().__init__()
        if rounds not in {2, 3}:
            raise ValueError("FEISTEL1 supports exactly two or three DES rounds")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if f_input_branch not in {"right", "left"}:
            raise ValueError("f_input_branch must be right or left")
        self.rounds = rounds
        self.channels = channels
        self.f_input_branch = f_input_branch
        self.register_buffer(
            "ip_source_for_destination",
            torch.tensor([position - 1 for position in DES_IP], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "fp_source_for_destination",
            torch.tensor([position - 1 for position in DES_FP], dtype=torch.long),
            persistent=False,
        )
        self.embedding = nn.Linear(1, channels)
        self.position_embedding = nn.Parameter(torch.empty(1, 64, channels))
        self.round_contexts = nn.Parameter(torch.empty(rounds, 48, channels))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.round_contexts, std=0.02)
        self.f_block = _SharedDesFBlock(channels)
        self.xor_mixer = _SharedDesXorMixer(channels)
        self.heads = nn.ModuleList(nn.Linear(channels, 1) for _ in range(64))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_flat_features(features)
        hidden = self.embedding(features.float().unsqueeze(-1))
        hidden = hidden + self.position_embedding
        permuted = hidden.index_select(1, self.ip_source_for_destination)
        left, right = permuted[:, :32, :], permuted[:, 32:, :]
        for round_index in range(self.rounds):
            branch = right if self.f_input_branch == "right" else left
            f_output = self.f_block(branch, self.round_contexts[round_index])
            left, right = right, self.xor_mixer(left, f_output)
        preoutput = torch.cat((right, left), dim=1)
        ciphertext_order = preoutput.index_select(1, self.fp_source_for_destination)
        logits = torch.cat(
            [
                head(ciphertext_order[:, index, :])
                for index, head in enumerate(self.heads)
            ],
            dim=1,
        )
        return torch.sigmoid(logits)


def split_des_halves(features: torch.Tensor) -> torch.Tensor:
    _validate_flat_features(features)
    return features.float().reshape(features.shape[0], 2, 32)


def build_des_feistel_recurrent(
    f_input_branch: str,
    *,
    rounds: int = 2,
    channels: int = FEISTEL_CHANNELS,
) -> DesFeistelRecurrent:
    return DesFeistelRecurrent(
        rounds=rounds,
        channels=channels,
        f_input_branch=f_input_branch,
    )


def des_output_model_protocols() -> dict[str, dict[str, Any]]:
    shared = {
        "task": "innovation2_output_prediction",
        "cipher": "DES",
        "input": "64_msb_first_plaintext_bits",
        "target": "64_msb_first_true_round_reduced_des_ciphertext_bits",
        "sample_classification": False,
        "output_activation": "sigmoid",
        "planned_loss": "binary_cross_entropy",
        "planned_optimizer": "adamw",
        "planned_learning_rate": 1e-3,
    }
    paper = {
        **shared,
        "paper": (
            "Jeong et al., Comprehensive Neural Cryptanalysis on Block Ciphers "
            "Using Different Encryption Methods"
        ),
        "doi": JEONG_2024_DOI,
        "paper_family": "Jeong 2024 encryption emulation",
        "paper_exact_reproduction": False,
        "paper_unspecified_shared_details": [
            "weight_initialization",
            "checkpoint_selection",
        ],
    }
    return {
        FCNN_MODEL_NAME: {
            **paper,
            "architecture": "fully_connected",
            "input_shape": ["batch", 64],
            "hidden_widths": [512, 1024, 512],
            "implementation_approximation": (
                "four dense layers follow the paper family; unreported DES widths "
                "and shared training details remain explicit"
            ),
        },
        BILSTM_MODEL_NAME: {
            **paper,
            "architecture": "bidirectional_lstm",
            "input_shape": ["batch", 2, 32],
            "half_order": ["left_msb", "right_lsb"],
            "layers": 3,
            "hidden_size_per_direction": 256,
            "sequence_reduction": "top_layer_final_forward_backward_hidden_concat",
            "implementation_approximation": (
                "reported layers and hidden size are implemented; sequence "
                "reduction and unreported training details remain explicit"
            ),
        },
        FEISTEL_MODEL_NAME: {
            **shared,
            "architecture": "shared_feistel_recurrent",
            "public_input_permutation": "DES_IP",
            "f_input_branch": "right",
            "shared_round_body": True,
            "public_output_permutation": "swap_then_DES_FP",
            "channels": FEISTEL_CHANNELS,
        },
        WRONG_F_MODEL_NAME: {
            **shared,
            "architecture": "shared_feistel_recurrent",
            "public_input_permutation": "DES_IP",
            "f_input_branch": "left",
            "shared_round_body": True,
            "public_output_permutation": "swap_then_DES_FP",
            "channels": FEISTEL_CHANNELS,
        },
    }


def parameter_counts(*, rounds: int = 2) -> dict[str, int]:
    models = {
        FCNN_MODEL_NAME: DesJeongFcnn(),
        BILSTM_MODEL_NAME: DesJeongBiLstm(),
        FEISTEL_MODEL_NAME: build_des_feistel_recurrent("right", rounds=rounds),
        WRONG_F_MODEL_NAME: build_des_feistel_recurrent("left", rounds=rounds),
    }
    return {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }


def _validate_flat_features(features: torch.Tensor) -> None:
    if features.ndim != 2 or features.shape[1] != 64:
        raise ValueError(f"expected [batch, 64] input, got {tuple(features.shape)}")


__all__ = [
    "BILSTM_MODEL_NAME",
    "BILSTM_PARAMETER_COUNT",
    "DesFeistelRecurrent",
    "DesJeongBiLstm",
    "DesJeongFcnn",
    "FCNN_MODEL_NAME",
    "FCNN_PARAMETER_COUNT",
    "FEISTEL_CHANNELS",
    "FEISTEL_MODEL_NAME",
    "FEISTEL_R2_PARAMETER_COUNT",
    "FEISTEL_R3_PARAMETER_COUNT",
    "JEONG_2024_DOI",
    "WRONG_F_MODEL_NAME",
    "build_des_feistel_recurrent",
    "des_output_model_protocols",
    "parameter_counts",
    "split_des_halves",
]
