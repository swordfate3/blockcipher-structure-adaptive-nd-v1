from __future__ import annotations

from typing import Any

import torch
from torch import nn

from blockcipher_nd.tasks.innovation2.speck32_output_prediction_models import (
    BILSTM_PARAMETER_COUNT,
)


ROTATION_CARRY_MODEL_NAME = "speck32_full32_rotation_carry_true_output"
WRONG_ROTATION_CARRY_MODEL_NAME = "speck32_full32_wrong_rotation_carry_true_output"
ROTATION_CARRY_SHUFFLE_MODEL_NAME = "speck32_full32_rotation_carry_label_shuffle"


class Speck32RotationCarryPredictor(nn.Module):
    def __init__(
        self,
        *,
        channels: int = 400,
        steps: int = 3,
        rotate_x_right: int = 7,
        rotate_y_left: int = 2,
    ) -> None:
        super().__init__()
        if channels <= 0 or steps <= 0:
            raise ValueError("channels and steps must be positive")
        if not 1 <= rotate_x_right < 16 or not 1 <= rotate_y_left < 16:
            raise ValueError("SPECK rotation constants must be in 1..15")
        self.channels = channels
        self.steps = steps
        self.rotate_x_right = rotate_x_right
        self.rotate_y_left = rotate_y_left
        self.input_projection = nn.Linear(1, channels)
        self.position_embedding = nn.Parameter(torch.empty(32, channels))
        self.round_key_context = nn.Parameter(torch.empty(steps, 16, channels))
        self.carry_scan = nn.GRU(
            input_size=channels * 2,
            hidden_size=channels,
            batch_first=True,
        )
        self.addition_mixer = nn.Sequential(
            nn.Linear(channels * 3, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.xor_mixer = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.x_normalization = nn.LayerNorm(channels)
        self.y_normalization = nn.LayerNorm(channels)
        self.output_weight = nn.Parameter(torch.empty(32, channels))
        self.output_bias = nn.Parameter(torch.zeros(32))
        self._reset_explicit_parameters()

    @classmethod
    def correct(cls, *, channels: int = 400) -> Speck32RotationCarryPredictor:
        return cls(
            channels=channels,
            steps=3,
            rotate_x_right=7,
            rotate_y_left=2,
        )

    @classmethod
    def wrong_rotation(cls, *, channels: int = 400) -> Speck32RotationCarryPredictor:
        return cls(
            channels=channels,
            steps=3,
            rotate_x_right=5,
            rotate_y_left=6,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 32:
            raise ValueError(f"expected [batch, 32] input, got {tuple(features.shape)}")
        tokens = self.input_projection(features.float().unsqueeze(-1))
        tokens = tokens + self.position_embedding.unsqueeze(0)
        x_tokens = tokens[:, :16]
        y_tokens = tokens[:, 16:]
        for step in range(self.steps):
            rotated_x = rotate_word_tokens(
                x_tokens,
                amount=self.rotate_x_right,
                direction="right",
            )
            carry_input = torch.cat((rotated_x, y_tokens), dim=-1)
            reversed_carry_input = torch.flip(carry_input, dims=(1,))
            reversed_carry, _ = self.carry_scan(reversed_carry_input)
            carry = torch.flip(reversed_carry, dims=(1,))
            addition = self.addition_mixer(
                torch.cat((rotated_x, y_tokens, carry), dim=-1)
            )
            x_tokens = self.x_normalization(
                addition + self.round_key_context[step].unsqueeze(0)
            )
            rotated_y = rotate_word_tokens(
                y_tokens,
                amount=self.rotate_y_left,
                direction="left",
            )
            y_tokens = self.y_normalization(
                self.xor_mixer(torch.cat((rotated_y, x_tokens), dim=-1))
            )
        output_tokens = torch.cat((x_tokens, y_tokens), dim=1)
        logits = torch.einsum("bpc,pc->bp", output_tokens, self.output_weight)
        logits = logits + self.output_bias.unsqueeze(0)
        return torch.sigmoid(logits)

    def _reset_explicit_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.round_key_context, std=0.02)
        nn.init.normal_(self.output_weight, std=0.02)


def rotate_word_tokens(
    tokens: torch.Tensor,
    *,
    amount: int,
    direction: str,
) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] != 16:
        raise ValueError(
            f"expected [batch, 16, channels] word tokens, got {tuple(tokens.shape)}"
        )
    if not 1 <= amount < 16:
        raise ValueError("word rotation amount must be in 1..15")
    if direction == "right":
        shift = amount
    elif direction == "left":
        shift = -amount
    else:
        raise ValueError("rotation direction must be right or left")
    return torch.roll(tokens, shifts=shift, dims=1)


def rotation_carry_protocols(channels: int = 400) -> dict[str, dict[str, Any]]:
    correct = Speck32RotationCarryPredictor.correct(channels=channels)
    parameters = sum(parameter.numel() for parameter in correct.parameters())
    parameter_ratio = parameters / BILSTM_PARAMETER_COUNT
    shared = {
        "task": "innovation2_output_prediction",
        "cipher": "SPECK32/64",
        "rounds": 3,
        "input": "32_msb_first_plaintext_bits_as_x_msw_then_y_lsw",
        "target": "32_msb_first_true_speck32_ciphertext_bits",
        "sample_classification": False,
        "architecture": "shared_three_step_rotation_carry_recurrence",
        "channels": channels,
        "steps": 3,
        "carry_direction": "lsb_to_msb_via_reversed_msb_first_tokens",
        "round_key_context": "three_step_specific_16_position_x_word_contexts",
        "output_head": "32_position_bound_weights",
        "parameters": parameters,
        "bilstm_anchor_parameters": BILSTM_PARAMETER_COUNT,
        "parameter_ratio_to_bilstm": parameter_ratio,
        "within_bilstm_five_percent": 0.95 <= parameter_ratio <= 1.05,
        "planned_loss": "binary_cross_entropy",
        "output_activation": "sigmoid",
    }
    return {
        ROTATION_CARRY_MODEL_NAME: {
            **shared,
            "rotate_x_right": 7,
            "rotate_y_left": 2,
            "training_labels": "true_speck32_ciphertext_outputs",
            "control": False,
        },
        WRONG_ROTATION_CARRY_MODEL_NAME: {
            **shared,
            "rotate_x_right": 5,
            "rotate_y_left": 6,
            "training_labels": "true_speck32_ciphertext_outputs",
            "control": True,
            "control_scope": "rotation_constants_only",
        },
        ROTATION_CARRY_SHUFFLE_MODEL_NAME: {
            **shared,
            "rotate_x_right": 7,
            "rotate_y_left": 2,
            "training_labels": "fixed_permutation_of_training_targets_only",
            "test_labels": "true_speck32_ciphertext_outputs",
            "control": True,
            "control_scope": "training_label_order_only",
        },
    }


__all__ = [
    "ROTATION_CARRY_MODEL_NAME",
    "ROTATION_CARRY_SHUFFLE_MODEL_NAME",
    "WRONG_ROTATION_CARRY_MODEL_NAME",
    "Speck32RotationCarryPredictor",
    "rotate_word_tokens",
    "rotation_carry_protocols",
]
