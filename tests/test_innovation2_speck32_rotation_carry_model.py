from __future__ import annotations

import pytest
import torch

from blockcipher_nd.ciphers.base import rol, ror
from blockcipher_nd.tasks.innovation2.speck32_output_prediction_models import (
    BILSTM_PARAMETER_COUNT,
)
from blockcipher_nd.tasks.innovation2.speck32_rotation_carry_model import (
    ROTATION_CARRY_MODEL_NAME,
    ROTATION_CARRY_SHUFFLE_MODEL_NAME,
    WRONG_ROTATION_CARRY_MODEL_NAME,
    Speck32RotationCarryPredictor,
    rotate_word_tokens,
    rotation_carry_protocols,
)


def test_rotation_carry_parameters_match_bilstm_five_percent_gate() -> None:
    protocols = rotation_carry_protocols()
    correct = protocols[ROTATION_CARRY_MODEL_NAME]
    wrong = protocols[WRONG_ROTATION_CARRY_MODEL_NAME]
    shuffled = protocols[ROTATION_CARRY_SHUFFLE_MODEL_NAME]

    assert correct["parameters"] == wrong["parameters"] == shuffled["parameters"]
    assert correct["bilstm_anchor_parameters"] == BILSTM_PARAMETER_COUNT
    assert correct["within_bilstm_five_percent"] is True
    assert 0.95 <= correct["parameter_ratio_to_bilstm"] <= 1.05


def test_correct_and_wrong_models_share_identical_initialized_state() -> None:
    torch.manual_seed(21)
    correct = Speck32RotationCarryPredictor.correct()
    torch.manual_seed(21)
    wrong = Speck32RotationCarryPredictor.wrong_rotation()

    assert correct.rotate_x_right == 7
    assert correct.rotate_y_left == 2
    assert wrong.rotate_x_right == 5
    assert wrong.rotate_y_left == 6
    assert correct.state_dict().keys() == wrong.state_dict().keys()
    for name, tensor in correct.state_dict().items():
        torch.testing.assert_close(tensor, wrong.state_dict()[name])


def test_wrong_protocol_changes_only_rotation_and_control_fields() -> None:
    protocols = rotation_carry_protocols()
    correct = dict(protocols[ROTATION_CARRY_MODEL_NAME])
    wrong = dict(protocols[WRONG_ROTATION_CARRY_MODEL_NAME])
    for field in (
        "rotate_x_right",
        "rotate_y_left",
        "control",
        "control_scope",
    ):
        correct.pop(field, None)
        wrong.pop(field, None)

    assert correct == wrong
    assert protocols[WRONG_ROTATION_CARRY_MODEL_NAME]["control_scope"] == (
        "rotation_constants_only"
    )
    assert protocols[ROTATION_CARRY_SHUFFLE_MODEL_NAME]["control_scope"] == (
        "training_label_order_only"
    )


def test_word_token_rotation_matches_msb_first_speck_directions() -> None:
    word = 0x1234
    tokens = torch.tensor(
        [[[(word >> shift) & 1] for shift in range(15, -1, -1)]],
        dtype=torch.float32,
    )

    right = rotate_word_tokens(tokens, amount=7, direction="right")
    left = rotate_word_tokens(tokens, amount=2, direction="left")

    assert _tokens_to_word(right) == ror(word, 7, 16)
    assert _tokens_to_word(left) == rol(word, 2, 16)


def test_rotation_carry_models_emit_position_bound_32_bit_probabilities() -> None:
    torch.manual_seed(21)
    correct = Speck32RotationCarryPredictor.correct(channels=16)
    torch.manual_seed(21)
    wrong = Speck32RotationCarryPredictor.wrong_rotation(channels=16)
    features = torch.randint(0, 2, size=(3, 32), dtype=torch.float32)

    correct.eval()
    wrong.eval()
    with torch.no_grad():
        correct_output = correct(features)
        wrong_output = wrong(features)

    assert correct_output.shape == (3, 32)
    assert torch.all(correct_output >= 0.0)
    assert torch.all(correct_output <= 1.0)
    assert not torch.equal(correct_output, wrong_output)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"channels": 0},
        {"steps": 0},
        {"rotate_x_right": 0},
        {"rotate_y_left": 16},
    ),
)
def test_invalid_rotation_carry_configurations_fail_closed(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        Speck32RotationCarryPredictor(**kwargs)


def test_rotation_carry_model_rejects_wrong_input_shape() -> None:
    model = Speck32RotationCarryPredictor.correct(channels=8)

    with pytest.raises(ValueError, match=r"expected \[batch, 32\]"):
        model(torch.zeros(2, 31))
    with pytest.raises(ValueError, match=r"expected \[batch, 16, channels\]"):
        rotate_word_tokens(torch.zeros(2, 15, 8), amount=7, direction="right")


def _tokens_to_word(tokens: torch.Tensor) -> int:
    value = 0
    for bit in tokens[0, :, 0]:
        value = (value << 1) | int(bit.item())
    return value
