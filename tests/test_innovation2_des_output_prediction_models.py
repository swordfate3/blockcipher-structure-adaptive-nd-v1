from __future__ import annotations

import pytest
import torch

from blockcipher_nd.ciphers.feistel.des import DES_E, DES_FP, DES_IP, DES_P
from blockcipher_nd.tasks.innovation2.feistel.des_output_prediction_models import (
    BILSTM_MODEL_NAME,
    BILSTM_PARAMETER_COUNT,
    FCNN_MODEL_NAME,
    FCNN_PARAMETER_COUNT,
    FEISTEL_MODEL_NAME,
    FEISTEL_R2_PARAMETER_COUNT,
    FEISTEL_R3_PARAMETER_COUNT,
    JEONG_2024_DOI,
    WRONG_F_MODEL_NAME,
    DesJeongBiLstm,
    DesJeongFcnn,
    build_des_feistel_recurrent,
    des_output_model_protocols,
    parameter_counts,
    split_des_halves,
)


def test_des_model_protocols_freeze_true_output_semantics_and_controls() -> None:
    protocols = des_output_model_protocols()

    assert protocols[FCNN_MODEL_NAME]["doi"] == JEONG_2024_DOI
    assert protocols[FCNN_MODEL_NAME]["hidden_widths"] == [512, 1024, 512]
    assert protocols[BILSTM_MODEL_NAME]["input_shape"] == ["batch", 2, 32]
    assert protocols[BILSTM_MODEL_NAME]["layers"] == 3
    assert protocols[BILSTM_MODEL_NAME]["hidden_size_per_direction"] == 256
    assert protocols[BILSTM_MODEL_NAME]["paper_exact_reproduction"] is False
    assert protocols[FEISTEL_MODEL_NAME]["f_input_branch"] == "right"
    assert protocols[WRONG_F_MODEL_NAME]["f_input_branch"] == "left"
    assert protocols[FEISTEL_MODEL_NAME]["sample_classification"] is False
    assert protocols[FEISTEL_MODEL_NAME]["target"].startswith(
        "64_msb_first_true_round_reduced_des"
    )


def test_des_half_split_preserves_serialized_left_then_right_roles() -> None:
    features = torch.arange(128, dtype=torch.float32).reshape(2, 64)

    halves = split_des_halves(features)

    assert halves.shape == (2, 2, 32)
    torch.testing.assert_close(halves[0, 0], features[0, :32])
    torch.testing.assert_close(halves[0, 1], features[0, 32:])


def test_des_fcnn_and_bilstm_emit_full64_probabilities() -> None:
    torch.manual_seed(31)
    features = torch.randint(0, 2, size=(4, 64), dtype=torch.float32)

    for model in (DesJeongFcnn(), DesJeongBiLstm()):
        model.eval()
        with torch.no_grad():
            outputs = model(features)
        assert outputs.shape == (4, 64)
        assert torch.all(outputs >= 0.0)
        assert torch.all(outputs <= 1.0)


def test_des_anchor_and_recurrent_parameter_counts_are_frozen_and_fair() -> None:
    counts = parameter_counts(rounds=2)
    round3_counts = parameter_counts(rounds=3)

    assert counts == {
        FCNN_MODEL_NAME: FCNN_PARAMETER_COUNT,
        BILSTM_MODEL_NAME: BILSTM_PARAMETER_COUNT,
        FEISTEL_MODEL_NAME: FEISTEL_R2_PARAMETER_COUNT,
        WRONG_F_MODEL_NAME: FEISTEL_R2_PARAMETER_COUNT,
    }
    relative_difference = (
        abs(FEISTEL_R2_PARAMETER_COUNT - BILSTM_PARAMETER_COUNT)
        / BILSTM_PARAMETER_COUNT
    )
    assert relative_difference <= 0.05
    assert round3_counts[FEISTEL_MODEL_NAME] == FEISTEL_R3_PARAMETER_COUNT
    assert round3_counts[WRONG_F_MODEL_NAME] == FEISTEL_R3_PARAMETER_COUNT
    round3_relative_difference = (
        abs(FEISTEL_R3_PARAMETER_COUNT - BILSTM_PARAMETER_COUNT)
        / BILSTM_PARAMETER_COUNT
    )
    assert round3_relative_difference <= 0.05


def test_des_recurrent_uses_public_ip_expansion_p_and_fp_routes() -> None:
    model = build_des_feistel_recurrent("right", rounds=2, channels=4)

    assert model.ip_source_for_destination.tolist() == [
        position - 1 for position in DES_IP
    ]
    assert model.fp_source_for_destination.tolist() == [
        position - 1 for position in DES_FP
    ]
    assert model.f_block.expanded_source_for_destination.tolist() == [
        position - 1 for position in DES_E
    ]
    assert model.f_block.p_source_for_destination.tolist() == [
        position - 1 for position in DES_P
    ]


def test_des_correct_and_wrong_f_models_are_parameter_matched() -> None:
    models = []
    for branch in ("right", "left"):
        torch.manual_seed(31)
        models.append(build_des_feistel_recurrent(branch, rounds=2, channels=8))

    correct, wrong = models
    assert correct.state_dict().keys() == wrong.state_dict().keys()
    assert all(
        torch.equal(correct.state_dict()[name], wrong.state_dict()[name])
        for name in correct.state_dict()
    )
    assert sum(parameter.numel() for parameter in correct.parameters()) == sum(
        parameter.numel() for parameter in wrong.parameters()
    )

    features = (torch.arange(128).reshape(2, 64) % 3 == 0).float()
    correct.eval()
    wrong.eval()
    with torch.no_grad():
        assert not torch.equal(correct(features), wrong(features))


def test_des_f_control_changes_only_the_first_round_input_branch() -> None:
    torch.manual_seed(31)
    correct = build_des_feistel_recurrent("right", rounds=2, channels=4)
    torch.manual_seed(31)
    wrong = build_des_feistel_recurrent("left", rounds=2, channels=4)
    features = (torch.arange(128).reshape(2, 64) % 5 == 0).float()
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            captured.setdefault(name, inputs[0].detach().clone())

        return hook

    handles = [
        correct.f_block.register_forward_pre_hook(capture("right")),
        wrong.f_block.register_forward_pre_hook(capture("left")),
    ]
    try:
        correct(features)
        wrong(features)
    finally:
        for handle in handles:
            handle.remove()

    hidden = correct.embedding(features.unsqueeze(-1)) + correct.position_embedding
    ip_hidden = hidden.index_select(1, correct.ip_source_for_destination)
    torch.testing.assert_close(captured["right"], ip_hidden[:, 32:, :])
    torch.testing.assert_close(captured["left"], ip_hidden[:, :32, :])


@pytest.mark.parametrize("rounds", (2, 3))
def test_des_recurrent_reuses_one_round_body_and_emits_full64(
    rounds: int,
) -> None:
    model = build_des_feistel_recurrent("right", rounds=rounds, channels=4)
    calls = 0

    def count_call(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        nonlocal calls
        calls += 1

    handle = model.f_block.register_forward_hook(count_call)
    try:
        outputs = model(torch.zeros((2, 64)))
    finally:
        handle.remove()

    assert calls == rounds
    assert len([name for name, _ in model.named_modules() if name == "f_block"]) == 1
    assert outputs.shape == (2, 64)
    assert model.round_contexts.shape == (rounds, 48, 4)


@pytest.mark.parametrize(
    "features",
    (
        torch.zeros(64),
        torch.zeros(2, 63),
        torch.zeros(2, 64, 1),
    ),
)
def test_des_output_models_reject_non_flat64_inputs(features: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"expected \[batch, 64\]"):
        DesJeongFcnn()(features)
    with pytest.raises(ValueError, match=r"expected \[batch, 64\]"):
        DesJeongBiLstm()(features)
    with pytest.raises(ValueError, match=r"expected \[batch, 64\]"):
        build_des_feistel_recurrent("right", channels=4)(features)


def test_des_recurrent_rejects_protocol_drift() -> None:
    with pytest.raises(ValueError, match="two or three"):
        build_des_feistel_recurrent("right", rounds=4)
    with pytest.raises(ValueError, match="right or left"):
        build_des_feistel_recurrent("identity")
    with pytest.raises(ValueError, match="channels must be positive"):
        build_des_feistel_recurrent("right", channels=0)
