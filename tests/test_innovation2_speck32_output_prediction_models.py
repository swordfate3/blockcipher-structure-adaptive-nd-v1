from __future__ import annotations

import pytest
import torch

from blockcipher_nd.tasks.innovation2.speck32_output_prediction_models import (
    BILSTM_MODEL_NAME,
    FCNN_MODEL_NAME,
    JEONG_2024_DOI,
    Speck32JeongBiLstm,
    Speck32JeongFcnn,
    jeong_anchor_protocols,
    parameter_counts,
    split_speck32_words,
)


def test_jeong_anchor_protocol_freezes_reported_fields_and_marks_unknowns() -> None:
    protocols = jeong_anchor_protocols()
    fcnn = protocols[FCNN_MODEL_NAME]
    bilstm = protocols[BILSTM_MODEL_NAME]

    assert fcnn["doi"] == JEONG_2024_DOI
    assert fcnn["hidden_widths"] == [512, 1024, 512]
    assert fcnn["batch_normalization_after_hidden_layers"] is True
    assert fcnn["hidden_activation"] == "relu"
    assert fcnn["output_activation"] == "sigmoid"
    assert bilstm["input_shape"] == ["batch", 2, 16]
    assert bilstm["word_order"] == ["x_msw", "y_lsw"]
    assert bilstm["layers"] == 3
    assert bilstm["hidden_size_per_direction"] == 256
    assert bilstm["paper_exact_reproduction"] is False
    assert (
        "bidirectional_sequence_reduction_before_output_layer"
        in bilstm["paper_unspecified_model_details"]
    )
    assert fcnn["sample_classification"] is False
    assert bilstm["target"] == "32_msb_first_true_speck32_ciphertext_bits"


def test_speck_word_split_preserves_msw_then_lsw_roles() -> None:
    features = torch.arange(64, dtype=torch.float32).reshape(2, 32)

    words = split_speck32_words(features)

    assert words.shape == (2, 2, 16)
    torch.testing.assert_close(words[0, 0], features[0, :16])
    torch.testing.assert_close(words[0, 1], features[0, 16:])


def test_fcnn_and_bilstm_emit_32_true_output_probabilities() -> None:
    torch.manual_seed(21)
    features = torch.randint(0, 2, size=(4, 32), dtype=torch.float32)
    models = (Speck32JeongFcnn(), Speck32JeongBiLstm())

    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(features)
        assert outputs.shape == (4, 32)
        assert torch.all(outputs >= 0.0)
        assert torch.all(outputs <= 1.0)


def test_anchor_parameter_counts_are_frozen() -> None:
    counts = parameter_counts()

    assert counts == {
        FCNN_MODEL_NAME: 1_087_520,
        BILSTM_MODEL_NAME: 3_731_488,
    }


@pytest.mark.parametrize(
    "features",
    (
        torch.zeros(32),
        torch.zeros(2, 31),
        torch.zeros(2, 32, 1),
    ),
)
def test_anchor_models_reject_non_flat_32_bit_inputs(features: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"expected \[batch, 32\]"):
        Speck32JeongFcnn()(features)
    with pytest.raises(ValueError, match=r"expected \[batch, 32\]"):
        Speck32JeongBiLstm()(features)
