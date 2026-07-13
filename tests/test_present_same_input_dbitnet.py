from __future__ import annotations

import pytest
import torch

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
    present_inverse_p_indices,
)
from blockcipher_nd.registry.model_factory import build_model


INPUT_BITS = 16 * 128
MODEL_MAPPINGS = (
    ("present_invp_dbitnet2023", "true"),
    ("present_shuffled_p_dbitnet2023", "shuffled"),
    ("present_raw_delta_dbitnet2023", "raw"),
)


def _build(model_key: str):
    return build_model(
        model_key,
        input_bits=INPUT_BITS,
        hidden_bits=32,
        pair_bits=128,
        structure="SPN",
        model_options={},
    )


def _features(batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260713)
    return torch.randint(0, 2, (batch, INPUT_BITS), generator=generator).float()


def _mapping_indices(mapping_mode: str) -> torch.Tensor:
    if mapping_mode == "raw":
        return torch.arange(64)
    return present_inverse_p_indices(mapping_mode)


@pytest.mark.parametrize(("model_key", "mapping_mode"), MODEL_MAPPINGS)
def test_same_input_dbitnet_builds_exact_mapped_delta_view(
    model_key: str, mapping_mode: str
) -> None:
    model = _build(model_key)
    features = _features()
    raw_pairs = features.reshape(2, 16, 2, 64)
    difference = (raw_pairs[:, :, 0] - raw_pairs[:, :, 1]).abs()
    expected = difference.index_select(2, _mapping_indices(mapping_mode)).reshape(
        2, 1024
    )

    actual = model.mapped_delta_view(features)

    assert model.mapping_mode == mapping_mode
    assert model.input_bits == 2048
    assert model.pair_bits == 128
    assert model.pairs_per_sample == 16
    assert model.mapped_input_bits == 1024
    assert actual.shape == (2, 1024)
    torch.testing.assert_close(actual, expected)


def test_same_input_dbitnet_variants_have_equal_capacity_and_initialization() -> None:
    models = []
    for model_key, _ in MODEL_MAPPINGS:
        torch.manual_seed(20260713)
        models.append(_build(model_key))

    assert len({sum(parameter.numel() for parameter in model.parameters()) for model in models}) == 1
    reference = models[0].state_dict()
    for model in models[1:]:
        assert model.state_dict().keys() == reference.keys()
        for key, value in reference.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    assert not torch.equal(models[0].mapping_indices, models[1].mapping_indices)
    assert not torch.equal(models[0].mapping_indices, models[2].mapping_indices)


@pytest.mark.parametrize("model_key", [row[0] for row in MODEL_MAPPINGS])
def test_same_input_dbitnet_forward_backward_and_auxiliary_loss_are_finite(
    model_key: str,
) -> None:
    model = _build(model_key)
    logits = model(_features())
    (logits.mean() + model.last_auxiliary_loss).backward()

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(model.last_auxiliary_loss)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_same_input_dbitnet_rejects_wrong_runtime_width() -> None:
    model = _build("present_invp_dbitnet2023")

    with pytest.raises(ValueError, match="expected 2048 input bits"):
        model.mapped_delta_view(torch.zeros(2, 2047))


def test_same_input_dbitnet_exposes_auditable_dbitnet_geometry() -> None:
    model = _build("present_invp_dbitnet2023")

    assert model_metadata(model) == {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "dilations": [511, 255, 127, 63, 31, 15, 7, 3],
        "output_width": 12,
        "output_channels": 144,
        "flattened_width": 1728,
        "l2_coefficient": 1e-5,
    }
