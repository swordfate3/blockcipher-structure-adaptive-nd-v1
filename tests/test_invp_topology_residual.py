import pytest
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.present_invp_state_matrix_conv2d import (
    PresentStateMatrixResidualBlock,
)
from blockcipher_nd.models.structure.spn.present_invp_topology_residual import (
    PresentNibbleDeltaTopologyResidualSpnOnlyDistinguisher,
    PresentNibbleInvPTopologyResidualSpnOnlyDistinguisher,
    PresentNibbleShuffledPTopologyResidualSpnOnlyDistinguisher,
    PresentNibbleTopologyResidualSpnOnlyDistinguisher,
)
from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
    present_inverse_p_indices,
)


INPUT_BITS = 16 * 128


def _raw_features(batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260712)
    return torch.randint(0, 2, (batch, INPUT_BITS), generator=generator).float()


@pytest.mark.parametrize("mapping_mode", ["true", "shuffled", "delta"])
def test_topology_residual_state_matrix_has_exact_mapping(mapping_mode: str) -> None:
    model = PresentNibbleTopologyResidualSpnOnlyDistinguisher(
        input_bits=INPUT_BITS,
        mapping_mode=mapping_mode,
    )
    features = _raw_features()
    raw_pairs = features.reshape(2, 16, 2, 64)
    difference = (raw_pairs[:, :, 0] - raw_pairs[:, :, 1]).abs()
    indices = (
        torch.arange(64)
        if mapping_mode == "delta"
        else present_inverse_p_indices(mapping_mode)
    )
    expected = (
        difference.index_select(2, indices).reshape(2, 16, 16, 4).permute(0, 1, 3, 2)
    )

    actual = model.local_state_matrix_view(features)

    assert actual.shape == (2, 16, 4, 16)
    torch.testing.assert_close(actual, expected)


def test_topology_residual_has_exact_frozen_architecture() -> None:
    model = PresentNibbleTopologyResidualSpnOnlyDistinguisher(input_bits=INPUT_BITS)

    assert model.input_bits == INPUT_BITS
    assert model.pair_bits == 128
    assert model.pairs_per_sample == 16
    assert model.spn_encoder.view_mode == "inv_p"
    assert model.spn_encoder.embedding_bits == 128
    assert len(model.spn_encoder.spn_mixers) == 2
    assert torch.equal(
        model.spn_encoder.inverse_p_indices,
        present_inverse_p_indices("true"),
    )
    assert list(model._modules) == [
        "spn_encoder",
        "classifier",
        "local_stem",
        "local_blocks",
        "local_projection",
    ]

    assert isinstance(model.classifier[0], nn.LayerNorm)
    assert model.classifier[0].normalized_shape == (256,)
    assert isinstance(model.classifier[1], nn.Linear)
    assert (model.classifier[1].in_features, model.classifier[1].out_features) == (
        256,
        256,
    )
    assert isinstance(model.classifier[2], nn.ReLU)
    assert isinstance(model.classifier[3], nn.Dropout)
    assert model.classifier[3].p == 0.0
    assert isinstance(model.classifier[4], nn.Linear)
    assert (model.classifier[4].in_features, model.classifier[4].out_features) == (
        256,
        1,
    )

    assert isinstance(model.local_stem[0], nn.Conv2d)
    assert model.local_stem[0].in_channels == 1
    assert model.local_stem[0].out_channels == 16
    assert model.local_stem[0].kernel_size == (1, 1)
    assert isinstance(model.local_stem[1], nn.BatchNorm2d)
    assert isinstance(model.local_stem[2], nn.ReLU)
    assert len(model.local_blocks) == 1
    block = model.local_blocks[0]
    assert isinstance(block, PresentStateMatrixResidualBlock)
    assert block.conv1.kernel_size == (3, 3)
    assert block.conv2.kernel_size == (3, 3)
    assert isinstance(block.norm1, nn.BatchNorm2d)
    assert isinstance(block.norm2, nn.BatchNorm2d)
    assert block.dropout.p == 0.0
    assert isinstance(model.local_projection, nn.Linear)
    assert (
        model.local_projection.in_features,
        model.local_projection.out_features,
    ) == (
        32,
        128,
    )
    assert model.alpha.shape == torch.Size([])
    torch.testing.assert_close(model.alpha.detach(), torch.tensor(0.1))


@pytest.mark.parametrize(
    ("model_class", "mapping_mode"),
    [
        (PresentNibbleInvPTopologyResidualSpnOnlyDistinguisher, "true"),
        (PresentNibbleShuffledPTopologyResidualSpnOnlyDistinguisher, "shuffled"),
        (PresentNibbleDeltaTopologyResidualSpnOnlyDistinguisher, "delta"),
    ],
)
def test_fixed_topology_residual_subclasses_default_to_fixed_mapping(
    model_class: type[PresentNibbleTopologyResidualSpnOnlyDistinguisher],
    mapping_mode: str,
) -> None:
    model = model_class(input_bits=INPUT_BITS)
    expected = (
        torch.arange(64)
        if mapping_mode == "delta"
        else present_inverse_p_indices(mapping_mode)
    )

    assert model.mapping_mode == mapping_mode
    assert torch.equal(model.mapping_indices, expected)


@pytest.mark.parametrize(
    ("model_class", "mapping_mode"),
    [
        (PresentNibbleInvPTopologyResidualSpnOnlyDistinguisher, "true"),
        (PresentNibbleShuffledPTopologyResidualSpnOnlyDistinguisher, "shuffled"),
        (PresentNibbleDeltaTopologyResidualSpnOnlyDistinguisher, "delta"),
    ],
)
def test_fixed_topology_residual_subclasses_accept_matching_mapping(
    model_class: type[PresentNibbleTopologyResidualSpnOnlyDistinguisher],
    mapping_mode: str,
) -> None:
    model = model_class(input_bits=INPUT_BITS, mapping_mode=mapping_mode)

    assert model.mapping_mode == mapping_mode


@pytest.mark.parametrize(
    ("model_class", "fixed_mapping", "conflicting_mapping"),
    [
        (PresentNibbleInvPTopologyResidualSpnOnlyDistinguisher, "true", "delta"),
        (
            PresentNibbleShuffledPTopologyResidualSpnOnlyDistinguisher,
            "shuffled",
            "true",
        ),
        (PresentNibbleDeltaTopologyResidualSpnOnlyDistinguisher, "delta", "shuffled"),
    ],
)
def test_fixed_topology_residual_subclasses_reject_conflicting_mapping(
    model_class: type[PresentNibbleTopologyResidualSpnOnlyDistinguisher],
    fixed_mapping: str,
    conflicting_mapping: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"fixed mapping.*{fixed_mapping}.*conflicting.*{conflicting_mapping}",
    ):
        model_class(input_bits=INPUT_BITS, mapping_mode=conflicting_mapping)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pair_bits": 64}, "expects raw 128-bit ciphertext pairs"),
        ({"input_bits": 0}, "input_bits must be positive"),
        ({"input_bits": -128}, "input_bits must be positive"),
        ({"input_bits": INPUT_BITS - 1}, "input_bits must be a multiple"),
        ({"base_channels": 0}, "base_channels must be positive"),
        ({"base_channels": -1}, "base_channels must be positive"),
        ({"local_channels": 0}, "local_channels must be positive"),
        ({"local_depth": 0}, "local_depth must equal 1"),
        ({"local_depth": 2}, "local_depth must equal 1"),
        ({"local_kernel_size": 0}, "positive odd integer"),
        ({"local_kernel_size": 2}, "positive odd integer"),
        ({"mapping_mode": "bad"}, "unsupported mapping_mode: bad"),
    ],
)
def test_topology_residual_rejects_invalid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    options = {"input_bits": INPUT_BITS, **kwargs}

    with pytest.raises(ValueError, match=message):
        PresentNibbleTopologyResidualSpnOnlyDistinguisher(**options)


@pytest.mark.parametrize(
    "shape",
    [(INPUT_BITS,), (2, INPUT_BITS - 1), (2, 16, 128)],
)
def test_topology_residual_rejects_invalid_feature_shape(
    shape: tuple[int, ...],
) -> None:
    model = PresentNibbleTopologyResidualSpnOnlyDistinguisher(input_bits=INPUT_BITS)

    with pytest.raises(ValueError, match=f"expected {INPUT_BITS} input bits"):
        model.local_state_matrix_view(torch.zeros(shape))


def test_topology_residual_public_encoding_paths_have_frozen_shapes() -> None:
    model = PresentNibbleTopologyResidualSpnOnlyDistinguisher(input_bits=INPUT_BITS)
    features = _raw_features()

    state_matrices = model.local_state_matrix_view(features)
    local_pairs = model.encode_local_pairs(state_matrices)
    fused_pairs = model.encode_fused_pairs(features)

    assert local_pairs.shape == (2, 16, 128)
    assert fused_pairs.shape == (2, 16, 128)
    assert torch.isfinite(local_pairs).all()
    assert torch.isfinite(fused_pairs).all()
    assert model.set_cipher_structure("SPN") is None
    assert model.set_structure_features(torch.ones(1)) is None


@pytest.mark.parametrize(
    "model_class",
    [
        PresentNibbleInvPTopologyResidualSpnOnlyDistinguisher,
        PresentNibbleShuffledPTopologyResidualSpnOnlyDistinguisher,
        PresentNibbleDeltaTopologyResidualSpnOnlyDistinguisher,
    ],
)
def test_topology_residual_variants_have_finite_forward_and_adapter_backward(
    model_class: type[PresentNibbleTopologyResidualSpnOnlyDistinguisher],
) -> None:
    model = model_class(input_bits=INPUT_BITS)

    logits = model(_raw_features())
    logits.square().mean().backward()

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    adapter_parameters = [
        model.alpha,
        *model.local_stem.parameters(),
        *model.local_blocks.parameters(),
        *model.local_projection.parameters(),
    ]
    assert sum(parameter.grad.abs().sum() for parameter in adapter_parameters) > 0
