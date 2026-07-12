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
    PresentNibbleInvPOnlySpnOnlyDistinguisher,
    present_inverse_p_indices,
)
from blockcipher_nd.registry.model_factory import build_model


INPUT_BITS = 16 * 128
TOPOLOGY_RESIDUAL_MODELS = (
    (
        "present_nibble_invp_topology_residual_spn_only",
        PresentNibbleInvPTopologyResidualSpnOnlyDistinguisher,
        "true",
    ),
    (
        "present_nibble_shuffled_p_topology_residual_spn_only",
        PresentNibbleShuffledPTopologyResidualSpnOnlyDistinguisher,
        "shuffled",
    ),
    (
        "present_nibble_delta_topology_residual_spn_only",
        PresentNibbleDeltaTopologyResidualSpnOnlyDistinguisher,
        "delta",
    ),
)


def _raw_features(batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260712)
    return torch.randint(0, 2, (batch, INPUT_BITS), generator=generator).float()


def _assert_state_dicts_equal(left: nn.Module, right: nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        torch.testing.assert_close(left_state[key], right_state[key], rtol=0, atol=0)


def _build_registered_topology_residual(
    model_key: str,
    *,
    model_options: dict[str, object] | None = None,
    hidden_bits: int = 32,
    pair_bits: int | None = 128,
) -> PresentNibbleTopologyResidualSpnOnlyDistinguisher:
    model = build_model(
        model_key,
        input_bits=INPUT_BITS,
        hidden_bits=hidden_bits,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=model_options,
    )
    assert isinstance(model, PresentNibbleTopologyResidualSpnOnlyDistinguisher)
    return model


@pytest.mark.parametrize("model_key", [row[0] for row in TOPOLOGY_RESIDUAL_MODELS])
def test_topology_residual_registry_defaults_missing_pair_bits_to_128(
    model_key: str,
) -> None:
    model = _build_registered_topology_residual(model_key, pair_bits=None)

    assert model.pair_bits == 128


@pytest.mark.parametrize("model_key", [row[0] for row in TOPOLOGY_RESIDUAL_MODELS])
@pytest.mark.parametrize(
    ("pair_bits", "model_options", "message"),
    [
        (0, None, "expects raw 128-bit ciphertext pairs"),
        (128, {"spn_mixer_depth": 0}, "spn_mixer_depth must be >= 1"),
        (128, {"token_mlp_ratio": 0}, "token_mlp_ratio must be >= 1"),
    ],
)
def test_topology_residual_registry_preserves_explicit_invalid_options(
    model_key: str,
    pair_bits: int,
    model_options: dict[str, object] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_registered_topology_residual(
            model_key,
            pair_bits=pair_bits,
            model_options=model_options,
        )


@pytest.mark.parametrize(
    ("model_key", "model_class", "mapping_mode"), TOPOLOGY_RESIDUAL_MODELS
)
def test_topology_residual_registry_builds_exact_class_and_forwards_options(
    model_key: str,
    model_class: type[PresentNibbleTopologyResidualSpnOnlyDistinguisher],
    mapping_mode: str,
) -> None:
    options = {
        "spn_token_dim": 24,
        "spn_mixer_depth": 3,
        "token_mlp_ratio": 3,
        "local_channels": 10,
        "local_depth": 1,
        "local_kernel_size": 5,
        "local_residual_scale_init": 0.25,
        "activation": "gelu",
        "norm": "layernorm",
        "local_norm": "batchnorm2d",
        "dropout": 0.2,
    }

    model = _build_registered_topology_residual(
        model_key, model_options=options, hidden_bits=12
    )

    assert type(model) is model_class
    assert model.mapping_mode == mapping_mode
    assert model.pair_bits == 128
    assert model.spn_encoder.spn_token_dim == 24
    assert model.spn_encoder.embedding_bits == 48
    assert len(model.spn_encoder.spn_mixers) == 3
    assert model.spn_encoder.spn_mixers[0].channel_mixer[0].out_features == 72
    assert model.local_stem[0].out_channels == 10
    assert model.local_blocks[0].conv1.kernel_size == (5, 5)
    assert isinstance(model.local_stem[1], nn.BatchNorm2d)
    assert model.local_blocks[0].dropout.p == 0.2
    assert model.classifier[3].p == 0.2
    torch.testing.assert_close(model.alpha.detach(), torch.tensor(0.25))


def test_registered_topology_residual_variants_have_equal_capacity_and_adapters() -> (
    None
):
    options = {
        "spn_mixer_depth": 2,
        "token_mlp_ratio": 2,
        "local_channels": 16,
        "local_depth": 1,
        "local_kernel_size": 3,
        "local_residual_scale_init": 0.1,
        "activation": "relu",
        "norm": "layernorm",
        "local_norm": "batchnorm2d",
        "dropout": 0.0,
    }
    models = [
        _build_registered_topology_residual(model_key, model_options=options)
        for model_key, _, _ in TOPOLOGY_RESIDUAL_MODELS
    ]

    total_counts = [
        sum(parameter.numel() for parameter in model.parameters()) for model in models
    ]
    trainable_counts = [
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        for model in models
    ]
    adapter_counts = [
        sum(
            parameter.numel()
            for module in (model.local_stem, model.local_blocks, model.local_projection)
            for parameter in module.parameters()
        )
        + model.alpha.numel()
        for model in models
    ]

    assert len(set(total_counts)) == 1
    assert len(set(trainable_counts)) == 1
    assert len(set(adapter_counts)) == 1
    assert [model.mapping_mode for model in models] == ["true", "shuffled", "delta"]
    assert torch.equal(models[0].mapping_indices, present_inverse_p_indices("true"))
    assert torch.equal(models[1].mapping_indices, present_inverse_p_indices("shuffled"))
    assert torch.equal(models[2].mapping_indices, torch.arange(64))


def test_registered_topology_residual_variants_share_common_initialization() -> None:
    models = []
    for model_key, _, _ in TOPOLOGY_RESIDUAL_MODELS:
        torch.manual_seed(20260712)
        models.append(_build_registered_topology_residual(model_key))

    for model in models[1:]:
        _assert_state_dicts_equal(models[0].spn_encoder, model.spn_encoder)
        _assert_state_dicts_equal(models[0].classifier, model.classifier)


def test_registered_topology_residual_matches_anchor_common_initialization() -> None:
    torch.manual_seed(20260712)
    anchor = PresentNibbleInvPOnlySpnOnlyDistinguisher(
        input_bits=INPUT_BITS,
        pair_bits=128,
        base_channels=32,
        spn_mixer_depth=2,
        token_mlp_ratio=2,
        activation="relu",
        norm="layernorm",
        dropout=0.0,
    )
    torch.manual_seed(20260712)
    candidate = _build_registered_topology_residual(
        "present_nibble_invp_topology_residual_spn_only"
    )

    _assert_state_dicts_equal(anchor.spn_encoder, candidate.spn_encoder)
    _assert_state_dicts_equal(anchor.classifier, candidate.classifier)


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
