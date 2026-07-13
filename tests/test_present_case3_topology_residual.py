import pytest
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.present_case3_topology_residual import (
    PresentNibbleCase3InvPTopologyResidualSpnOnlyDistinguisher,
    PresentNibbleCase3RawTopologyResidualSpnOnlyDistinguisher,
    PresentNibbleCase3ShuffledPTopologyResidualSpnOnlyDistinguisher,
    PresentNibbleCase3TopologyResidualSpnOnlyDistinguisher,
)
from blockcipher_nd.models.structure.spn.present_invp_state_matrix_conv2d import (
    PresentStateMatrixResidualBlock,
)
from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
    PresentNibbleInvPOnlySpnOnlyDistinguisher,
    present_inverse_p_indices,
)
from blockcipher_nd.registry.model_factory import build_model


INPUT_BITS = 16 * 128
CASE3_MODELS = (
    (
        "present_nibble_case3_invp_topology_residual_spn_only",
        PresentNibbleCase3InvPTopologyResidualSpnOnlyDistinguisher,
        "true",
    ),
    (
        "present_nibble_case3_shuffled_p_topology_residual_spn_only",
        PresentNibbleCase3ShuffledPTopologyResidualSpnOnlyDistinguisher,
        "shuffled",
    ),
    (
        "present_nibble_case3_raw_topology_residual_spn_only",
        PresentNibbleCase3RawTopologyResidualSpnOnlyDistinguisher,
        "raw",
    ),
)


def _raw_features(batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260713)
    return torch.randint(0, 2, (batch, INPUT_BITS), generator=generator).float()


def _mapping_indices(mapping_mode: str) -> torch.Tensor:
    if mapping_mode == "raw":
        return torch.arange(64)
    return present_inverse_p_indices(mapping_mode)


def _assert_state_dicts_equal(left: nn.Module, right: nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        torch.testing.assert_close(left_state[key], right_state[key], rtol=0, atol=0)


@pytest.mark.parametrize("mapping_mode", ["true", "shuffled", "raw"])
def test_case3_local_view_has_exact_channels(mapping_mode: str) -> None:
    model = PresentNibbleCase3TopologyResidualSpnOnlyDistinguisher(
        input_bits=INPUT_BITS,
        mapping_mode=mapping_mode,
    )
    features = _raw_features()
    pairs = features.reshape(2, 16, 2, 64)
    difference = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
    channels = torch.stack(
        [
            pairs[:, :, 0],
            pairs[:, :, 1],
            difference.index_select(2, _mapping_indices(mapping_mode)),
        ],
        dim=2,
    )
    expected = channels.reshape(2, 16, 3, 16, 4).permute(0, 1, 2, 4, 3)

    actual = model.local_case3_view(features)

    assert actual.shape == (2, 16, 3, 4, 16)
    torch.testing.assert_close(actual, expected)


def test_case3_mapping_changes_only_difference_channel() -> None:
    features = _raw_features()
    views = [
        PresentNibbleCase3TopologyResidualSpnOnlyDistinguisher(
            input_bits=INPUT_BITS,
            mapping_mode=mode,
        ).local_case3_view(features)
        for mode in ("true", "shuffled", "raw")
    ]

    for view in views[1:]:
        torch.testing.assert_close(view[:, :, :2], views[0][:, :, :2])
    assert not torch.equal(views[0][:, :, 2], views[1][:, :, 2])
    assert not torch.equal(views[0][:, :, 2], views[2][:, :, 2])


def test_case3_has_exact_frozen_architecture() -> None:
    model = PresentNibbleCase3TopologyResidualSpnOnlyDistinguisher(
        input_bits=INPUT_BITS
    )

    assert model.pair_bits == 128
    assert model.pairs_per_sample == 16
    assert model.spn_encoder.view_mode == "inv_p"
    assert len(model.spn_encoder.spn_mixers) == 2
    assert isinstance(model.local_stem[0], nn.Conv2d)
    assert model.local_stem[0].in_channels == 3
    assert model.local_stem[0].out_channels == 16
    assert model.local_stem[0].kernel_size == (1, 1)
    assert isinstance(model.local_stem[1], nn.BatchNorm2d)
    assert len(model.local_blocks) == 1
    assert isinstance(model.local_blocks[0], PresentStateMatrixResidualBlock)
    assert model.local_blocks[0].conv1.kernel_size == (3, 3)
    assert model.local_projection.in_features == 32
    assert model.local_projection.out_features == model.spn_encoder.embedding_bits
    torch.testing.assert_close(model.alpha.detach(), torch.tensor(0.1))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pair_bits": 64}, "raw 128-bit ciphertext pairs"),
        ({"input_bits": 0}, "input_bits must be positive"),
        ({"input_bits": 129}, "multiple of pair_bits"),
        ({"mapping_mode": "invalid"}, "unsupported mapping_mode"),
        ({"local_channels": 0}, "local_channels must be positive"),
        ({"local_depth": 2}, "local_depth must equal 1"),
        ({"local_kernel_size": 2}, "positive odd integer"),
    ],
)
def test_case3_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    options = {"input_bits": INPUT_BITS, **kwargs}
    with pytest.raises(ValueError, match=message):
        PresentNibbleCase3TopologyResidualSpnOnlyDistinguisher(**options)


def test_case3_rejects_invalid_runtime_shape() -> None:
    model = PresentNibbleCase3TopologyResidualSpnOnlyDistinguisher(
        input_bits=INPUT_BITS
    )
    with pytest.raises(ValueError, match="expected 2048 input bits"):
        model.local_case3_view(torch.zeros(2, INPUT_BITS - 1))


@pytest.mark.parametrize("model_class", [row[1] for row in CASE3_MODELS])
def test_case3_forward_backward_is_finite(model_class: type[nn.Module]) -> None:
    model = model_class(input_bits=INPUT_BITS, base_channels=8, local_channels=4)
    features = _raw_features()

    output = model(features)
    output.mean().backward()

    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    ("model_class", "mapping_mode"),
    [(row[1], row[2]) for row in CASE3_MODELS],
)
def test_case3_fixed_variants_reject_conflicting_mapping(
    model_class: type[nn.Module], mapping_mode: str
) -> None:
    model = model_class(input_bits=INPUT_BITS, mapping_mode=mapping_mode)
    assert model.mapping_mode == mapping_mode
    conflicting = next(
        mode for mode in ("true", "shuffled", "raw") if mode != mapping_mode
    )
    with pytest.raises(ValueError, match="conflicting value"):
        model_class(input_bits=INPUT_BITS, mapping_mode=conflicting)


def _build_registered(
    model_key: str,
    *,
    model_options: dict[str, object] | None = None,
    hidden_bits: int = 32,
    pair_bits: int | None = 128,
) -> PresentNibbleCase3TopologyResidualSpnOnlyDistinguisher:
    model = build_model(
        model_key,
        input_bits=INPUT_BITS,
        hidden_bits=hidden_bits,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=model_options,
    )
    assert isinstance(model, PresentNibbleCase3TopologyResidualSpnOnlyDistinguisher)
    return model


@pytest.mark.parametrize("model_key", [row[0] for row in CASE3_MODELS])
def test_case3_registry_builds_variants_and_forwards_options(model_key: str) -> None:
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
    model = _build_registered(model_key, model_options=options, hidden_bits=12)

    assert model.pair_bits == 128
    assert model.spn_encoder.spn_token_dim == 24
    assert len(model.spn_encoder.spn_mixers) == 3
    assert model.local_stem[0].out_channels == 10
    assert model.local_blocks[0].conv1.kernel_size == (5, 5)
    assert model.local_blocks[0].dropout.p == 0.2
    torch.testing.assert_close(model.alpha.detach(), torch.tensor(0.25))


def test_case3_registered_variants_have_equal_capacity_and_initialization() -> None:
    models = []
    for model_key, _, _ in CASE3_MODELS:
        torch.manual_seed(20260713)
        models.append(_build_registered(model_key))

    assert len({sum(p.numel() for p in model.parameters()) for model in models}) == 1
    assert (
        len(
            {
                sum(p.numel() for p in model.parameters() if p.requires_grad)
                for model in models
            }
        )
        == 1
    )
    for model in models[1:]:
        _assert_state_dicts_equal(models[0], model)


def test_case3_candidate_preserves_anchor_common_initialization() -> None:
    torch.manual_seed(20260713)
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
    torch.manual_seed(20260713)
    candidate = _build_registered(CASE3_MODELS[0][0])

    _assert_state_dicts_equal(anchor.spn_encoder, candidate.spn_encoder)
    _assert_state_dicts_equal(anchor.classifier, candidate.classifier)


@pytest.mark.parametrize("model_key", [row[0] for row in CASE3_MODELS])
@pytest.mark.parametrize(
    ("pair_bits", "model_options", "message"),
    [
        (0, None, "raw 128-bit ciphertext pairs"),
        (128, {"spn_mixer_depth": 0}, "spn_mixer_depth must be >= 1"),
        (128, {"token_mlp_ratio": 0}, "token_mlp_ratio must be >= 1"),
    ],
)
def test_case3_registry_preserves_invalid_options(
    model_key: str,
    pair_bits: int,
    model_options: dict[str, object] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_registered(
            model_key,
            pair_bits=pair_bits,
            model_options=model_options,
        )

