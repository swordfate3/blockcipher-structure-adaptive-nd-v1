from __future__ import annotations

import pytest
import torch
from torch import nn

from blockcipher_nd.ciphers.spn.gift import Gift64
from blockcipher_nd.features.encoders.bitwise import int_to_bits
from blockcipher_nd.models.structure.spn.cross_spn_typed_cell import (
    CrossSpnTypedCellPairSetDistinguisher,
    GiftAlignedTokenMixerRawInputDistinguisher,
    GiftCrossSpnTypedCellE5FromPresentTrueShuffledDistinguisher,
    GiftCrossSpnTypedCellE6FromPresentFunctionalMarginDistinguisher,
    GiftCrossSpnTypedCellEquivariantMixerDistinguisher,
    GiftCrossSpnTypedCellNoPositionDistinguisher,
    GiftCrossSpnTypedCellSharedViewEncoderDistinguisher,
    GiftCrossSpnTypedCellRawDistinguisher,
    GiftCrossSpnTypedCellShuffledDistinguisher,
    GiftCrossSpnTypedCellTrueDistinguisher,
    PresentCrossSpnTypedCellRawDistinguisher,
    PresentCrossSpnTypedCellE5OffDistinguisher,
    PresentCrossSpnTypedCellE5ShuffledPlaceboDistinguisher,
    PresentCrossSpnTypedCellE5TrueShuffledDistinguisher,
    PresentCrossSpnTypedCellE6FunctionalMarginDistinguisher,
    PresentCrossSpnTypedCellE6OffDistinguisher,
    PresentCrossSpnTypedCellE6ShuffledPlaceboDistinguisher,
    PresentCrossSpnTypedCellShuffledDistinguisher,
    PresentCrossSpnTypedCellTrueDistinguisher,
    cipher_inverse_permutation_indices,
)
from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
    present_inverse_p_indices,
)
from blockcipher_nd.models.structure.spn.token_mixer_pairset import (
    EquivariantSpnTokenMixerBlock,
    SpnTokenMixerBlock,
    SpnTokenMixerPairSetDistinguisher,
)
from blockcipher_nd.registry.model_factory import build_model


def _apply_indices(value: int, indices: torch.Tensor) -> list[int]:
    bits = torch.tensor(int_to_bits(value, 64), dtype=torch.long)
    return bits.index_select(0, indices).tolist()


def test_cross_spn_mapping_indices_are_permutations() -> None:
    for cipher_key in ("present80", "gift64"):
        for mapping_mode in ("true", "shuffled", "raw"):
            indices = cipher_inverse_permutation_indices(cipher_key, mapping_mode)

            assert indices.dtype == torch.long
            assert sorted(indices.tolist()) == list(range(64))


def test_cross_spn_present_true_mapping_matches_existing_adapter() -> None:
    torch.testing.assert_close(
        cipher_inverse_permutation_indices("present80", "true"),
        present_inverse_p_indices("true"),
    )


def test_cross_spn_gift_true_mapping_matches_cipher_implementation() -> None:
    value = 0xD35A81C709E246BF
    indices = cipher_inverse_permutation_indices("gift64", "true")

    actual = _apply_indices(value, indices)
    expected = int_to_bits(Gift64.inverse_permutation_layer(value), 64)

    assert actual == expected


def test_cross_spn_shuffled_mapping_is_shared_across_ciphers() -> None:
    assert torch.equal(
        cipher_inverse_permutation_indices("present80", "shuffled"),
        cipher_inverse_permutation_indices("gift64", "shuffled"),
    )


TYPED_VARIANTS: tuple[type[nn.Module], ...] = (
    PresentCrossSpnTypedCellTrueDistinguisher,
    PresentCrossSpnTypedCellShuffledDistinguisher,
    PresentCrossSpnTypedCellRawDistinguisher,
    GiftCrossSpnTypedCellTrueDistinguisher,
    GiftCrossSpnTypedCellShuffledDistinguisher,
    GiftCrossSpnTypedCellRawDistinguisher,
)


def _raw_features(batch: int = 2, pairs_per_sample: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260713)
    return torch.randint(
        0,
        2,
        (batch, pairs_per_sample * 128),
        generator=generator,
    ).float()


@pytest.mark.parametrize("mapping_mode", ["true", "shuffled", "raw"])
def test_cross_spn_typed_cell_view_has_exact_current_and_previous_cells(
    mapping_mode: str,
) -> None:
    model = CrossSpnTypedCellPairSetDistinguisher(
        input_bits=4 * 128,
        cipher_key="gift64",
        mapping_mode=mapping_mode,
        base_channels=8,
    )
    features = _raw_features()
    pairs = features.reshape(2, 4, 2, 64)
    difference = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
    expected_current = difference.reshape(2, 4, 16, 4)
    expected_previous = difference.index_select(
        2, model.mapping_indices
    ).reshape(2, 4, 16, 4)

    current, previous = model.typed_cell_view(features)

    assert current.shape == previous.shape == (2, 4, 16, 4)
    torch.testing.assert_close(current, expected_current)
    torch.testing.assert_close(previous, expected_previous)


def test_gift_position_ablation_keeps_geometry_and_removes_position_effect() -> None:
    torch.manual_seed(17)
    learned = GiftCrossSpnTypedCellTrueDistinguisher(
        input_bits=4 * 128,
        base_channels=8,
        token_dim=16,
        mixer_depth=1,
        dropout=0.0,
    ).eval()
    torch.manual_seed(17)
    zero = GiftCrossSpnTypedCellNoPositionDistinguisher(
        input_bits=4 * 128,
        base_channels=8,
        token_dim=16,
        mixer_depth=1,
        dropout=0.0,
    ).eval()
    assert {
        name: tuple(value.shape) for name, value in learned.state_dict().items()
    } == {name: tuple(value.shape) for name, value in zero.state_dict().items()}
    assert sum(parameter.numel() for parameter in learned.parameters()) == sum(
        parameter.numel() for parameter in zero.parameters()
    )
    features = _raw_features()
    distinct_positions = torch.randn(
        learned.position_embedding.shape,
        generator=torch.Generator().manual_seed(29),
    )
    with torch.no_grad():
        zero_before = zero(features)
        zero.position_embedding.copy_(distinct_positions)
        zero_after = zero(features)
        learned.position_embedding.copy_(distinct_positions)
        learned_after = learned(features)
    torch.testing.assert_close(zero_before, zero_after)
    assert not torch.allclose(zero_before, learned_after)


def test_gift_shared_view_ablation_keeps_dormant_encoder_geometry() -> None:
    torch.manual_seed(31)
    separate = GiftCrossSpnTypedCellNoPositionDistinguisher(
        input_bits=4 * 128, base_channels=8, token_dim=16, mixer_depth=1
    ).eval()
    torch.manual_seed(31)
    shared = GiftCrossSpnTypedCellSharedViewEncoderDistinguisher(
        input_bits=4 * 128, base_channels=8, token_dim=16, mixer_depth=1
    ).eval()
    assert {
        name: tuple(value.shape) for name, value in separate.state_dict().items()
    } == {name: tuple(value.shape) for name, value in shared.state_dict().items()}
    features = _raw_features()
    with torch.no_grad():
        shared_before = shared(features)
        for parameter in shared.previous_cell_encoder.parameters():
            parameter.fill_(100.0)
        shared_after = shared(features)
        for parameter in separate.previous_cell_encoder.parameters():
            parameter.fill_(100.0)
        separate_after = separate(features)
    torch.testing.assert_close(shared_before, shared_after)
    assert not torch.allclose(shared_before, separate_after)


def test_equivariant_token_mixer_matches_fixed_parameter_budget() -> None:
    fixed = SpnTokenMixerBlock(
        nibbles_per_pair=16, token_dim=128, token_mlp_ratio=2
    )
    equivariant = EquivariantSpnTokenMixerBlock(
        nibbles_per_pair=16, token_dim=128, token_mlp_ratio=2
    )

    assert sum(parameter.numel() for parameter in fixed.parameters()) == sum(
        parameter.numel() for parameter in equivariant.parameters()
    )


def test_equivariant_token_mixer_commutes_with_cell_permutation() -> None:
    torch.manual_seed(37)
    model = EquivariantSpnTokenMixerBlock(
        nibbles_per_pair=16,
        token_dim=32,
        token_mlp_ratio=2,
        dropout=0.0,
    ).eval()
    features = torch.randn(3, 16, 32)
    permutation = torch.randperm(16, generator=torch.Generator().manual_seed(41))

    with torch.no_grad():
        original = model(features)
        permuted = model(features.index_select(1, permutation))

    torch.testing.assert_close(
        permuted,
        original.index_select(1, permutation),
        rtol=0.0,
        atol=1e-6,
    )


def test_gift_equivariant_mixer_matches_anchor_parameter_count() -> None:
    anchor = GiftCrossSpnTypedCellSharedViewEncoderDistinguisher(
        input_bits=4 * 128, base_channels=64, mixer_depth=2
    )
    control = GiftCrossSpnTypedCellEquivariantMixerDistinguisher(
        input_bits=4 * 128, base_channels=64, mixer_depth=2
    )

    assert sum(parameter.numel() for parameter in anchor.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )
    assert all(
        isinstance(block, EquivariantSpnTokenMixerBlock)
        for block in control.mixer_blocks
    )


def test_cross_spn_typed_variants_have_identical_trainable_state() -> None:
    models: list[nn.Module] = []
    for model_class in TYPED_VARIANTS:
        torch.manual_seed(20260713)
        models.append(model_class(input_bits=4 * 128, base_channels=8))

    assert len({sum(parameter.numel() for parameter in model.parameters()) for model in models}) == 1
    assert (
        len(
            {
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                for model in models
            }
        )
        == 1
    )
    reference = models[0].state_dict()
    assert "mapping_indices" not in reference
    for model in models[1:]:
        assert model.state_dict().keys() == reference.keys()
        for key, value in reference.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


E5_SOURCE_VARIANTS: tuple[type[nn.Module], ...] = (
    PresentCrossSpnTypedCellE5OffDistinguisher,
    PresentCrossSpnTypedCellE5TrueShuffledDistinguisher,
    PresentCrossSpnTypedCellE5ShuffledPlaceboDistinguisher,
)


def test_cross_spn_e5_source_variants_have_identical_state() -> None:
    models: list[nn.Module] = []
    for model_class in E5_SOURCE_VARIANTS:
        torch.manual_seed(20260715)
        models.append(model_class(input_bits=4 * 128, base_channels=8))

    reference = models[0].state_dict()
    assert any(key.startswith("topology_auxiliary_head") for key in reference)
    for model in models[1:]:
        assert model.state_dict().keys() == reference.keys()
        for key, value in reference.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_cross_spn_e5_does_not_change_legacy_state_dict() -> None:
    legacy = PresentCrossSpnTypedCellTrueDistinguisher(
        input_bits=4 * 128,
        base_channels=8,
    )

    assert not any(
        key.startswith("topology_auxiliary_head") for key in legacy.state_dict()
    )


@pytest.mark.parametrize(
    "model_class",
    (
        PresentCrossSpnTypedCellE5TrueShuffledDistinguisher,
        PresentCrossSpnTypedCellE5ShuffledPlaceboDistinguisher,
    ),
)
def test_cross_spn_e5_auxiliary_loss_is_finite_and_backpropagates(
    model_class: type[nn.Module],
) -> None:
    model = model_class(input_bits=4 * 128, base_channels=8)
    model.train()

    logits = model(_raw_features())
    assert model.last_auxiliary_loss is not None
    loss = logits.mean() + model.last_auxiliary_loss
    loss.backward()

    assert torch.isfinite(model.last_auxiliary_loss)
    gradients = [
        parameter.grad
        for parameter in model.topology_auxiliary_head.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_cross_spn_e5_off_disables_loss_but_preserves_transfer_state() -> None:
    source = PresentCrossSpnTypedCellE5OffDistinguisher(
        input_bits=16 * 128,
        base_channels=8,
    )
    target = GiftCrossSpnTypedCellE5FromPresentTrueShuffledDistinguisher(
        input_bits=4 * 128,
        base_channels=8,
    )

    source(_raw_features(pairs_per_sample=16))
    assert source.last_auxiliary_loss is None
    target.load_state_dict(source.state_dict(), strict=True)


def test_cross_spn_e5_counterfactual_mappings_are_distinct() -> None:
    candidate = PresentCrossSpnTypedCellE5TrueShuffledDistinguisher(
        input_bits=4 * 128,
        base_channels=8,
    )
    placebo = PresentCrossSpnTypedCellE5ShuffledPlaceboDistinguisher(
        input_bits=4 * 128,
        base_channels=8,
    )

    torch.testing.assert_close(
        candidate.topology_auxiliary_positive_indices,
        candidate.mapping_indices,
    )
    assert not torch.equal(
        candidate.topology_auxiliary_positive_indices,
        candidate.topology_auxiliary_negative_indices,
    )
    assert not torch.equal(
        placebo.topology_auxiliary_positive_indices,
        placebo.topology_auxiliary_negative_indices,
    )
    assert not torch.equal(
        placebo.topology_auxiliary_positive_indices,
        candidate.topology_auxiliary_positive_indices,
    )


E6_SOURCE_VARIANTS: tuple[type[nn.Module], ...] = (
    PresentCrossSpnTypedCellE6OffDistinguisher,
    PresentCrossSpnTypedCellE6FunctionalMarginDistinguisher,
    PresentCrossSpnTypedCellE6ShuffledPlaceboDistinguisher,
)


def test_cross_spn_e6_source_variants_preserve_e5_transfer_state() -> None:
    torch.manual_seed(20260715)
    reference = PresentCrossSpnTypedCellE5OffDistinguisher(
        input_bits=4 * 128,
        base_channels=8,
    )
    for model_class in E6_SOURCE_VARIANTS:
        torch.manual_seed(20260715)
        model = model_class(input_bits=4 * 128, base_channels=8)
        assert model.state_dict().keys() == reference.state_dict().keys()
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "model_class",
    (
        PresentCrossSpnTypedCellE6FunctionalMarginDistinguisher,
        PresentCrossSpnTypedCellE6ShuffledPlaceboDistinguisher,
    ),
)
def test_cross_spn_e6_functional_margin_is_finite_and_backpropagates(
    model_class: type[nn.Module],
) -> None:
    model = model_class(input_bits=4 * 128, base_channels=8)
    model.train()
    labels = torch.tensor([0.0, 1.0])

    logits = model(_raw_features()).squeeze(1)
    auxiliary_loss = model.compute_auxiliary_loss(logits, labels, "mse")

    assert auxiliary_loss is not None
    assert torch.isfinite(auxiliary_loss)
    assert model._last_functional_logits is not None
    assert len(model._last_functional_logits) == 2
    assert set(model.last_auxiliary_metrics) == {
        "functional_preferred_loss",
        "functional_comparison_loss",
        "functional_loss_gap",
        "functional_margin_loss",
        "functional_violation_rate",
    }
    (logits.mean() + auxiliary_loss).backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("topology_auxiliary_head")
        and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_cross_spn_e6_target_disables_margin_and_strictly_loads_source() -> None:
    source = PresentCrossSpnTypedCellE6FunctionalMarginDistinguisher(
        input_bits=16 * 128,
        base_channels=8,
    )
    target = GiftCrossSpnTypedCellE6FromPresentFunctionalMarginDistinguisher(
        input_bits=4 * 128,
        base_channels=8,
    )

    target.load_state_dict(source.state_dict(), strict=True)
    target.train()
    logits = target(_raw_features()).squeeze(1)
    auxiliary_loss = target.compute_auxiliary_loss(
        logits,
        torch.tensor([0.0, 1.0]),
        "mse",
    )

    assert auxiliary_loss is None
    assert target._last_functional_logits is None
    assert target.last_auxiliary_metrics == {}


@pytest.mark.parametrize("model_class", TYPED_VARIANTS)
def test_cross_spn_typed_variants_have_finite_forward_and_backward(
    model_class: type[nn.Module],
) -> None:
    model = model_class(input_bits=4 * 128, base_channels=8)

    logits = model(_raw_features())
    logits.mean().backward()

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pair_bits": 64}, "raw 128-bit ciphertext pairs"),
        ({"input_bits": 129}, "multiple of pair_bits"),
        ({"mixer_depth": 0}, "mixer_depth must be >= 1"),
        ({"pooling": "unsupported"}, "unsupported pooling"),
    ],
)
def test_cross_spn_typed_model_rejects_invalid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    options: dict[str, object] = {
        "input_bits": 4 * 128,
        "cipher_key": "present80",
        "mapping_mode": "true",
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        CrossSpnTypedCellPairSetDistinguisher(**options)


def test_cross_spn_typed_model_rejects_wrong_runtime_width() -> None:
    model = PresentCrossSpnTypedCellTrueDistinguisher(input_bits=4 * 128)

    with pytest.raises(ValueError, match="expected 512 input bits"):
        model.typed_cell_view(torch.zeros(2, 511))


def test_gift_raw_anchor_is_logit_equivalent_to_historical_aligned_input() -> None:
    options = {
        "input_bits": 4 * 128,
        "base_channels": 8,
        "mixer_depth": 1,
        "activation": "relu",
        "norm": "layernorm",
        "pooling": "topk_logsumexp",
        "top_k": 2,
        "lse_temperature": 1.0,
    }
    wrapper = GiftAlignedTokenMixerRawInputDistinguisher(**options)
    external = SpnTokenMixerPairSetDistinguisher(
        input_bits=4 * 256,
        pair_bits=256,
        base_channels=8,
        mixer_depth=1,
        activation="relu",
        norm="layernorm",
        pooling="topk_logsumexp",
        top_k=2,
        lse_temperature=1.0,
    )
    external.load_state_dict(wrapper.delegate.state_dict(), strict=True)
    features = _raw_features()

    actual = wrapper(features)
    expected = external(wrapper.aligned_view(features))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gift_raw_anchor_builds_exact_historical_word_order() -> None:
    wrapper = GiftAlignedTokenMixerRawInputDistinguisher(
        input_bits=4 * 128,
        base_channels=8,
    )
    features = _raw_features()
    pairs = features.reshape(2, 4, 2, 64)
    difference = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
    mapped = difference.index_select(2, wrapper.mapping_indices)
    expected = torch.cat(
        [pairs[:, :, 0], pairs[:, :, 1], difference, mapped], dim=2
    ).reshape(2, 4 * 256)

    actual = wrapper.aligned_view(features)

    torch.testing.assert_close(actual, expected)
    assert "mapping_indices" not in wrapper.state_dict()


def test_gift_raw_anchor_rejects_wrong_runtime_width() -> None:
    wrapper = GiftAlignedTokenMixerRawInputDistinguisher(input_bits=4 * 128)

    with pytest.raises(ValueError, match="expected 512 input bits"):
        wrapper.aligned_view(torch.zeros(2, 511))


REGISTERED_MODELS: tuple[tuple[str, type[nn.Module]], ...] = (
    (
        "present_cross_spn_typed_cell_true",
        PresentCrossSpnTypedCellTrueDistinguisher,
    ),
    (
        "present_cross_spn_typed_cell_shuffled",
        PresentCrossSpnTypedCellShuffledDistinguisher,
    ),
    (
        "present_cross_spn_typed_cell_raw",
        PresentCrossSpnTypedCellRawDistinguisher,
    ),
    ("gift_cross_spn_typed_cell_true", GiftCrossSpnTypedCellTrueDistinguisher),
    (
        "gift_cross_spn_typed_cell_no_position",
        GiftCrossSpnTypedCellNoPositionDistinguisher,
    ),
    (
        "gift_cross_spn_typed_cell_shared_view_encoder",
        GiftCrossSpnTypedCellSharedViewEncoderDistinguisher,
    ),
    (
        "gift_cross_spn_typed_cell_equivariant_mixer",
        GiftCrossSpnTypedCellEquivariantMixerDistinguisher,
    ),
    (
        "gift_cross_spn_typed_cell_shuffled",
        GiftCrossSpnTypedCellShuffledDistinguisher,
    ),
    ("gift_cross_spn_typed_cell_raw", GiftCrossSpnTypedCellRawDistinguisher),
    (
        "gift_cross_spn_aligned_token_mixer_raw_anchor",
        GiftAlignedTokenMixerRawInputDistinguisher,
    ),
    (
        "present_cross_spn_typed_cell_e5_off",
        PresentCrossSpnTypedCellE5OffDistinguisher,
    ),
    (
        "present_cross_spn_typed_cell_e5_true_shuffled",
        PresentCrossSpnTypedCellE5TrueShuffledDistinguisher,
    ),
    (
        "present_cross_spn_typed_cell_e5_shuffled_placebo",
        PresentCrossSpnTypedCellE5ShuffledPlaceboDistinguisher,
    ),
    (
        "gift_cross_spn_typed_cell_e5_from_present_true_shuffled",
        GiftCrossSpnTypedCellE5FromPresentTrueShuffledDistinguisher,
    ),
    (
        "present_cross_spn_typed_cell_e6_functional_margin",
        PresentCrossSpnTypedCellE6FunctionalMarginDistinguisher,
    ),
    (
        "present_cross_spn_typed_cell_e6_shuffled_placebo",
        PresentCrossSpnTypedCellE6ShuffledPlaceboDistinguisher,
    ),
    (
        "gift_cross_spn_typed_cell_e6_from_present_functional_margin",
        GiftCrossSpnTypedCellE6FromPresentFunctionalMarginDistinguisher,
    ),
)


@pytest.mark.parametrize(("model_key", "expected_type"), REGISTERED_MODELS)
def test_cross_spn_models_build_through_public_registry(
    model_key: str,
    expected_type: type[nn.Module],
) -> None:
    model = build_model(
        model_key,
        input_bits=4 * 128,
        hidden_bits=8,
        pair_bits=128,
        structure="SPN",
        model_options={
            "mixer_depth": 1,
            "token_mlp_ratio": 2,
            "activation": "relu",
            "norm": "layernorm",
            "dropout": 0.0,
        },
    )

    assert isinstance(model, expected_type)
    if isinstance(model, CrossSpnTypedCellPairSetDistinguisher):
        assert len(model.mixer_blocks) == 1
    else:
        assert len(model.delegate.mixer_blocks) == 1
