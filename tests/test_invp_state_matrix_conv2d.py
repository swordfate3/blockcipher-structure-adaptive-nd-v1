import pytest
import torch

from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
    PresentNibbleInvPOnlySpnOnlyDistinguisher,
    present_inverse_p_indices,
)
from blockcipher_nd.models.structure.spn.present_invp_state_matrix_conv2d import (
    PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher,
    PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher,
    PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher,
    PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher,
)


INPUT_BITS = 16 * 128


def _raw_features(batch: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260711)
    return torch.randint(0, 2, (batch, INPUT_BITS), generator=generator).float()


def test_present_inverse_p_indices_are_deterministic_distinct_permutations():
    true_first = present_inverse_p_indices("true")
    true_second = present_inverse_p_indices("true")
    shuffled_first = present_inverse_p_indices("shuffled")
    shuffled_second = present_inverse_p_indices("shuffled")

    expected = list(range(64))
    assert true_first.dtype == torch.long
    assert shuffled_first.dtype == torch.long
    assert sorted(true_first.tolist()) == expected
    assert sorted(shuffled_first.tolist()) == expected
    assert torch.equal(true_first, true_second)
    assert torch.equal(shuffled_first, shuffled_second)
    assert not torch.equal(true_first, shuffled_first)


def test_present_inverse_p_indices_rejects_unknown_alignment():
    with pytest.raises(ValueError) as exc_info:
        present_inverse_p_indices("unknown")

    assert str(exc_info.value) == "unsupported p_alignment: unknown"


def test_invp_anchor_uses_shared_true_inverse_p_indices():
    model = PresentNibbleInvPOnlySpnOnlyDistinguisher(
        input_bits=16 * 128,
        pair_bits=128,
        base_channels=32,
    )

    assert torch.equal(model.spn_encoder.inverse_p_indices, present_inverse_p_indices("true"))


def test_true_state_matrix_view_matches_existing_invp_anchor():
    features = _raw_features()
    candidate = PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher(
        input_bits=INPUT_BITS
    )
    anchor = PresentNibbleInvPOnlySpnOnlyDistinguisher(input_bits=INPUT_BITS)

    expected = anchor.spn_encoder.present_nibble_paligned_view(features).reshape(3, 16, 4, 16)

    assert torch.equal(candidate.state_matrix_view(features), expected)


def test_delta_state_matrix_view_uses_unaligned_raw_difference():
    features = _raw_features()
    candidate = PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher(
        input_bits=INPUT_BITS
    )
    raw_pairs = features.reshape(3, 16, 2, 64)
    expected = (
        (raw_pairs[:, :, 0, :] - raw_pairs[:, :, 1, :])
        .abs()
        .reshape(3, 16, 16, 4)
        .permute(0, 1, 3, 2)
    )

    assert torch.equal(candidate.state_matrix_view(features), expected)


def test_shuffled_state_matrix_view_is_deterministic_and_differs_from_true():
    features = _raw_features()
    first = PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher(
        input_bits=INPUT_BITS
    )
    second = PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher(
        input_bits=INPUT_BITS
    )
    true = PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher(input_bits=INPUT_BITS)

    assert torch.equal(first.state_matrix_view(features), second.state_matrix_view(features))
    assert not torch.equal(first.state_matrix_view(features), true.state_matrix_view(features))


def test_shuffled_state_matrix_mapping_identity_is_locked():
    expected = torch.tensor(
        [
            34,
            54,
            25,
            2,
            42,
            32,
            33,
            23,
            11,
            13,
            48,
            45,
            38,
            43,
            10,
            56,
            36,
            58,
            5,
            9,
            51,
            59,
            29,
            17,
            53,
            27,
            6,
            15,
            57,
            41,
            61,
            40,
            20,
            37,
            19,
            26,
            12,
            63,
            30,
            22,
            49,
            0,
            3,
            14,
            21,
            18,
            39,
            7,
            1,
            28,
            31,
            24,
            52,
            60,
            44,
            50,
            8,
            47,
            35,
            55,
            46,
            4,
            62,
            16,
        ],
        dtype=torch.long,
    )
    candidate = PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher(
        input_bits=INPUT_BITS
    )

    assert torch.equal(candidate.mapping_indices, expected)


def test_state_matrix_conv2d_variants_have_equal_parameter_counts():
    variants = [
        PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher(input_bits=INPUT_BITS),
        PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher(input_bits=INPUT_BITS),
        PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher(input_bits=INPUT_BITS),
    ]

    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in variants]

    assert counts[0] == counts[1] == counts[2]


@pytest.mark.parametrize(
    "model_class",
    [
        PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher,
        PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher,
        PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher,
    ],
)
def test_state_matrix_conv2d_variants_support_finite_backward(model_class):
    model = model_class(input_bits=INPUT_BITS)

    logits = model(_raw_features(batch=2))
    logits.sum().backward()

    assert logits.shape == (2, 1)
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pair_bits": 64}, "expects raw 128-bit ciphertext pairs"),
        ({"conv_depth": 0}, "conv_depth must be >= 1"),
        ({"kernel_size": 2}, "kernel_size must be a positive odd integer"),
        ({"mapping_mode": "bad"}, "unsupported mapping_mode: bad"),
    ],
)
def test_state_matrix_conv2d_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher(
            input_bits=INPUT_BITS,
            **kwargs,
        )


def test_state_matrix_view_rejects_invalid_feature_shape():
    model = PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher(input_bits=INPUT_BITS)

    with pytest.raises(ValueError, match=f"expected {INPUT_BITS} input bits"):
        model.state_matrix_view(torch.zeros(2, INPUT_BITS - 1))
