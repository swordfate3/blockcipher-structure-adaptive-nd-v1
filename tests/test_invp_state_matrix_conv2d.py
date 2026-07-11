import pytest
import torch

from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
    PresentNibbleInvPOnlySpnOnlyDistinguisher,
    present_inverse_p_indices,
)


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
