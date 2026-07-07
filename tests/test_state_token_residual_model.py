from __future__ import annotations

import pytest
import torch

from blockcipher_nd.models.registry import get_model_class
from blockcipher_nd.models.structure.spn import PresentStateTokenResidualDistinguisher


def test_present_state_token_residual_forward_uses_span_tokens():
    model = PresentStateTokenResidualDistinguisher(
        input_bits=3708,
        token_dim=8,
        hidden_bits=16,
        dropout=0.0,
    )
    features = torch.randn(3, 3708)

    logits = model(features)

    assert logits.shape == (3, 1)
    assert model.selected_span_feature_bits == 731
    assert model.last_attention_weights is not None
    assert model.last_attention_weights.shape == (3, 731)


def test_present_state_token_residual_is_registered():
    model_cls = get_model_class("present_state_token_residual")

    assert model_cls is PresentStateTokenResidualDistinguisher


def test_present_state_token_residual_rejects_unexpected_feature_layout():
    with pytest.raises(ValueError, match="expects trail_position_stats feature layout"):
        PresentStateTokenResidualDistinguisher(input_bits=117)
