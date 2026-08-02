from __future__ import annotations

import torch

from blockcipher_nd.models.structure.spn.present_zhang_wang_keras import (
    PresentZhangWangKerasMCNDDistinguisher,
    SpnZhangWangMCNDAdapterDistinguisher,
)
from blockcipher_nd.models.structure.spn.published_architecture_adapters import (
    SpnLiuCase3Conv2DAdapterDistinguisher,
)
from blockcipher_nd.registry.model_factory import build_model


def test_zhang_wang_adapter_exactly_preserves_present_port() -> None:
    torch.manual_seed(7)
    adapted = SpnZhangWangMCNDAdapterDistinguisher(
        input_bits=256, pair_bits=128, base_channels=8, blocks=2
    )
    legacy = PresentZhangWangKerasMCNDDistinguisher(
        input_bits=256, pair_bits=128, base_channels=8, blocks=2
    )
    legacy.load_state_dict(adapted.state_dict(), strict=True)
    adapted.eval()
    legacy.eval()
    features = torch.randint(0, 2, (3, 256)).float()

    assert torch.equal(adapted(features), legacy(features))


def test_published_adapters_support_uknit_and_dialga_pair_widths() -> None:
    for pair_bits in (128, 256):
        input_bits = pair_bits * 2
        features = torch.randint(0, 2, (2, input_bits)).float()
        models = (
            SpnZhangWangMCNDAdapterDistinguisher(
                input_bits=input_bits,
                pair_bits=pair_bits,
                base_channels=8,
                blocks=1,
            ),
            SpnLiuCase3Conv2DAdapterDistinguisher(
                input_bits=input_bits,
                pair_bits=pair_bits,
                base_channels=8,
                conv_depth=1,
            ),
        )
        for model in models:
            logits = model(features)
            logits.square().mean().backward()
            assert logits.shape == (2, 1)
            assert torch.isfinite(logits).all()
            assert any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )


def test_liu_case3_view_is_c_cprime_and_raw_difference() -> None:
    model = SpnLiuCase3Conv2DAdapterDistinguisher(
        input_bits=128, pair_bits=128, base_channels=8, conv_depth=1
    )
    features = torch.zeros(1, 128)
    features[0, 0] = 1
    features[0, 67] = 1

    view = model.case3_view(features)
    recovered = view.permute(0, 1, 2, 4, 3).reshape(1, 1, 3, 64)

    assert view.shape == (1, 1, 3, 4, 16)
    assert torch.equal(recovered[0, 0, 0], features[0, :64])
    assert torch.equal(recovered[0, 0, 1], features[0, 64:])
    assert torch.equal(
        recovered[0, 0, 2],
        (features[0, :64] - features[0, 64:]).abs(),
    )


def test_model_factory_builds_both_published_adapters() -> None:
    for name in (
        "spn_zhang_wang_mcnd_adapter",
        "spn_liu_case3_conv2d_adapter",
    ):
        model = build_model(
            name,
            input_bits=2048,
            hidden_bits=8,
            pair_bits=128,
            structure="SPN",
            model_options={"cell_bits": 4},
        )
        assert model(torch.zeros(2, 2048)).shape == (2, 1)


def test_legacy_present_port_still_rejects_non_present_pair_width() -> None:
    try:
        PresentZhangWangKerasMCNDDistinguisher(input_bits=256, pair_bits=256)
    except ValueError as exc:
        assert "expects 128 bits" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("legacy PRESENT port accepted a non-PRESENT pair width")
