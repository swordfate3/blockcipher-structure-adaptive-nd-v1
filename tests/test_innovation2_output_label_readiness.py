from __future__ import annotations

from blockcipher_nd.tasks.innovation2.integral_output_label_readiness import (
    OutputLabelReadinessConfig,
    adjudicate_output_label_readiness,
    bounded_span,
    matched_negative_masks,
)


def test_bounded_span_and_matched_controls_are_safe() -> None:
    basis = (1, 2, 4)
    span = bounded_span(basis, max_dimension=3)

    assert span == set(range(8))
    controls = matched_negative_masks(
        basis,
        excluded_masks=span,
        seed=0,
    )
    assert len(controls) == len(basis)
    assert [mask.bit_count() for mask in controls] == [1, 1, 1]
    assert set(controls).isdisjoint(span)


def test_output_label_gate_holds_for_perfect_additive_shortcut() -> None:
    config = OutputLabelReadinessConfig(run_id="test")
    baselines = [
        {"baseline": "active_block_marginal", "accuracy": 0.75},
        {"baseline": "mask_weight_marginal", "accuracy": 0.75},
        {"baseline": "mask_identity_marginal", "accuracy": 0.94},
        {"baseline": "block_mask_additive", "accuracy": 1.0},
    ]

    gate = adjudicate_output_label_readiness(
        config,
        baselines,
        {"ok": True},
        positive_rate=0.3,
        flipping_masks=4,
        distinct_block_label_signatures=2,
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_output_label_shortcut_dominated"
    assert gate["shortcut_checks"][
        "block_mask_additive_accuracy_below_0p98"
    ] is False
