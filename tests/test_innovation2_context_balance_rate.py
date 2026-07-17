from __future__ import annotations

import numpy as np

from blockcipher_nd.tasks.innovation2 import integral_context_balance_rate as rate
from blockcipher_nd.tasks.innovation2.integral_context_balance_rate import (
    ContextBalanceRateConfig,
    additive_residuals,
    adjudicate_context_balance_rates,
    output_nibble_masks,
    pearson_correlation,
)
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import gf2_kernel_basis


def test_output_nibble_masks_cover_all_nonzero_four_bit_masks() -> None:
    masks = output_nibble_masks()

    assert len(masks) == 240
    assert len({mask_value for _, _, mask_value in masks}) == 240
    assert masks[0] == (0, 1, 1)
    assert masks[14] == (0, 15, 15)
    assert masks[-1] == (15, 15, 15 << 60)


def test_additive_residuals_remove_context_and_mask_marginals() -> None:
    context_effect = np.asarray([0.1, 0.3, 0.5])[:, None]
    mask_effect = np.asarray([0.2, 0.4, 0.6, 0.8])[None, :]
    values = context_effect + mask_effect

    residuals = additive_residuals(values)

    np.testing.assert_allclose(residuals, 0.0, atol=1e-12)
    assert pearson_correlation(values, values) == 1.0


def test_balance_rate_gate_requires_reproducible_excess_interaction() -> None:
    config = ContextBalanceRateConfig(run_id="test")
    metrics = {
        "rate_half_correlation": 0.8,
        "interaction_residual_half_correlation": 0.4,
        "mean_absolute_half_rate_difference": 0.1,
        "validation_residual_standard_deviation": 0.08,
        "interaction_excess_variance": 0.002,
        "context_shuffle_residual_correlation": 0.02,
        "label_shuffle_residual_correlation": -0.03,
    }
    ready = adjudicate_context_balance_rates(
        config,
        {"ok": True},
        metrics,
        reproduced_signatures=64,
    )
    noisy = adjudicate_context_balance_rates(
        config,
        {"ok": True},
        {**metrics, "interaction_excess_variance": 0.0},
        reproduced_signatures=64,
    )

    assert ready["decision"] == "innovation2_balance_rate_interaction_ready"
    assert noisy["decision"] == (
        "innovation2_balance_rate_interaction_not_reproducible"
    )


def test_balance_rate_runner_returns_valid_no_interaction_result(monkeypatch) -> None:
    constrained = {0, 4, 12, 16, 48, 20, 28, 52, 60}
    orthogonal_rows = [1 << bit for bit in range(64) if bit not in constrained]
    orthogonal_rows.extend(
        [
            (1 << 4) | (1 << 12),
            (1 << 16) | (1 << 48),
            (1 << 20) | (1 << 28),
            (1 << 20) | (1 << 52),
            (1 << 20) | (1 << 60),
        ]
    )
    words = np.asarray(
        orthogonal_rows + orthogonal_rows + orthogonal_rows[:8],
        dtype=np.uint64,
    )
    monkeypatch.setattr(
        rate,
        "_collect_xor_words",
        lambda structure, keys, **kwargs: words.copy(),
    )
    monkeypatch.setattr(
        rate,
        "scalar_bit_integral_output_xor",
        lambda structure, rounds, key: int(words[0]),
    )
    signature = ":".join(
        f"{vector:016X}" for vector in gf2_kernel_basis(words)
    )
    source_rows = [
        {
            "run_id": "source",
            "context_id": context_id,
            "joint_basis_signature": signature,
        }
        for context_id in range(64)
    ]
    result = rate.run_context_balance_rate_audit(
        ContextBalanceRateConfig(run_id="runner"),
        source_gate={
            "status": "hold",
            "decision": "innovation2_context_kernel_fresh_key_unstable",
        },
        source_metadata={
            "task": "innovation2_present_r7_fresh_expanded_context_kernel_diversity",
            "training_performed": False,
            "fresh_keys": 128,
            "fresh_key_generation_seed": 8801,
            "contexts": [f"0x{context_id:012X}" for context_id in range(64)],
        },
        source_result_rows=source_rows,
    )

    assert result["xor_words"].shape == (64, 128)
    assert len(result["cell_rows"]) == 64 * 240
    assert result["gate"]["reproduced_e18_joint_signatures"] == 64
    assert result["gate"]["status"] == "hold"
    assert all(result["gate"]["readiness_checks"].values())
