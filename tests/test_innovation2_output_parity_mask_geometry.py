from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_output_parity_mask_geometry import (
    render_output_parity_mask_geometry,
)
from blockcipher_nd.tasks.innovation2.output_parity_mask_geometry import (
    ALIGNED_MASKS,
    CONTIGUOUS_MASKS,
    adjudicate_mask_geometry,
    build_mask_geometry_data,
    mask_positions,
    validate_mask_geometry_contract,
)
from blockcipher_nd.tasks.innovation2.output_parity_prediction import (
    OutputParityPredictionConfig,
    parity_targets_from_full_bits,
)


def _small_config() -> OutputParityPredictionConfig:
    return OutputParityPredictionConfig(
        run_id="test-output-parity-mask-geometry",
        train_rows=128,
        validation_rows=64,
        test_rows=64,
        hidden_dim=16,
        epochs=1,
        batch_size=32,
    )


def test_aligned_masks_are_weight_four_present_p_layer_partition() -> None:
    assert len(set(ALIGNED_MASKS)) == 16
    assert all(mask.bit_count() == 4 for mask in ALIGNED_MASKS)
    assert sum(ALIGNED_MASKS) == (1 << 64) - 1
    assert mask_positions(CONTIGUOUS_MASKS[0]) == (0, 1, 2, 3)
    assert mask_positions(ALIGNED_MASKS[0]) == (0, 16, 32, 48)


def test_mask_geometry_changes_only_real_ciphertext_output_masks() -> None:
    config = _small_config()
    datasets = build_mask_geometry_data(config)
    checks = validate_mask_geometry_contract(config, datasets)

    assert all(checks.values())
    assert np.array_equal(
        datasets["contiguous"]["train"].full_targets,
        datasets["aligned"]["train"].full_targets,
    )
    assert np.array_equal(
        datasets["aligned"]["test"].parity_targets,
        parity_targets_from_full_bits(
            datasets["aligned"]["test"].full_targets,
            ALIGNED_MASKS,
        ),
    )
    assert checks["labels_are_ciphertext_outputs_not_sample_classes"] is True


def test_mask_geometry_gate_requires_candidate_anchor_and_shuffle_margins() -> None:
    config = _small_config()

    def row(model: str, auc: float, seed: int) -> dict[str, object]:
        return {
            "model": model,
            "model_seed": seed,
            "test_loss": 0.6,
            "test_accuracy": auc,
            "test_macro_auc": auc,
            "test_exact_match": 0.0,
        }

    rows = [
        row("full_output_mlp", 0.64, 0),
        row("contiguous_parity_mlp", 0.50, 1000),
        row("aligned_parity_mlp", 0.60, 1000),
        row("aligned_parity_label_shuffle", 0.50, 2000),
    ]
    training = {
        "rows": rows,
        "history": [{} for _ in range(4)],
        "trained": {
            "aligned_parity_label_shuffle": {
                "test_target_identity": "true_parity_targets"
            }
        },
    }

    passed = adjudicate_mask_geometry(config, {"protocol": True}, training)
    rows[2]["test_macro_auc"] = 0.52
    held = adjudicate_mask_geometry(config, {"protocol": True}, training)

    assert passed["status"] == "pass"
    assert passed["decision"] == "innovation2_output_parity_mask_geometry_supported"
    assert passed["next_action"]["sample_classification"] is False
    assert held["status"] == "hold"
    assert held["decision"] == (
        "innovation2_output_parity_mask_geometry_not_calibrated"
    )


def test_mask_geometry_plot_explains_real_output_target(tmp_path: Path) -> None:
    summary = {
        "trained_rows": [
            {
                "model": "contiguous_parity_mlp",
                "test_accuracy": 0.50,
                "test_macro_auc": 0.49,
            },
            {
                "model": "aligned_parity_mlp",
                "test_accuracy": 0.68,
                "test_macro_auc": 0.72,
            },
            {
                "model": "aligned_parity_label_shuffle",
                "test_accuracy": 0.50,
                "test_macro_auc": 0.50,
            },
        ],
        "gate": {
            "status": "pass",
            "decision": "innovation2_output_parity_mask_geometry_supported",
            "metrics": {
                "aligned_minus_contiguous_macro_auc": 0.23,
                "aligned_minus_shuffled_macro_auc": 0.22,
                "full_bit_macro_auc": 0.64,
            },
        },
    }
    output = tmp_path / "curves.svg"

    render_output_parity_mask_geometry(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "真实密文输出parity" in svg
    assert "没有real-vs-random" in svg
    assert "同一S-box对齐" in svg
    assert "不是高轮攻击" in svg
