from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_output_parity_mask_geometry import (
    render_output_parity_mask_geometry,
)
from blockcipher_nd.cli.plot_innovation2_output_parity_independent_key import (
    render_output_parity_independent_key,
)
from blockcipher_nd.tasks.innovation2.output_parity_mask_geometry import (
    ALIGNED_MASKS,
    CONTIGUOUS_MASKS,
    adjudicate_mask_geometry,
    adjudicate_two_key_confirmation,
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


def test_two_key_confirmation_requires_both_supported_and_independent() -> None:
    def gate(seed: int, *, supported: bool = True) -> dict[str, object]:
        return {
            "status": "pass" if supported else "hold",
            "decision": (
                "innovation2_output_parity_mask_geometry_supported"
                if supported
                else "innovation2_output_parity_mask_geometry_not_calibrated"
            ),
            "metrics": {
                "aligned_parity_macro_auc": 0.95 - seed * 0.01,
                "aligned_minus_contiguous_macro_auc": 0.44,
                "aligned_minus_shuffled_macro_auc": 0.43,
            },
            "thresholds": {"aligned_macro_auc_min": 0.55},
        }

    passed = adjudicate_two_key_confirmation(
        "op3",
        gate(0),
        gate(1),
        {"keys_differ": True, "plaintexts_disjoint": True},
    )
    held = adjudicate_two_key_confirmation(
        "op3",
        gate(0),
        gate(1, supported=False),
        {"keys_differ": True, "plaintexts_disjoint": True},
    )
    invalid = adjudicate_two_key_confirmation(
        "op3",
        gate(0),
        gate(1),
        {"keys_differ": False, "plaintexts_disjoint": True},
    )

    assert passed["status"] == "pass"
    assert passed["decision"] == (
        "innovation2_output_parity_mask_geometry_two_key_confirmed"
    )
    assert passed["next_action"]["next_adjudication"] == (
        "op4_present_r2_two_key_round_step"
    )
    assert held["status"] == "hold"
    assert invalid["status"] == "fail"


def test_independent_key_plot_preserves_output_prediction_scope(
    tmp_path: Path,
) -> None:
    metrics = {
        "contiguous_parity_macro_auc": 0.50,
        "aligned_parity_macro_auc": 0.95,
        "shuffled_aligned_parity_macro_auc": 0.51,
    }
    summary = {
        "gate": {
            "status": "pass",
            "decision": "innovation2_output_parity_mask_geometry_two_key_confirmed",
            "metrics": {
                "seed0": metrics,
                "seed1": {**metrics, "aligned_parity_macro_auc": 0.94},
                "minimum_aligned_parity_macro_auc": 0.94,
                "mean_aligned_parity_macro_auc": 0.945,
                "aligned_parity_macro_auc_range": 0.01,
            },
        }
    }
    output = tmp_path / "curves.svg"

    render_output_parity_independent_key(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "独立固定密钥确认" in svg
    assert "真实密文四位置异或输出" in svg
    assert "不是真假或平衡类别" in svg
    assert "不是高轮攻击" in svg
