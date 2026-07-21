from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_output_parity_prediction import (
    render_output_parity_readiness,
)
from blockcipher_nd.tasks.innovation2.output_parity_prediction import (
    MASKS,
    OutputParityPredictionConfig,
    adjudicate_output_prediction_readiness,
    generate_output_prediction_data,
    parity_probabilities_from_bit_probabilities,
    parity_targets_from_full_bits,
    validate_output_prediction_contract,
)


def _small_config() -> OutputParityPredictionConfig:
    return OutputParityPredictionConfig(
        run_id="test-output-parity",
        train_rows=128,
        validation_rows=64,
        test_rows=64,
        hidden_dim=16,
        epochs=1,
        batch_size=32,
    )


def test_output_parity_data_is_fixed_key_output_prediction_not_classification() -> None:
    config = _small_config()
    data = generate_output_prediction_data(config)
    checks = validate_output_prediction_contract(config, data)

    assert all(checks.values())
    assert len(MASKS) == 16
    assert data["train"].features.shape == (128, 64)
    assert data["train"].full_targets.shape == (128, 64)
    assert data["train"].parity_targets.shape == (128, 16)
    assert np.array_equal(
        data["train"].parity_targets,
        parity_targets_from_full_bits(data["train"].full_targets),
    )
    assert checks["no_sample_classification_label"] is True


def test_parity_probability_matches_known_independent_bit_cases() -> None:
    probabilities = np.zeros((2, 64), dtype=np.float32)
    probabilities[0, :4] = (1.0, 0.0, 1.0, 1.0)
    probabilities[1, :4] = 0.5

    parity = parity_probabilities_from_bit_probabilities(probabilities)

    assert parity.shape == (2, 16)
    assert parity[0, 0] == 1.0
    assert parity[0, 1] == 0.0
    assert parity[1, 0] == 0.5


def test_output_parity_gate_opens_only_for_complete_protocol() -> None:
    config = _small_config()
    rows = [
        {
            "model": "full_output_mlp",
            "test_loss": 0.69,
            "test_accuracy": 0.51,
            "test_macro_auc": 0.52,
            "test_exact_match": 0.0,
            "derived_parity_accuracy": 0.50,
            "derived_parity_macro_auc": 0.51,
        },
        {
            "model": "direct_parity_mlp",
            "test_loss": 0.68,
            "test_accuracy": 0.53,
            "test_macro_auc": 0.54,
            "test_exact_match": 0.0,
            "parity_accuracy": 0.53,
        },
        {
            "model": "direct_parity_label_shuffle",
            "test_loss": 0.70,
            "test_accuracy": 0.49,
            "test_macro_auc": 0.48,
            "test_exact_match": 0.0,
            "parity_accuracy": 0.49,
        },
    ]
    training = {
        "rows": rows,
        "history": [{} for _ in range(3)],
        "trained": {
            "direct_parity_label_shuffle": {
                "test_target_identity": "true_parity_targets"
            }
        },
    }

    gate = adjudicate_output_prediction_readiness(
        config, {"fixed_key": True}, training
    )
    invalid = adjudicate_output_prediction_readiness(
        config, {"fixed_key": False}, training
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_output_parity_prediction_readiness_passed"
    assert gate["next_action"]["sample_classification"] is False
    assert invalid["status"] == "fail"


def test_output_parity_plot_names_output_target_and_scope(tmp_path: Path) -> None:
    summary = {
        "trained_rows": [
            {
                "model": "full_output_mlp",
                "derived_parity_accuracy": 0.51,
                "derived_parity_macro_auc": 0.52,
            },
            {
                "model": "direct_parity_mlp",
                "parity_accuracy": 0.55,
                "parity_macro_auc": 0.56,
            },
            {
                "model": "direct_parity_label_shuffle",
                "parity_accuracy": 0.50,
                "parity_macro_auc": 0.49,
            },
        ],
        "gate": {
            "status": "pass",
            "metrics": {
                "full_bit_accuracy": 0.58,
                "direct_minus_derived_parity": 0.04,
                "direct_minus_shuffled_parity": 0.05,
            },
        },
    }
    output = tmp_path / "curves.svg"

    render_output_parity_readiness(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "直接预测密文输出parity" in svg
    assert "没有真假样本" in svg
    assert "下一步只开放一轮mask几何校准" in svg
    assert "不是高轮结果" in svg
