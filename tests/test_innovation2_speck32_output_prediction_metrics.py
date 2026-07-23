from __future__ import annotations

import numpy as np
import pytest

from blockcipher_nd.tasks.innovation2.speck32_output_prediction_metrics import (
    evaluate_speck32_output_scores,
)


def test_perfect_scores_report_bap_auc_and_full_exact_match() -> None:
    targets = np.asarray(
        [
            [0, 1] * 16,
            [1, 0] * 16,
            [0, 1] * 16,
            [1, 0] * 16,
        ],
        dtype=np.float32,
    )
    probabilities = targets * 0.98 + (1.0 - targets) * 0.02

    result = evaluate_speck32_output_scores(
        "model",
        "test",
        probabilities,
        targets,
    )

    summary = result["summary"]
    assert summary["bap_avg"] == 1.0
    assert summary["bit_match"] == 1.0
    assert summary["macro_auc"] == 1.0
    assert summary["exact_match"] == 1.0
    assert summary["exact_match_count"] == 4
    assert summary["majority_bap_avg"] == 0.5
    assert summary["sample_classification"] is False
    assert len(result["per_bit_rows"]) == 32


def test_paper_threshold_treats_exactly_half_as_zero() -> None:
    targets = np.zeros((3, 32), dtype=np.float32)
    probabilities = np.full((3, 32), 0.5, dtype=np.float32)

    result = evaluate_speck32_output_scores(
        "model",
        "test",
        probabilities,
        targets,
    )

    assert result["summary"]["threshold_rule"] == "probability_gt_0.5_is_one"
    assert result["summary"]["bap_avg"] == 1.0
    assert result["summary"]["exact_match_count"] == 3


def test_bit_rows_preserve_msb_integer_and_speck_word_roles() -> None:
    targets = np.tile(np.asarray([[0], [1]], dtype=np.float32), (1, 32))
    probabilities = targets * 0.8 + (1.0 - targets) * 0.2

    rows = evaluate_speck32_output_scores(
        "model",
        "test",
        probabilities,
        targets,
    )["per_bit_rows"]

    assert rows[0]["msb_index"] == 0
    assert rows[0]["integer_bit"] == 31
    assert rows[0]["word_msb_index"] == 0
    assert rows[0]["word_role"] == "x_msw"
    assert rows[0]["bit_in_word_msb"] == 0
    assert rows[15]["word_role"] == "x_msw"
    assert rows[15]["bit_in_word_msb"] == 15
    assert rows[16]["word_msb_index"] == 1
    assert rows[16]["word_role"] == "y_lsw"
    assert rows[31]["integer_bit"] == 0
    assert rows[31]["bit_in_word_msb"] == 15


def test_majority_baseline_and_invalid_probability_rate_are_explicit() -> None:
    targets = np.zeros((4, 32), dtype=np.float32)
    targets[-1] = 1.0
    probabilities = np.full((4, 32), 0.2, dtype=np.float32)
    probabilities[0, 0] = -0.1
    probabilities[1, 0] = 1.1

    result = evaluate_speck32_output_scores(
        "model",
        "test",
        probabilities,
        targets,
    )

    first = result["per_bit_rows"][0]
    assert first["majority_accuracy"] == 0.75
    assert first["invalid_probability_rate"] == 0.5
    assert result["summary"]["invalid_probability_rate"] == pytest.approx(2 / 128)
    assert np.isfinite(result["summary"]["bce"])


@pytest.mark.parametrize(
    ("probabilities", "targets", "message"),
    (
        (np.zeros((2, 31)), np.zeros((2, 31)), r"\[rows, 32\]"),
        (np.zeros((0, 32)), np.zeros((0, 32)), r"\[rows, 32\]"),
        (
            np.full((2, 32), np.nan),
            np.zeros((2, 32)),
            "must be finite",
        ),
        (
            np.zeros((2, 32)),
            np.full((2, 32), 0.25),
            "only zero or one",
        ),
    ),
)
def test_invalid_metric_inputs_fail_closed(
    probabilities: np.ndarray,
    targets: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_speck32_output_scores("model", "test", probabilities, targets)
