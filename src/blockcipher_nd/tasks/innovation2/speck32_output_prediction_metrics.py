from __future__ import annotations

from typing import Any

import numpy as np

from blockcipher_nd.training.metrics import binary_auc


def evaluate_speck32_output_scores(
    model_name: str,
    split: str,
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    if not model_name or not split:
        raise ValueError("model_name and split must be non-empty")
    scores, labels = _validated_arrays(probabilities, targets)
    predictions = scores > 0.5
    matches = predictions == labels
    exact_matches = np.all(matches, axis=1)
    prevalence = labels.mean(axis=0)
    majority_accuracy = np.maximum(prevalence, 1.0 - prevalence)
    per_bit_rows: list[dict[str, Any]] = []
    for msb_index in range(32):
        bit_scores = scores[:, msb_index]
        bit_labels = labels[:, msb_index]
        bit_accuracy = float(np.mean(matches[:, msb_index]))
        bit_majority = float(majority_accuracy[msb_index])
        per_bit_rows.append(
            {
                "model": model_name,
                "split": split,
                "target": "single_true_speck32_ciphertext_output_bit",
                "sample_classification": False,
                "msb_index": msb_index,
                "integer_bit": 31 - msb_index,
                "word_msb_index": msb_index // 16,
                "word_role": "x_msw" if msb_index < 16 else "y_lsw",
                "bit_in_word_msb": msb_index % 16,
                "rows": len(bit_labels),
                "prevalence": float(prevalence[msb_index]),
                "threshold_rule": "probability_gt_0.5_is_one",
                "threshold_accuracy": bit_accuracy,
                "majority_accuracy": bit_majority,
                "accuracy_minus_majority": bit_accuracy - bit_majority,
                "auc": float(binary_auc(bit_labels, bit_scores)),
                "bce": _binary_cross_entropy(bit_scores, bit_labels),
                "mse": float(np.mean(np.square(bit_scores - bit_labels))),
                "invalid_probability_rate": float(
                    np.mean((bit_scores < 0.0) | (bit_scores > 1.0))
                ),
            }
        )
    summary = {
        "model": model_name,
        "split": split,
        "target": "full_32_bit_true_speck32_ciphertext_output",
        "sample_classification": False,
        "rows": len(labels),
        "output_bits": 32,
        "threshold_rule": "probability_gt_0.5_is_one",
        "bap_avg": float(np.mean(matches)),
        "bit_match": float(np.mean(matches)),
        "majority_bap_avg": float(np.mean(majority_accuracy)),
        "macro_auc": float(np.mean([row["auc"] for row in per_bit_rows])),
        "bce": _binary_cross_entropy(scores, labels),
        "mse": float(np.mean(np.square(scores - labels))),
        "exact_match": float(np.mean(exact_matches)),
        "exact_match_count": int(np.sum(exact_matches)),
        "invalid_probability_rate": float(np.mean((scores < 0.0) | (scores > 1.0))),
    }
    return {"summary": summary, "per_bit_rows": per_bit_rows}


def _validated_arrays(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    if (
        scores.shape != labels.shape
        or scores.ndim != 2
        or scores.shape[1] != 32
        or len(scores) == 0
    ):
        raise ValueError("probabilities and targets must be non-empty [rows, 32]")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(labels)):
        raise ValueError("probabilities and targets must be finite")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("SPECK32 output targets must contain only zero or one")
    return scores, labels


def _binary_cross_entropy(scores: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    return float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    )


__all__ = ["evaluate_speck32_output_scores"]
