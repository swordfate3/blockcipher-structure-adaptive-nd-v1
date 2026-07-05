from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from blockcipher_nd.engine.datasets import make_task_dataset
from blockcipher_nd.engine.modeling import configure_structure_aware_model, infer_pair_bits
from blockcipher_nd.engine.task_config import build_dataset_config, build_training_config, resolve_task_keys
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.training import predict_binary_probabilities, train_binary_classifier
from blockcipher_nd.training.metrics import binary_auc, best_threshold_accuracy_and_threshold


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small projection-feature matrix and evaluate whether weak "
            "PRESENT/SPN views improve when ensembled on the same validation set."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-bits", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "lion"])
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", default="bce", choices=["bce", "mse"])
    parser.add_argument(
        "--lr-scheduler",
        default="none",
        choices=["none", "cyclic", "cosine_warmup", "official_cyclic"],
    )
    parser.add_argument("--max-learning-rate", type=float, default=None)
    parser.add_argument(
        "--checkpoint-metric",
        default="val_auc",
        choices=["val_accuracy", "val_auc", "val_loss"],
    )
    parser.add_argument("--restore-best-checkpoint", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--train-eval-interval", type=int, default=0)
    parser.add_argument("--dataset-cache-root", default=None)
    parser.add_argument("--dataset-cache-chunk-size", type=int, default=8192)
    parser.add_argument("--dataset-cache-workers", type=int, default=1)
    parser.add_argument("--progress-output", default=None)
    parser.add_argument("--checkpoint-output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = build_tasks(_task_args(args))
    if len(tasks) < 2:
        raise ValueError("ensemble evaluation requires at least two plan rows")

    row_reports: list[dict[str, Any]] = []
    probability_columns: list[np.ndarray] = []
    reference_labels: np.ndarray | None = None
    for index, task in enumerate(tasks, start=1):
        report, labels, probabilities = train_projection_row(task, args, index=index, total=len(tasks))
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise ValueError(
                "ensemble rows produced different validation label orders; "
                "keep seed/sample_structure/negative_mode/validation split fixed"
            )
        row_reports.append(report)
        probability_columns.append(probabilities)

    assert reference_labels is not None
    probability_matrix = np.stack(probability_columns, axis=1)
    ensemble_reports = ensemble_metrics(reference_labels, probability_matrix, row_reports)
    diversity_report = diversity_metrics(reference_labels, probability_matrix, row_reports)
    best_single = max(row_reports, key=lambda row: row["metrics"]["auc"])
    best_ensemble = max(ensemble_reports, key=lambda row: row["metrics"]["auc"])
    summary = {
        "status": "pass",
        "plan": str(args.plan),
        "rows": row_reports,
        "ensembles": ensemble_reports,
        "diversity": diversity_report,
        "best_single": best_single,
        "best_ensemble": best_ensemble,
        "delta_best_ensemble_vs_single_auc": float(
            best_ensemble["metrics"]["auc"] - best_single["metrics"]["auc"]
        ),
        "claim_scope": (
            "same-validation projection ensemble diagnostic only; not a formal "
            "PRESENT/SPN result and not a replacement for held-out confirmation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


def _task_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        plan=str(args.plan),
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
        key_rotation_interval=0,
        sample_structure="independent_pairs",
        integral_active_nibble=0,
    )


def train_projection_row(
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    index: int,
    total: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    train_key, validation_key = resolve_task_keys(task)
    train_cipher = build_cipher(task["cipher_key"], task["rounds"], key=train_key)
    validation_cipher = build_cipher(task["cipher_key"], task["rounds"], key=validation_key)
    pair_bits = infer_pair_bits(train_cipher.block_bits, task["feature_encoding"])
    train_dataset = make_task_dataset(
        build_dataset_config(
            task,
            cipher=train_cipher,
            samples_per_class=task["samples_per_class"],
            seed=task["seed"],
        ),
        args,
        task,
        split="ensemble_train",
        index=index,
        total=total,
    )
    validation_dataset = make_task_dataset(
        build_dataset_config(
            task,
            cipher=validation_cipher,
            samples_per_class=max(8, task["samples_per_class"] // 2),
            seed=task["seed"] + 10_000,
        ),
        args,
        task,
        split="ensemble_validation",
        index=index,
        total=total,
    )
    model = build_model(
        task["model_key"],
        input_bits=train_dataset.features.shape[1],
        hidden_bits=args.hidden_bits,
        pair_bits=pair_bits,
        structure=train_cipher.structure,
        model_options=task.get("model_options"),
    )
    configure_structure_aware_model(model, task["cipher_key"], task["rounds"])
    training = train_binary_classifier(
        model,
        train_dataset,
        validation_dataset,
        build_training_config(task, args, epochs=args.epochs, seed=task["seed"]),
    )
    probabilities = predict_binary_probabilities(
        model,
        validation_dataset,
        batch_size=args.batch_size,
        device=args.device,
    )
    labels = validation_dataset.labels.astype(np.float32, copy=False)
    report = {
        "row_index": index,
        "model_key": task["model_key"],
        "architecture": task["architecture"],
        "rounds": task["rounds"],
        "seed": task["seed"],
        "samples_per_class": task["samples_per_class"],
        "validation_samples_per_class": max(8, task["samples_per_class"] // 2),
        "pairs_per_sample": task["pairs_per_sample"],
        "feature_encoding": task["feature_encoding"],
        "selected_bit_indices": list(task["selected_bit_indices"]),
        "input_bits": int(train_dataset.features.shape[1]),
        "metrics": training.final_metrics,
        "training": training.metadata,
    }
    return report, labels, probabilities


def ensemble_metrics(
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    row_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    logits = probability_to_logit(probability_matrix)
    return [
        {
            "mode": "probability_mean",
            "weights": uniform_weights(probability_matrix.shape[1]).tolist(),
            "metrics": metrics_from_probabilities(labels, probability_matrix.mean(axis=1)),
        },
        {
            "mode": "logit_mean",
            "weights": uniform_weights(probability_matrix.shape[1]).tolist(),
            "metrics": metrics_from_probabilities(labels, sigmoid(logits.mean(axis=1))),
        },
        {
            "mode": "auc_weighted_logit_mean",
            "weights": auc_positive_weights(row_reports).tolist(),
            "metrics": metrics_from_probabilities(
                labels,
                sigmoid(logits @ auc_positive_weights(row_reports)),
            ),
        },
    ]


def diversity_metrics(
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    row_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions = (probability_matrix >= 0.5).astype(np.float32)
    correctness = predictions == labels.reshape(-1, 1)
    return {
        "oracle_accuracy_at_0_5": float(correctness.any(axis=1).mean()) if len(labels) else 0.0,
        "all_models_wrong_rate_at_0_5": float((~correctness.any(axis=1)).mean()) if len(labels) else 0.0,
        "pairwise": pairwise_diversity_metrics(labels, probability_matrix, row_reports),
        "interpretation": (
            "High disagreement with low double-fault/error overlap supports complementary weak views; "
            "high correlation and high shared errors suggest ensembling is unlikely to add evidence."
        ),
    }


def pairwise_diversity_metrics(
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    row_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    logits = probability_to_logit(probability_matrix)
    predictions = probability_matrix >= 0.5
    correct = predictions == labels.reshape(-1, 1)
    for left in range(probability_matrix.shape[1]):
        for right in range(left + 1, probability_matrix.shape[1]):
            left_wrong = ~correct[:, left]
            right_wrong = ~correct[:, right]
            either_wrong = left_wrong | right_wrong
            both_wrong = left_wrong & right_wrong
            reports.append(
                {
                    "left": _row_label(row_reports[left], left),
                    "right": _row_label(row_reports[right], right),
                    "probability_correlation": safe_correlation(
                        probability_matrix[:, left],
                        probability_matrix[:, right],
                    ),
                    "logit_correlation": safe_correlation(logits[:, left], logits[:, right]),
                    "disagreement_rate_at_0_5": float((predictions[:, left] != predictions[:, right]).mean()),
                    "double_fault_rate_at_0_5": float(both_wrong.mean()),
                    "error_jaccard_at_0_5": (
                        float(both_wrong.sum() / either_wrong.sum()) if int(either_wrong.sum()) else 0.0
                    ),
                }
            )
    return reports


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 0.0 or right_std <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _row_label(row: dict[str, Any], index: int) -> str:
    return str(row.get("architecture") or row.get("model_key") or f"row_{index + 1}")


def metrics_from_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = probabilities.astype(np.float32, copy=False)
    predictions = (probabilities >= 0.5).astype(np.float32)
    accuracy = float((predictions == labels).mean()) if len(labels) else 0.0
    calibrated_accuracy, threshold = best_threshold_accuracy_and_threshold(labels, probabilities)
    return {
        "accuracy": accuracy,
        "advantage": 2.0 * accuracy - 1.0,
        "auc": binary_auc(labels, probabilities),
        "best_accuracy": calibrated_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_advantage": 2.0 * calibrated_accuracy - 1.0,
        "calibrated_threshold": threshold,
    }


def uniform_weights(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("ensemble requires at least one probability column")
    return np.full((count,), 1.0 / float(count), dtype=np.float64)


def auc_positive_weights(row_reports: list[dict[str, Any]]) -> np.ndarray:
    weights = np.array(
        [max(0.0, float(row["metrics"]["auc"]) - 0.5) for row in row_reports],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 0.0:
        return uniform_weights(len(row_reports))
    return weights / total


def probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


if __name__ == "__main__":
    raise SystemExit(main())
