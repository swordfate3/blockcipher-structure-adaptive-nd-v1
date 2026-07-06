from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    evaluate_frozen_score_ensemble,
    load_score_artifact,
)
from blockcipher_nd.training.metrics import binary_auc, best_threshold_accuracy_and_threshold


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a logistic stacking layer on aligned train score artifacts and "
            "evaluate it on aligned validation score artifacts."
        )
    )
    parser.add_argument("--train-artifacts", nargs="+", required=True, type=Path)
    parser.add_argument("--validation-artifacts", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-space", choices=["logits", "probabilities"], default="logits")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--no-standardize", action="store_true")
    return parser.parse_args(argv)


def evaluate_stacked_ensemble(
    *,
    train_artifacts: list[EnsembleScoreArtifact],
    validation_artifacts: list[EnsembleScoreArtifact],
    feature_space: str = "logits",
    steps: int = 2000,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    standardize: bool = True,
) -> dict[str, Any]:
    if len(train_artifacts) < 2:
        raise ValueError("stacked ensemble requires at least two train artifacts")
    if len(train_artifacts) != len(validation_artifacts):
        raise ValueError("train and validation artifact counts must match")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")

    _validate_matching_model_order(train_artifacts, validation_artifacts)
    train_summary = evaluate_frozen_score_ensemble(train_artifacts)
    validation_summary = evaluate_frozen_score_ensemble(validation_artifacts)

    train_features = _feature_matrix(train_artifacts, feature_space)
    validation_features = _feature_matrix(validation_artifacts, feature_space)
    fitted = _fit_logistic_stack(
        train_features,
        train_artifacts[0].labels.astype(np.float64, copy=False),
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
    )
    validation_probabilities = _predict_logistic_stack(validation_features, fitted)
    train_probabilities = _predict_logistic_stack(train_features, fitted)
    validation_metrics = _metrics_from_probabilities(validation_artifacts[0].labels, validation_probabilities)
    train_metrics = _metrics_from_probabilities(train_artifacts[0].labels, train_probabilities)
    best_validation_single_auc = float(validation_summary["best_single"]["metrics"]["auc"])
    best_validation_fixed_auc = float(validation_summary["best_ensemble"]["metrics"]["auc"])

    return {
        "status": "pass",
        "decision": (
            "stacked_ensemble_improves_validation_best_single"
            if validation_metrics["auc"] > best_validation_single_auc
            else "stacked_ensemble_diagnostic_no_best_single_gain"
        ),
        "model_order": [str(item.metadata.get("model_key", "")) for item in validation_artifacts],
        "feature_space": feature_space,
        "fit": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "standardize": bool(standardize),
            "weights": [float(value) for value in fitted["weights"]],
            "bias": float(fitted["bias"]),
            "feature_mean": [float(value) for value in fitted["mean"]],
            "feature_scale": [float(value) for value in fitted["scale"]],
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_best_single": validation_summary["best_single"],
        "validation_best_fixed_ensemble": validation_summary["best_ensemble"],
        "delta_stacked_vs_validation_best_single_auc": float(
            validation_metrics["auc"] - best_validation_single_auc
        ),
        "delta_stacked_vs_validation_best_fixed_ensemble_auc": float(
            validation_metrics["auc"] - best_validation_fixed_auc
        ),
        "train_fixed_summary": train_summary,
        "validation_fixed_summary": validation_summary,
        "claim_scope": (
            "train-fitted validation-evaluated frozen-score stacking diagnostic only; "
            "not raw single-sample SOTA, not remote evidence, and not formal SPN/PRESENT evidence"
        ),
    }


def _validate_matching_model_order(
    train_artifacts: list[EnsembleScoreArtifact],
    validation_artifacts: list[EnsembleScoreArtifact],
) -> None:
    train_keys = [str(item.metadata.get("model_key", "")) for item in train_artifacts]
    validation_keys = [str(item.metadata.get("model_key", "")) for item in validation_artifacts]
    if train_keys != validation_keys:
        raise ValueError(f"train/validation model order differs: {train_keys} != {validation_keys}")


def _feature_matrix(artifacts: list[EnsembleScoreArtifact], feature_space: str) -> np.ndarray:
    if feature_space == "logits":
        return np.stack([item.logits for item in artifacts], axis=1).astype(np.float64, copy=False)
    if feature_space == "probabilities":
        return np.stack([item.probabilities for item in artifacts], axis=1).astype(np.float64, copy=False)
    raise ValueError(f"unsupported feature_space: {feature_space}")


def _fit_logistic_stack(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    l2: float,
    standardize: bool,
) -> dict[str, np.ndarray | float]:
    if standardize:
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        x = (features - mean) / scale
    else:
        mean = np.zeros(features.shape[1], dtype=np.float64)
        scale = np.ones(features.shape[1], dtype=np.float64)
        x = features
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    y = labels.astype(np.float64, copy=False)
    n = max(1, len(y))
    for _ in range(steps):
        probabilities = _sigmoid(x @ weights + bias)
        error = probabilities - y
        grad_w = (x.T @ error) / n + l2 * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return {"weights": weights, "bias": bias, "mean": mean, "scale": scale}


def _predict_logistic_stack(features: np.ndarray, fitted: dict[str, np.ndarray | float]) -> np.ndarray:
    weights = np.asarray(fitted["weights"], dtype=np.float64)
    mean = np.asarray(fitted["mean"], dtype=np.float64)
    scale = np.asarray(fitted["scale"], dtype=np.float64)
    bias = float(fitted["bias"])
    return _sigmoid(((features - mean) / scale) @ weights + bias).astype(np.float32, copy=False)


def _metrics_from_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    label_array = labels.astype(np.float32, copy=False)
    probability_array = probabilities.astype(np.float32, copy=False)
    predictions = (probability_array >= 0.5).astype(np.float32)
    accuracy = float((predictions == label_array).mean()) if len(label_array) else 0.0
    calibrated_accuracy, threshold = best_threshold_accuracy_and_threshold(label_array, probability_array)
    return {
        "accuracy": accuracy,
        "advantage": 2.0 * accuracy - 1.0,
        "auc": binary_auc(label_array, probability_array),
        "best_accuracy": calibrated_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_advantage": 2.0 * calibrated_accuracy - 1.0,
        "calibrated_threshold": threshold,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_artifacts = [load_score_artifact(path) for path in args.train_artifacts]
    validation_artifacts = [load_score_artifact(path) for path in args.validation_artifacts]
    summary = evaluate_stacked_ensemble(
        train_artifacts=train_artifacts,
        validation_artifacts=validation_artifacts,
        feature_space=args.feature_space,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        standardize=not args.no_standardize,
    )
    summary["train_artifact_dirs"] = [str(path) for path in args.train_artifacts]
    summary["validation_artifact_dirs"] = [str(path) for path in args.validation_artifacts]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
