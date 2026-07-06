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
    parser.add_argument(
        "--train-holdout-fraction",
        type=float,
        default=0.0,
        help=(
            "If positive, select stacking calibration settings on a deterministic "
            "holdout split carved only from train artifacts, then refit on all train data."
        ),
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument(
        "--candidate-feature-spaces",
        nargs="+",
        choices=["logits", "probabilities"],
        default=None,
        help="Optional train-holdout feature-space grid. Defaults to --feature-space.",
    )
    parser.add_argument(
        "--candidate-l2",
        nargs="+",
        type=float,
        default=None,
        help="Optional train-holdout L2 grid. Defaults to --l2.",
    )
    parser.add_argument(
        "--candidate-standardize",
        choices=["current", "true", "false", "both"],
        default="current",
        help="Optional train-holdout standardization grid. Defaults to current --no-standardize setting.",
    )
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
    train_holdout_fraction: float = 0.0,
    selection_seed: int = 0,
    candidate_feature_spaces: list[str] | None = None,
    candidate_l2_values: list[float] | None = None,
    candidate_standardize_values: list[bool] | None = None,
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
    if train_holdout_fraction < 0.0 or train_holdout_fraction >= 1.0:
        raise ValueError("train_holdout_fraction must be in [0, 1)")

    _validate_matching_model_order(train_artifacts, validation_artifacts)
    train_summary = evaluate_frozen_score_ensemble(train_artifacts)
    validation_summary = evaluate_frozen_score_ensemble(validation_artifacts)

    selection_summary = _select_stack_candidate(
        train_artifacts=train_artifacts,
        default_feature_space=feature_space,
        default_l2=l2,
        default_standardize=standardize,
        steps=steps,
        learning_rate=learning_rate,
        train_holdout_fraction=train_holdout_fraction,
        selection_seed=selection_seed,
        candidate_feature_spaces=candidate_feature_spaces,
        candidate_l2_values=candidate_l2_values,
        candidate_standardize_values=candidate_standardize_values,
    )
    selected_feature_space = str(selection_summary["selected"]["feature_space"])
    selected_l2 = float(selection_summary["selected"]["l2"])
    selected_standardize = bool(selection_summary["selected"]["standardize"])
    train_features = _feature_matrix(train_artifacts, selected_feature_space)
    validation_features = _feature_matrix(validation_artifacts, selected_feature_space)
    fitted = _fit_logistic_stack(
        train_features,
        train_artifacts[0].labels.astype(np.float64, copy=False),
        steps=steps,
        learning_rate=learning_rate,
        l2=selected_l2,
        standardize=selected_standardize,
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
        "feature_space": selected_feature_space,
        "fit": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "l2": selected_l2,
            "standardize": selected_standardize,
            "weights": [float(value) for value in fitted["weights"]],
            "bias": float(fitted["bias"]),
            "feature_mean": [float(value) for value in fitted["mean"]],
            "feature_scale": [float(value) for value in fitted["scale"]],
        },
        "selection": selection_summary,
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


def _select_stack_candidate(
    *,
    train_artifacts: list[EnsembleScoreArtifact],
    default_feature_space: str,
    default_l2: float,
    default_standardize: bool,
    steps: int,
    learning_rate: float,
    train_holdout_fraction: float,
    selection_seed: int,
    candidate_feature_spaces: list[str] | None,
    candidate_l2_values: list[float] | None,
    candidate_standardize_values: list[bool] | None,
) -> dict[str, Any]:
    feature_spaces = candidate_feature_spaces or [default_feature_space]
    l2_values = candidate_l2_values or [default_l2]
    standardize_values = candidate_standardize_values or [default_standardize]
    for value in l2_values:
        if value < 0.0:
            raise ValueError("candidate l2 values must be non-negative")
    if train_holdout_fraction <= 0.0:
        return {
            "mode": "fixed",
            "selected": {
                "feature_space": default_feature_space,
                "l2": float(default_l2),
                "standardize": bool(default_standardize),
            },
            "claim_scope": "fixed train-fit settings; no train-holdout selection",
        }

    labels = train_artifacts[0].labels.astype(np.float32, copy=False)
    fit_indices, holdout_indices = _stratified_train_holdout_indices(
        labels, fraction=train_holdout_fraction, seed=selection_seed
    )
    candidates: list[dict[str, Any]] = []
    for candidate_feature_space in feature_spaces:
        features = _feature_matrix(train_artifacts, candidate_feature_space)
        for candidate_l2 in l2_values:
            for candidate_standardize in standardize_values:
                fitted = _fit_logistic_stack(
                    features[fit_indices],
                    labels[fit_indices].astype(np.float64, copy=False),
                    steps=steps,
                    learning_rate=learning_rate,
                    l2=candidate_l2,
                    standardize=candidate_standardize,
                )
                probabilities = _predict_logistic_stack(features[holdout_indices], fitted)
                metrics = _metrics_from_probabilities(labels[holdout_indices], probabilities)
                candidates.append(
                    {
                        "feature_space": candidate_feature_space,
                        "l2": float(candidate_l2),
                        "standardize": bool(candidate_standardize),
                        "holdout_metrics": metrics,
                    }
                )
    best = max(
        candidates,
        key=lambda item: (
            float(item["holdout_metrics"]["auc"]),
            float(item["holdout_metrics"]["calibrated_accuracy"]),
            -float(item["l2"]),
        ),
    )
    return {
        "mode": "train_holdout",
        "train_holdout_fraction": float(train_holdout_fraction),
        "selection_seed": int(selection_seed),
        "fit_rows": int(len(fit_indices)),
        "holdout_rows": int(len(holdout_indices)),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": {
            "feature_space": str(best["feature_space"]),
            "l2": float(best["l2"]),
            "standardize": bool(best["standardize"]),
            "holdout_metrics": best["holdout_metrics"],
        },
        "claim_scope": (
            "calibration settings selected only on a train split holdout; "
            "held-out validation artifacts are used only for final evaluation"
        ),
    }


def _stratified_train_holdout_indices(
    labels: np.ndarray, *, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit_parts: list[np.ndarray] = []
    holdout_parts: list[np.ndarray] = []
    for label in sorted(float(value) for value in np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("train-holdout selection needs at least two rows per class")
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        holdout_count = int(round(len(shuffled) * fraction))
        holdout_count = max(1, min(len(shuffled) - 1, holdout_count))
        holdout_parts.append(shuffled[:holdout_count])
        fit_parts.append(shuffled[holdout_count:])
    fit_indices = np.sort(np.concatenate(fit_parts))
    holdout_indices = np.sort(np.concatenate(holdout_parts))
    if len(np.unique(labels[fit_indices])) < 2 or len(np.unique(labels[holdout_indices])) < 2:
        raise ValueError("train-holdout split must preserve both classes")
    return fit_indices, holdout_indices


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
    standardize = not args.no_standardize
    candidate_standardize_values = _candidate_standardize_values(args.candidate_standardize, standardize)
    summary = evaluate_stacked_ensemble(
        train_artifacts=train_artifacts,
        validation_artifacts=validation_artifacts,
        feature_space=args.feature_space,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        standardize=standardize,
        train_holdout_fraction=args.train_holdout_fraction,
        selection_seed=args.selection_seed,
        candidate_feature_spaces=args.candidate_feature_spaces,
        candidate_l2_values=args.candidate_l2,
        candidate_standardize_values=candidate_standardize_values,
    )
    summary["train_artifact_dirs"] = [str(path) for path in args.train_artifacts]
    summary["validation_artifact_dirs"] = [str(path) for path in args.validation_artifacts]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


def _candidate_standardize_values(option: str, current: bool) -> list[bool] | None:
    if option == "current":
        return None
    if option == "true":
        return [True]
    if option == "false":
        return [False]
    if option == "both":
        return [True, False]
    raise ValueError(f"unsupported candidate_standardize option: {option}")


if __name__ == "__main__":
    raise SystemExit(main())
