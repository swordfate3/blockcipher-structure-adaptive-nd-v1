from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.cli.decode_compressed_feature_sparsity import _family_for_name, _feature_names_from_metadata
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, write_score_artifact
from blockcipher_nd.training.metrics import binary_auc, best_threshold_accuracy_and_threshold


DEFAULT_MODEL_KEY = "compressed_feature_logistic_expert"
DEFAULT_EXPERT_FAMILY = "compressed_spn_structural_stats"
DEFAULT_CANDIDATE_STATUS = "compressed_feature_screen"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a small logistic expert on train feature artifacts and score held-out "
            "validation feature artifacts."
        )
    )
    parser.add_argument("--train-feature-dir", required=True, type=Path)
    parser.add_argument("--validation-feature-dir", required=True, type=Path)
    parser.add_argument("--output-validation-dir", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-train-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--expert-family", default=DEFAULT_EXPERT_FAMILY)
    parser.add_argument("--candidate-status", default=DEFAULT_CANDIDATE_STATUS)
    parser.add_argument("--shuffle-train-labels", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument(
        "--include-feature-family",
        action="append",
        default=[],
        help="Restrict fitting/scoring to decoded compressed-feature families; repeat for multiple families.",
    )
    return parser.parse_args(argv)


def fit_compressed_feature_expert(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    steps: int = 2000,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    standardize: bool = True,
    run_id: str = "",
    model_key: str = DEFAULT_MODEL_KEY,
    expert_family: str = DEFAULT_EXPERT_FAMILY,
    candidate_status: str = DEFAULT_CANDIDATE_STATUS,
    shuffle_train_labels: bool = False,
    shuffle_seed: int = 0,
    include_feature_families: list[str] | None = None,
) -> tuple[EnsembleScoreArtifact, EnsembleScoreArtifact, dict[str, Any]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")

    train = _load_feature_dir(train_feature_dir)
    validation = _load_feature_dir(validation_feature_dir)
    _validate_feature_dirs(train, validation)
    feature_selection = _select_feature_columns(
        train["metadata"],
        feature_count=int(train["features"].shape[1]),
        include_feature_families=include_feature_families or [],
    )
    selected_indices = np.asarray(feature_selection["selected_feature_indices"], dtype=np.int64)
    train_features = train["features"][:, selected_indices]
    validation_features = validation["features"][:, selected_indices]
    fit_labels = train["labels"].astype(np.float64, copy=True)
    if shuffle_train_labels:
        fit_labels = np.random.default_rng(shuffle_seed).permutation(fit_labels)
    fitted = _fit_logistic(
        train_features.astype(np.float64, copy=False),
        fit_labels,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
    )
    train_logits = _predict_logits(train_features, fitted)
    validation_logits = _predict_logits(validation_features, fitted)
    train_probabilities = _sigmoid(train_logits)
    validation_probabilities = _sigmoid(validation_logits)

    common_metadata = {
        "model_key": model_key,
        "expert_family": expert_family,
        "candidate_status": candidate_status,
        "run_id": run_id,
        "feature_fit_split": "train",
        "feature_train_dir": str(train_feature_dir),
        "feature_validation_dir": str(validation_feature_dir),
        "feature_model": "logistic",
        "feature_steps": int(steps),
        "feature_learning_rate": float(learning_rate),
        "feature_l2": float(l2),
        "feature_standardize": bool(standardize),
        "fit_train_labels_shuffled": bool(shuffle_train_labels),
        "fit_label_shuffle_seed": int(shuffle_seed),
        "feature_count": int(train_features.shape[1]),
        "feature_original_count": int(train["features"].shape[1]),
        "feature_selection": _feature_selection_metadata(feature_selection),
        "claim_scope": (
            "train-fitted compressed SPN feature expert diagnostic only; validation is "
            "held out for final scoring, and this is not remote or formal SPN/PRESENT evidence"
        ),
    }
    train_artifact = EnsembleScoreArtifact(
        labels=train["labels"],
        probabilities=train_probabilities.astype(np.float32, copy=False),
        logits=train_logits.astype(np.float32, copy=False),
        sample_ids=train["sample_ids"],
        metadata=_score_metadata(train["metadata"], common_metadata, score_split="train"),
    )
    validation_artifact = EnsembleScoreArtifact(
        labels=validation["labels"],
        probabilities=validation_probabilities.astype(np.float32, copy=False),
        logits=validation_logits.astype(np.float32, copy=False),
        sample_ids=validation["sample_ids"],
        metadata=_score_metadata(validation["metadata"], common_metadata, score_split="validation"),
    )
    train_metrics = _metrics(train_artifact.labels, train_artifact.probabilities)
    validation_metrics = _metrics(validation_artifact.labels, validation_artifact.probabilities)
    report = {
        "status": "pass",
        "decision": (
            "compressed_feature_expert_shuffle_train_labels_control"
            if shuffle_train_labels
            else "compressed_feature_expert_local_screen_positive_needs_controls"
        ),
        "model_key": model_key,
        "expert_family": expert_family,
        "train_feature_dir": str(train_feature_dir),
        "validation_feature_dir": str(validation_feature_dir),
        "train_rows": int(len(train["labels"])),
        "validation_rows": int(len(validation["labels"])),
        "feature_count": int(train_features.shape[1]),
        "feature_selection": _feature_selection_metadata(feature_selection),
        "fit": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "standardize": bool(standardize),
            "weight_count": int(len(np.asarray(fitted["weights"]))),
            "weight_l2_norm": float(np.linalg.norm(np.asarray(fitted["weights"], dtype=np.float64))),
            "weight_abs_max": float(np.max(np.abs(np.asarray(fitted["weights"], dtype=np.float64)))),
            "bias": float(fitted["bias"]),
            "feature_mean_l2_norm": float(np.linalg.norm(np.asarray(fitted["mean"], dtype=np.float64))),
            "feature_scale_min": float(np.min(np.asarray(fitted["scale"], dtype=np.float64))),
            "feature_scale_max": float(np.max(np.asarray(fitted["scale"], dtype=np.float64))),
        },
        "label_control": {
            "shuffle_train_labels": bool(shuffle_train_labels),
            "shuffle_seed": int(shuffle_seed),
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "guardrails": [
            "fit_split_must_be_train",
            "validation_split_final_score_only",
            "strict_negative_mode_required",
            "compare_against_same_input_global_control",
            "compare_against_trail_position_anchor",
        ],
        "claim_scope": common_metadata["claim_scope"],
    }
    return train_artifact, validation_artifact, report


def _select_feature_columns(
    metadata: dict[str, Any],
    *,
    feature_count: int,
    include_feature_families: list[str],
) -> dict[str, Any]:
    families = sorted({family for family in include_feature_families if family})
    if not families:
        return {
            "mode": "all",
            "include_feature_families": [],
            "original_feature_count": int(feature_count),
            "selected_feature_count": int(feature_count),
            "selected_feature_indices": [int(index) for index in range(feature_count)],
        }
    view_metadata = metadata.get("feature_view_metadata", metadata)
    names = _feature_names_from_metadata(view_metadata)
    if len(names) != feature_count:
        raise ValueError(f"expected {feature_count} feature names, got {len(names)}")
    family_set = set(families)
    selected = [index for index, name in enumerate(names) if _family_for_name(name) in family_set]
    if not selected:
        raise ValueError(f"include_feature_family matched no features: {families}")
    return {
        "mode": "family_filter",
        "include_feature_families": families,
        "original_feature_count": int(feature_count),
        "selected_feature_count": int(len(selected)),
        "selected_feature_indices": [int(index) for index in selected],
    }


def _feature_selection_metadata(feature_selection: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(feature_selection)
    if metadata.get("mode") == "all":
        metadata.pop("selected_feature_indices", None)
    return metadata


def _score_metadata(
    feature_metadata: dict[str, Any],
    common_metadata: dict[str, Any],
    *,
    score_split: str,
) -> dict[str, Any]:
    metadata = {**feature_metadata, **common_metadata, "score_split": score_split}
    score_samples_per_class = int(feature_metadata.get("samples_per_class", 0))
    metadata["score_samples_per_class"] = score_samples_per_class
    if score_split == "train":
        metadata["train_samples_per_class"] = score_samples_per_class
    if score_split == "validation":
        metadata["validation_samples_per_class"] = score_samples_per_class
    return metadata


def _load_feature_dir(path: Path) -> dict[str, Any]:
    features = np.load(path / "features.npy")
    labels = np.load(path / "labels.npy").astype(np.float32, copy=False)
    sample_ids = np.load(path / "sample_ids.npy").astype(str, copy=False)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if features.ndim < 2:
        raise ValueError("features must have at least two dimensions: rows x features")
    feature_matrix = features.reshape(features.shape[0], -1).astype(np.float32, copy=False)
    if len({len(feature_matrix), len(labels), len(sample_ids)}) != 1:
        raise ValueError("feature rows, labels, and sample_ids must have equal length")
    return {
        "features": feature_matrix,
        "labels": labels,
        "sample_ids": sample_ids,
        "metadata": metadata,
    }


def _validate_feature_dirs(train: dict[str, Any], validation: dict[str, Any]) -> None:
    if train["metadata"].get("split") != "train":
        raise ValueError("train feature dir must have split=train")
    if validation["metadata"].get("split") != "validation":
        raise ValueError("validation feature dir must have split=validation")
    if train["metadata"].get("negative_mode") != "encrypted_random_plaintexts":
        raise ValueError("negative_mode must be encrypted_random_plaintexts")
    if train["features"].shape[1] != validation["features"].shape[1]:
        raise ValueError("train and validation feature counts differ")
    if len(np.unique(train["labels"])) < 2:
        raise ValueError("train labels must contain both classes")
    for field in (
        "cipher",
        "rounds",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "train_key",
        "validation_key",
        "feature_view",
    ):
        if train["metadata"].get(field) != validation["metadata"].get(field):
            raise ValueError(f"train and validation feature metadata differs: {field}")


def _fit_logistic(
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
    n = max(1, len(labels))
    for _ in range(steps):
        probabilities = _sigmoid(x @ weights + bias)
        error = probabilities - labels
        grad_w = (x.T @ error) / n + l2 * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return {"weights": weights, "bias": bias, "mean": mean, "scale": scale}


def _predict_logits(features: np.ndarray, fitted: dict[str, np.ndarray | float]) -> np.ndarray:
    weights = np.asarray(fitted["weights"], dtype=np.float64)
    mean = np.asarray(fitted["mean"], dtype=np.float64)
    scale = np.asarray(fitted["scale"], dtype=np.float64)
    bias = float(fitted["bias"])
    return ((features.astype(np.float64, copy=False) - mean) / scale) @ weights + bias


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    label_array = labels.astype(np.float32, copy=False)
    probability_array = probabilities.astype(np.float32, copy=False)
    predictions = (probability_array >= 0.5).astype(np.float32)
    accuracy = float((predictions == label_array).mean()) if len(label_array) else 0.0
    calibrated_accuracy, threshold = best_threshold_accuracy_and_threshold(label_array, probability_array)
    return {
        "accuracy": accuracy,
        "advantage": 2.0 * accuracy - 1.0,
        "auc": binary_auc(label_array, probability_array),
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_advantage": 2.0 * calibrated_accuracy - 1.0,
        "calibrated_threshold": threshold,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_artifact, validation_artifact, report = fit_compressed_feature_expert(
        train_feature_dir=args.train_feature_dir,
        validation_feature_dir=args.validation_feature_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        standardize=not args.no_standardize,
        run_id=args.run_id,
        model_key=args.model_key,
        expert_family=args.expert_family,
        candidate_status=args.candidate_status,
        shuffle_train_labels=args.shuffle_train_labels,
        shuffle_seed=args.shuffle_seed,
        include_feature_families=args.include_feature_family,
    )
    if args.output_train_dir is not None:
        write_score_artifact(args.output_train_dir, train_artifact)
        report["output_train_dir"] = str(args.output_train_dir)
    write_score_artifact(args.output_validation_dir, validation_artifact)
    report["output_validation_dir"] = str(args.output_validation_dir)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
