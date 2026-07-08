from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.cli.analyze_reliability_residual_buckets import _residual_feature_views
from blockcipher_nd.cli.fit_bucket_conditioned_feature_expert import (
    _edge_report,
    _quantile_edges,
    _validate_feature_score_alignment,
)
from blockcipher_nd.cli.fit_compressed_feature_expert import (
    _load_feature_dir,
    _metrics,
    _score_metadata,
    _select_feature_columns,
    _sigmoid,
    _validate_feature_dirs,
)
from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    evaluate_frozen_score_ensemble,
    load_score_artifact,
    write_score_artifact,
)


DEFAULT_BUCKET_FEATURE = "logit_gap_abs"
DEFAULT_MODEL_KEY = "residual_correction_feature_expert"
DEFAULT_EXPERT_FAMILY = "spn_aux_residual_correction"
DEFAULT_CANDIDATE_STATUS = "local_residual_correction_probe"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a compressed SPN feature correction on top of frozen base logits. "
            "The base score artifacts stay frozen; the fitted model only learns an "
            "additive residual correction from train features and optional reliability buckets."
        )
    )
    parser.add_argument("--train-feature-dir", required=True, type=Path)
    parser.add_argument("--validation-feature-dir", required=True, type=Path)
    parser.add_argument("--train-base-artifacts", nargs=2, required=True, type=Path)
    parser.add_argument("--validation-base-artifacts", nargs=2, required=True, type=Path)
    parser.add_argument("--output-validation-dir", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-train-dir", type=Path, default=None)
    parser.add_argument(
        "--bucket-feature",
        choices=[
            "min_confidence",
            "confidence_gap_abs",
            "logit_gap_abs",
            "signed_logit_delta_model1_minus_model0",
        ],
        default=DEFAULT_BUCKET_FEATURE,
    )
    parser.add_argument(
        "--bucket-count",
        type=int,
        default=0,
        help="Use train-derived bucket one-hot and feature interactions when >=2; 0 disables buckets.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument(
        "--residual-focus-fraction",
        type=float,
        default=0.0,
        help=(
            "If positive, give top train rows by frozen-base residual loss full "
            "weight and down-weight the background rows."
        ),
    )
    parser.add_argument(
        "--residual-focus-background-weight",
        type=float,
        default=0.1,
        help="Background train-row weight when --residual-focus-fraction is positive.",
    )
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--no-reliability-features", action="store_true")
    parser.add_argument("--no-bucket-interactions", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--expert-family", default=DEFAULT_EXPERT_FAMILY)
    parser.add_argument("--candidate-status", default=DEFAULT_CANDIDATE_STATUS)
    parser.add_argument("--shuffle-train-labels", action="store_true")
    parser.add_argument("--shuffle-train-bucket-values", action="store_true")
    parser.add_argument("--shuffle-validation-bucket-values", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument(
        "--include-feature-family",
        action="append",
        default=[],
        help="Restrict fitting/scoring to decoded compressed-feature families; repeat for multiple families.",
    )
    parser.add_argument(
        "--include-feature-prefix",
        action="append",
        default=[],
        help="Restrict fitting/scoring to explicit metadata feature_names prefixes; repeat for multiple prefixes.",
    )
    parser.add_argument(
        "--include-feature-prefixes-from-summary",
        action="append",
        default=[],
        type=Path,
        help=(
            "Load recommended_feature_prefixes from a train-only residual axis-spectrum "
            "summary; repeat for multiple summaries."
        ),
    )
    return parser.parse_args(argv)


def fit_residual_correction_feature_expert(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    train_base_artifact_dirs: list[Path],
    validation_base_artifact_dirs: list[Path],
    bucket_feature: str = DEFAULT_BUCKET_FEATURE,
    bucket_count: int = 0,
    steps: int = 1000,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    residual_focus_fraction: float = 0.0,
    residual_focus_background_weight: float = 0.1,
    standardize: bool = True,
    include_reliability_features: bool = True,
    include_bucket_interactions: bool = True,
    run_id: str = "",
    model_key: str = DEFAULT_MODEL_KEY,
    expert_family: str = DEFAULT_EXPERT_FAMILY,
    candidate_status: str = DEFAULT_CANDIDATE_STATUS,
    shuffle_train_labels: bool = False,
    shuffle_train_bucket_values: bool = False,
    shuffle_validation_bucket_values: bool = False,
    shuffle_seed: int = 0,
    include_feature_families: list[str] | None = None,
    include_feature_prefixes: list[str] | None = None,
) -> tuple[EnsembleScoreArtifact, EnsembleScoreArtifact, dict[str, Any]]:
    if len(train_base_artifact_dirs) != 2 or len(validation_base_artifact_dirs) != 2:
        raise ValueError("residual correction requires exactly two base artifacts per split")
    if bucket_count != 0 and bucket_count < 2:
        raise ValueError("bucket_count must be 0 or at least 2")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    if residual_focus_fraction < 0.0 or residual_focus_fraction >= 1.0:
        raise ValueError("residual_focus_fraction must be in [0, 1)")
    if residual_focus_background_weight < 0.0 or residual_focus_background_weight > 1.0:
        raise ValueError("residual_focus_background_weight must be in [0, 1]")
    if bucket_count == 0 and (shuffle_train_bucket_values or shuffle_validation_bucket_values):
        raise ValueError("bucket shuffles require bucket_count >= 2")

    train_features = _load_feature_dir(train_feature_dir)
    validation_features = _load_feature_dir(validation_feature_dir)
    _validate_feature_dirs(train_features, validation_features)
    train_base_artifacts = [load_score_artifact(path) for path in train_base_artifact_dirs]
    validation_base_artifacts = [load_score_artifact(path) for path in validation_base_artifact_dirs]
    for index, artifact in enumerate(train_base_artifacts):
        _validate_feature_score_alignment(train_features, artifact, split=f"train base artifact {index}")
    for index, artifact in enumerate(validation_base_artifacts):
        _validate_feature_score_alignment(
            validation_features,
            artifact,
            split=f"validation base artifact {index}",
        )
    train_base_summary = evaluate_frozen_score_ensemble(train_base_artifacts)
    validation_base_summary = evaluate_frozen_score_ensemble(validation_base_artifacts)

    feature_selection = _select_feature_columns(
        train_features["metadata"],
        feature_count=int(train_features["features"].shape[1]),
        include_feature_families=include_feature_families or [],
        include_feature_prefixes=include_feature_prefixes or [],
    )
    selected_indices = np.asarray(feature_selection["selected_feature_indices"], dtype=np.int64)
    train_selected = train_features["features"][:, selected_indices].astype(np.float64, copy=False)
    validation_selected = validation_features["features"][:, selected_indices].astype(np.float64, copy=False)

    train_base_logits = _base_logit_mean(train_base_artifacts)
    validation_base_logits = _base_logit_mean(validation_base_artifacts)
    train_design, validation_design, design_report = _build_correction_design_matrices(
        train_selected=train_selected,
        validation_selected=validation_selected,
        train_base_artifacts=train_base_artifacts,
        validation_base_artifacts=validation_base_artifacts,
        bucket_feature=bucket_feature,
        bucket_count=bucket_count,
        include_reliability_features=include_reliability_features,
        include_bucket_interactions=include_bucket_interactions,
        shuffle_train_bucket_values=shuffle_train_bucket_values,
        shuffle_validation_bucket_values=shuffle_validation_bucket_values,
        shuffle_seed=shuffle_seed,
    )

    fit_labels = train_features["labels"].astype(np.float64, copy=True)
    train_sample_weights, sample_weight_report = _residual_focus_weights(
        labels=train_features["labels"].astype(np.float64, copy=False),
        base_logits=train_base_logits,
        residual_focus_fraction=residual_focus_fraction,
        background_weight=residual_focus_background_weight,
    )
    if shuffle_train_labels:
        fit_labels = np.random.default_rng(shuffle_seed).permutation(fit_labels)
    fitted = _fit_logit_correction(
        train_design,
        train_base_logits,
        fit_labels,
        sample_weights=train_sample_weights,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
    )
    train_correction_logits = _predict_correction_logits(train_design, fitted)
    validation_correction_logits = _predict_correction_logits(validation_design, fitted)
    train_corrected_logits = train_base_logits + train_correction_logits
    validation_corrected_logits = validation_base_logits + validation_correction_logits
    train_probabilities = _sigmoid(train_corrected_logits)
    validation_probabilities = _sigmoid(validation_corrected_logits)

    model_order = [str(item.metadata.get("model_key", "")) for item in validation_base_artifacts]
    run_order = [str(item.metadata.get("run_id", "")) for item in validation_base_artifacts]
    common_metadata = {
        "model_key": model_key,
        "expert_family": expert_family,
        "candidate_status": candidate_status,
        "run_id": run_id,
        "feature_fit_split": "train",
        "feature_train_dir": str(train_feature_dir),
        "feature_validation_dir": str(validation_feature_dir),
        "feature_model": "residual_logit_correction",
        "feature_steps": int(steps),
        "feature_learning_rate": float(learning_rate),
        "feature_l2": float(l2),
        "feature_standardize": bool(standardize),
        "residual_focus_fraction": float(residual_focus_fraction),
        "residual_focus_background_weight": float(residual_focus_background_weight),
        "feature_count": int(train_selected.shape[1]),
        "feature_original_count": int(train_features["features"].shape[1]),
        "correction_feature_count": int(train_design.shape[1]),
        "feature_selection": _feature_selection_metadata(feature_selection),
        "base_model_order": model_order,
        "base_run_order": run_order,
        "base_fusion": "logit_mean",
        "fit_train_labels_shuffled": bool(shuffle_train_labels),
        "fit_label_shuffle_seed": int(shuffle_seed),
        "bucket_train_values_shuffled": bool(shuffle_train_bucket_values),
        "bucket_validation_values_shuffled": bool(shuffle_validation_bucket_values),
        "bucket_shuffle_seed": int(shuffle_seed),
        "bucket_feature": bucket_feature,
        "bucket_count": int(bucket_count),
        "include_reliability_features": bool(include_reliability_features),
        "include_bucket_interactions": bool(include_bucket_interactions and bucket_count >= 2),
        "claim_scope": (
            "train-fitted residual logit correction over frozen base scores; "
            "validation is held out for final scoring, and this is not remote or formal "
            "SPN/PRESENT evidence"
        ),
    }
    if design_report.get("bucket_edges") is not None:
        common_metadata["bucket_edges"] = design_report["bucket_edges"]

    train_artifact = EnsembleScoreArtifact(
        labels=train_features["labels"],
        probabilities=train_probabilities.astype(np.float32, copy=False),
        logits=train_corrected_logits.astype(np.float32, copy=False),
        sample_ids=train_features["sample_ids"],
        metadata=_score_metadata(train_features["metadata"], common_metadata, score_split="train"),
    )
    validation_artifact = EnsembleScoreArtifact(
        labels=validation_features["labels"],
        probabilities=validation_probabilities.astype(np.float32, copy=False),
        logits=validation_corrected_logits.astype(np.float32, copy=False),
        sample_ids=validation_features["sample_ids"],
        metadata=_score_metadata(validation_features["metadata"], common_metadata, score_split="validation"),
    )
    train_base_metrics = _metrics(train_artifact.labels, _sigmoid(train_base_logits))
    validation_base_metrics = _metrics(validation_artifact.labels, _sigmoid(validation_base_logits))
    train_metrics = _metrics(train_artifact.labels, train_artifact.probabilities)
    validation_metrics = _metrics(validation_artifact.labels, validation_artifact.probabilities)
    correction_weights = np.asarray(fitted["weights"], dtype=np.float64)
    report = {
        "status": "pass",
        "decision": (
            "residual_correction_local_candidate_needs_controls"
            if validation_metrics["auc"] > validation_base_metrics["auc"]
            else "residual_correction_diagnostic_no_base_gain"
        ),
        "model_key": model_key,
        "expert_family": expert_family,
        "train_feature_dir": str(train_feature_dir),
        "validation_feature_dir": str(validation_feature_dir),
        "train_base_artifact_dirs": [str(path) for path in train_base_artifact_dirs],
        "validation_base_artifact_dirs": [str(path) for path in validation_base_artifact_dirs],
        "base_model_order": model_order,
        "base_run_order": run_order,
        "base_fusion": "logit_mean",
        "train_rows": int(len(train_features["labels"])),
        "validation_rows": int(len(validation_features["labels"])),
        "selected_feature_count": int(train_selected.shape[1]),
        "correction_feature_count": int(train_design.shape[1]),
        "feature_selection": _feature_selection_metadata(feature_selection),
        "design": design_report,
        "fit": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "standardize": bool(standardize),
            "residual_focus": sample_weight_report,
            "weight_count": int(len(correction_weights)),
            "weight_l2_norm": float(np.linalg.norm(correction_weights)),
            "weight_abs_max": float(np.max(np.abs(correction_weights))) if len(correction_weights) else 0.0,
            "bias": float(fitted["bias"]),
        },
        "label_control": {
            "shuffle_train_labels": bool(shuffle_train_labels),
            "shuffle_seed": int(shuffle_seed),
        },
        "bucket_source_control": {
            "shuffle_train_bucket_values": bool(shuffle_train_bucket_values),
            "shuffle_validation_bucket_values": bool(shuffle_validation_bucket_values),
            "shuffle_seed": int(shuffle_seed),
        },
        "train_base_logit_mean_metrics": train_base_metrics,
        "validation_base_logit_mean_metrics": validation_base_metrics,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "delta_train_corrected_vs_base_logit_mean_auc": float(
            train_metrics["auc"] - train_base_metrics["auc"]
        ),
        "delta_validation_corrected_vs_base_logit_mean_auc": float(
            validation_metrics["auc"] - validation_base_metrics["auc"]
        ),
        "train_base_summary": train_base_summary,
        "validation_base_summary": validation_base_summary,
        "guardrails": [
            "base_scores_must_stay_frozen",
            "correction_fit_split_must_be_train",
            "bucket_edges_must_be_train_derived_when_enabled",
            "validation_split_final_score_only",
            "strict_negative_mode_required",
            "shuffle_label_and_bucket_controls_required_before_remote_scaleup",
        ],
        "claim_scope": common_metadata["claim_scope"],
    }
    if design_report.get("bucket_edges") is not None:
        report["bucket_edges"] = design_report["bucket_edges"]
    report["bucket_feature"] = bucket_feature
    report["bucket_count"] = int(bucket_count)
    return train_artifact, validation_artifact, report


def _base_logit_mean(artifacts: list[EnsembleScoreArtifact]) -> np.ndarray:
    return np.stack([artifact.logits for artifact in artifacts], axis=1).mean(axis=1).astype(np.float64)


def _build_correction_design_matrices(
    *,
    train_selected: np.ndarray,
    validation_selected: np.ndarray,
    train_base_artifacts: list[EnsembleScoreArtifact],
    validation_base_artifacts: list[EnsembleScoreArtifact],
    bucket_feature: str,
    bucket_count: int,
    include_reliability_features: bool,
    include_bucket_interactions: bool,
    shuffle_train_bucket_values: bool,
    shuffle_validation_bucket_values: bool,
    shuffle_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_parts = [train_selected]
    validation_parts = [validation_selected]
    train_views = _residual_feature_views(train_base_artifacts)
    validation_views = _residual_feature_views(validation_base_artifacts)
    report: dict[str, Any] = {
        "selected_feature_columns": int(train_selected.shape[1]),
        "include_reliability_features": bool(include_reliability_features),
        "include_bucket_interactions": bool(include_bucket_interactions and bucket_count >= 2),
        "bucket_feature": bucket_feature,
        "bucket_count": int(bucket_count),
        "bucket_edges": None,
    }
    if include_reliability_features:
        reliability_names = [
            "min_confidence",
            "confidence_gap_abs",
            "logit_gap_abs",
            "signed_logit_delta_model1_minus_model0",
        ]
        train_parts.append(np.stack([train_views[name] for name in reliability_names], axis=1))
        validation_parts.append(np.stack([validation_views[name] for name in reliability_names], axis=1))
        report["reliability_feature_names"] = reliability_names
    if bucket_count >= 2:
        train_bucket_values = train_views[bucket_feature].astype(np.float64, copy=True)
        validation_bucket_values = validation_views[bucket_feature].astype(np.float64, copy=True)
        rng = np.random.default_rng(shuffle_seed)
        if shuffle_train_bucket_values:
            train_bucket_values = rng.permutation(train_bucket_values)
        if shuffle_validation_bucket_values:
            validation_bucket_values = rng.permutation(validation_bucket_values)
        edges = _quantile_edges(train_bucket_values, bucket_count)
        train_bucket_ids = np.searchsorted(edges[1:-1], train_bucket_values, side="right")
        validation_bucket_ids = np.searchsorted(edges[1:-1], validation_bucket_values, side="right")
        train_one_hot = _one_hot(train_bucket_ids, bucket_count)
        validation_one_hot = _one_hot(validation_bucket_ids, bucket_count)
        train_parts.append(train_one_hot)
        validation_parts.append(validation_one_hot)
        if include_bucket_interactions:
            train_parts.append(_bucket_interactions(train_selected, train_one_hot))
            validation_parts.append(_bucket_interactions(validation_selected, validation_one_hot))
        report.update(
            {
                "bucket_edges": _edge_report(edges),
                "train_bucket_rows": _bucket_rows(train_bucket_ids, bucket_count),
                "validation_bucket_rows": _bucket_rows(validation_bucket_ids, bucket_count),
            }
        )
    train_design = np.concatenate(train_parts, axis=1).astype(np.float64, copy=False)
    validation_design = np.concatenate(validation_parts, axis=1).astype(np.float64, copy=False)
    report["correction_feature_columns"] = int(train_design.shape[1])
    return train_design, validation_design, report


def _one_hot(bucket_ids: np.ndarray, bucket_count: int) -> np.ndarray:
    matrix = np.zeros((len(bucket_ids), bucket_count), dtype=np.float64)
    matrix[np.arange(len(bucket_ids)), bucket_ids.astype(np.int64, copy=False)] = 1.0
    return matrix


def _bucket_interactions(features: np.ndarray, one_hot: np.ndarray) -> np.ndarray:
    return (features[:, :, None] * one_hot[:, None, :]).reshape(features.shape[0], -1)


def _bucket_rows(bucket_ids: np.ndarray, bucket_count: int) -> list[dict[str, int]]:
    return [
        {"bucket": int(bucket), "rows": int(np.sum(bucket_ids == bucket))}
        for bucket in range(bucket_count)
    ]


def _residual_focus_weights(
    *,
    labels: np.ndarray,
    base_logits: np.ndarray,
    residual_focus_fraction: float,
    background_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_probabilities = _sigmoid(base_logits)
    residual_loss = np.abs(labels.astype(np.float64, copy=False) - base_probabilities)
    if residual_focus_fraction <= 0.0:
        weights = np.ones_like(residual_loss, dtype=np.float64)
        return weights, {
            "mode": "uniform",
            "residual_focus_fraction": 0.0,
            "background_weight": 1.0,
            "focused_rows": int(len(weights)),
            "total_rows": int(len(weights)),
            "effective_weight_sum": float(weights.sum()),
            "residual_loss_mean": float(residual_loss.mean()) if len(residual_loss) else 0.0,
            "residual_loss_threshold": None,
        }

    focus_count = int(np.ceil(len(residual_loss) * residual_focus_fraction))
    focus_count = max(1, min(len(residual_loss), focus_count))
    order = np.argsort(residual_loss)
    focus_indices = order[-focus_count:]
    weights = np.full(len(residual_loss), float(background_weight), dtype=np.float64)
    weights[focus_indices] = 1.0
    mean_weight = float(weights.mean()) if len(weights) else 1.0
    if mean_weight > 0.0:
        weights = weights / mean_weight
    threshold = float(residual_loss[focus_indices].min()) if len(focus_indices) else None
    return weights, {
        "mode": "top_base_residual_loss",
        "residual_focus_fraction": float(residual_focus_fraction),
        "background_weight": float(background_weight),
        "focused_rows": int(focus_count),
        "total_rows": int(len(weights)),
        "effective_weight_sum": float(weights.sum()),
        "residual_loss_mean": float(residual_loss.mean()) if len(residual_loss) else 0.0,
        "focused_residual_loss_mean": float(residual_loss[focus_indices].mean())
        if len(focus_indices)
        else 0.0,
        "residual_loss_threshold": threshold,
    }


def _fit_logit_correction(
    features: np.ndarray,
    base_logits: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
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
    row_weights = sample_weights.astype(np.float64, copy=False)
    weight_sum = max(float(row_weights.sum()), 1.0)
    for _ in range(steps):
        probabilities = _sigmoid(base_logits + x @ weights + bias)
        error = (probabilities - y) * row_weights
        grad_w = (x.T @ error) / weight_sum + l2 * weights
        grad_b = float(error.sum() / weight_sum)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return {"weights": weights, "bias": bias, "mean": mean, "scale": scale}


def _predict_correction_logits(features: np.ndarray, fitted: dict[str, np.ndarray | float]) -> np.ndarray:
    weights = np.asarray(fitted["weights"], dtype=np.float64)
    mean = np.asarray(fitted["mean"], dtype=np.float64)
    scale = np.asarray(fitted["scale"], dtype=np.float64)
    bias = float(fitted["bias"])
    return ((features.astype(np.float64, copy=False) - mean) / scale) @ weights + bias


def _feature_selection_metadata(feature_selection: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(feature_selection)
    if metadata.get("mode") == "all":
        metadata.pop("selected_feature_indices", None)
    return metadata


def _feature_prefixes_from_summaries(paths: list[Path]) -> list[str]:
    prefixes: list[str] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected JSON object")
        loaded = [str(prefix) for prefix in payload.get("recommended_feature_prefixes", []) if str(prefix)]
        if not loaded:
            raise ValueError(f"{path}: recommended_feature_prefixes is empty")
        prefixes.extend(loaded)
    return _dedupe_strings(prefixes)


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    include_feature_prefixes = _dedupe_strings(
        [
            *[str(prefix) for prefix in args.include_feature_prefix],
            *_feature_prefixes_from_summaries(args.include_feature_prefixes_from_summary),
        ]
    )
    train_artifact, validation_artifact, report = fit_residual_correction_feature_expert(
        train_feature_dir=args.train_feature_dir,
        validation_feature_dir=args.validation_feature_dir,
        train_base_artifact_dirs=args.train_base_artifacts,
        validation_base_artifact_dirs=args.validation_base_artifacts,
        bucket_feature=args.bucket_feature,
        bucket_count=args.bucket_count,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        residual_focus_fraction=args.residual_focus_fraction,
        residual_focus_background_weight=args.residual_focus_background_weight,
        standardize=not args.no_standardize,
        include_reliability_features=not args.no_reliability_features,
        include_bucket_interactions=not args.no_bucket_interactions,
        run_id=args.run_id,
        model_key=args.model_key,
        expert_family=args.expert_family,
        candidate_status=args.candidate_status,
        shuffle_train_labels=args.shuffle_train_labels,
        shuffle_train_bucket_values=args.shuffle_train_bucket_values,
        shuffle_validation_bucket_values=args.shuffle_validation_bucket_values,
        shuffle_seed=args.shuffle_seed,
        include_feature_families=args.include_feature_family,
        include_feature_prefixes=include_feature_prefixes,
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
