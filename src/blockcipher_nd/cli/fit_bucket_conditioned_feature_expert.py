from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.cli.analyze_reliability_residual_buckets import _residual_feature_views
from blockcipher_nd.cli.fit_compressed_feature_expert import (
    _fit_logistic,
    _load_feature_dir,
    _metrics,
    _predict_logits,
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
DEFAULT_BUCKET_COUNT = 5
DEFAULT_MODEL_KEY = "bucket_conditioned_feature_expert"
DEFAULT_EXPERT_FAMILY = "bucket_conditioned_spn_residual"
DEFAULT_CANDIDATE_STATUS = "local_residual_candidate"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit bucket-conditioned compressed SPN feature experts. Bucket edges are "
            "derived only from train frozen scores and applied unchanged to validation."
        )
    )
    parser.add_argument("--train-feature-dir", required=True, type=Path)
    parser.add_argument("--validation-feature-dir", required=True, type=Path)
    parser.add_argument("--train-bucket-artifacts", nargs=2, required=True, type=Path)
    parser.add_argument("--validation-bucket-artifacts", nargs=2, required=True, type=Path)
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
    parser.add_argument("--bucket-count", type=int, default=DEFAULT_BUCKET_COUNT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.0003)
    parser.add_argument("--no-standardize", action="store_true")
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
    return parser.parse_args(argv)


def fit_bucket_conditioned_feature_expert(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    train_bucket_artifact_dirs: list[Path],
    validation_bucket_artifact_dirs: list[Path],
    bucket_feature: str = DEFAULT_BUCKET_FEATURE,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    steps: int = 1000,
    learning_rate: float = 0.05,
    l2: float = 0.0003,
    standardize: bool = True,
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
    if len(train_bucket_artifact_dirs) != 2 or len(validation_bucket_artifact_dirs) != 2:
        raise ValueError("bucket-conditioned expert requires exactly two bucket artifacts per split")
    if bucket_count < 2:
        raise ValueError("bucket_count must be at least 2")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")

    train_features = _load_feature_dir(train_feature_dir)
    validation_features = _load_feature_dir(validation_feature_dir)
    _validate_feature_dirs(train_features, validation_features)
    train_bucket_artifacts = [load_score_artifact(path) for path in train_bucket_artifact_dirs]
    validation_bucket_artifacts = [load_score_artifact(path) for path in validation_bucket_artifact_dirs]
    for index, artifact in enumerate(train_bucket_artifacts):
        _validate_feature_score_alignment(train_features, artifact, split=f"train bucket artifact {index}")
    for index, artifact in enumerate(validation_bucket_artifacts):
        _validate_feature_score_alignment(
            validation_features,
            artifact,
            split=f"validation bucket artifact {index}",
        )
    train_bucket_summary = evaluate_frozen_score_ensemble(train_bucket_artifacts)
    validation_bucket_summary = evaluate_frozen_score_ensemble(validation_bucket_artifacts)

    train_bucket_values = _residual_feature_views(train_bucket_artifacts)[bucket_feature].astype(
        np.float64,
        copy=True,
    )
    validation_bucket_values = _residual_feature_views(validation_bucket_artifacts)[bucket_feature].astype(
        np.float64,
        copy=True,
    )
    bucket_rng = np.random.default_rng(shuffle_seed)
    if shuffle_train_bucket_values:
        train_bucket_values = bucket_rng.permutation(train_bucket_values)
    if shuffle_validation_bucket_values:
        validation_bucket_values = bucket_rng.permutation(validation_bucket_values)
    bucket_edges = _quantile_edges(train_bucket_values, bucket_count)
    train_bucket_ids = np.searchsorted(bucket_edges[1:-1], train_bucket_values, side="right")
    validation_bucket_ids = np.searchsorted(bucket_edges[1:-1], validation_bucket_values, side="right")

    feature_selection = _select_feature_columns(
        train_features["metadata"],
        feature_count=int(train_features["features"].shape[1]),
        include_feature_families=include_feature_families or [],
        include_feature_prefixes=include_feature_prefixes or [],
    )
    selected_indices = np.asarray(feature_selection["selected_feature_indices"], dtype=np.int64)
    train_matrix = train_features["features"][:, selected_indices]
    validation_matrix = validation_features["features"][:, selected_indices]
    fit_labels = train_features["labels"].astype(np.float64, copy=True)
    if shuffle_train_labels:
        fit_labels = np.random.default_rng(shuffle_seed).permutation(fit_labels)

    train_logits, bucket_reports = _fit_predict_by_bucket(
        train_matrix=train_matrix,
        train_labels=fit_labels,
        eval_matrix=train_matrix,
        train_bucket_ids=train_bucket_ids,
        eval_bucket_ids=train_bucket_ids,
        bucket_count=bucket_count,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
    )
    validation_logits, validation_bucket_reports = _fit_predict_by_bucket(
        train_matrix=train_matrix,
        train_labels=fit_labels,
        eval_matrix=validation_matrix,
        train_bucket_ids=train_bucket_ids,
        eval_bucket_ids=validation_bucket_ids,
        bucket_count=bucket_count,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
    )
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
        "feature_model": "bucket_conditioned_logistic",
        "feature_steps": int(steps),
        "feature_learning_rate": float(learning_rate),
        "feature_l2": float(l2),
        "feature_standardize": bool(standardize),
        "feature_count": int(train_matrix.shape[1]),
        "feature_original_count": int(train_features["features"].shape[1]),
        "feature_selection": _feature_selection_metadata(feature_selection),
        "fit_train_labels_shuffled": bool(shuffle_train_labels),
        "fit_label_shuffle_seed": int(shuffle_seed),
        "bucket_train_values_shuffled": bool(shuffle_train_bucket_values),
        "bucket_validation_values_shuffled": bool(shuffle_validation_bucket_values),
        "bucket_shuffle_seed": int(shuffle_seed),
        "bucket_feature": bucket_feature,
        "bucket_count": int(bucket_count),
        "bucket_edges": _edge_report(bucket_edges),
        "bucket_source_model_order": [
            str(artifact.metadata.get("model_key", "")) for artifact in validation_bucket_artifacts
        ],
        "claim_scope": (
            "train-fitted bucket-conditioned compressed SPN residual expert diagnostic only; "
            "bucket edges are train-derived, validation is held out for final scoring, and "
            "this is not remote or formal SPN/PRESENT evidence"
        ),
    }
    train_artifact = EnsembleScoreArtifact(
        labels=train_features["labels"],
        probabilities=train_probabilities.astype(np.float32, copy=False),
        logits=train_logits.astype(np.float32, copy=False),
        sample_ids=train_features["sample_ids"],
        metadata=_score_metadata(train_features["metadata"], common_metadata, score_split="train"),
    )
    validation_artifact = EnsembleScoreArtifact(
        labels=validation_features["labels"],
        probabilities=validation_probabilities.astype(np.float32, copy=False),
        logits=validation_logits.astype(np.float32, copy=False),
        sample_ids=validation_features["sample_ids"],
        metadata=_score_metadata(validation_features["metadata"], common_metadata, score_split="validation"),
    )
    report = {
        "status": "pass",
        "decision": "bucket_conditioned_feature_expert_local_candidate_needs_controls",
        "model_key": model_key,
        "expert_family": expert_family,
        "train_feature_dir": str(train_feature_dir),
        "validation_feature_dir": str(validation_feature_dir),
        "train_bucket_artifact_dirs": [str(path) for path in train_bucket_artifact_dirs],
        "validation_bucket_artifact_dirs": [str(path) for path in validation_bucket_artifact_dirs],
        "bucket_feature": bucket_feature,
        "bucket_count": int(bucket_count),
        "bucket_edges": _edge_report(bucket_edges),
        "bucket_source_train_summary": train_bucket_summary,
        "bucket_source_validation_summary": validation_bucket_summary,
        "train_rows": int(len(train_features["labels"])),
        "validation_rows": int(len(validation_features["labels"])),
        "feature_count": int(train_matrix.shape[1]),
        "feature_selection": _feature_selection_metadata(feature_selection),
        "fit": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "standardize": bool(standardize),
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
        "train_bucket_reports": bucket_reports,
        "validation_bucket_reports": validation_bucket_reports,
        "train_metrics": _metrics(train_artifact.labels, train_artifact.probabilities),
        "validation_metrics": _metrics(validation_artifact.labels, validation_artifact.probabilities),
        "guardrails": [
            "bucket_edges_must_be_train_derived",
            "validation_split_final_score_only",
            "strict_negative_mode_required",
            "compare_against_raw117_and_fixed_fusion",
            "shuffle_or_mismatch_controls_required_before_remote_scaleup",
        ],
        "claim_scope": common_metadata["claim_scope"],
    }
    return train_artifact, validation_artifact, report


def _fit_predict_by_bucket(
    *,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    eval_matrix: np.ndarray,
    train_bucket_ids: np.ndarray,
    eval_bucket_ids: np.ndarray,
    bucket_count: int,
    steps: int,
    learning_rate: float,
    l2: float,
    standardize: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output_logits = np.zeros(eval_matrix.shape[0], dtype=np.float64)
    reports: list[dict[str, Any]] = []
    for bucket in range(bucket_count):
        train_mask = train_bucket_ids == bucket
        eval_mask = eval_bucket_ids == bucket
        fit_mask = train_mask
        fallback = False
        if int(train_mask.sum()) < 20 or len(np.unique(train_labels[train_mask])) < 2:
            fit_mask = np.ones_like(train_mask, dtype=bool)
            fallback = True
        fitted = _fit_logistic(
            train_matrix[fit_mask],
            train_labels[fit_mask],
            steps=steps,
            learning_rate=learning_rate,
            l2=l2,
            standardize=standardize,
        )
        if int(eval_mask.sum()):
            output_logits[eval_mask] = _predict_logits(eval_matrix[eval_mask], fitted)
        weights = np.asarray(fitted["weights"], dtype=np.float64)
        reports.append(
            {
                "bucket": int(bucket),
                "train_rows": int(train_mask.sum()),
                "eval_rows": int(eval_mask.sum()),
                "fit_rows": int(fit_mask.sum()),
                "global_fallback": bool(fallback),
                "weight_l2_norm": float(np.linalg.norm(weights)),
                "weight_abs_max": float(np.max(np.abs(weights))) if len(weights) else 0.0,
                "bias": float(fitted["bias"]),
            }
        )
    return output_logits, reports


def _validate_feature_score_alignment(
    feature: dict[str, Any],
    artifact: EnsembleScoreArtifact,
    *,
    split: str,
) -> None:
    if not np.array_equal(feature["labels"].astype(np.float32, copy=False), artifact.labels):
        raise ValueError(f"{split} feature labels differ from bucket artifact labels")
    if not np.array_equal(feature["sample_ids"].astype(str, copy=False), artifact.sample_ids):
        raise ValueError(f"{split} feature sample_ids differ from bucket artifact sample_ids")


def _quantile_edges(values: np.ndarray, bucket_count: int) -> np.ndarray:
    edges = np.quantile(values.astype(np.float64, copy=False), np.linspace(0.0, 1.0, bucket_count + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _edge_report(edges: np.ndarray) -> list[dict[str, float | None]]:
    reports = []
    for index in range(len(edges) - 1):
        lower = None if np.isneginf(edges[index]) else float(edges[index])
        upper = None if np.isposinf(edges[index + 1]) else float(edges[index + 1])
        reports.append({"bucket": int(index), "lower": lower, "upper": upper})
    return reports


def _feature_selection_metadata(feature_selection: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(feature_selection)
    if metadata.get("mode") == "all":
        metadata.pop("selected_feature_indices", None)
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_artifact, validation_artifact, report = fit_bucket_conditioned_feature_expert(
        train_feature_dir=args.train_feature_dir,
        validation_feature_dir=args.validation_feature_dir,
        train_bucket_artifact_dirs=args.train_bucket_artifacts,
        validation_bucket_artifact_dirs=args.validation_bucket_artifacts,
        bucket_feature=args.bucket_feature,
        bucket_count=args.bucket_count,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        standardize=not args.no_standardize,
        run_id=args.run_id,
        model_key=args.model_key,
        expert_family=args.expert_family,
        candidate_status=args.candidate_status,
        shuffle_train_labels=args.shuffle_train_labels,
        shuffle_train_bucket_values=args.shuffle_train_bucket_values,
        shuffle_validation_bucket_values=args.shuffle_validation_bucket_values,
        shuffle_seed=args.shuffle_seed,
        include_feature_families=args.include_feature_family,
        include_feature_prefixes=args.include_feature_prefix,
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
