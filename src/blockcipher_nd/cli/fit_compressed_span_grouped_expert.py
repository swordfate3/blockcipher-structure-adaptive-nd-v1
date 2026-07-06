from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.cli.fit_compressed_feature_expert import (
    _fit_logistic,
    _metrics,
    _predict_logits,
    _sigmoid,
    fit_compressed_feature_expert,
)
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, write_score_artifact


DEFAULT_MODEL_KEY = "compressed_span_grouped_logistic_expert"
DEFAULT_EXPERT_FAMILY = "compressed_spn_span_grouped_summary"
DEFAULT_CANDIDATE_STATUS = "compressed_span_grouped_screen"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a two-branch grouped expert over compressed SPN span-summary features."
    )
    parser.add_argument("--train-feature-dir", required=True, type=Path)
    parser.add_argument("--validation-feature-dir", required=True, type=Path)
    parser.add_argument("--output-validation-dir", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-train-dir", type=Path, default=None)
    parser.add_argument("--primary-prefix", default="primary_")
    parser.add_argument("--auxiliary-prefix", default="aux_")
    parser.add_argument("--branch-steps", type=int, default=800)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--expert-family", default=DEFAULT_EXPERT_FAMILY)
    parser.add_argument("--candidate-status", default=DEFAULT_CANDIDATE_STATUS)
    return parser.parse_args(argv)


def fit_compressed_span_grouped_expert(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    primary_prefix: str = "primary_",
    auxiliary_prefix: str = "aux_",
    branch_steps: int = 800,
    steps: int = 800,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    standardize: bool = True,
    run_id: str = "",
    model_key: str = DEFAULT_MODEL_KEY,
    expert_family: str = DEFAULT_EXPERT_FAMILY,
    candidate_status: str = DEFAULT_CANDIDATE_STATUS,
) -> tuple[EnsembleScoreArtifact, EnsembleScoreArtifact, dict[str, Any]]:
    if not primary_prefix:
        raise ValueError("primary_prefix must be non-empty")
    if not auxiliary_prefix:
        raise ValueError("auxiliary_prefix must be non-empty")
    if branch_steps <= 0:
        raise ValueError("branch_steps must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    primary_train, primary_validation, primary_report = fit_compressed_feature_expert(
        train_feature_dir=train_feature_dir,
        validation_feature_dir=validation_feature_dir,
        steps=branch_steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
        run_id=run_id,
        model_key=f"{model_key}_primary_branch",
        expert_family=f"{expert_family}_primary",
        candidate_status="primary_branch",
        include_feature_prefixes=[primary_prefix],
    )
    auxiliary_train, auxiliary_validation, auxiliary_report = fit_compressed_feature_expert(
        train_feature_dir=train_feature_dir,
        validation_feature_dir=validation_feature_dir,
        steps=branch_steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
        run_id=run_id,
        model_key=f"{model_key}_auxiliary_branch",
        expert_family=f"{expert_family}_auxiliary",
        candidate_status="auxiliary_branch",
        include_feature_prefixes=[auxiliary_prefix],
    )
    _validate_branch_alignment(primary_train, auxiliary_train)
    _validate_branch_alignment(primary_validation, auxiliary_validation)

    train_branch_features = np.stack([primary_train.logits, auxiliary_train.logits], axis=1)
    validation_branch_features = np.stack(
        [primary_validation.logits, auxiliary_validation.logits],
        axis=1,
    )
    fitted = _fit_logistic(
        train_branch_features.astype(np.float64, copy=False),
        primary_train.labels.astype(np.float64, copy=False),
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
    )
    train_logits = _predict_logits(train_branch_features, fitted)
    validation_logits = _predict_logits(validation_branch_features, fitted)
    train_probabilities = _sigmoid(train_logits)
    validation_probabilities = _sigmoid(validation_logits)

    common_metadata = {
        **primary_validation.metadata,
        "model_key": model_key,
        "expert_family": expert_family,
        "candidate_status": candidate_status,
        "run_id": run_id,
        "feature_fit_split": "train",
        "feature_model": "two_branch_logistic",
        "feature_count": 2,
        "feature_original_count": int(primary_report["feature_selection"]["original_feature_count"]),
        "feature_train_dir": str(train_feature_dir),
        "feature_validation_dir": str(validation_feature_dir),
        "feature_steps": int(steps),
        "feature_learning_rate": float(learning_rate),
        "feature_l2": float(l2),
        "feature_standardize": bool(standardize),
        "branch_prefixes": {"primary": primary_prefix, "auxiliary": auxiliary_prefix},
        "branch_feature_counts": {
            "primary": int(primary_report["feature_count"]),
            "auxiliary": int(auxiliary_report["feature_count"]),
        },
        "claim_scope": (
            "train-fitted two-branch compressed SPN span-summary expert diagnostic only; "
            "branch and combiner fits use train labels only, validation is held out for final scoring, "
            "and this is not remote or formal SPN/PRESENT evidence"
        ),
    }
    train_artifact = EnsembleScoreArtifact(
        labels=primary_train.labels,
        probabilities=train_probabilities.astype(np.float32, copy=False),
        logits=train_logits.astype(np.float32, copy=False),
        sample_ids=primary_train.sample_ids,
        metadata={**common_metadata, "score_split": "train"},
    )
    validation_artifact = EnsembleScoreArtifact(
        labels=primary_validation.labels,
        probabilities=validation_probabilities.astype(np.float32, copy=False),
        logits=validation_logits.astype(np.float32, copy=False),
        sample_ids=primary_validation.sample_ids,
        metadata={**common_metadata, "score_split": "validation"},
    )
    train_metrics = _metrics(train_artifact.labels, train_artifact.probabilities)
    validation_metrics = _metrics(validation_artifact.labels, validation_artifact.probabilities)
    report = {
        "status": "pass",
        "decision": "compressed_span_grouped_expert_local_screen_positive_needs_controls",
        "model_key": model_key,
        "expert_family": expert_family,
        "train_feature_dir": str(train_feature_dir),
        "validation_feature_dir": str(validation_feature_dir),
        "train_rows": int(len(train_artifact.labels)),
        "validation_rows": int(len(validation_artifact.labels)),
        "feature_count": 2,
        "branch_prefixes": {"primary": primary_prefix, "auxiliary": auxiliary_prefix},
        "branch_reports": {
            "primary": _branch_report(primary_report),
            "auxiliary": _branch_report(auxiliary_report),
        },
        "fit": {
            "branch_steps": int(branch_steps),
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "standardize": bool(standardize),
            "weight_count": int(len(np.asarray(fitted["weights"]))),
            "weight_l2_norm": float(np.linalg.norm(np.asarray(fitted["weights"], dtype=np.float64))),
            "weight_abs_max": float(np.max(np.abs(np.asarray(fitted["weights"], dtype=np.float64)))),
            "bias": float(fitted["bias"]),
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "guardrails": [
            "branch_fit_split_must_be_train",
            "combiner_fit_split_must_be_train",
            "validation_split_final_score_only",
            "strict_negative_mode_required",
            "compare_against_full_summary_flat_logistic",
            "compare_against_primary_and_auxiliary_branches",
        ],
        "claim_scope": common_metadata["claim_scope"],
    }
    return train_artifact, validation_artifact, report


def _validate_branch_alignment(left: EnsembleScoreArtifact, right: EnsembleScoreArtifact) -> None:
    if not np.array_equal(left.labels, right.labels):
        raise ValueError("branch labels differ")
    if not np.array_equal(left.sample_ids, right.sample_ids):
        raise ValueError("branch sample_ids differ")


def _branch_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_key": report["model_key"],
        "expert_family": report["expert_family"],
        "feature_count": int(report["feature_count"]),
        "feature_selection": report["feature_selection"],
        "train_metrics": report["train_metrics"],
        "validation_metrics": report["validation_metrics"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_artifact, validation_artifact, report = fit_compressed_span_grouped_expert(
        train_feature_dir=args.train_feature_dir,
        validation_feature_dir=args.validation_feature_dir,
        primary_prefix=args.primary_prefix,
        auxiliary_prefix=args.auxiliary_prefix,
        branch_steps=args.branch_steps,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        standardize=not args.no_standardize,
        run_id=args.run_id,
        model_key=args.model_key,
        expert_family=args.expert_family,
        candidate_status=args.candidate_status,
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
