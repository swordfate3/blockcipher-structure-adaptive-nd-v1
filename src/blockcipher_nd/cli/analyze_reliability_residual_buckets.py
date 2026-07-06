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


DEFAULT_BUCKET_COUNT = 5
DEFAULT_MIN_DISAGREEMENT_RATE = 0.005
DEFAULT_MIN_BUCKET_FRACTION = 0.02
DEFAULT_MIN_CORRECTION_GAP = 0.005
DEFAULT_MIN_BOTH_WRONG_LIFT = 0.005


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze train-derived reliability/residual buckets for two aligned frozen "
            "score artifacts, then apply the same bucket edges to validation scores."
        )
    )
    parser.add_argument("--train-artifacts", nargs=2, required=True, type=Path)
    parser.add_argument("--validation-artifacts", nargs=2, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bucket-count", type=int, default=DEFAULT_BUCKET_COUNT)
    parser.add_argument("--min-disagreement-rate", type=float, default=DEFAULT_MIN_DISAGREEMENT_RATE)
    parser.add_argument("--min-bucket-fraction", type=float, default=DEFAULT_MIN_BUCKET_FRACTION)
    parser.add_argument("--min-correction-gap", type=float, default=DEFAULT_MIN_CORRECTION_GAP)
    parser.add_argument("--min-both-wrong-lift", type=float, default=DEFAULT_MIN_BOTH_WRONG_LIFT)
    return parser.parse_args(argv)


def analyze_reliability_residual_buckets(
    *,
    train_artifacts: list[EnsembleScoreArtifact],
    validation_artifacts: list[EnsembleScoreArtifact],
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    min_disagreement_rate: float = DEFAULT_MIN_DISAGREEMENT_RATE,
    min_bucket_fraction: float = DEFAULT_MIN_BUCKET_FRACTION,
    min_correction_gap: float = DEFAULT_MIN_CORRECTION_GAP,
    min_both_wrong_lift: float = DEFAULT_MIN_BOTH_WRONG_LIFT,
) -> dict[str, Any]:
    if len(train_artifacts) != 2 or len(validation_artifacts) != 2:
        raise ValueError("reliability/residual bucket analysis requires exactly two train and validation artifacts")
    if bucket_count < 2:
        raise ValueError("bucket_count must be at least 2")
    if min_bucket_fraction < 0.0 or min_bucket_fraction >= 1.0:
        raise ValueError("min_bucket_fraction must be in [0, 1)")
    _validate_matching_model_order(train_artifacts, validation_artifacts)

    train_summary = evaluate_frozen_score_ensemble(train_artifacts)
    validation_summary = evaluate_frozen_score_ensemble(validation_artifacts)
    model_order = [str(item.metadata.get("model_key", "")) for item in validation_artifacts]

    train_features = _residual_feature_views(train_artifacts)
    validation_features = _residual_feature_views(validation_artifacts)
    bucket_reports: dict[str, Any] = {}
    candidate_buckets: list[dict[str, Any]] = []
    validation_overlap = _pair_overlap(validation_artifacts)
    min_rows = max(1, int(np.ceil(len(validation_artifacts[0].labels) * min_bucket_fraction)))

    for feature_name, train_values in train_features.items():
        edges = _quantile_edges(train_values, bucket_count)
        train_buckets = _bucket_summary(
            artifacts=train_artifacts,
            values=train_values,
            edges=edges,
            model_order=model_order,
        )
        validation_buckets = _bucket_summary(
            artifacts=validation_artifacts,
            values=validation_features[feature_name],
            edges=edges,
            model_order=model_order,
        )
        bucket_reports[feature_name] = {
            "train_derived_edges": _edge_report(edges),
            "train": train_buckets,
            "validation": validation_buckets,
        }
        candidate_buckets.extend(
            _candidate_bucket_rows(
                feature_name=feature_name,
                buckets=validation_buckets,
                min_rows=min_rows,
                min_correction_gap=min_correction_gap,
                min_both_wrong_lift=min_both_wrong_lift,
                global_both_wrong_rate=float(validation_overlap["both_wrong_rate_at_0_5"]),
                model_order=model_order,
            )
        )

    validation_pairwise = validation_summary["diversity"]["pairwise"][0]
    validation_disagreement = float(validation_pairwise["disagreement_rate_at_0_5"])
    if validation_disagreement < min_disagreement_rate:
        decision = "reliability_residual_bucket_diagnostic_low_disagreement"
        action = "do_not_build_third_expert_from_current_two_score_residuals"
    elif candidate_buckets:
        decision = "reliability_residual_bucket_route_candidate_local"
        action = "design_a_frozen_residual_expert_or_control_gate_before_training_scaleup"
    else:
        decision = "reliability_residual_bucket_diagnostic_only"
        action = "keep_as_error_analysis; no third-expert promotion yet"

    return {
        "status": "pass",
        "decision": decision,
        "action": action,
        "model_order": model_order,
        "rows": {
            "train": int(len(train_artifacts[0].labels)),
            "validation": int(len(validation_artifacts[0].labels)),
        },
        "bucket_count": int(bucket_count),
        "thresholds": {
            "min_disagreement_rate": float(min_disagreement_rate),
            "min_bucket_fraction": float(min_bucket_fraction),
            "min_bucket_rows": int(min_rows),
            "min_correction_gap": float(min_correction_gap),
            "min_both_wrong_lift": float(min_both_wrong_lift),
        },
        "train_fixed_summary": train_summary,
        "validation_fixed_summary": validation_summary,
        "validation_overlap_at_0_5": validation_overlap,
        "bucket_reports": bucket_reports,
        "candidate_buckets": candidate_buckets,
        "claim_scope": (
            "local frozen-score reliability/residual bucket diagnostic only; "
            "uses train-derived bucket edges and held-out validation scoring; "
            "not a trained third expert, not remote evidence, and not formal SPN/PRESENT evidence"
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


def _residual_feature_views(artifacts: list[EnsembleScoreArtifact]) -> dict[str, np.ndarray]:
    left, right = artifacts
    left_confidence = np.abs(left.probabilities.astype(np.float64, copy=False) - 0.5)
    right_confidence = np.abs(right.probabilities.astype(np.float64, copy=False) - 0.5)
    left_logits = left.logits.astype(np.float64, copy=False)
    right_logits = right.logits.astype(np.float64, copy=False)
    return {
        "min_confidence": np.minimum(left_confidence, right_confidence),
        "confidence_gap_abs": np.abs(left_confidence - right_confidence),
        "logit_gap_abs": np.abs(left_logits - right_logits),
        "signed_logit_delta_model1_minus_model0": right_logits - left_logits,
    }


def _quantile_edges(values: np.ndarray, bucket_count: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, bucket_count + 1)
    finite_edges = np.quantile(values.astype(np.float64, copy=False), quantiles)
    edges = finite_edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _edge_report(edges: np.ndarray) -> list[dict[str, float | None]]:
    reports = []
    for index in range(len(edges) - 1):
        lower = None if np.isneginf(edges[index]) else float(edges[index])
        upper = None if np.isposinf(edges[index + 1]) else float(edges[index + 1])
        reports.append({"bucket": index, "lower": lower, "upper": upper})
    return reports


def _bucket_summary(
    *,
    artifacts: list[EnsembleScoreArtifact],
    values: np.ndarray,
    edges: np.ndarray,
    model_order: list[str],
) -> list[dict[str, Any]]:
    labels = artifacts[0].labels.astype(np.float32, copy=False)
    probabilities = np.stack([item.probabilities for item in artifacts], axis=1).astype(
        np.float64, copy=False
    )
    predictions = probabilities >= 0.5
    correct = predictions == labels.reshape(-1, 1)
    bucket_ids = np.searchsorted(edges[1:-1], values, side="right")
    reports: list[dict[str, Any]] = []
    n = max(1, len(labels))
    for bucket in range(len(edges) - 1):
        mask = bucket_ids == bucket
        count = int(mask.sum())
        row = {
            "bucket": bucket,
            "rows": count,
            "row_fraction": float(count / n),
            "feature_mean": _masked_float(values, mask, "mean"),
            "feature_min": _masked_float(values, mask, "min"),
            "feature_max": _masked_float(values, mask, "max"),
            "positive_rate": _masked_rate(labels == 1.0, mask),
            "disagreement_rate_at_0_5": _masked_rate(predictions[:, 0] != predictions[:, 1], mask),
            "both_wrong_rate_at_0_5": _masked_rate(~correct[:, 0] & ~correct[:, 1], mask),
            f"{model_order[0]}_error_rate_at_0_5": _masked_rate(~correct[:, 0], mask),
            f"{model_order[1]}_error_rate_at_0_5": _masked_rate(~correct[:, 1], mask),
            f"{model_order[0]}_wrong_{model_order[1]}_correct_rate_at_0_5": _masked_rate(
                ~correct[:, 0] & correct[:, 1], mask
            ),
            f"{model_order[1]}_wrong_{model_order[0]}_correct_rate_at_0_5": _masked_rate(
                ~correct[:, 1] & correct[:, 0], mask
            ),
        }
        reports.append(row)
    return reports


def _masked_rate(mask_values: np.ndarray, bucket_mask: np.ndarray) -> float | None:
    count = int(bucket_mask.sum())
    if count == 0:
        return None
    return float(mask_values[bucket_mask].mean())


def _masked_float(values: np.ndarray, bucket_mask: np.ndarray, mode: str) -> float | None:
    if int(bucket_mask.sum()) == 0:
        return None
    selected = values[bucket_mask]
    if mode == "mean":
        return float(np.mean(selected))
    if mode == "min":
        return float(np.min(selected))
    if mode == "max":
        return float(np.max(selected))
    raise ValueError(f"unsupported masked float mode: {mode}")


def _pair_overlap(artifacts: list[EnsembleScoreArtifact]) -> dict[str, Any]:
    labels = artifacts[0].labels.astype(np.float32, copy=False)
    probabilities = np.stack([item.probabilities for item in artifacts], axis=1)
    predictions = probabilities >= 0.5
    correct = predictions == labels.reshape(-1, 1)
    left_wrong = ~correct[:, 0]
    right_wrong = ~correct[:, 1]
    either_wrong = left_wrong | right_wrong
    both_wrong = left_wrong & right_wrong
    n = max(1, len(labels))
    either_wrong_count = int(either_wrong.sum())
    return {
        "both_correct_rate_at_0_5": float((correct[:, 0] & correct[:, 1]).sum() / n),
        "both_wrong_rate_at_0_5": float(both_wrong.sum() / n),
        "model0_wrong_model1_correct_rate_at_0_5": float((left_wrong & correct[:, 1]).sum() / n),
        "model1_wrong_model0_correct_rate_at_0_5": float((right_wrong & correct[:, 0]).sum() / n),
        "model0_wrong_count_at_0_5": int(left_wrong.sum()),
        "model1_wrong_count_at_0_5": int(right_wrong.sum()),
        "both_wrong_count_at_0_5": int(both_wrong.sum()),
        "either_wrong_count_at_0_5": either_wrong_count,
        "error_jaccard_at_0_5": float(both_wrong.sum() / either_wrong_count) if either_wrong_count else 0.0,
    }


def _candidate_bucket_rows(
    *,
    feature_name: str,
    buckets: list[dict[str, Any]],
    min_rows: int,
    min_correction_gap: float,
    min_both_wrong_lift: float,
    global_both_wrong_rate: float,
    model_order: list[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    left_corrects_key = f"{model_order[0]}_wrong_{model_order[1]}_correct_rate_at_0_5"
    right_corrects_key = f"{model_order[1]}_wrong_{model_order[0]}_correct_rate_at_0_5"
    for row in buckets:
        if int(row["rows"]) < min_rows:
            continue
        left_corrects = _none_as_zero(row[left_corrects_key])
        right_corrects = _none_as_zero(row[right_corrects_key])
        both_wrong_rate = _none_as_zero(row["both_wrong_rate_at_0_5"])
        correction_gap = abs(left_corrects - right_corrects)
        both_wrong_lift = both_wrong_rate - global_both_wrong_rate
        reasons = []
        if correction_gap >= min_correction_gap:
            reasons.append("directional_correction_imbalance")
        if both_wrong_lift >= min_both_wrong_lift:
            reasons.append("hard_case_concentration")
        if reasons:
            candidates.append(
                {
                    "feature": feature_name,
                    "bucket": int(row["bucket"]),
                    "rows": int(row["rows"]),
                    "reasons": reasons,
                    "correction_gap": float(correction_gap),
                    "both_wrong_lift": float(both_wrong_lift),
                    "bucket_report": row,
                }
            )
    return candidates


def _none_as_zero(value: Any) -> float:
    return 0.0 if value is None else float(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_artifacts = [load_score_artifact(path) for path in args.train_artifacts]
    validation_artifacts = [load_score_artifact(path) for path in args.validation_artifacts]
    report = analyze_reliability_residual_buckets(
        train_artifacts=train_artifacts,
        validation_artifacts=validation_artifacts,
        bucket_count=args.bucket_count,
        min_disagreement_rate=args.min_disagreement_rate,
        min_bucket_fraction=args.min_bucket_fraction,
        min_correction_gap=args.min_correction_gap,
        min_both_wrong_lift=args.min_both_wrong_lift,
    )
    report["train_artifact_dirs"] = [str(path) for path in args.train_artifacts]
    report["validation_artifact_dirs"] = [str(path) for path in args.validation_artifacts]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
