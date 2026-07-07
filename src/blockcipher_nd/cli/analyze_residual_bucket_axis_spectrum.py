from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.cli.analyze_reliability_residual_buckets import (
    _edge_report,
    _quantile_edges,
    _residual_feature_views,
)
from blockcipher_nd.cli.decode_compressed_feature_sparsity import _feature_names_from_metadata
from blockcipher_nd.cli.fit_compressed_feature_expert import _load_feature_dir
from blockcipher_nd.evaluation.neural_ensemble import load_score_artifact
from blockcipher_nd.training.metrics import binary_auc


DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_bucket_axis_spectrum.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze which semantic SPN feature groups explain residual errors inside "
            "train-derived frozen-score reliability buckets."
        )
    )
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--bucket-artifacts", nargs=2, type=Path, required=True)
    parser.add_argument(
        "--bucket-feature",
        choices=[
            "min_confidence",
            "confidence_gap_abs",
            "logit_gap_abs",
            "signed_logit_delta_model1_minus_model0",
        ],
        default="logit_gap_abs",
    )
    parser.add_argument("--bucket-count", type=int, default=5)
    parser.add_argument("--top-groups", type=int, default=10)
    parser.add_argument(
        "--target",
        choices=[
            "residual_error_at_0_5",
            "residual_loss",
            "signed_margin",
            "global_candidate_gap",
        ],
        default="residual_error_at_0_5",
        help=(
            "Residual axis target. residual_error_at_0_5 uses hard mistakes; "
            "continuous targets are binarized at the median for per-group AUC ranking."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def analyze_residual_bucket_axis_spectrum(
    *,
    feature_dir: Path,
    bucket_artifact_dirs: list[Path],
    bucket_feature: str = "logit_gap_abs",
    bucket_count: int = 5,
    top_groups: int = 10,
    target: str = "residual_error_at_0_5",
) -> dict[str, Any]:
    if len(bucket_artifact_dirs) != 2:
        raise ValueError("residual bucket spectrum requires exactly two bucket artifacts")
    if bucket_count < 2:
        raise ValueError("bucket_count must be at least 2")
    if top_groups <= 0:
        raise ValueError("top_groups must be positive")

    feature_artifact = _load_feature_dir(feature_dir)
    artifacts = [load_score_artifact(path) for path in bucket_artifact_dirs]
    _validate_alignment(feature_artifact, artifacts)

    feature_matrix = feature_artifact["features"].astype(np.float64, copy=False)
    labels = feature_artifact["labels"].astype(np.float32, copy=False)
    metadata = feature_artifact["metadata"]
    view_metadata = metadata.get("feature_view_metadata", metadata)
    feature_names = _feature_names(view_metadata)
    if len(feature_names) != feature_matrix.shape[1]:
        raise ValueError(f"expected {feature_matrix.shape[1]} feature names, got {len(feature_names)}")

    groups = _semantic_groups(feature_names)
    probability_mean = np.mean([artifact.probabilities for artifact in artifacts], axis=0)
    residual_errors = ((probability_mean >= 0.5).astype(np.float32) != labels).astype(np.float32)
    target_values = _target_values(target, labels, probability_mean, artifacts)
    target_labels = _target_labels(target_values)
    bucket_values = _residual_feature_views(artifacts)[bucket_feature]
    bucket_edges = _quantile_edges(bucket_values, bucket_count)
    bucket_ids = np.searchsorted(bucket_edges[1:-1], bucket_values, side="right")
    global_group_reports = [
        _group_report(
            name=name,
            indices=indices,
            feature_matrix=feature_matrix,
            labels=labels,
            residual_errors=residual_errors,
            target_values=target_values,
            target_labels=target_labels,
            mask=np.ones(len(labels), dtype=bool),
        )
        for name, indices in groups
    ]
    global_ranked = sorted(
        global_group_reports,
        key=lambda row: (
            float(row["target_score"]),
            float(row["residual_error_score"]),
            float(row["label_score"]),
        ),
        reverse=True,
    )

    bucket_reports = []
    for bucket in range(bucket_count):
        mask = bucket_ids == bucket
        rows = int(mask.sum())
        group_reports = [
            _group_report(
                name=name,
                indices=indices,
                feature_matrix=feature_matrix,
                labels=labels,
                residual_errors=residual_errors,
                target_values=target_values,
                target_labels=target_labels,
                mask=mask,
            )
            for name, indices in groups
        ]
        ranked = sorted(
            group_reports,
            key=lambda row: (
                float(row["target_score"]),
                float(row["residual_error_score"]),
                float(row["label_score"]),
            ),
            reverse=True,
        )
        bucket_reports.append(
            {
                "bucket": int(bucket),
                "rows": rows,
                "positive_rows": int(labels[mask].sum()),
                "residual_error_rows": int(residual_errors[mask].sum()),
                "residual_error_rate_at_0_5": float(residual_errors[mask].mean()) if rows else 0.0,
                "target_mean": float(target_values[mask].mean()) if rows else 0.0,
                "top_groups": ranked[:top_groups],
            }
        )

    return {
        "status": "pass",
        "decision": "residual_bucket_axis_spectrum_ready",
        "feature_dir": str(feature_dir),
        "bucket_artifact_dirs": [str(path) for path in bucket_artifact_dirs],
        "bucket_feature": bucket_feature,
        "bucket_count": int(bucket_count),
        "bucket_edges": _edge_report(bucket_edges),
        "top_groups": int(top_groups),
        "target": target,
        "target_mean": float(target_values.mean()),
        "row_count": int(len(labels)),
        "group_count": int(len(groups)),
        "residual_error_rate_at_0_5": float(residual_errors.mean()),
        "global_top_groups": global_ranked[:top_groups],
        "bucket_reports": bucket_reports,
        "claim_scope": (
            "train-only or validation-only diagnostic over frozen score buckets and existing SPN "
            "feature artifacts; does not train, alter labels, alter negatives, launch remote work, "
            "or provide formal SPN/PRESENT evidence"
        ),
    }


def _validate_alignment(feature_artifact: dict[str, Any], artifacts: list[Any]) -> None:
    labels = feature_artifact["labels"].astype(np.float32, copy=False)
    sample_ids = feature_artifact["sample_ids"].astype(str, copy=False)
    for index, artifact in enumerate(artifacts):
        if not np.array_equal(labels, artifact.labels.astype(np.float32, copy=False)):
            raise ValueError(f"bucket artifact {index} labels differ from feature labels")
        if not np.array_equal(sample_ids, artifact.sample_ids.astype(str, copy=False)):
            raise ValueError(f"bucket artifact {index} sample_ids differ from feature sample_ids")


def _semantic_groups(feature_names: list[str]) -> list[tuple[str, list[int]]]:
    grouped: dict[str, list[int]] = {}
    for index, name in enumerate(feature_names):
        group = _semantic_group_name(name)
        grouped.setdefault(group, []).append(index)
    return [(name, indices) for name, indices in sorted(grouped.items())]


def _feature_names(view_metadata: dict[str, Any]) -> list[str]:
    names = view_metadata.get("feature_names")
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return _feature_names_from_metadata(view_metadata)


def _semantic_group_name(feature_name: str) -> str:
    return re.sub(r"_(depth|cell|word|trailword)\d+$", "", feature_name)


def _group_report(
    *,
    name: str,
    indices: list[int],
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    residual_errors: np.ndarray,
    target_values: np.ndarray,
    target_labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    values = feature_matrix[:, indices].mean(axis=1)
    bucket_values = values[mask]
    bucket_labels = labels[mask]
    bucket_errors = residual_errors[mask]
    bucket_target_values = target_values[mask]
    bucket_target_labels = target_labels[mask]
    label_auc = _safe_auc(bucket_labels, bucket_values)
    residual_error_auc = _safe_auc(bucket_errors, bucket_values)
    target_auc = _safe_auc(bucket_target_labels, bucket_values)
    return {
        "group": name,
        "feature_count": int(len(indices)),
        "mean": float(bucket_values.mean()) if len(bucket_values) else 0.0,
        "std": float(bucket_values.std()) if len(bucket_values) else 0.0,
        "target_mean": float(bucket_target_values.mean()) if len(bucket_target_values) else 0.0,
        "label_auc": label_auc,
        "label_score": _auc_score(label_auc),
        "residual_error_auc": residual_error_auc,
        "residual_error_score": _auc_score(residual_error_auc),
        "target_auc": target_auc,
        "target_score": _auc_score(target_auc),
    }


def _target_values(
    target: str,
    labels: np.ndarray,
    probability_mean: np.ndarray,
    artifacts: list[Any],
) -> np.ndarray:
    if target == "residual_error_at_0_5":
        return ((probability_mean >= 0.5).astype(np.float32) != labels).astype(np.float32)
    if target == "residual_loss":
        return np.abs(labels.astype(np.float64, copy=False) - probability_mean.astype(np.float64, copy=False))
    if target == "signed_margin":
        signed_labels = labels.astype(np.float64, copy=False) * 2.0 - 1.0
        return -signed_labels * (probability_mean.astype(np.float64, copy=False) - 0.5)
    if target == "global_candidate_gap":
        if len(artifacts) != 2:
            raise ValueError("global_candidate_gap requires exactly two artifacts")
        return np.abs(
            artifacts[1].probabilities.astype(np.float64, copy=False)
            - artifacts[0].probabilities.astype(np.float64, copy=False)
        )
    raise ValueError(f"unsupported target: {target}")


def _target_labels(target_values: np.ndarray) -> np.ndarray:
    if len(target_values) == 0:
        return target_values.astype(np.float32, copy=False)
    threshold = float(np.median(target_values))
    labels = (target_values > threshold).astype(np.float32)
    if len(np.unique(labels)) < 2:
        labels = (target_values >= threshold).astype(np.float32)
    return labels


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(binary_auc(labels.astype(np.float32, copy=False), scores.astype(np.float32, copy=False)))


def _auc_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return abs(float(value) - 0.5)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze_residual_bucket_axis_spectrum(
        feature_dir=args.feature_dir,
        bucket_artifact_dirs=args.bucket_artifacts,
        bucket_feature=args.bucket_feature,
        bucket_count=args.bucket_count,
        top_groups=args.top_groups,
        target=args.target,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
