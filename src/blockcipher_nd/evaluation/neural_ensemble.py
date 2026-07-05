from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.training.metrics import (
    best_threshold_accuracy_and_threshold,
    binary_auc,
)


DEFAULT_NEAR_NEIGHBOR_FAMILIES = frozenset({"invp_cell", "ddt_graph", "p_layer_graph"})


@dataclass(frozen=True)
class EnsembleScoreArtifact:
    labels: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray
    sample_ids: np.ndarray
    metadata: dict[str, Any]


def write_score_artifact(output_dir: Path, artifact: EnsembleScoreArtifact) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_artifact_arrays(artifact)
    np.save(output_dir / "labels.npy", artifact.labels.astype(np.float32, copy=False))
    np.save(output_dir / "probabilities.npy", artifact.probabilities.astype(np.float32, copy=False))
    np.save(output_dir / "logits.npy", artifact.logits.astype(np.float32, copy=False))
    np.save(output_dir / "sample_ids.npy", artifact.sample_ids.astype(str, copy=False))
    (output_dir / "models.json").write_text(
        json.dumps(artifact.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_score_artifact(input_dir: Path) -> EnsembleScoreArtifact:
    metadata = json.loads((input_dir / "models.json").read_text(encoding="utf-8"))
    return EnsembleScoreArtifact(
        labels=np.load(input_dir / "labels.npy").astype(np.float32, copy=False),
        probabilities=np.load(input_dir / "probabilities.npy").astype(np.float32, copy=False),
        logits=np.load(input_dir / "logits.npy").astype(np.float32, copy=False),
        sample_ids=np.load(input_dir / "sample_ids.npy").astype(str, copy=False),
        metadata=metadata,
    )


def evaluate_frozen_score_ensemble(artifacts: list[EnsembleScoreArtifact]) -> dict[str, Any]:
    if len(artifacts) < 2:
        raise ValueError("neural ensemble requires at least two score artifacts")
    _validate_artifact_alignment(artifacts)
    labels = artifacts[0].labels.astype(np.float32, copy=False)
    probability_matrix = np.stack([item.probabilities for item in artifacts], axis=1)
    logit_matrix = np.stack([item.logits for item in artifacts], axis=1)
    row_reports = [_single_report(item) for item in artifacts]
    diversity = _diversity_report(labels, probability_matrix, logit_matrix, row_reports)
    ensemble_reports = [
        _ensemble_report("probability_mean", labels, probability_matrix.mean(axis=1)),
        _ensemble_report("logit_mean", labels, _sigmoid(logit_matrix.mean(axis=1))),
        _ensemble_report("sum_logodds", labels, _sigmoid(logit_matrix.sum(axis=1))),
        _ensemble_report(
            "auc_positive_weighted_logit_mean",
            labels,
            _sigmoid(logit_matrix @ _auc_positive_weights(row_reports)),
        ),
        _ensemble_report("rank_average", labels, _rank_average_probabilities(probability_matrix)),
    ]
    best_single = max(row_reports, key=lambda row: row["metrics"]["auc"])
    best_ensemble = max(ensemble_reports, key=lambda row: row["metrics"]["auc"])
    return {
        "status": "pass",
        "models": row_reports,
        "ensembles": ensemble_reports,
        "best_single": best_single,
        "best_ensemble": best_ensemble,
        "delta_best_ensemble_vs_single_auc": float(
            best_ensemble["metrics"]["auc"] - best_single["metrics"]["auc"]
        ),
        "diversity": diversity,
        "diverse_expert_pool": assess_diverse_expert_pool(
            {
                "status": "pass",
                "models": row_reports,
                "diversity": diversity,
            }
        ),
        "claim_scope": (
            "application-level frozen score aggregation diagnostic only; "
            "not raw single-sample SOTA, not architecture evidence by itself"
        ),
    }


def assess_diverse_expert_pool(
    summary: dict[str, Any],
    *,
    min_expert_auc: float = 0.52,
    min_family_count: int = 3,
    max_error_jaccard: float = 0.75,
    near_neighbor_families: set[str] | frozenset[str] = DEFAULT_NEAR_NEIGHBOR_FAMILIES,
) -> dict[str, Any]:
    models = summary.get("models", [])
    pairwise = (summary.get("diversity") or {}).get("pairwise", [])
    errors: list[str] = []
    eligible_models: list[dict[str, Any]] = []
    model_families: dict[str, str] = {}

    if not isinstance(models, list) or not models:
        errors.append("missing_models")
        models = []
    if not isinstance(pairwise, list) or not pairwise:
        errors.append("missing_pairwise_diversity")
        pairwise = []

    for model in models:
        if not isinstance(model, dict):
            continue
        model_key = str(model.get("model_key", ""))
        metadata = model.get("metadata") if isinstance(model.get("metadata"), dict) else {}
        family = str(metadata.get("expert_family", "")).strip()
        if not family:
            errors.append(f"missing_expert_family:{model_key}")
            continue
        model_families[model_key] = family
        status = str(metadata.get("candidate_status", "weak_positive"))
        auc = _metric_auc(model)
        if auc is None or auc < min_expert_auc:
            continue
        if status == "rejected":
            continue
        eligible_models.append(model)

    eligible_families = sorted({model_families.get(str(model.get("model_key", "")), "") for model in eligible_models})
    eligible_families = [family for family in eligible_families if family]
    non_neighbor_families = sorted(
        family for family in eligible_families if family not in set(near_neighbor_families)
    )
    if len(eligible_families) < min_family_count:
        errors.append("too_few_eligible_families")
    if not non_neighbor_families:
        errors.append("missing_non_neighbor_expert")

    non_neighbor_pairs = []
    for row in pairwise:
        if not isinstance(row, dict):
            continue
        left = str(row.get("left", ""))
        right = str(row.get("right", ""))
        left_family = model_families.get(left)
        right_family = model_families.get(right)
        if not left_family or not right_family:
            continue
        if left_family in set(near_neighbor_families) and right_family in set(near_neighbor_families):
            continue
        error_jaccard = _float_or_none(row.get("error_jaccard_at_0_5"))
        if error_jaccard is None:
            continue
        non_neighbor_pairs.append({**row, "left_family": left_family, "right_family": right_family})

    best_non_neighbor_pair = min(
        non_neighbor_pairs,
        key=lambda row: float(row["error_jaccard_at_0_5"]),
        default=None,
    )
    if best_non_neighbor_pair is None:
        errors.append("missing_non_neighbor_pairwise_diversity")
    elif float(best_non_neighbor_pair["error_jaccard_at_0_5"]) > max_error_jaccard:
        errors.append("non_neighbor_error_overlap_too_high")

    return {
        "status": "pass" if not errors else "fail",
        "decision": "diverse_expert_pool_ready" if not errors else "diverse_expert_pool_not_ready",
        "errors": errors,
        "eligible_models": [str(model.get("model_key", "")) for model in eligible_models],
        "eligible_family_count": len(eligible_families),
        "eligible_families": eligible_families,
        "non_neighbor_families": non_neighbor_families,
        "min_expert_auc": min_expert_auc,
        "min_family_count": min_family_count,
        "max_error_jaccard_at_0_5": max_error_jaccard,
        "near_neighbor_families": sorted(set(near_neighbor_families)),
        "best_non_neighbor_pair": best_non_neighbor_pair,
        "claim_scope": (
            "diverse expert pool readiness gate only; requires aligned frozen score artifacts "
            "and does not claim raw single-sample SOTA or formal SPN/PRESENT evidence"
        ),
    }


def _validate_artifact_arrays(artifact: EnsembleScoreArtifact) -> None:
    lengths = {
        len(artifact.labels),
        len(artifact.probabilities),
        len(artifact.logits),
        len(artifact.sample_ids),
    }
    if len(lengths) != 1:
        raise ValueError("labels, probabilities, logits, and sample_ids must have equal length")


def _validate_artifact_alignment(artifacts: list[EnsembleScoreArtifact]) -> None:
    first = artifacts[0]
    for artifact in artifacts[1:]:
        if not np.array_equal(first.labels, artifact.labels):
            raise ValueError("score artifact labels differ")
        if not np.array_equal(first.sample_ids, artifact.sample_ids):
            raise ValueError("score artifact sample_ids differ")
        for field in (
            "cipher",
            "rounds",
            "validation_samples_per_class",
            "pairs_per_sample",
            "feature_encoding",
            "negative_mode",
            "sample_structure",
            "difference_profile",
            "difference_member",
            "validation_key",
        ):
            if first.metadata.get(field) != artifact.metadata.get(field):
                raise ValueError(f"score artifact protocol field differs: {field}")


def _single_report(artifact: EnsembleScoreArtifact) -> dict[str, Any]:
    return {
        "model_key": str(artifact.metadata.get("model_key", "")),
        "run_id": str(artifact.metadata.get("run_id", "")),
        "checkpoint_path": str(artifact.metadata.get("checkpoint_path", "")),
        "metrics": _metrics_from_probabilities(artifact.labels, artifact.probabilities),
        "metadata": artifact.metadata,
    }


def _ensemble_report(mode: str, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    return {
        "mode": mode,
        "metrics": _metrics_from_probabilities(labels, probabilities),
    }


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


def _auc_positive_weights(row_reports: list[dict[str, Any]]) -> np.ndarray:
    weights = np.array(
        [max(0.0, float(row["metrics"]["auc"]) - 0.5) for row in row_reports],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 0.0:
        return np.full((len(row_reports),), 1.0 / float(len(row_reports)), dtype=np.float64)
    return weights / total


def _rank_average_probabilities(probability_matrix: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(probability_matrix, axis=0), axis=0).astype(np.float64)
    denom = max(1.0, float(probability_matrix.shape[0] - 1))
    return ranks.mean(axis=1) / denom


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def _diversity_report(
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    logit_matrix: np.ndarray,
    row_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions = probability_matrix >= 0.5
    correct = predictions == labels.reshape(-1, 1)
    return {
        "oracle_accuracy_at_0_5": float(correct.any(axis=1).mean()) if len(labels) else 0.0,
        "all_models_wrong_rate_at_0_5": float((~correct.any(axis=1)).mean()) if len(labels) else 0.0,
        "pairwise": _pairwise_diversity(labels, probability_matrix, logit_matrix, row_reports),
    }


def _pairwise_diversity(
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    logit_matrix: np.ndarray,
    row_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
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
                    "left": str(row_reports[left]["model_key"]),
                    "right": str(row_reports[right]["model_key"]),
                    "probability_correlation": _safe_correlation(
                        probability_matrix[:, left],
                        probability_matrix[:, right],
                    ),
                    "logit_correlation": _safe_correlation(logit_matrix[:, left], logit_matrix[:, right]),
                    "disagreement_rate_at_0_5": float((predictions[:, left] != predictions[:, right]).mean()),
                    "double_fault_rate_at_0_5": float(both_wrong.mean()),
                    "error_jaccard_at_0_5": (
                        float(both_wrong.sum() / either_wrong.sum()) if int(either_wrong.sum()) else 0.0
                    ),
                }
            )
    return reports


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 0.0 or right_std <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _metric_auc(row: dict[str, Any]) -> float | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return _float_or_none(metrics.get("auc"))


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
