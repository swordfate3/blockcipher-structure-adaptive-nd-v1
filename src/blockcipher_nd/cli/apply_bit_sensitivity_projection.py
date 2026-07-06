from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    load_score_artifact,
    write_score_artifact,
)
from blockcipher_nd.training.metrics import binary_auc


DEFAULT_MODEL_KEY = "present_r8_bit_sensitivity_projection_expert"
DEFAULT_EXPERT_FAMILY = "bit_sensitivity_projection"
DEFAULT_CANDIDATE_STATUS = "projection_screen"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a frozen bit-sensitivity projection mask and write a score artifact."
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--reference-artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--expert-family", default=DEFAULT_EXPERT_FAMILY)
    parser.add_argument("--candidate-status", default=DEFAULT_CANDIDATE_STATUS)
    return parser.parse_args(argv)


def apply_bit_sensitivity_projection(
    *,
    features_path: Path,
    mask_path: Path,
    reference_artifact_dir: Path,
    run_id: str = "",
    model_key: str = DEFAULT_MODEL_KEY,
    expert_family: str = DEFAULT_EXPERT_FAMILY,
    candidate_status: str = DEFAULT_CANDIDATE_STATUS,
) -> tuple[EnsembleScoreArtifact, dict[str, Any]]:
    reference = load_score_artifact(reference_artifact_dir)
    features = _load_feature_matrix(features_path)
    if features.shape[0] != len(reference.labels):
        raise ValueError(
            "feature row count must match reference artifact rows: "
            f"{features.shape[0]} != {len(reference.labels)}"
        )
    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    if mask.get("selection_split") != "train":
        raise ValueError("bit-sensitivity projection mask must be selected on the train split")
    projection_unit = str(mask.get("projection_unit") or "axis")
    projection_parameters = _projection_parameters(mask)
    logits = _projection_logits(features, projection_parameters)
    probabilities = _sigmoid(logits)
    probability_array = probabilities.astype(np.float32, copy=False)
    logit_array = logits.astype(np.float32, copy=False)
    selected_axes = sorted({axis for params in projection_parameters for axis in params["axes"]})
    metadata = {
        **reference.metadata,
        "model_key": model_key,
        "expert_family": expert_family,
        "candidate_status": candidate_status,
        "run_id": run_id,
        "projection_mask": str(mask_path),
        "projection_features": str(features_path),
        "projection_unit": projection_unit,
        "projection_axis_count": len(selected_axes),
        "projection_group_count": len(projection_parameters) if projection_unit != "axis" else 0,
        "projection_source": "bit_sensitivity_train_only_mask",
        "claim_scope": (
            "frozen bit-sensitivity projection score artifact only; not a trained neural model, "
            "not a remote-launch gate, and not formal SPN/PRESENT evidence"
        ),
    }
    artifact = EnsembleScoreArtifact(
        labels=reference.labels,
        probabilities=probability_array,
        logits=logit_array,
        sample_ids=reference.sample_ids,
        metadata=metadata,
    )
    report = {
        "status": "pass",
        "decision": "projection_score_artifact_ready_for_local_gate",
        "rows": int(len(reference.labels)),
        "feature_shape": [int(value) for value in features.shape],
        "axis_count": len(selected_axes),
        "group_count": len(projection_parameters) if projection_unit != "axis" else 0,
        "mask": str(mask_path),
        "reference_artifact": str(reference_artifact_dir),
        "metrics": {
            "auc": binary_auc(reference.labels.astype(np.float32, copy=False), probability_array),
        },
        "guardrails": [
            "mask_must_be_train_selected",
            "compare_against_same_input_global_control",
            "require_low_error_overlap_before_diverse_pool_promotion",
            "require_active_nibble_and_input_difference_mismatch_controls",
        ],
        "claim_scope": metadata["claim_scope"],
    }
    return artifact, report


def _load_feature_matrix(path: Path) -> np.ndarray:
    features = np.load(path)
    if features.ndim < 2:
        raise ValueError("features must have at least two dimensions: rows x axes")
    return features.reshape(features.shape[0], -1).astype(np.float32, copy=False)


def _projection_parameters(mask: dict[str, Any]) -> list[dict[str, Any]]:
    selected_groups = mask.get("selected_groups")
    if isinstance(selected_groups, list) and selected_groups:
        return _group_parameters(selected_groups)
    return _axis_parameters(mask)


def _axis_parameters(mask: dict[str, Any]) -> list[dict[str, Any]]:
    selected_axes = [int(axis) for axis in mask.get("selected_axes", [])]
    if not selected_axes:
        raise ValueError("projection mask must contain selected_axes")
    score_rows = {
        int(row["axis"]): row
        for row in mask.get("axis_scores", [])
        if isinstance(row, dict) and "axis" in row
    }
    parameters: list[dict[str, Any]] = []
    for axis in selected_axes:
        row = score_rows.get(axis, {})
        positive_mean = _float(row.get("positive_mean"), 1.0)
        negative_mean = _float(row.get("negative_mean"), 0.0)
        delta = positive_mean - negative_mean
        direction = 1.0 if delta >= 0.0 else -1.0
        center = (positive_mean + negative_mean) / 2.0
        scale = max(abs(delta), 1e-6)
        weight = max(_float(row.get("score"), 1.0), 1e-6)
        parameters.append(
            {
                "axes": [axis],
                "direction": direction,
                "center": center,
                "scale": scale,
                "weight": weight,
            }
        )
    return parameters


def _group_parameters(groups: list[Any]) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        axes = [int(axis) for axis in group.get("axes", [])]
        if not axes:
            continue
        positive_mean = _float(group.get("positive_mean"), 1.0)
        negative_mean = _float(group.get("negative_mean"), 0.0)
        delta = positive_mean - negative_mean
        direction = 1.0 if delta >= 0.0 else -1.0
        center = (positive_mean + negative_mean) / 2.0
        scale = max(abs(delta), 1e-6)
        weight = max(_float(group.get("score"), 1.0), 1e-6)
        parameters.append(
            {
                "axes": axes,
                "direction": direction,
                "center": center,
                "scale": scale,
                "weight": weight,
            }
        )
    if not parameters:
        raise ValueError("projection mask selected_groups must contain axes")
    return parameters


def _projection_logits(features: np.ndarray, projection_parameters: list[dict[str, Any]]) -> np.ndarray:
    logits = np.zeros((features.shape[0],), dtype=np.float64)
    total_weight = 0.0
    for params in projection_parameters:
        axes = [int(axis) for axis in params["axes"]]
        for axis in axes:
            if axis < 0 or axis >= features.shape[1]:
                raise ValueError(f"projection axis out of range: {axis}")
        values = features[:, axes].mean(axis=1).astype(np.float64, copy=False)
        weight = float(params["weight"])
        total_weight += weight
        logits += (
            weight
            * float(params["direction"])
            * (values - float(params["center"]))
            / float(params["scale"])
        )
    return logits / max(total_weight, 1e-6)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact, report = apply_bit_sensitivity_projection(
        features_path=args.features,
        mask_path=args.mask,
        reference_artifact_dir=args.reference_artifact,
        run_id=args.run_id,
        model_key=args.model_key,
        expert_family=args.expert_family,
        candidate_status=args.candidate_status,
    )
    write_score_artifact(args.output_dir, artifact)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
