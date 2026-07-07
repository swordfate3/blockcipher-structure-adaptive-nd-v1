from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.cli.fit_compressed_feature_expert import _metrics, _sigmoid
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, load_score_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a corrected score artifact on validation rows selected by a "
            "train-derived frozen-base residual-loss threshold."
        )
    )
    parser.add_argument("--train-base-artifacts", nargs=2, required=True, type=Path)
    parser.add_argument("--validation-base-artifacts", nargs=2, required=True, type=Path)
    parser.add_argument("--validation-corrected-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--focus-fraction", type=float, default=0.1)
    return parser.parse_args(argv)


def evaluate_residual_slice_correction(
    *,
    train_base_artifacts: list[EnsembleScoreArtifact],
    validation_base_artifacts: list[EnsembleScoreArtifact],
    validation_corrected_artifact: EnsembleScoreArtifact,
    focus_fraction: float = 0.1,
) -> dict[str, Any]:
    if len(train_base_artifacts) != 2 or len(validation_base_artifacts) != 2:
        raise ValueError("residual slice evaluation requires exactly two base artifacts per split")
    if focus_fraction <= 0.0 or focus_fraction >= 1.0:
        raise ValueError("focus_fraction must be in (0, 1)")
    _validate_artifact_alignment(train_base_artifacts, split="train base")
    _validate_artifact_alignment(validation_base_artifacts, split="validation base")
    _validate_corrected_alignment(validation_base_artifacts[0], validation_corrected_artifact)

    train_base_probabilities = _sigmoid(_base_logit_mean(train_base_artifacts))
    validation_base_probabilities = _sigmoid(_base_logit_mean(validation_base_artifacts))
    validation_corrected_probabilities = validation_corrected_artifact.probabilities.astype(
        np.float64,
        copy=False,
    )
    train_labels = train_base_artifacts[0].labels.astype(np.float64, copy=False)
    validation_labels = validation_base_artifacts[0].labels.astype(np.float64, copy=False)
    train_residual_loss = np.abs(train_labels - train_base_probabilities)
    validation_residual_loss = np.abs(validation_labels - validation_base_probabilities)
    threshold = _train_focus_threshold(train_residual_loss, focus_fraction)
    focus_mask = validation_residual_loss >= threshold
    if int(focus_mask.sum()) < 2 or len(np.unique(validation_labels[focus_mask])) < 2:
        focus_mask = _top_validation_residual_mask(validation_residual_loss, validation_labels, focus_fraction)
        focus_mode = "validation_top_fallback_due_to_single_class_or_too_few_rows"
    else:
        focus_mode = "train_derived_base_residual_loss_threshold"

    global_base_metrics = _metrics(validation_labels, validation_base_probabilities)
    global_corrected_metrics = _metrics(validation_labels, validation_corrected_probabilities)
    focus_base_metrics = _slice_metrics(validation_labels, validation_base_probabilities, focus_mask)
    focus_corrected_metrics = _slice_metrics(validation_labels, validation_corrected_probabilities, focus_mask)
    base_loss = np.abs(validation_labels - validation_base_probabilities)
    corrected_loss = np.abs(validation_labels - validation_corrected_probabilities)
    return {
        "status": "pass",
        "decision": (
            "residual_slice_correction_improves_focus_loss"
            if focus_corrected_metrics["residual_loss_mean"] < focus_base_metrics["residual_loss_mean"]
            else "residual_slice_correction_diagnostic_no_focus_loss_gain"
        ),
        "focus": {
            "mode": focus_mode,
            "focus_fraction": float(focus_fraction),
            "threshold": float(threshold),
            "train_rows": int(len(train_labels)),
            "validation_rows": int(len(validation_labels)),
            "validation_focus_rows": int(focus_mask.sum()),
            "validation_focus_fraction": float(focus_mask.mean()) if len(focus_mask) else 0.0,
            "train_base_residual_loss_mean": float(train_residual_loss.mean()),
            "validation_base_residual_loss_mean": float(validation_residual_loss.mean()),
        },
        "base_model_order": [str(item.metadata.get("model_key", "")) for item in validation_base_artifacts],
        "base_run_order": [str(item.metadata.get("run_id", "")) for item in validation_base_artifacts],
        "corrected_model_key": str(validation_corrected_artifact.metadata.get("model_key", "")),
        "corrected_run_id": str(validation_corrected_artifact.metadata.get("run_id", "")),
        "validation_global_base_metrics": {
            **global_base_metrics,
            "residual_loss_mean": float(base_loss.mean()),
        },
        "validation_global_corrected_metrics": {
            **global_corrected_metrics,
            "residual_loss_mean": float(corrected_loss.mean()),
        },
        "validation_global_delta": _delta_metrics(
            {**global_base_metrics, "residual_loss_mean": float(base_loss.mean())},
            {**global_corrected_metrics, "residual_loss_mean": float(corrected_loss.mean())},
        ),
        "validation_focus_metrics": {
            "rows": int(focus_mask.sum()),
            "row_fraction": float(focus_mask.mean()) if len(focus_mask) else 0.0,
        },
        "validation_focus_base_metrics": focus_base_metrics,
        "validation_focus_corrected_metrics": focus_corrected_metrics,
        "validation_focus_delta": _delta_metrics(focus_base_metrics, focus_corrected_metrics),
        "claim_scope": (
            "local score-artifact diagnostic only; validation is sliced by train-derived "
            "base residual threshold and remains held out; not remote evidence and not "
            "formal SPN/PRESENT evidence"
        ),
    }


def _base_logit_mean(artifacts: list[EnsembleScoreArtifact]) -> np.ndarray:
    return np.stack([artifact.logits for artifact in artifacts], axis=1).mean(axis=1).astype(np.float64)


def _train_focus_threshold(residual_loss: np.ndarray, focus_fraction: float) -> float:
    focus_count = max(1, int(np.ceil(len(residual_loss) * focus_fraction)))
    order = np.argsort(residual_loss)
    return float(residual_loss[order[-focus_count]])


def _top_validation_residual_mask(
    residual_loss: np.ndarray,
    labels: np.ndarray,
    focus_fraction: float,
) -> np.ndarray:
    focus_count = max(2, int(np.ceil(len(residual_loss) * focus_fraction)))
    focus_count = min(len(residual_loss), focus_count)
    order = np.argsort(residual_loss)
    for count in range(focus_count, len(residual_loss) + 1):
        mask = np.zeros(len(residual_loss), dtype=bool)
        mask[order[-count:]] = True
        if len(np.unique(labels[mask])) >= 2:
            return mask
    mask = np.zeros(len(residual_loss), dtype=bool)
    mask[order[-focus_count:]] = True
    return mask


def _slice_metrics(labels: np.ndarray, probabilities: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected_labels = labels[mask]
    selected_probabilities = probabilities[mask]
    metrics = _metrics(selected_labels, selected_probabilities)
    metrics["rows"] = int(mask.sum())
    metrics["residual_loss_mean"] = (
        float(np.abs(selected_labels - selected_probabilities).mean()) if len(selected_labels) else 0.0
    )
    return metrics


def _delta_metrics(base: dict[str, float], corrected: dict[str, float]) -> dict[str, float]:
    keys = ["auc", "accuracy", "calibrated_accuracy", "residual_loss_mean"]
    return {
        key: float(corrected.get(key, 0.0) - base.get(key, 0.0))
        for key in keys
        if key in base and key in corrected
    }


def _validate_artifact_alignment(artifacts: list[EnsembleScoreArtifact], *, split: str) -> None:
    first = artifacts[0]
    for index, artifact in enumerate(artifacts[1:], start=1):
        if not np.array_equal(first.labels, artifact.labels):
            raise ValueError(f"{split} artifact {index} labels differ")
        if not np.array_equal(first.sample_ids, artifact.sample_ids):
            raise ValueError(f"{split} artifact {index} sample_ids differ")


def _validate_corrected_alignment(
    base_artifact: EnsembleScoreArtifact,
    corrected_artifact: EnsembleScoreArtifact,
) -> None:
    if not np.array_equal(base_artifact.labels, corrected_artifact.labels):
        raise ValueError("validation corrected artifact labels differ from base")
    if not np.array_equal(base_artifact.sample_ids, corrected_artifact.sample_ids):
        raise ValueError("validation corrected artifact sample_ids differ from base")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_residual_slice_correction(
        train_base_artifacts=[load_score_artifact(path) for path in args.train_base_artifacts],
        validation_base_artifacts=[load_score_artifact(path) for path in args.validation_base_artifacts],
        validation_corrected_artifact=load_score_artifact(args.validation_corrected_artifact),
        focus_fraction=args.focus_fraction,
    )
    report["train_base_artifact_dirs"] = [str(path) for path in args.train_base_artifacts]
    report["validation_base_artifact_dirs"] = [str(path) for path in args.validation_base_artifacts]
    report["validation_corrected_artifact_dir"] = str(args.validation_corrected_artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
