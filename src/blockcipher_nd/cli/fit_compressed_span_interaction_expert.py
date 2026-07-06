from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.cli.fit_compressed_feature_expert import (
    _fit_logistic,
    _load_feature_dir,
    _metrics,
    _predict_logits,
    _sigmoid,
    _validate_feature_dirs,
)
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, write_score_artifact


DEFAULT_MODEL_KEY = "compressed_span_interaction_logistic_expert"
DEFAULT_EXPERT_FAMILY = "compressed_spn_span_raw_interaction_summary"
DEFAULT_CANDIDATE_STATUS = "compressed_span_interaction_screen"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a raw compressed span-summary expert with train-selected "
            "primary/auxiliary cross-group interaction terms."
        )
    )
    parser.add_argument("--train-feature-dir", required=True, type=Path)
    parser.add_argument("--validation-feature-dir", required=True, type=Path)
    parser.add_argument("--output-validation-dir", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-train-dir", type=Path, default=None)
    parser.add_argument("--primary-prefix", default="primary_")
    parser.add_argument("--auxiliary-prefix", default="aux_")
    parser.add_argument("--top-primary", type=int, default=8)
    parser.add_argument("--top-auxiliary", type=int, default=8)
    parser.add_argument("--selection-holdout-fraction", type=float, default=0.0)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--include-raw-feature-prefix", action="append", default=[])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--shuffle-train-labels", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--expert-family", default=DEFAULT_EXPERT_FAMILY)
    parser.add_argument("--candidate-status", default=DEFAULT_CANDIDATE_STATUS)
    return parser.parse_args(argv)


def fit_compressed_span_interaction_expert(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    primary_prefix: str = "primary_",
    auxiliary_prefix: str = "aux_",
    top_primary: int = 8,
    top_auxiliary: int = 8,
    selection_holdout_fraction: float = 0.0,
    selection_seed: int = 0,
    include_raw_feature_prefixes: list[str] | None = None,
    steps: int = 2000,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    standardize: bool = True,
    shuffle_train_labels: bool = False,
    shuffle_seed: int = 0,
    run_id: str = "",
    model_key: str = DEFAULT_MODEL_KEY,
    expert_family: str = DEFAULT_EXPERT_FAMILY,
    candidate_status: str = DEFAULT_CANDIDATE_STATUS,
) -> tuple[EnsembleScoreArtifact, EnsembleScoreArtifact, dict[str, Any]]:
    if not primary_prefix:
        raise ValueError("primary_prefix must be non-empty")
    if not auxiliary_prefix:
        raise ValueError("auxiliary_prefix must be non-empty")
    if top_primary <= 0:
        raise ValueError("top_primary must be positive")
    if top_auxiliary <= 0:
        raise ValueError("top_auxiliary must be positive")
    if selection_holdout_fraction < 0.0 or selection_holdout_fraction >= 1.0:
        raise ValueError("selection_holdout_fraction must be in [0.0, 1.0)")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    raw_feature_prefixes = sorted(set(include_raw_feature_prefixes or []))

    train = _load_feature_dir(train_feature_dir)
    validation = _load_feature_dir(validation_feature_dir)
    _validate_feature_dirs(train, validation)
    names = _feature_names(train["metadata"], feature_count=int(train["features"].shape[1]))
    if raw_feature_prefixes:
        selected_raw_feature_indices = _select_raw_feature_indices(names, prefixes=raw_feature_prefixes)
        raw_feature_selection = "prefix_filter"
    else:
        selected_raw_feature_indices = list(range(int(train["features"].shape[1])))
        raw_feature_selection = "all"
    fit_labels = train["labels"].astype(np.float64, copy=True)
    if shuffle_train_labels:
        fit_labels = np.random.default_rng(shuffle_seed).permutation(fit_labels)
    split = _selection_fit_split(
        fit_labels,
        selection_holdout_fraction=selection_holdout_fraction,
        selection_seed=selection_seed,
    )

    selection = _interaction_selection(
        train_features=train["features"].astype(np.float64, copy=False)[split["selection_indices"]],
        fit_labels=fit_labels[split["selection_indices"]],
        feature_names=names,
        primary_prefix=primary_prefix,
        auxiliary_prefix=auxiliary_prefix,
        top_primary=top_primary,
        top_auxiliary=top_auxiliary,
    )
    train_augmented, validation_augmented = _augment_features(
        train["features"].astype(np.float64, copy=False),
        validation["features"].astype(np.float64, copy=False),
        selection=selection,
        raw_feature_indices=selected_raw_feature_indices,
    )
    fitted = _fit_logistic(
        train_augmented[split["fit_indices"]],
        fit_labels[split["fit_indices"]],
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
    )
    train_logits = _predict_logits(train_augmented, fitted)
    validation_logits = _predict_logits(validation_augmented, fitted)
    train_probabilities = _sigmoid(train_logits)
    validation_probabilities = _sigmoid(validation_logits)

    feature_model = (
        "selected_raw_plus_primary_auxiliary_interactions_logistic"
        if raw_feature_prefixes
        else "raw_plus_primary_auxiliary_interactions_logistic"
    )
    common_metadata = {
        "model_key": model_key,
        "expert_family": expert_family,
        "candidate_status": candidate_status,
        "run_id": run_id,
        "feature_fit_split": "train",
        "selection_fit_split_mode": split["mode"],
        "selection_holdout_fraction": float(selection_holdout_fraction),
        "selection_seed": int(selection_seed),
        "interaction_selection_rows": int(len(split["selection_indices"])),
        "fit_rows": int(len(split["fit_indices"])),
        "feature_train_dir": str(train_feature_dir),
        "feature_validation_dir": str(validation_feature_dir),
        "feature_model": feature_model,
        "feature_steps": int(steps),
        "feature_learning_rate": float(learning_rate),
        "feature_l2": float(l2),
        "feature_standardize": bool(standardize),
        "fit_train_labels_shuffled": bool(shuffle_train_labels),
        "fit_label_shuffle_seed": int(shuffle_seed),
        "feature_count": int(train_augmented.shape[1]),
        "raw_feature_count": int(train["features"].shape[1]),
        "raw_feature_selection": raw_feature_selection,
        "selected_raw_feature_count": int(len(selected_raw_feature_indices)),
        "selected_raw_feature_prefixes": raw_feature_prefixes,
        "selected_raw_feature_indices": [int(index) for index in selected_raw_feature_indices],
        "selected_raw_feature_names": [names[index] for index in selected_raw_feature_indices],
        "interaction_count": int(selection["interaction_count"]),
        "interaction_selection": _selection_metadata(selection, include_indices=False),
        "claim_scope": (
            "train-selected raw compressed SPN span-summary interaction expert diagnostic only; "
            "interaction selection and logistic fit use train labels only, validation is held out for final scoring, "
            "and this is not remote or formal SPN/PRESENT evidence"
        ),
    }
    train_artifact = EnsembleScoreArtifact(
        labels=train["labels"],
        probabilities=train_probabilities.astype(np.float32, copy=False),
        logits=train_logits.astype(np.float32, copy=False),
        sample_ids=train["sample_ids"],
        metadata={**train["metadata"], **common_metadata, "score_split": "train"},
    )
    validation_artifact = EnsembleScoreArtifact(
        labels=validation["labels"],
        probabilities=validation_probabilities.astype(np.float32, copy=False),
        logits=validation_logits.astype(np.float32, copy=False),
        sample_ids=validation["sample_ids"],
        metadata={**validation["metadata"], **common_metadata, "score_split": "validation"},
    )
    train_metrics = _metrics(train_artifact.labels, train_artifact.probabilities)
    validation_metrics = _metrics(validation_artifact.labels, validation_artifact.probabilities)
    report = {
        "status": "pass",
        "decision": (
            "compressed_span_interaction_shuffle_train_labels_control"
            if shuffle_train_labels
            else "compressed_span_interaction_local_screen_positive_needs_controls"
        ),
        "model_key": model_key,
        "expert_family": expert_family,
        "train_feature_dir": str(train_feature_dir),
        "validation_feature_dir": str(validation_feature_dir),
        "train_rows": int(len(train_artifact.labels)),
        "validation_rows": int(len(validation_artifact.labels)),
        "selection_fit_split_mode": split["mode"],
        "selection_holdout_fraction": float(selection_holdout_fraction),
        "selection_seed": int(selection_seed),
        "interaction_selection_rows": int(len(split["selection_indices"])),
        "fit_rows": int(len(split["fit_indices"])),
        "feature_model": feature_model,
        "feature_count": int(train_augmented.shape[1]),
        "raw_feature_count": int(train["features"].shape[1]),
        "raw_feature_selection": raw_feature_selection,
        "selected_raw_feature_count": int(len(selected_raw_feature_indices)),
        "selected_raw_feature_prefixes": raw_feature_prefixes,
        "selected_raw_feature_indices": [int(index) for index in selected_raw_feature_indices],
        "selected_raw_feature_names": [names[index] for index in selected_raw_feature_indices],
        "interaction_count": int(selection["interaction_count"]),
        "interaction_selection": _selection_metadata(selection, include_indices=True),
        "fit": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "standardize": bool(standardize),
            "weight_count": int(len(np.asarray(fitted["weights"]))),
            "weight_l2_norm": float(np.linalg.norm(np.asarray(fitted["weights"], dtype=np.float64))),
            "weight_abs_max": float(np.max(np.abs(np.asarray(fitted["weights"], dtype=np.float64)))),
            "bias": float(fitted["bias"]),
        },
        "label_control": {
            "shuffle_train_labels": bool(shuffle_train_labels),
            "shuffle_seed": int(shuffle_seed),
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "guardrails": [
            "interaction_selection_split_must_be_train",
            "fit_split_must_be_train",
            "validation_split_final_score_only",
            "strict_negative_mode_required",
            "compare_against_full_summary_flat_logistic",
            "compare_against_shuffle_train_labels_control",
        ],
        "claim_scope": common_metadata["claim_scope"],
    }
    return train_artifact, validation_artifact, report


def _feature_names(metadata: dict[str, Any], *, feature_count: int) -> list[str]:
    view_metadata = metadata.get("feature_view_metadata", metadata)
    names = list(view_metadata.get("feature_names", []))
    if len(names) != feature_count:
        raise ValueError(f"expected {feature_count} feature_names, got {len(names)}")
    return [str(name) for name in names]


def _select_raw_feature_indices(names: list[str], *, prefixes: list[str]) -> list[int]:
    indices = [
        index
        for index, name in enumerate(names)
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    if not indices:
        raise ValueError("no raw features matched --include-raw-feature-prefix")
    return indices


def _interaction_selection(
    *,
    train_features: np.ndarray,
    fit_labels: np.ndarray,
    feature_names: list[str],
    primary_prefix: str,
    auxiliary_prefix: str,
    top_primary: int,
    top_auxiliary: int,
) -> dict[str, Any]:
    primary_indices = [index for index, name in enumerate(feature_names) if name.startswith(primary_prefix)]
    auxiliary_indices = [index for index, name in enumerate(feature_names) if name.startswith(auxiliary_prefix)]
    if len(primary_indices) < top_primary:
        raise ValueError(f"requested top_primary={top_primary}, but only {len(primary_indices)} primary features exist")
    if len(auxiliary_indices) < top_auxiliary:
        raise ValueError(
            f"requested top_auxiliary={top_auxiliary}, but only {len(auxiliary_indices)} auxiliary features exist"
        )
    selected_primary = _top_mean_gap_indices(train_features, fit_labels, primary_indices, top_primary)
    selected_auxiliary = _top_mean_gap_indices(train_features, fit_labels, auxiliary_indices, top_auxiliary)
    interaction_pairs = [
        (primary_index, auxiliary_index)
        for primary_index in selected_primary
        for auxiliary_index in selected_auxiliary
    ]
    interaction_names = [
        f"{feature_names[primary_index]}*{feature_names[auxiliary_index]}"
        for primary_index, auxiliary_index in interaction_pairs
    ]
    return {
        "primary_prefix": primary_prefix,
        "auxiliary_prefix": auxiliary_prefix,
        "top_primary": int(top_primary),
        "top_auxiliary": int(top_auxiliary),
        "primary_indices": [int(index) for index in selected_primary],
        "auxiliary_indices": [int(index) for index in selected_auxiliary],
        "primary_feature_names": [feature_names[index] for index in selected_primary],
        "auxiliary_feature_names": [feature_names[index] for index in selected_auxiliary],
        "interaction_pairs": [(int(left), int(right)) for left, right in interaction_pairs],
        "interaction_feature_names": interaction_names,
        "interaction_count": int(len(interaction_pairs)),
    }


def _selection_fit_split(
    labels: np.ndarray,
    *,
    selection_holdout_fraction: float,
    selection_seed: int,
) -> dict[str, Any]:
    row_count = int(len(labels))
    if selection_holdout_fraction == 0.0:
        all_indices = np.arange(row_count, dtype=np.int64)
        return {
            "mode": "same_train_rows",
            "selection_indices": all_indices,
            "fit_indices": all_indices,
        }

    rng = np.random.default_rng(selection_seed)
    selection_parts: list[np.ndarray] = []
    fit_parts: list[np.ndarray] = []
    for label in sorted(float(value) for value in np.unique(labels)):
        label_indices = np.flatnonzero(labels == label)
        if len(label_indices) < 2:
            raise ValueError("selection holdout requires at least two train rows for each label")
        shuffled = rng.permutation(label_indices)
        select_count = int(round(len(shuffled) * selection_holdout_fraction))
        select_count = max(1, min(len(shuffled) - 1, select_count))
        selection_parts.append(shuffled[:select_count])
        fit_parts.append(shuffled[select_count:])

    selection_indices = rng.permutation(np.concatenate(selection_parts)).astype(np.int64, copy=False)
    fit_indices = rng.permutation(np.concatenate(fit_parts)).astype(np.int64, copy=False)
    return {
        "mode": "train_internal_holdout",
        "selection_indices": selection_indices,
        "fit_indices": fit_indices,
    }


def _top_mean_gap_indices(
    features: np.ndarray,
    labels: np.ndarray,
    candidate_indices: list[int],
    top_k: int,
) -> list[int]:
    positive = features[labels == 1.0][:, candidate_indices]
    negative = features[labels == 0.0][:, candidate_indices]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("both positive and negative train labels are required")
    scores = np.abs(positive.mean(axis=0) - negative.mean(axis=0))
    order = np.argsort(scores)[::-1][:top_k]
    return [candidate_indices[int(position)] for position in order]


def _augment_features(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    *,
    selection: dict[str, Any],
    raw_feature_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    train_standardized = (train_features - mean) / scale
    validation_standardized = (validation_features - mean) / scale
    interaction_pairs = list(selection["interaction_pairs"])
    train_interactions = np.stack(
        [train_standardized[:, left] * train_standardized[:, right] for left, right in interaction_pairs],
        axis=1,
    )
    validation_interactions = np.stack(
        [validation_standardized[:, left] * validation_standardized[:, right] for left, right in interaction_pairs],
        axis=1,
    )
    train_raw = train_features[:, raw_feature_indices]
    validation_raw = validation_features[:, raw_feature_indices]
    return (
        np.concatenate([train_raw, train_interactions], axis=1),
        np.concatenate([validation_raw, validation_interactions], axis=1),
    )


def _selection_metadata(selection: dict[str, Any], *, include_indices: bool) -> dict[str, Any]:
    metadata = {
        "primary_prefix": selection["primary_prefix"],
        "auxiliary_prefix": selection["auxiliary_prefix"],
        "top_primary": int(selection["top_primary"]),
        "top_auxiliary": int(selection["top_auxiliary"]),
        "primary_feature_names": list(selection["primary_feature_names"]),
        "auxiliary_feature_names": list(selection["auxiliary_feature_names"]),
        "interaction_feature_names": list(selection["interaction_feature_names"]),
    }
    if include_indices:
        metadata["primary_indices"] = list(selection["primary_indices"])
        metadata["auxiliary_indices"] = list(selection["auxiliary_indices"])
        metadata["interaction_pairs"] = [list(pair) for pair in selection["interaction_pairs"]]
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_artifact, validation_artifact, report = fit_compressed_span_interaction_expert(
        train_feature_dir=args.train_feature_dir,
        validation_feature_dir=args.validation_feature_dir,
        primary_prefix=args.primary_prefix,
        auxiliary_prefix=args.auxiliary_prefix,
        top_primary=args.top_primary,
        top_auxiliary=args.top_auxiliary,
        selection_holdout_fraction=args.selection_holdout_fraction,
        selection_seed=args.selection_seed,
        include_raw_feature_prefixes=args.include_raw_feature_prefix,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        standardize=not args.no_standardize,
        shuffle_train_labels=args.shuffle_train_labels,
        shuffle_seed=args.shuffle_seed,
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
