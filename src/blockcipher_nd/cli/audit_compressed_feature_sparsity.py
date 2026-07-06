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
    _select_feature_columns,
    _sigmoid,
    _validate_feature_dirs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank compressed SPN structural features using train split only, then fit "
            "top-k sparse logistic experts and score held-out validation features."
        )
    )
    parser.add_argument("--train-feature-dir", required=True, type=Path)
    parser.add_argument("--validation-feature-dir", required=True, type=Path)
    parser.add_argument("--top-k", nargs="+", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--min-positive-auc", type=float, default=0.99)
    parser.add_argument(
        "--include-feature-family",
        action="append",
        default=[],
        help="Restrict ranking/fitting to decoded compressed-feature families; repeat for multiple families.",
    )
    parser.add_argument(
        "--include-feature-prefix",
        action="append",
        default=[],
        help="Restrict ranking/fitting to metadata feature_names prefixes; repeat for multiple prefixes.",
    )
    return parser.parse_args(argv)


def audit_compressed_feature_sparsity(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    top_k_values: list[int],
    steps: int = 800,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    standardize: bool = True,
    min_positive_auc: float = 0.99,
    include_feature_families: list[str] | None = None,
    include_feature_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    if not top_k_values:
        raise ValueError("at least one top-k value is required")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("top-k values must be positive")
    if min_positive_auc < 0.0 or min_positive_auc > 1.0:
        raise ValueError("min_positive_auc must be in [0, 1]")

    train = _load_feature_dir(train_feature_dir)
    validation = _load_feature_dir(validation_feature_dir)
    _validate_feature_dirs(train, validation)
    feature_selection = _select_feature_columns(
        train["metadata"],
        feature_count=int(train["features"].shape[1]),
        include_feature_families=include_feature_families or [],
        include_feature_prefixes=include_feature_prefixes or [],
    )
    selected_original_indices = np.asarray(feature_selection["selected_feature_indices"], dtype=np.int64)
    selected_feature_names = _feature_names_for_indices(
        train["metadata"],
        feature_count=int(train["features"].shape[1]),
        indices=selected_original_indices,
    )
    train_features = train["features"][:, selected_original_indices].astype(np.float64, copy=False)
    validation_features = validation["features"][:, selected_original_indices].astype(np.float64, copy=False)
    labels = train["labels"].astype(np.float64, copy=False)
    ranking = _rank_features_by_train_separation(train_features, labels)
    rows = [
        _sparse_row(
            train_features=train_features,
            validation_features=validation_features,
            train_labels=train["labels"],
            validation_labels=validation["labels"],
            ranking=ranking,
            selected_original_indices=selected_original_indices,
            selected_feature_names=selected_feature_names,
            top_k=top_k,
            steps=steps,
            learning_rate=learning_rate,
            l2=l2,
            standardize=standardize,
        )
        for top_k in sorted(set(top_k_values))
    ]
    best_row = max(rows, key=lambda row: (row["validation_metrics"]["auc"], -row["top_k"]))
    decision = (
        "sparse_compressed_feature_local_screen_positive"
        if float(best_row["validation_metrics"]["auc"]) >= min_positive_auc
        else "sparse_compressed_feature_hold"
    )
    return {
        "status": "pass",
        "decision": decision,
        "train_feature_dir": str(train_feature_dir),
        "validation_feature_dir": str(validation_feature_dir),
        "feature_count": int(train_features.shape[1]),
        "feature_original_count": int(train["features"].shape[1]),
        "feature_selection": _feature_selection_metadata(feature_selection),
        "top_k_values": [int(value) for value in sorted(set(top_k_values))],
        "ranking_method": "abs_train_class_mean_difference_over_train_std",
        "fit": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "standardize": bool(standardize),
        },
        "thresholds": {
            "min_positive_auc": float(min_positive_auc),
        },
        "best_row": best_row,
        "rows": rows,
        "claim_scope": (
            "train-ranked compressed SPN feature sparsity diagnostic only; feature "
            "selection uses train labels only, scoring uses held-out validation labels, "
            "and this is not remote or formal SPN/PRESENT evidence"
        ),
    }


def _rank_features_by_train_separation(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positives = features[labels == 1.0]
    negatives = features[labels == 0.0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("train labels must contain both classes")
    separation = np.abs(positives.mean(axis=0) - negatives.mean(axis=0))
    scale = features.std(axis=0)
    score = separation / np.where(scale > 1e-8, scale, 1.0)
    return np.argsort(-score, kind="stable")


def _sparse_row(
    *,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    ranking: np.ndarray,
    selected_original_indices: np.ndarray,
    selected_feature_names: list[str],
    top_k: int,
    steps: int,
    learning_rate: float,
    l2: float,
    standardize: bool,
) -> dict[str, Any]:
    feature_count = train_features.shape[1]
    selected = ranking[: min(top_k, feature_count)].astype(np.int64)
    fitted = _fit_logistic(
        train_features[:, selected],
        train_labels.astype(np.float64, copy=False),
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
    )
    train_probabilities = _sigmoid(_predict_logits(train_features[:, selected], fitted))
    validation_probabilities = _sigmoid(_predict_logits(validation_features[:, selected], fitted))
    selected_original = selected_original_indices[selected]
    selected_names = [selected_feature_names[int(index)] for index in selected] if selected_feature_names else []
    return {
        "top_k": int(top_k),
        "effective_top_k": int(len(selected)),
        "selected_feature_indices": [int(index) for index in selected_original],
        "selected_feature_indices_within_selection": [int(index) for index in selected],
        "selected_feature_names": selected_names,
        "train_metrics": _metrics(train_labels, train_probabilities),
        "validation_metrics": _metrics(validation_labels, validation_probabilities),
    }


def _feature_selection_metadata(feature_selection: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(feature_selection)
    if metadata.get("mode") == "all":
        metadata.pop("selected_feature_indices", None)
    return metadata


def _feature_names_for_indices(
    metadata: dict[str, Any],
    *,
    feature_count: int,
    indices: np.ndarray,
) -> list[str]:
    view_metadata = metadata.get("feature_view_metadata", metadata)
    names = list(view_metadata.get("feature_names", []))
    if len(names) != feature_count:
        return []
    return [str(names[int(index)]) for index in indices]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_compressed_feature_sparsity(
        train_feature_dir=args.train_feature_dir,
        validation_feature_dir=args.validation_feature_dir,
        top_k_values=args.top_k,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        standardize=not args.no_standardize,
        min_positive_auc=args.min_positive_auc,
        include_feature_families=args.include_feature_family,
        include_feature_prefixes=args.include_feature_prefix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
