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


DEFAULT_MODEL_KEY = "compressed_span_block_interaction_logistic_expert"
DEFAULT_EXPERT_FAMILY = "compressed_spn_span_block_interaction_summary"
DEFAULT_CANDIDATE_STATUS = "compressed_span_block_interaction_screen"
INTERACTION_STAT_NAMES = ["mean_product", "abs_mean_product", "rms_product", "maxabs_product"]

SEMANTIC_GROUP_PREFIXES = [
    ("primary_depth", ["primary_depth_mean_"]),
    ("primary_trailword", ["primary_trailword_mean_"]),
    ("primary_cell", ["primary_cell_mean_"]),
    ("primary_depth_cell", ["primary_depth_cell_"]),
    ("primary_depth_trailword", ["primary_depth_trailword_"]),
    ("primary_global", ["primary_global_"]),
    ("aux_depth_cell", ["aux_depth_cell_"]),
    ("aux_word", ["aux_word_mean_"]),
    ("aux_depth_word", ["aux_depth_word_"]),
    ("aux_cell", ["aux_cell_mean_"]),
    ("aux_word_global", ["aux_word_global_"]),
    ("aux_cell_global", ["aux_cell_global_"]),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a raw compressed span-summary expert with semantic block interaction summaries."
    )
    parser.add_argument("--train-feature-dir", required=True, type=Path)
    parser.add_argument("--validation-feature-dir", required=True, type=Path)
    parser.add_argument("--output-validation-dir", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-train-dir", type=Path, default=None)
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


def fit_compressed_span_block_interaction_expert(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
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
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")

    train = _load_feature_dir(train_feature_dir)
    validation = _load_feature_dir(validation_feature_dir)
    _validate_feature_dirs(train, validation)
    names = _feature_names(train["metadata"], feature_count=int(train["features"].shape[1]))
    groups = _semantic_groups(names)
    primary_groups = [group for group in groups if group["kind"] == "primary"]
    auxiliary_groups = [group for group in groups if group["kind"] == "auxiliary"]
    if not primary_groups:
        raise ValueError("no primary semantic span groups found")
    if not auxiliary_groups:
        raise ValueError("no auxiliary semantic span groups found")

    train_augmented, validation_augmented, interaction_metadata = _augment_with_block_interactions(
        train["features"].astype(np.float64, copy=False),
        validation["features"].astype(np.float64, copy=False),
        primary_groups=primary_groups,
        auxiliary_groups=auxiliary_groups,
    )
    fit_labels = train["labels"].astype(np.float64, copy=True)
    if shuffle_train_labels:
        fit_labels = np.random.default_rng(shuffle_seed).permutation(fit_labels)
    fitted = _fit_logistic(
        train_augmented,
        fit_labels,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
    )
    train_logits = _predict_logits(train_augmented, fitted)
    validation_logits = _predict_logits(validation_augmented, fitted)
    train_probabilities = _sigmoid(train_logits)
    validation_probabilities = _sigmoid(validation_logits)

    feature_model = "raw_plus_semantic_block_interactions_logistic"
    common_metadata = {
        "model_key": model_key,
        "expert_family": expert_family,
        "candidate_status": candidate_status,
        "run_id": run_id,
        "feature_fit_split": "train",
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
        "primary_group_count": int(len(primary_groups)),
        "auxiliary_group_count": int(len(auxiliary_groups)),
        "block_pair_count": int(interaction_metadata["block_pair_count"]),
        "block_interaction_stat_count": int(len(INTERACTION_STAT_NAMES)),
        "block_interaction_feature_count": int(interaction_metadata["feature_count"]),
        "block_groups": _group_metadata(groups),
        "claim_scope": (
            "train-fitted raw compressed SPN span-summary block interaction expert diagnostic only; "
            "semantic block interactions are label-independent, validation is held out for final scoring, "
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
            "compressed_span_block_interaction_shuffle_train_labels_control"
            if shuffle_train_labels
            else "compressed_span_block_interaction_local_screen_positive_needs_controls"
        ),
        "model_key": model_key,
        "expert_family": expert_family,
        "train_feature_dir": str(train_feature_dir),
        "validation_feature_dir": str(validation_feature_dir),
        "train_rows": int(len(train_artifact.labels)),
        "validation_rows": int(len(validation_artifact.labels)),
        "feature_model": feature_model,
        "feature_count": int(train_augmented.shape[1]),
        "raw_feature_count": int(train["features"].shape[1]),
        "primary_group_count": int(len(primary_groups)),
        "auxiliary_group_count": int(len(auxiliary_groups)),
        "block_pair_count": int(interaction_metadata["block_pair_count"]),
        "block_interaction_stat_count": int(len(INTERACTION_STAT_NAMES)),
        "block_interaction_feature_count": int(interaction_metadata["feature_count"]),
        "block_interaction_feature_names": interaction_metadata["feature_names"],
        "block_groups": _group_metadata(groups),
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
            "block_interactions_must_be_label_independent",
            "fit_split_must_be_train",
            "validation_split_final_score_only",
            "strict_negative_mode_required",
            "compare_against_full_summary_flat_logistic",
            "compare_against_flat_train_selected_interaction_expert",
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


def _semantic_groups(feature_names: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for group_name, prefixes in SEMANTIC_GROUP_PREFIXES:
        indices = [
            index
            for index, feature_name in enumerate(feature_names)
            if any(feature_name.startswith(prefix) for prefix in prefixes)
        ]
        if not indices:
            continue
        kind = "primary" if group_name.startswith("primary_") else "auxiliary"
        groups.append(
            {
                "name": group_name,
                "kind": kind,
                "prefixes": prefixes,
                "indices": indices,
            }
        )
    return groups


def _augment_with_block_interactions(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    *,
    primary_groups: list[dict[str, Any]],
    auxiliary_groups: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    train_standardized = (train_features - mean) / scale
    validation_standardized = (validation_features - mean) / scale

    train_interactions: list[np.ndarray] = []
    validation_interactions: list[np.ndarray] = []
    feature_names: list[str] = []
    for primary_group in primary_groups:
        for auxiliary_group in auxiliary_groups:
            train_block = _block_interaction_stats(
                train_standardized[:, primary_group["indices"]],
                train_standardized[:, auxiliary_group["indices"]],
            )
            validation_block = _block_interaction_stats(
                validation_standardized[:, primary_group["indices"]],
                validation_standardized[:, auxiliary_group["indices"]],
            )
            train_interactions.append(train_block)
            validation_interactions.append(validation_block)
            feature_names.extend(
                f"block_{primary_group['name']}__{auxiliary_group['name']}__{stat_name}"
                for stat_name in INTERACTION_STAT_NAMES
            )

    train_block_features = np.concatenate(train_interactions, axis=1)
    validation_block_features = np.concatenate(validation_interactions, axis=1)
    return (
        np.concatenate([train_features, train_block_features], axis=1),
        np.concatenate([validation_features, validation_block_features], axis=1),
        {
            "block_pair_count": int(len(primary_groups) * len(auxiliary_groups)),
            "feature_count": int(train_block_features.shape[1]),
            "feature_names": feature_names,
        },
    )


def _block_interaction_stats(primary_block: np.ndarray, auxiliary_block: np.ndarray) -> np.ndarray:
    primary_mean = primary_block.mean(axis=1)
    auxiliary_mean = auxiliary_block.mean(axis=1)
    primary_abs_mean = np.abs(primary_block).mean(axis=1)
    auxiliary_abs_mean = np.abs(auxiliary_block).mean(axis=1)
    primary_rms = np.sqrt(np.mean(primary_block * primary_block, axis=1))
    auxiliary_rms = np.sqrt(np.mean(auxiliary_block * auxiliary_block, axis=1))
    primary_maxabs = np.max(np.abs(primary_block), axis=1)
    auxiliary_maxabs = np.max(np.abs(auxiliary_block), axis=1)
    return np.stack(
        [
            primary_mean * auxiliary_mean,
            primary_abs_mean * auxiliary_abs_mean,
            primary_rms * auxiliary_rms,
            primary_maxabs * auxiliary_maxabs,
        ],
        axis=1,
    )


def _group_metadata(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(group["name"]),
            "kind": str(group["kind"]),
            "prefixes": list(group["prefixes"]),
            "feature_count": int(len(group["indices"])),
            "indices": [int(index) for index in group["indices"]],
        }
        for group in groups
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_artifact, validation_artifact, report = fit_compressed_span_block_interaction_expert(
        train_feature_dir=args.train_feature_dir,
        validation_feature_dir=args.validation_feature_dir,
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
