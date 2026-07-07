from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from blockcipher_nd.cli.fit_bucket_conditioned_feature_expert import _validate_feature_score_alignment
from blockcipher_nd.cli.fit_compressed_feature_expert import (
    _load_feature_dir,
    _metrics,
    _score_metadata,
    _sigmoid,
    _validate_feature_dirs,
)
from blockcipher_nd.cli.fit_residual_correction_feature_expert import _base_logit_mean, _residual_focus_weights
from blockcipher_nd.cli.fit_state_token_residual_expert import (
    _shuffle_token_coordinates,
    _standardization_stats,
    _validate_state_token_feature_view,
)
from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    evaluate_frozen_score_ensemble,
    load_score_artifact,
    write_score_artifact,
)
from blockcipher_nd.models.structure.spn.present_state_token_residual import (
    PresentStateTokenResidualDistinguisher,
)


DEFAULT_MODEL_KEY = "present_state_token_residual_correction"
DEFAULT_EXPERT_FAMILY = "state_token_residual_graph"
DEFAULT_CANDIDATE_STATUS = "state_token_residual_correction_screen"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a PRESENT state-token additive logit correction on top of two "
            "frozen base score artifacts."
        )
    )
    parser.add_argument("--train-feature-dir", required=True, type=Path)
    parser.add_argument("--validation-feature-dir", required=True, type=Path)
    parser.add_argument("--train-base-artifacts", nargs=2, required=True, type=Path)
    parser.add_argument("--validation-base-artifacts", nargs=2, required=True, type=Path)
    parser.add_argument("--output-validation-dir", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-train-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--hidden-bits", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--residual-focus-fraction",
        type=float,
        default=0.0,
        help="If positive, up-weight train rows with the largest frozen-base residual loss.",
    )
    parser.add_argument(
        "--residual-focus-background-weight",
        type=float,
        default=0.1,
        help="Background train-row weight when --residual-focus-fraction is positive.",
    )
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--shuffle-train-labels", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--shuffle-token-coordinates", action="store_true")
    parser.add_argument("--token-coordinate-shuffle-seed", type=int, default=0)
    parser.add_argument("--no-zero-init-correction-head", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--expert-family", default=DEFAULT_EXPERT_FAMILY)
    parser.add_argument("--candidate-status", default=DEFAULT_CANDIDATE_STATUS)
    return parser.parse_args(argv)


def fit_state_token_residual_correction_expert(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    train_base_artifact_dirs: list[Path],
    validation_base_artifact_dirs: list[Path],
    steps: int = 200,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    batch_size: int = 256,
    token_dim: int = 32,
    hidden_bits: int = 64,
    dropout: float = 0.0,
    residual_focus_fraction: float = 0.0,
    residual_focus_background_weight: float = 0.1,
    standardize: bool = True,
    shuffle_train_labels: bool = False,
    shuffle_seed: int = 0,
    shuffle_token_coordinates: bool = False,
    token_coordinate_shuffle_seed: int = 0,
    zero_initialize_correction_head: bool = True,
    seed: int = 0,
    run_id: str = "",
    model_key: str = DEFAULT_MODEL_KEY,
    expert_family: str = DEFAULT_EXPERT_FAMILY,
    candidate_status: str = DEFAULT_CANDIDATE_STATUS,
) -> tuple[EnsembleScoreArtifact, EnsembleScoreArtifact, dict[str, Any]]:
    _validate_args(
        train_base_artifact_dirs=train_base_artifact_dirs,
        validation_base_artifact_dirs=validation_base_artifact_dirs,
        steps=steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        token_dim=token_dim,
        hidden_bits=hidden_bits,
        residual_focus_fraction=residual_focus_fraction,
        residual_focus_background_weight=residual_focus_background_weight,
    )

    train = _load_feature_dir(train_feature_dir)
    validation = _load_feature_dir(validation_feature_dir)
    _validate_feature_dirs(train, validation)
    _validate_state_token_feature_view(train, validation)
    train_base_artifacts = [load_score_artifact(path) for path in train_base_artifact_dirs]
    validation_base_artifacts = [load_score_artifact(path) for path in validation_base_artifact_dirs]
    for index, artifact in enumerate(train_base_artifacts):
        _validate_feature_score_alignment(train, artifact, split=f"train base artifact {index}")
    for index, artifact in enumerate(validation_base_artifacts):
        _validate_feature_score_alignment(validation, artifact, split=f"validation base artifact {index}")

    train_base_summary = evaluate_frozen_score_ensemble(train_base_artifacts)
    validation_base_summary = evaluate_frozen_score_ensemble(validation_base_artifacts)
    train_base_logits = _base_logit_mean(train_base_artifacts)
    validation_base_logits = _base_logit_mean(validation_base_artifacts)

    train_features = train["features"].astype(np.float32, copy=False)
    validation_features = validation["features"].astype(np.float32, copy=False)
    feature_mean, feature_scale = _standardization_stats(train_features, standardize=standardize)
    train_model_features = ((train_features - feature_mean) / feature_scale).astype(np.float32, copy=False)
    validation_model_features = ((validation_features - feature_mean) / feature_scale).astype(np.float32, copy=False)

    fit_labels = train["labels"].astype(np.float32, copy=True)
    if shuffle_train_labels:
        fit_labels = np.random.default_rng(shuffle_seed).permutation(fit_labels).astype(np.float32, copy=False)
    sample_weights, sample_weight_report = _residual_focus_weights(
        labels=train["labels"].astype(np.float64, copy=False),
        base_logits=train_base_logits,
        residual_focus_fraction=residual_focus_fraction,
        background_weight=residual_focus_background_weight,
    )

    torch.manual_seed(seed)
    model = PresentStateTokenResidualDistinguisher(
        input_bits=int(train_features.shape[1]),
        token_dim=token_dim,
        hidden_bits=hidden_bits,
        dropout=dropout,
    )
    if shuffle_token_coordinates:
        _shuffle_token_coordinates(model, seed=token_coordinate_shuffle_seed)
    if zero_initialize_correction_head:
        _zero_initialize_correction_head(model)
    fit = _fit_correction_model(
        model,
        train_model_features,
        train_base_logits,
        fit_labels,
        sample_weights,
        steps=steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
    )

    train_correction_logits = _predict_correction_logits(model, train_model_features)
    validation_correction_logits = _predict_correction_logits(model, validation_model_features)
    train_corrected_logits = train_base_logits + train_correction_logits
    validation_corrected_logits = validation_base_logits + validation_correction_logits
    train_probabilities = _sigmoid(train_corrected_logits)
    validation_probabilities = _sigmoid(validation_corrected_logits)

    model_order = [str(item.metadata.get("model_key", "")) for item in validation_base_artifacts]
    run_order = [str(item.metadata.get("run_id", "")) for item in validation_base_artifacts]
    common_metadata = {
        "model_key": model_key,
        "expert_family": expert_family,
        "candidate_status": candidate_status,
        "run_id": run_id,
        "feature_fit_split": "train",
        "feature_train_dir": str(train_feature_dir),
        "feature_validation_dir": str(validation_feature_dir),
        "feature_model": "state_token_residual_logit_correction",
        "feature_steps": int(steps),
        "feature_learning_rate": float(learning_rate),
        "feature_weight_decay": float(weight_decay),
        "feature_batch_size": int(batch_size),
        "feature_standardize": bool(standardize),
        "residual_focus_fraction": float(residual_focus_fraction),
        "residual_focus_background_weight": float(residual_focus_background_weight),
        "fit_train_labels_shuffled": bool(shuffle_train_labels),
        "fit_label_shuffle_seed": int(shuffle_seed),
        "token_coordinates_shuffled": bool(shuffle_token_coordinates),
        "token_coordinate_shuffle_seed": int(token_coordinate_shuffle_seed),
        "correction_head_zero_initialized": bool(zero_initialize_correction_head),
        "seed": int(seed),
        "feature_count": int(train_features.shape[1]),
        "selected_span_feature_bits": int(model.selected_span_feature_bits),
        "token_dim": int(model.token_dim),
        "hidden_bits": int(model.hidden_bits),
        "base_model_order": model_order,
        "base_run_order": run_order,
        "base_fusion": "logit_mean",
        "claim_scope": (
            "train-fitted PRESENT state-token additive logit correction over frozen base scores; "
            "validation is held out for final scoring, and this is not remote or formal SPN/PRESENT evidence"
        ),
    }
    train_artifact = EnsembleScoreArtifact(
        labels=train["labels"],
        probabilities=train_probabilities.astype(np.float32, copy=False),
        logits=train_corrected_logits.astype(np.float32, copy=False),
        sample_ids=train["sample_ids"],
        metadata=_score_metadata(train["metadata"], common_metadata, score_split="train"),
    )
    validation_artifact = EnsembleScoreArtifact(
        labels=validation["labels"],
        probabilities=validation_probabilities.astype(np.float32, copy=False),
        logits=validation_corrected_logits.astype(np.float32, copy=False),
        sample_ids=validation["sample_ids"],
        metadata=_score_metadata(validation["metadata"], common_metadata, score_split="validation"),
    )
    train_base_metrics = _metrics(train_artifact.labels, _sigmoid(train_base_logits))
    validation_base_metrics = _metrics(validation_artifact.labels, _sigmoid(validation_base_logits))
    train_metrics = _metrics(train_artifact.labels, train_artifact.probabilities)
    validation_metrics = _metrics(validation_artifact.labels, validation_artifact.probabilities)
    report = {
        "status": "pass",
        "decision": _decision(
            validation_auc=validation_metrics["auc"],
            validation_base_auc=validation_base_metrics["auc"],
            shuffle_train_labels=shuffle_train_labels,
            shuffle_token_coordinates=shuffle_token_coordinates,
        ),
        "model_key": model_key,
        "expert_family": expert_family,
        "train_feature_dir": str(train_feature_dir),
        "validation_feature_dir": str(validation_feature_dir),
        "train_base_artifact_dirs": [str(path) for path in train_base_artifact_dirs],
        "validation_base_artifact_dirs": [str(path) for path in validation_base_artifact_dirs],
        "base_model_order": model_order,
        "base_run_order": run_order,
        "base_fusion": "logit_mean",
        "train_rows": int(len(train_artifact.labels)),
        "validation_rows": int(len(validation_artifact.labels)),
        "feature_view": str(train["metadata"].get("feature_view")),
        "feature_count": int(train_features.shape[1]),
        "selected_span_feature_bits": int(model.selected_span_feature_bits),
        "fit": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "standardize": bool(standardize),
            "seed": int(seed),
            "residual_focus": sample_weight_report,
            "final_loss": float(fit["final_loss"]),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "feature_mean_l2_norm": float(np.linalg.norm(feature_mean.astype(np.float64, copy=False))),
            "feature_scale_min": float(np.min(feature_scale)),
            "feature_scale_max": float(np.max(feature_scale)),
        },
        "label_control": {
            "shuffle_train_labels": bool(shuffle_train_labels),
            "shuffle_seed": int(shuffle_seed),
        },
        "token_coordinate_control": {
            "shuffle_token_coordinates": bool(shuffle_token_coordinates),
            "token_coordinate_shuffle_seed": int(token_coordinate_shuffle_seed),
        },
        "correction_initialization": {
            "zero_initialize_correction_head": bool(zero_initialize_correction_head),
        },
        "train_base_logit_mean_metrics": train_base_metrics,
        "validation_base_logit_mean_metrics": validation_base_metrics,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "delta_train_corrected_vs_base_logit_mean_auc": float(train_metrics["auc"] - train_base_metrics["auc"]),
        "delta_validation_corrected_vs_base_logit_mean_auc": float(
            validation_metrics["auc"] - validation_base_metrics["auc"]
        ),
        "train_base_summary": train_base_summary,
        "validation_base_summary": validation_base_summary,
        "guardrails": [
            "base_scores_must_stay_frozen",
            "correction_fit_split_must_be_train",
            "validation_split_final_score_only",
            "feature_view_must_be_trail_position_stats",
            "strict_negative_mode_required",
            "compare_against_label_shuffle_control",
            "compare_against_token_coordinate_shuffle_control",
        ],
        "claim_scope": common_metadata["claim_scope"],
    }
    return train_artifact, validation_artifact, report


def _validate_args(
    *,
    train_base_artifact_dirs: list[Path],
    validation_base_artifact_dirs: list[Path],
    steps: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    token_dim: int,
    hidden_bits: int,
    residual_focus_fraction: float,
    residual_focus_background_weight: float,
) -> None:
    if len(train_base_artifact_dirs) != 2 or len(validation_base_artifact_dirs) != 2:
        raise ValueError("state-token residual correction requires exactly two base artifacts per split")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if token_dim <= 0:
        raise ValueError("token_dim must be positive")
    if hidden_bits <= 0:
        raise ValueError("hidden_bits must be positive")
    if residual_focus_fraction < 0.0 or residual_focus_fraction >= 1.0:
        raise ValueError("residual_focus_fraction must be in [0, 1)")
    if residual_focus_background_weight < 0.0 or residual_focus_background_weight > 1.0:
        raise ValueError("residual_focus_background_weight must be in [0, 1]")


def _decision(
    *,
    validation_auc: float,
    validation_base_auc: float,
    shuffle_train_labels: bool,
    shuffle_token_coordinates: bool,
) -> str:
    if shuffle_token_coordinates:
        return "state_token_residual_correction_token_coordinate_shuffle_control"
    if shuffle_train_labels:
        return "state_token_residual_correction_shuffle_train_labels_control"
    if validation_auc > validation_base_auc:
        return "state_token_residual_correction_local_candidate_needs_controls"
    return "state_token_residual_correction_diagnostic_no_base_gain"


def _zero_initialize_correction_head(model: PresentStateTokenResidualDistinguisher) -> None:
    output = model.classifier[-1]
    if not isinstance(output, nn.Linear) or output.out_features != 1:
        raise TypeError("expected final state-token classifier layer to be a single-output Linear")
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)


def _fit_correction_model(
    model: PresentStateTokenResidualDistinguisher,
    features: np.ndarray,
    base_logits: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    x = torch.as_tensor(features, dtype=torch.float32)
    base = torch.as_tensor(base_logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    weights = torch.as_tensor(sample_weights, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator().manual_seed(seed)
    final_loss = 0.0
    model.train()
    for _ in range(steps):
        order = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            correction = model(x[indices]).squeeze(1)
            loss_rows = criterion(base[indices] + correction, y[indices]) * weights[indices]
            loss = loss_rows.mean()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    return {"final_loss": final_loss}


def _predict_correction_logits(model: PresentStateTokenResidualDistinguisher, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(features, dtype=torch.float32)).squeeze(1)
    return logits.detach().cpu().numpy().astype(np.float64, copy=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_artifact, validation_artifact, report = fit_state_token_residual_correction_expert(
        train_feature_dir=args.train_feature_dir,
        validation_feature_dir=args.validation_feature_dir,
        train_base_artifact_dirs=args.train_base_artifacts,
        validation_base_artifact_dirs=args.validation_base_artifacts,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        token_dim=args.token_dim,
        hidden_bits=args.hidden_bits,
        dropout=args.dropout,
        residual_focus_fraction=args.residual_focus_fraction,
        residual_focus_background_weight=args.residual_focus_background_weight,
        standardize=not args.no_standardize,
        shuffle_train_labels=args.shuffle_train_labels,
        shuffle_seed=args.shuffle_seed,
        shuffle_token_coordinates=args.shuffle_token_coordinates,
        token_coordinate_shuffle_seed=args.token_coordinate_shuffle_seed,
        zero_initialize_correction_head=not args.no_zero_init_correction_head,
        seed=args.seed,
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
