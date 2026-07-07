from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from blockcipher_nd.cli.fit_compressed_feature_expert import (
    _load_feature_dir,
    _metrics,
    _sigmoid,
    _validate_feature_dirs,
)
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, write_score_artifact
from blockcipher_nd.models.structure.spn.present_state_token_residual import (
    PresentStateTokenResidualDistinguisher,
)


DEFAULT_MODEL_KEY = "present_state_token_residual"
DEFAULT_EXPERT_FAMILY = "state_token_residual_graph"
DEFAULT_CANDIDATE_STATUS = "state_token_residual_feature_artifact_screen"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a PRESENT state-token residual expert on trail_position_stats feature artifacts."
    )
    parser.add_argument("--train-feature-dir", required=True, type=Path)
    parser.add_argument("--validation-feature-dir", required=True, type=Path)
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
    parser.add_argument("--coordinate-mode", choices=["additive", "film"], default="additive")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--shuffle-train-labels", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--shuffle-token-coordinates", action="store_true")
    parser.add_argument("--token-coordinate-shuffle-seed", type=int, default=0)
    parser.add_argument("--drop-token-coordinates", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--expert-family", default=DEFAULT_EXPERT_FAMILY)
    parser.add_argument("--candidate-status", default=DEFAULT_CANDIDATE_STATUS)
    return parser.parse_args(argv)


def fit_state_token_residual_expert(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    steps: int = 200,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    batch_size: int = 256,
    token_dim: int = 32,
    hidden_bits: int = 64,
    dropout: float = 0.0,
    coordinate_mode: str = "additive",
    standardize: bool = True,
    shuffle_train_labels: bool = False,
    shuffle_seed: int = 0,
    shuffle_token_coordinates: bool = False,
    token_coordinate_shuffle_seed: int = 0,
    drop_token_coordinates: bool = False,
    seed: int = 0,
    run_id: str = "",
    model_key: str = DEFAULT_MODEL_KEY,
    expert_family: str = DEFAULT_EXPERT_FAMILY,
    candidate_status: str = DEFAULT_CANDIDATE_STATUS,
) -> tuple[EnsembleScoreArtifact, EnsembleScoreArtifact, dict[str, Any]]:
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
    if shuffle_token_coordinates and drop_token_coordinates:
        raise ValueError("shuffle_token_coordinates and drop_token_coordinates are mutually exclusive")

    train = _load_feature_dir(train_feature_dir)
    validation = _load_feature_dir(validation_feature_dir)
    _validate_feature_dirs(train, validation)
    _validate_state_token_feature_view(train, validation)

    train_features = train["features"].astype(np.float32, copy=False)
    validation_features = validation["features"].astype(np.float32, copy=False)
    feature_mean, feature_scale = _standardization_stats(train_features, standardize=standardize)
    train_model_features = ((train_features - feature_mean) / feature_scale).astype(np.float32, copy=False)
    validation_model_features = ((validation_features - feature_mean) / feature_scale).astype(np.float32, copy=False)

    fit_labels = train["labels"].astype(np.float32, copy=True)
    if shuffle_train_labels:
        fit_labels = np.random.default_rng(shuffle_seed).permutation(fit_labels).astype(np.float32, copy=False)

    torch.manual_seed(seed)
    model = PresentStateTokenResidualDistinguisher(
        input_bits=int(train_features.shape[1]),
        token_dim=token_dim,
        hidden_bits=hidden_bits,
        dropout=dropout,
        coordinate_mode=coordinate_mode,
    )
    if shuffle_token_coordinates:
        _shuffle_token_coordinates(model, seed=token_coordinate_shuffle_seed)
    if drop_token_coordinates:
        _drop_token_coordinates(model)
    fit = _fit_model(
        model,
        train_model_features,
        fit_labels,
        steps=steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
    )
    train_logits = _predict_logits(model, train_model_features)
    validation_logits = _predict_logits(model, validation_model_features)
    train_probabilities = _sigmoid(train_logits)
    validation_probabilities = _sigmoid(validation_logits)

    common_metadata = {
        "model_key": model_key,
        "expert_family": expert_family,
        "candidate_status": candidate_status,
        "run_id": run_id,
        "feature_fit_split": "train",
        "feature_train_dir": str(train_feature_dir),
        "feature_validation_dir": str(validation_feature_dir),
        "feature_model": "state_token_residual",
        "feature_steps": int(steps),
        "feature_learning_rate": float(learning_rate),
        "feature_weight_decay": float(weight_decay),
        "feature_batch_size": int(batch_size),
        "feature_standardize": bool(standardize),
        "coordinate_mode": coordinate_mode,
        "fit_train_labels_shuffled": bool(shuffle_train_labels),
        "fit_label_shuffle_seed": int(shuffle_seed),
        "token_coordinates_shuffled": bool(shuffle_token_coordinates),
        "token_coordinate_shuffle_seed": int(token_coordinate_shuffle_seed),
        "token_coordinates_dropped": bool(drop_token_coordinates),
        "seed": int(seed),
        "feature_count": int(train_features.shape[1]),
        "selected_span_feature_bits": int(model.selected_span_feature_bits),
        "token_dim": int(model.token_dim),
        "hidden_bits": int(model.hidden_bits),
        "claim_scope": (
            "train-fitted PRESENT state-token residual feature-artifact diagnostic only; "
            "validation is held out for final scoring, and this is not remote or formal "
            "SPN/PRESENT evidence"
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
    report = {
        "status": "pass",
        "decision": _decision(
            shuffle_train_labels=shuffle_train_labels,
            shuffle_token_coordinates=shuffle_token_coordinates,
            drop_token_coordinates=drop_token_coordinates,
        ),
        "model_key": model_key,
        "expert_family": expert_family,
        "train_feature_dir": str(train_feature_dir),
        "validation_feature_dir": str(validation_feature_dir),
        "train_rows": int(len(train_artifact.labels)),
        "validation_rows": int(len(validation_artifact.labels)),
        "feature_view": str(train["metadata"].get("feature_view")),
        "feature_count": int(train_features.shape[1]),
        "coordinate_mode": coordinate_mode,
        "selected_span_feature_bits": int(model.selected_span_feature_bits),
        "fit": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "standardize": bool(standardize),
            "seed": int(seed),
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
            "drop_token_coordinates": bool(drop_token_coordinates),
        },
        "train_metrics": _metrics(train_artifact.labels, train_artifact.probabilities),
        "validation_metrics": _metrics(validation_artifact.labels, validation_artifact.probabilities),
        "guardrails": [
            "fit_split_must_be_train",
            "validation_split_final_score_only",
            "feature_view_must_be_trail_position_stats",
            "strict_negative_mode_required",
            "compare_against_compressed_feature_logistic_expert",
            "compare_against_label_shuffle_control",
            "compare_against_token_coordinate_shuffle_control",
            "compare_against_token_coordinate_drop_control",
        ],
        "claim_scope": common_metadata["claim_scope"],
    }
    return train_artifact, validation_artifact, report


def _validate_state_token_feature_view(train: dict[str, Any], validation: dict[str, Any]) -> None:
    if train["metadata"].get("feature_view") != "trail_position_stats":
        raise ValueError("feature_view must be trail_position_stats")
    if validation["metadata"].get("feature_view") != "trail_position_stats":
        raise ValueError("feature_view must be trail_position_stats")
    if int(train["features"].shape[1]) != 3708:
        raise ValueError(f"state-token residual expert expects 3708 features, got {train['features'].shape[1]}")


def _standardization_stats(features: np.ndarray, *, standardize: bool) -> tuple[np.ndarray, np.ndarray]:
    if not standardize:
        return (
            np.zeros(features.shape[1], dtype=np.float32),
            np.ones(features.shape[1], dtype=np.float32),
        )
    mean = features.mean(axis=0).astype(np.float32, copy=False)
    scale = features.std(axis=0).astype(np.float32, copy=False)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32, copy=False)
    return mean, scale


def _decision(*, shuffle_train_labels: bool, shuffle_token_coordinates: bool, drop_token_coordinates: bool) -> str:
    if drop_token_coordinates:
        return "state_token_residual_drop_token_coordinates_control"
    if shuffle_token_coordinates:
        return "state_token_residual_token_coordinate_shuffle_control"
    if shuffle_train_labels:
        return "state_token_residual_shuffle_train_labels_control"
    return "state_token_residual_feature_artifact_local_diagnostic"


def _shuffle_token_coordinates(model: PresentStateTokenResidualDistinguisher, *, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(model.selected_span_feature_bits, generator=generator)
    model.span_family_ids.copy_(model.span_family_ids[permutation])
    model.span_depth_ids.copy_(model.span_depth_ids[permutation])
    model.span_word_ids.copy_(model.span_word_ids[permutation])
    model.span_cell_ids.copy_(model.span_cell_ids[permutation])


def _drop_token_coordinates(model: PresentStateTokenResidualDistinguisher) -> None:
    for embedding in [
        model.family_embedding,
        model.depth_embedding,
        model.word_embedding,
        model.cell_embedding,
    ]:
        nn.init.zeros_(embedding.weight)
        embedding.weight.requires_grad_(False)


def _fit_model(
    model: PresentStateTokenResidualDistinguisher,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    final_loss = 0.0
    model.train()
    for _ in range(steps):
        order = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[indices]).squeeze(1)
            loss = criterion(logits, y[indices])
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    return {"final_loss": final_loss}


def _predict_logits(model: PresentStateTokenResidualDistinguisher, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(features, dtype=torch.float32)).squeeze(1)
    return logits.detach().cpu().numpy().astype(np.float64, copy=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_artifact, validation_artifact, report = fit_state_token_residual_expert(
        train_feature_dir=args.train_feature_dir,
        validation_feature_dir=args.validation_feature_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        token_dim=args.token_dim,
        hidden_bits=args.hidden_bits,
        dropout=args.dropout,
        coordinate_mode=args.coordinate_mode,
        standardize=not args.no_standardize,
        shuffle_train_labels=args.shuffle_train_labels,
        shuffle_seed=args.shuffle_seed,
        shuffle_token_coordinates=args.shuffle_token_coordinates,
        token_coordinate_shuffle_seed=args.token_coordinate_shuffle_seed,
        drop_token_coordinates=args.drop_token_coordinates,
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
