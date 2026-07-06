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
from blockcipher_nd.cli.fit_compressed_span_block_interaction_expert import (
    _feature_names,
    _group_metadata,
    _semantic_groups,
)
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, write_score_artifact


DEFAULT_MODEL_KEY = "compressed_span_learned_low_rank_interaction_expert"
DEFAULT_EXPERT_FAMILY = "compressed_spn_span_learned_low_rank_interaction_summary"
DEFAULT_CANDIDATE_STATUS = "compressed_span_learned_low_rank_interaction_screen"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a small learned low-rank block interaction expert on compressed span-summary features."
    )
    parser.add_argument("--train-feature-dir", required=True, type=Path)
    parser.add_argument("--validation-feature-dir", required=True, type=Path)
    parser.add_argument("--output-validation-dir", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-train-dir", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--projection-init", choices=["random", "svd"], default="random")
    parser.add_argument("--freeze-projections", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--shuffle-train-labels", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--expert-family", default=DEFAULT_EXPERT_FAMILY)
    parser.add_argument("--candidate-status", default=DEFAULT_CANDIDATE_STATUS)
    return parser.parse_args(argv)


def fit_compressed_span_learned_low_rank_interaction_expert(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    rank: int = 1,
    steps: int = 2000,
    learning_rate: float = 0.01,
    weight_decay: float = 0.001,
    projection_init: str = "random",
    freeze_projections: bool = False,
    standardize: bool = True,
    shuffle_train_labels: bool = False,
    shuffle_seed: int = 0,
    seed: int = 0,
    run_id: str = "",
    model_key: str = DEFAULT_MODEL_KEY,
    expert_family: str = DEFAULT_EXPERT_FAMILY,
    candidate_status: str = DEFAULT_CANDIDATE_STATUS,
) -> tuple[EnsembleScoreArtifact, EnsembleScoreArtifact, dict[str, Any]]:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if projection_init not in {"random", "svd"}:
        raise ValueError("projection_init must be random or svd")

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

    train_features = train["features"].astype(np.float32, copy=False)
    validation_features = validation["features"].astype(np.float32, copy=False)
    feature_mean, feature_scale = _standardization_stats(train_features, standardize=standardize)
    train_model_features = ((train_features - feature_mean) / feature_scale).astype(np.float32, copy=False)
    validation_model_features = ((validation_features - feature_mean) / feature_scale).astype(np.float32, copy=False)

    fit_labels = train["labels"].astype(np.float32, copy=True)
    if shuffle_train_labels:
        fit_labels = np.random.default_rng(shuffle_seed).permutation(fit_labels).astype(np.float32, copy=False)

    torch.manual_seed(seed)
    model = _LearnedLowRankBlockInteractionModel(
        raw_feature_count=int(train_features.shape[1]),
        primary_groups=primary_groups,
        auxiliary_groups=auxiliary_groups,
        rank=rank,
    )
    if projection_init == "svd":
        _initialize_projection_weights_from_svd(model, train_model_features)
    if freeze_projections:
        _set_projection_trainability(model, trainable=False)
    fit = _fit_model(
        model,
        train_model_features,
        fit_labels,
        steps=steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    train_logits, train_interactions = _predict(model, train_model_features)
    validation_logits, validation_interactions = _predict(model, validation_model_features)
    train_probabilities = _sigmoid(train_logits)
    validation_probabilities = _sigmoid(validation_logits)

    feature_model = "raw_plus_learned_semantic_low_rank_block_interactions"
    interaction_count = int(train_interactions.shape[1])
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
        "feature_weight_decay": float(weight_decay),
        "projection_initialization": projection_init,
        "freeze_projections": bool(freeze_projections),
        "feature_standardize": bool(standardize),
        "fit_train_labels_shuffled": bool(shuffle_train_labels),
        "fit_label_shuffle_seed": int(shuffle_seed),
        "seed": int(seed),
        "feature_count": int(train_features.shape[1] + interaction_count),
        "raw_feature_count": int(train_features.shape[1]),
        "primary_group_count": int(len(primary_groups)),
        "auxiliary_group_count": int(len(auxiliary_groups)),
        "block_pair_count": int(len(primary_groups) * len(auxiliary_groups)),
        "low_rank_projection_rank": int(rank),
        "learned_low_rank_interaction_count": interaction_count,
        "projection_parameter_count": int(_projection_parameter_count(model)),
        "trainable_projection_parameter_count": int(_trainable_projection_parameter_count(model)),
        "block_groups": _group_metadata(groups),
        "claim_scope": (
            "train-fitted learned compressed SPN span-summary low-rank block interaction expert diagnostic only; "
            "low-rank projections and final interaction weights are learned on the train split, validation is "
            "held out for final scoring, and this is not remote or formal SPN/PRESENT evidence"
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
            "compressed_span_learned_low_rank_interaction_shuffle_train_labels_control"
            if shuffle_train_labels
            else "compressed_span_learned_low_rank_interaction_local_diagnostic"
        ),
        "model_key": model_key,
        "expert_family": expert_family,
        "train_feature_dir": str(train_feature_dir),
        "validation_feature_dir": str(validation_feature_dir),
        "train_rows": int(len(train_artifact.labels)),
        "validation_rows": int(len(validation_artifact.labels)),
        "feature_model": feature_model,
        "feature_count": int(train_features.shape[1] + interaction_count),
        "raw_feature_count": int(train_features.shape[1]),
        "primary_group_count": int(len(primary_groups)),
        "auxiliary_group_count": int(len(auxiliary_groups)),
        "block_pair_count": int(len(primary_groups) * len(auxiliary_groups)),
        "low_rank_projection_rank": int(rank),
        "learned_low_rank_interaction_count": interaction_count,
        "projection_initialization": projection_init,
        "freeze_projections": bool(freeze_projections),
        "learned_low_rank_interaction_feature_names": _interaction_feature_names(
            primary_groups,
            auxiliary_groups,
            rank=rank,
        ),
        "block_groups": _group_metadata(groups),
        "fit": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "standardize": bool(standardize),
            "seed": int(seed),
            "final_loss": float(fit["final_loss"]),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "projection_parameter_count": int(_projection_parameter_count(model)),
            "trainable_projection_parameter_count": int(_trainable_projection_parameter_count(model)),
            "raw_feature_mean_l2_norm": float(np.linalg.norm(feature_mean.astype(np.float64, copy=False))),
            "raw_feature_scale_min": float(np.min(feature_scale)),
            "raw_feature_scale_max": float(np.max(feature_scale)),
        },
        "label_control": {
            "shuffle_train_labels": bool(shuffle_train_labels),
            "shuffle_seed": int(shuffle_seed),
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "guardrails": [
            "fit_split_must_be_train",
            "validation_split_final_score_only",
            "strict_negative_mode_required",
            "compare_against_full_summary_flat_logistic",
            "compare_against_semantic_block_stat_interaction_expert",
            "compare_against_unsupervised_low_rank_interaction_expert",
            "compare_svd_frozen_against_unsupervised_low_rank_interaction_expert",
            "compare_against_shuffle_train_labels_control",
        ],
        "claim_scope": common_metadata["claim_scope"],
    }
    return train_artifact, validation_artifact, report


class _LearnedLowRankBlockInteractionModel(nn.Module):
    def __init__(
        self,
        *,
        raw_feature_count: int,
        primary_groups: list[dict[str, Any]],
        auxiliary_groups: list[dict[str, Any]],
        rank: int,
    ) -> None:
        super().__init__()
        self.raw_feature_count = int(raw_feature_count)
        self.rank = int(rank)
        self.primary_groups = [
            {"name": str(group["name"]), "indices": [int(index) for index in group["indices"]]}
            for group in primary_groups
        ]
        self.auxiliary_groups = [
            {"name": str(group["name"]), "indices": [int(index) for index in group["indices"]]}
            for group in auxiliary_groups
        ]
        self.primary_projections = nn.ModuleList(
            nn.Linear(len(group["indices"]), rank, bias=False) for group in self.primary_groups
        )
        self.auxiliary_projections = nn.ModuleList(
            nn.Linear(len(group["indices"]), rank, bias=False) for group in self.auxiliary_groups
        )
        interaction_count = len(self.primary_groups) * len(self.auxiliary_groups) * rank * rank
        self.classifier = nn.Linear(raw_feature_count + interaction_count, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        primary_latents = [
            projection(features[:, group["indices"]])
            for projection, group in zip(self.primary_projections, self.primary_groups, strict=True)
        ]
        auxiliary_latents = [
            projection(features[:, group["indices"]])
            for projection, group in zip(self.auxiliary_projections, self.auxiliary_groups, strict=True)
        ]
        interactions = [
            (primary[:, :, None] * auxiliary[:, None, :]).reshape(features.shape[0], -1)
            for primary in primary_latents
            for auxiliary in auxiliary_latents
        ]
        interaction_features = torch.cat(interactions, dim=1)
        logits = self.classifier(torch.cat([features, interaction_features], dim=1)).squeeze(1)
        return logits, interaction_features


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


def _fit_model(
    model: _LearnedLowRankBlockInteractionModel,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, float]:
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    final_loss = 0.0
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    return {"final_loss": final_loss}


def _initialize_projection_weights_from_svd(
    model: _LearnedLowRankBlockInteractionModel,
    train_features: np.ndarray,
) -> None:
    for projection, group in zip(model.primary_projections, model.primary_groups, strict=True):
        _copy_svd_components_to_projection(
            projection,
            train_features[:, group["indices"]],
            rank=model.rank,
        )
    for projection, group in zip(model.auxiliary_projections, model.auxiliary_groups, strict=True):
        _copy_svd_components_to_projection(
            projection,
            train_features[:, group["indices"]],
            rank=model.rank,
        )


def _copy_svd_components_to_projection(projection: nn.Linear, train_block: np.ndarray, *, rank: int) -> None:
    actual_rank = min(rank, int(train_block.shape[0]), int(train_block.shape[1]))
    if actual_rank <= 0:
        raise ValueError("SVD projection initialization requires a non-empty block")
    _, _, vh = np.linalg.svd(train_block.astype(np.float64, copy=False), full_matrices=False)
    components = np.zeros((train_block.shape[1], rank), dtype=np.float32)
    components[:, :actual_rank] = vh[:actual_rank].T.astype(np.float32, copy=False)
    with torch.no_grad():
        projection.weight.copy_(torch.as_tensor(components.T, dtype=projection.weight.dtype))


def _set_projection_trainability(model: _LearnedLowRankBlockInteractionModel, *, trainable: bool) -> None:
    for projection in [*model.primary_projections, *model.auxiliary_projections]:
        for parameter in projection.parameters():
            parameter.requires_grad = trainable


def _projection_parameter_count(model: _LearnedLowRankBlockInteractionModel) -> int:
    return int(
        sum(
            parameter.numel()
            for projection in [*model.primary_projections, *model.auxiliary_projections]
            for parameter in projection.parameters()
        )
    )


def _trainable_projection_parameter_count(model: _LearnedLowRankBlockInteractionModel) -> int:
    return int(
        sum(
            parameter.numel()
            for projection in [*model.primary_projections, *model.auxiliary_projections]
            for parameter in projection.parameters()
            if parameter.requires_grad
        )
    )


def _predict(model: _LearnedLowRankBlockInteractionModel, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits, interaction_features = model(torch.as_tensor(features, dtype=torch.float32))
    return (
        logits.detach().cpu().numpy().astype(np.float64, copy=False),
        interaction_features.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _interaction_feature_names(
    primary_groups: list[dict[str, Any]],
    auxiliary_groups: list[dict[str, Any]],
    *,
    rank: int,
) -> list[str]:
    names: list[str] = []
    for primary_group in primary_groups:
        for auxiliary_group in auxiliary_groups:
            for primary_rank in range(rank):
                for auxiliary_rank in range(rank):
                    names.append(
                        "learned_low_rank_"
                        f"{primary_group['name']}_r{primary_rank}__"
                        f"{auxiliary_group['name']}_r{auxiliary_rank}"
                    )
    return names


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_artifact, validation_artifact, report = fit_compressed_span_learned_low_rank_interaction_expert(
        train_feature_dir=args.train_feature_dir,
        validation_feature_dir=args.validation_feature_dir,
        rank=args.rank,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        projection_init=args.projection_init,
        freeze_projections=args.freeze_projections,
        standardize=not args.no_standardize,
        shuffle_train_labels=args.shuffle_train_labels,
        shuffle_seed=args.shuffle_seed,
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
